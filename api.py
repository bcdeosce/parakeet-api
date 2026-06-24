import os
import asyncio
import tempfile
import logging
import time
import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse
import torch
import nemo.collections.asr as nemo_asr

# ==================== CONFIGURAÇÕES COM VARIÁVEIS DE AMBIENTE ====================
PARAKEET_MODEL_FILE = os.getenv("PARAKEET_MODEL_FILE", "/app/model_cache/parakeet-tdt-0.6b-v3.nemo")
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.03"))          # tempo para agrupar lote
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))             # tamanho máximo do lote
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TEMP_DIR = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"

# Novas variáveis para paralelismo (já com valores padrão)
NUM_PREPROCESS_WORKERS = int(os.getenv("NUM_PREPROCESS_WORKERS", "2"))  # workers de CPU
FFMPEG_THREADS = int(os.getenv("FFMPEG_THREADS", "2"))                  # threads por FFmpeg

# Configuração do logging com mais detalhes
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("stt-api")

# ==================== CARREGAMENTO DO MODELO ====================
logger.info(f"=== INICIANDO API ===")
logger.info(f"Dispositivo: {DEVICE}")
logger.info(f"Arquivo do modelo: {PARAKEET_MODEL_FILE}")
logger.info(f"Tamanho máximo do lote: {MAX_BATCH_SIZE}")
logger.info(f"Timeout para formação de lote: {BATCH_TIMEOUT}s")
logger.info(f"Número de workers de pré-processamento: {NUM_PREPROCESS_WORKERS}")
logger.info(f"Threads por FFmpeg: {FFMPEG_THREADS}")
logger.info(f"Diretório temporário: {TEMP_DIR}")

try:
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=PARAKEET_MODEL_FILE)
    model = model.to(DEVICE)
    model.eval()
    MODEL_LOADED = True
    logger.info("✅ Modelo Parakeet carregado com sucesso!")
except Exception as e:
    MODEL_LOADED = False
    logger.error(f"❌ Falha ao carregar o modelo: {e}")
    raise

# ==================== FILAS ====================
preprocess_queue = asyncio.Queue()
inference_queue = asyncio.Queue()

# ==================== WORKER DE PRÉ-PROCESSAMENTO (CPU) ====================
async def preprocessor_worker(worker_id: int):
    """
    Converte .webm para .wav (16kHz, mono, PCM) usando FFmpeg com múltiplas threads.
    """
    logger.info(f"🧵 Worker de pré-processamento #{worker_id} iniciado e aguardando tarefas.")
    while True:
        try:
            # Pega um item da fila
            future, raw_path, suffix = await preprocess_queue.get()
            logger.debug(f"[Worker {worker_id}] Iniciando processamento de {raw_path} (suffix={suffix})")

            wav_path = raw_path
            start_conv = time.perf_counter()

            if suffix == ".webm":
                # Cria arquivo .wav temporário
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR) as wav_tmp:
                    wav_path = wav_tmp.name

                # Comando FFmpeg otimizado com -threads
                cmd = [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-threads", str(FFMPEG_THREADS),
                    "-y", "-i", raw_path,
                    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path
                ]
                logger.debug(f"[Worker {worker_id}] Executando FFmpeg: {' '.join(cmd)}")
                proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)

                if proc.returncode != 0:
                    raise RuntimeError(f"FFmpeg falhou (código {proc.returncode}): {proc.stderr}")

                # Remove o original (webm) para liberar espaço
                try:
                    os.unlink(raw_path)
                except Exception:
                    pass

                conv_time = (time.perf_counter() - start_conv) * 1000
                logger.info(f"⚡ [Worker {worker_id}] Conversão concluída em {conv_time:.1f}ms: {raw_path} -> {wav_path}")

            else:
                # Para outros formatos, usamos o caminho original
                logger.debug(f"[Worker {worker_id}] Arquivo {raw_path} não é webm, usando diretamente.")

            # Coloca na fila de inferência
            await inference_queue.put((future, wav_path))
            logger.debug(f"[Worker {worker_id}] Áudio colocado na fila de inferência. Tamanho atual da fila de inferência: {inference_queue.qsize()}")

        except Exception as e:
            logger.exception(f"❌ [Worker {worker_id}] Erro ao processar {raw_path}: {e}")
            if not future.done():
                future.set_exception(e)
            # Limpeza
            try:
                if os.path.exists(raw_path):
                    os.unlink(raw_path)
            except Exception:
                pass
            if wav_path != raw_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

# ==================== WORKER DE INFERÊNCIA (GPU) ====================
async def inference_worker():
    """
    Pega lotes da fila de inferência e chama model.transcribe(lista_de_paths).
    """
    logger.info("🚀 Worker de inferência iniciado e aguardando lotes.")
    while True:
        # Monta o lote
        batch = []
        first_future, first_path = await inference_queue.get()
        batch.append((first_future, first_path))
        deadline = asyncio.get_event_loop().time() + BATCH_TIMEOUT

        while len(batch) < MAX_BATCH_SIZE and asyncio.get_event_loop().time() < deadline:
            try:
                future, path = await asyncio.wait_for(inference_queue.get(), timeout=0.005)
                batch.append((future, path))
            except asyncio.TimeoutError:
                break

        # Log do lote formado
        logger.info(f"📦 Lote formado com {len(batch)} áudios (tamanho máximo: {MAX_BATCH_SIZE})")

        futures = [item[0] for item in batch]
        paths = [item[1] for item in batch]

        try:
            start_inf = time.perf_counter()
            logger.info(f"🧠 Enviando lote para o modelo Parakeet...")
            hypotheses = model.transcribe(paths)
            results = [hyp.text for hyp in hypotheses]
            inf_time = (time.perf_counter() - start_inf) * 1000
            logger.info(f"✅ Inferência concluída em {inf_time:.1f}ms para {len(paths)} áudios (média: {inf_time/len(paths):.1f}ms/áudio)")

            # Entrega os resultados
            for future, text in zip(futures, results):
                future.set_result(text)

        except Exception as e:
            logger.exception(f"❌ Erro na inferência em lote: {e}")
            for future in futures:
                if not future.done():
                    future.set_exception(e)
        finally:
            # Limpeza dos arquivos temporários
            for _, path in batch:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                        logger.debug(f"🗑️ Arquivo removido: {path}")
                except Exception:
                    pass

# ==================== FASTAPI LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicia múltiplos workers de pré-processamento
    for i in range(NUM_PREPROCESS_WORKERS):
        asyncio.create_task(preprocessor_worker(i + 1))
        logger.info(f"✅ Worker de pré-processamento #{i+1} agendado.")
    
    # Inicia o worker de inferência
    asyncio.create_task(inference_worker())
    logger.info("✅ Worker de inferência agendado.")
    
    logger.info(f"🎯 Sistema pronto! {NUM_PREPROCESS_WORKERS} preprocessadores + 1 inferência.")
    yield

app = FastAPI(lifespan=lifespan, title="STT API (Parakeet)")

# ==================== HEALTH CHECKS ====================
@app.get("/started")
async def started():
    return Response(status_code=200, content="started")

@app.get("/ready")
async def ready():
    return Response(status_code=200, content="ready") if MODEL_LOADED else Response(status_code=503, content="loading")

@app.get("/live")
async def live():
    return Response(status_code=200, content="alive")

@app.get("/health")
async def health():
    return Response(status_code=200, content="healthy")

# ==================== ENDPOINT PRINCIPAL ====================
@app.post("/v1/listen")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Recebe um arquivo de áudio, coloca na fila de pré-processamento e aguarda o resultado.
    """
    original_suffix = os.path.splitext(file.filename)[1].lower() or ".wav"
    logger.info(f"📥 Requisição recebida: {file.filename} (suffix={original_suffix})")

    # Salva o arquivo enviado
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix, dir=TEMP_DIR) as tmp:
        content = await file.read()
        tmp.write(content)
        raw_path = tmp.name
    logger.debug(f"💾 Arquivo salvo em: {raw_path} ({len(content)} bytes)")

    loop = asyncio.get_event_loop()
    future = loop.create_future()

    # Coloca na fila de pré-processamento
    await preprocess_queue.put((future, raw_path, original_suffix))
    logger.debug(f"📤 Requisição enfileirada. Tamanho da fila de pré-processamento: {preprocess_queue.qsize()}")

    try:
        start_wait = time.perf_counter()
        transcript = await asyncio.wait_for(future, timeout=60.0)
        total_time = (time.perf_counter() - start_wait) * 1000
        logger.info(f"✅ Transcrição concluída em {total_time:.1f}ms: '{transcript[:50]}...'")
        response = {
            "results": {
                "channels": [{
                    "alternatives": [{"transcript": transcript, "confidence": 1.0}]
                }]
            }
        }
        return JSONResponse(content=response)
    except asyncio.TimeoutError:
        logger.error(f"⏰ Timeout para {raw_path}")
        raise HTTPException(status_code=504, detail="Tempo limite excedido")
    except Exception as e:
        logger.exception(f"❌ Erro na requisição: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpeza extra (caso o arquivo ainda exista)
        if os.path.exists(raw_path):
            try:
                os.unlink(raw_path)
            except Exception:
                pass
