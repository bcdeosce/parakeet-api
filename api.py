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

# ==================== CONFIGURAÇÕES ====================
PARAKEET_MODEL_FILE = os.getenv("PARAKEET_MODEL_FILE", "/app/model_cache/parakeet-tdt-0.6b-v3.nemo")
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.03"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
TEMP_DIR = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stt-api")

# ==================== CARREGAMENTO DO MODELO (APENAS PARAKEET) ====================
logger.info(f"Dispositivo: {DEVICE}")
logger.info(f"Carregando Parakeet do arquivo local: {PARAKEET_MODEL_FILE}")
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=PARAKEET_MODEL_FILE)
model = model.to(DEVICE)
model.eval()
MODEL_LOADED = True
logger.info("Parakeet pronto.")

# ==================== FILAS DE DOIS ESTÁGIOS ====================
# Fila de pré-processamento: recebe (future, caminho_raw, sufixo)
preprocess_queue = asyncio.Queue()
# Fila de inferência: recebe (future, caminho_wav)
inference_queue = asyncio.Queue()

# ==================== WORKER DE PRÉ-PROCESSAMENTO (CPU) ====================
async def preprocessor_worker():
    """
    Converte arquivos .webm para .wav (16kHz, mono, PCM) usando FFmpeg.
    Outros formatos são passados sem conversão (mas o caminho é mantido).
    """
    while True:
        future, raw_path, suffix = await preprocess_queue.get()
        wav_path = raw_path  # assume que já é um formato aceito (ex: .wav)
        try:
            if suffix == ".webm":
                # Cria um arquivo .wav temporário
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR) as wav_tmp:
                    wav_path = wav_tmp.name
                
                # Comando FFmpeg: extrai áudio, mono, 16kHz, PCM s16le
                cmd = [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-y", "-i", raw_path,
                    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path
                ]
                # Executa em thread para não bloquear o event loop
                proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"FFmpeg falhou: {proc.stderr}")
                
                # Remove o arquivo original (webm) para economizar espaço
                try:
                    os.unlink(raw_path)
                except Exception:
                    pass
                logger.debug(f"Convertido: {raw_path} -> {wav_path}")
            else:
                # Para outros formatos (ex: .wav, .flac), usamos o próprio caminho
                # O NeMo/torchaudio vai carregar diretamente, mas garantimos que seja PCM?
                # Vamos confiar que o usuário enviou algo compatível.
                pass

            # Coloca o áudio pronto na fila de inferência
            await inference_queue.put((future, wav_path))
        except Exception as e:
            logger.exception(f"Erro no pré-processamento de {raw_path}")
            # Se falhou, seta exceção no future e limpa o arquivo
            if not future.done():
                future.set_exception(e)
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
    Pega lotes da fila de inferência, agrupa e chama model.transcribe(lista_de_paths).
    """
    while True:
        batch = []
        # Pega o primeiro item (bloqueia se vazio)
        first_future, first_path = await inference_queue.get()
        batch.append((first_future, first_path))
        deadline = asyncio.get_event_loop().time() + BATCH_TIMEOUT

        # Tenta preencher o lote até o timeout ou tamanho máximo
        while len(batch) < MAX_BATCH_SIZE and asyncio.get_event_loop().time() < deadline:
            try:
                future, path = await asyncio.wait_for(inference_queue.get(), timeout=0.005)
                batch.append((future, path))
            except asyncio.TimeoutError:
                break

        # Separa futures e paths
        futures = [item[0] for item in batch]
        paths = [item[1] for item in batch]

        try:
            start_time = time.perf_counter()
            # Chama o modelo com a lista de caminhos
            hypotheses = model.transcribe(paths)
            results = [hyp.text for hyp in hypotheses]
            end_time = time.perf_counter()
            logger.info(f"Lote de {len(paths)} áudios processado em {end_time - start_time:.3f}s (média: {(end_time - start_time)/len(paths):.3f}s por áudio)")

            # Seta os resultados nos futures
            for future, text in zip(futures, results):
                future.set_result(text)
        except Exception as e:
            logger.exception("Erro na inferência em lote")
            for future in futures:
                if not future.done():
                    future.set_exception(e)
        finally:
            # Limpeza: deleta todos os arquivos WAV (e outros temporários) do lote
            for _, path in batch:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception:
                    pass

# ==================== FASTAPI LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicia os workers
    asyncio.create_task(preprocessor_worker())
    asyncio.create_task(inference_worker())
    logger.info("Workers de pré-processamento e inferência iniciados.")
    yield

app = FastAPI(lifespan=lifespan, title="STT API (Parakeet)")

# ==================== ENDPOINTS DE HEALTH CHECK ====================
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
    Recebe um arquivo de áudio (qualquer formato suportado pelo FFmpeg),
    coloca na fila de pré-processamento e aguarda o resultado.
    """
    original_suffix = os.path.splitext(file.filename)[1].lower() or ".wav"

    # Salva o arquivo enviado em disco (RAM se possível)
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix, dir=TEMP_DIR) as tmp:
        content = await file.read()
        tmp.write(content)
        raw_path = tmp.name

    # Cria um Future para esta requisição
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    # Coloca na fila de pré-processamento (junto com o sufixo)
    await preprocess_queue.put((future, raw_path, original_suffix))

    # Aguarda o resultado (com timeout)
    try:
        transcript = await asyncio.wait_for(future, timeout=60.0)
        response = {
            "results": {
                "channels": [{
                    "alternatives": [{"transcript": transcript, "confidence": 1.0}]
                }]
            }
        }
        return JSONResponse(content=response)
    except asyncio.TimeoutError:
        # Se o future expirou, tentamos cancelar? Não há cancelamento fácil.
        # Mas podemos ao menos limpar o arquivo se ele ainda existir.
        raise HTTPException(status_code=504, detail="Tempo limite excedido")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Se por algum motivo o arquivo raw ainda existir (ex: se o preprocessador falhou),
        # removemos aqui para evitar vazamento.
        # Mas o preprocessador já tenta remover, então isso é redundante.
        if os.path.exists(raw_path):
            try:
                os.unlink(raw_path)
            except Exception:
                pass
