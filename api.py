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

# Novas variáveis para paralelismo
NUM_PREPROCESS_WORKERS = int(os.getenv("NUM_PREPROCESS_WORKERS", "2"))  # padrão 2
FFMPEG_THREADS = int(os.getenv("FFMPEG_THREADS", "2"))                  # threads por conversão

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
preprocess_queue = asyncio.Queue()
inference_queue = asyncio.Queue()

# ==================== WORKER DE PRÉ-PROCESSAMENTO (CPU) ====================
async def preprocessor_worker(worker_id: int):
    """
    Converte arquivos .webm para .wav (16kHz, mono, PCM) usando FFmpeg com múltiplas threads.
    Outros formatos são passados sem conversão.
    """
    logger.info(f"Worker de pré-processamento {worker_id} iniciado.")
    while True:
        future, raw_path, suffix = await preprocess_queue.get()
        wav_path = raw_path
        try:
            if suffix == ".webm":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR) as wav_tmp:
                    wav_path = wav_tmp.name

                # Comando FFmpeg otimizado com -threads
                cmd = [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-threads", str(FFMPEG_THREADS),
                    "-y", "-i", raw_path,
                    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path
                ]
                proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"FFmpeg falhou: {proc.stderr}")

                # Remove o original (webm)
                try:
                    os.unlink(raw_path)
                except Exception:
                    pass
                logger.debug(f"Worker {worker_id} converteu: {raw_path} -> {wav_path}")

            # Coloca o áudio pronto na fila de inferência
            await inference_queue.put((future, wav_path))

        except Exception as e:
            logger.exception(f"Worker {worker_id} erro em {raw_path}: {e}")
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
    logger.info("Worker de inferência iniciado.")
    while True:
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

        futures = [item[0] for item in batch]
        paths = [item[1] for item in batch]

        try:
            start_time = time.perf_counter()
            hypotheses = model.transcribe(paths)
            results = [hyp.text for hyp in hypotheses]
            end_time = time.perf_counter()
            logger.info(f"Lote de {len(paths)} áudios processado em {end_time - start_time:.3f}s (média: {(end_time - start_time)/len(paths):.3f}s/áudio)")

            for future, text in zip(futures, results):
                future.set_result(text)
        except Exception as e:
            logger.exception("Erro na inferência em lote")
            for future in futures:
                if not future.done():
                    future.set_exception(e)
        finally:
            for _, path in batch:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception:
                    pass

# ==================== FASTAPI LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicia múltiplos workers de pré-processamento (CPU paralela)
    for i in range(NUM_PREPROCESS_WORKERS):
        asyncio.create_task(preprocessor_worker(i + 1))
        logger.info(f"Worker de pré-processamento {i+1} agendado.")
    
    # Inicia o worker de inferência (GPU)
    asyncio.create_task(inference_worker())
    
    logger.info(f"Total: {NUM_PREPROCESS_WORKERS} preprocessadores, 1 inferência.")
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
    Recebe um arquivo de áudio, coloca na fila de pré-processamento e aguarda o resultado.
    """
    original_suffix = os.path.splitext(file.filename)[1].lower() or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix, dir=TEMP_DIR) as tmp:
        content = await file.read()
        tmp.write(content)
        raw_path = tmp.name

    loop = asyncio.get_event_loop()
    future = loop.create_future()

    await preprocess_queue.put((future, raw_path, original_suffix))

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
        raise HTTPException(status_code=504, detail="Tempo limite excedido")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpeza extra por segurança
        if os.path.exists(raw_path):
            try:
                os.unlink(raw_path)
            except Exception:
                pass
