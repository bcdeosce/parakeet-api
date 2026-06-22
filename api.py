import os
import sys
import subprocess
import logging

# ==================== AUTO-INSTALAÇÃO DE DEPENDÊNCIAS ====================
try:
    import librosa
    import audioread
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("installer")
    logger.warning(f"Dependência faltando: {e}. Instalando via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "librosa", "audioread"])
    import librosa
    import audioread
    logger.info("Librosa e audioread instalados com sucesso.")

# Opcional: instalar soundfile para evitar warning do PySoundFile
try:
    import soundfile
except ImportError:
    logging.warning("soundfile não encontrado. Instalando para melhor desempenho...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
    import soundfile

# Agora o resto do código
import asyncio
import tempfile
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse
import torch
import numpy as np

# ==================== CONFIGURAÇÕES ====================
MODEL_TYPE = os.getenv("MODEL_TYPE", "parakeet")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "large-v3-turbo")
PARAKEET_MODEL_FILE = os.getenv("PARAKEET_MODEL_FILE", "/app/model_cache/parakeet-tdt-0.6b-v3.nemo")
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.03"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
TEMP_DIR = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stt-api")

# ==================== CARREGAMENTO DO MODELO ====================
logger.info(f"Dispositivo: {DEVICE}, Modelo: {MODEL_TYPE}")

if MODEL_TYPE == "whisper":
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    logger.info(f"Carregando Whisper '{WHISPER_SIZE}'...")
    model = WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    model = BatchedInferencePipeline(model=model)
    MODEL_LOADED = True
    logger.info("Whisper pronto.")

elif MODEL_TYPE == "parakeet":
    import nemo.collections.asr as nemo_asr
    logger.info(f"Carregando Parakeet do arquivo local: {PARAKEET_MODEL_FILE}")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=PARAKEET_MODEL_FILE)
    model = model.to(DEVICE)
    model.eval()
    MODEL_LOADED = True
    logger.info("Parakeet pronto.")

else:
    raise ValueError(f"MODEL_TYPE inválido: {MODEL_TYPE}")

# ==================== FILA DE BATCH ====================
request_queue = asyncio.Queue()

async def batch_worker():
    while True:
        batch = []
        first = await request_queue.get()
        batch.append(first)
        deadline = asyncio.get_event_loop().time() + BATCH_TIMEOUT
        while len(batch) < MAX_BATCH_SIZE and asyncio.get_event_loop().time() < deadline:
            try:
                item = await asyncio.wait_for(request_queue.get(), timeout=0.005)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        futures = [item[0] for item in batch]
        paths = [item[1] for item in batch]

        try:
            start_time = time.perf_counter()

            # Carrega todos os áudios com librosa (força 16 kHz, mono)
            audio_data = []
            for path in paths:
                y, sr = librosa.load(path, sr=16000, mono=True)
                audio_data.append((y, sr))  # mantemos a tupla para compatibilidade

            if MODEL_TYPE == "whisper":
                waveforms = [y for y, _ in audio_data]
                segments_batch = model.transcribe(waveforms, language="pt", task="transcribe")
                results = []
                for segments in segments_batch:
                    results.append(" ".join(seg.text for seg in segments).strip())
            else:  # Parakeet
                # Converte cada waveform para torch tensor (float32) e move para o device do modelo
                waveforms = [torch.from_numpy(y).float().to(DEVICE) for y, _ in audio_data]
                # Chama transcribe com sample_rate explícito
                hypotheses = model.transcribe(waveforms, sample_rate=16000)
                results = [hyp.text for hyp in hypotheses]

            end_time = time.perf_counter()
            per_audio_time = (end_time - start_time) / len(paths)
            logger.info(f"Lote de {len(paths)} áudios processado em {end_time - start_time:.3f}s (média por áudio: {per_audio_time:.3f}s)")

            for future, text in zip(futures, results):
                future.set_result(text)

        except Exception as e:
            logger.exception("Erro na transcrição em lote")
            for future in futures:
                future.set_exception(e)
        finally:
            for _, tmp_path in batch:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

# ==================== FASTAPI ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(batch_worker())
    logger.info("Batch worker iniciado.")
    yield

app = FastAPI(lifespan=lifespan, title="STT API (Whisper/Parakeet)")

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

@app.post("/v1/listen")
async def transcribe_audio(file: UploadFile = File(...)):
    original_suffix = os.path.splitext(file.filename)[1].lower() or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix, dir=TEMP_DIR) as tmp:
        content = await file.read()
        tmp.write(content)
        audio_path = tmp.name

    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await request_queue.put((future, audio_path))

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
        raise HTTPException(status_code=504, detail="Transcrição expirou")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
