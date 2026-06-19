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
from pydub import AudioSegment
from faster_whisper import WhisperModel, BatchedInferencePipeline

# ==================== CONFIGURAÇÕES ====================
MODEL_TYPE = os.getenv("MODEL_TYPE", "parakeet")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "large-v3-turbo")
# PARAKEET_MODEL_FILE: caminho absoluto para o arquivo .nemo dentro da imagem
PARAKEET_MODEL_FILE = os.getenv("PARAKEET_MODEL_FILE", "/app/model_cache/parakeet-tdt-0.6b-v3.nemo")
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.03"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stt-api")

# ==================== CARREGAMENTO DO MODELO ====================
logger.info(f"Dispositivo: {DEVICE}, Modelo: {MODEL_TYPE}")

if MODEL_TYPE == "whisper":
    logger.info(f"Carregando Whisper '{WHISPER_SIZE}'...")
    model = WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    model = BatchedInferencePipeline(model=model)
    MODEL_LOADED = True
    logger.info("Whisper pronto.")

elif MODEL_TYPE == "parakeet":
    import nemo.collections.asr as nemo_asr
    logger.info(f"Carregando Parakeet do arquivo local: {PARAKEET_MODEL_FILE}")
    # Carrega o modelo diretamente do arquivo .nemo, sem tocar na internet
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
        durations = [item[2] for item in batch]

        try:
            start_time = time.perf_counter()
            if MODEL_TYPE == "whisper":
                results = []
                for path in paths:
                    segments, _ = model.transcribe(path, language="pt", task="transcribe")
                    results.append(" ".join(seg.text for seg in segments).strip())
            else:
                hypotheses = model.transcribe(paths)
                results = [hyp.text for hyp in hypotheses]
            end_time = time.perf_counter()

            for idx, (path, dur) in enumerate(zip(paths, durations)):
                per_audio_time = (end_time - start_time) / len(paths)
                rtf = per_audio_time / dur if dur > 0 else 0
                logger.info(f"RTF para {os.path.basename(path)}: {rtf:.4f} (tempo estimado: {per_audio_time:.3f}s, duração: {dur:.3f}s)")

            for future, text in zip(futures, results):
                future.set_result(text)
        except Exception as e:
            logger.exception("Erro na transcrição em lote")
            for future in futures:
                future.set_exception(e)
        finally:
            for _, tmp_path, _ in batch:
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
    # Obtém a extensão do arquivo enviado (case insensitive)
    original_suffix = os.path.splitext(file.filename)[1].lower() or ".wav"

    # Salva o arquivo original em um temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_original_path = tmp.name

    # Se o arquivo for WebM, converte para OGG (áudio apenas) usando ffmpeg
    if original_suffix == ".webm":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as ogg_tmp:
            ogg_path = ogg_tmp.name
        try:
            cmd = [
                "ffmpeg", "-y", "-i", tmp_original_path,
                "-vn", "-c:a", "libvorbis", ogg_path
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Falha na conversão de WebM para OGG: {proc.stderr}")
        except Exception as e:
            # Em caso de erro, remove os arquivos temporários
            os.unlink(tmp_original_path)
            if os.path.exists(ogg_path):
                os.unlink(ogg_path)
            raise HTTPException(status_code=400, detail=f"Erro ao converter .webm: {str(e)}")

        # Define o caminho do áudio convertido para uso no restante do fluxo
        audio_path = ogg_path
    else:
        audio_path = tmp_original_path
        ogg_path = None  # nenhum arquivo OGG gerado

    try:
        # Carrega o áudio com pydub para obter a duração
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0
    except Exception as e:
        if os.path.exists(tmp_original_path):
            os.unlink(tmp_original_path)
        if ogg_path and os.path.exists(ogg_path):
            os.unlink(ogg_path)
        raise HTTPException(status_code=400, detail=f"Erro ao ler arquivo de áudio: {str(e)}")

    # Coloca na fila de batch para transcrição
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await request_queue.put((future, audio_path, duration))

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
        # Garante que todos os arquivos temporários sejam removidos
        if os.path.exists(tmp_original_path):
            os.unlink(tmp_original_path)
        if ogg_path and os.path.exists(ogg_path):
            os.unlink(ogg_path)