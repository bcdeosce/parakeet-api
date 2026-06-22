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
from faster_whisper import WhisperModel, BatchedInferencePipeline

# ==================== CONFIGURAÇÕES ====================
MODEL_TYPE = os.getenv("MODEL_TYPE", "parakeet")           # "whisper" ou "parakeet"
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "large-v3-turbo")
PARAKEET_MODEL_FILE = os.getenv("PARAKEET_MODEL_FILE", "/app/model_cache/parakeet-tdt-0.6b-v3.nemo")
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.03"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")

# Diretório para arquivos temporários (RAM disk se existir)
TEMP_DIR = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"

# Configuração de logging (nível DEBUG para diagnóstico)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("stt-api")

# ==================== FUNÇÃO AUXILIAR PARA INSPECIONAR ÁUDIO ====================
def log_audio_info(filepath: str, label: str = "Áudio"):
    """Usa ffprobe para obter e logar informações detalhadas do arquivo."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_name,channels,sample_rate,duration",
        "-of", "default=noprint_wrappers=1",
        filepath
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        logger.debug(f"[{label}] Informações do arquivo {filepath}:\n{output}")
        # Extrai valores para log mais legível
        info = {}
        for line in output.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k] = v
        if info:
            logger.debug(
                f"[{label}] Resumo: codec={info.get('codec_name')}, "
                f"canais={info.get('channels')}, taxa={info.get('sample_rate')} Hz, "
                f"duração={info.get('duration')}s"
            )
    except subprocess.CalledProcessError as e:
        logger.warning(f"[{label}] Falha ao obter info com ffprobe: {e.output}")

# ==================== CARREGAMENTO DO MODELO ====================
logger.info(f"Dispositivo: {DEVICE}, Modelo: {MODEL_TYPE}")
logger.info(f"MODEL_TYPE={MODEL_TYPE}, WHISPER_SIZE={WHISPER_SIZE}, "
            f"PARAKEET_MODEL_FILE={PARAKEET_MODEL_FILE}")
logger.info(f"DEVICE={DEVICE}, COMPUTE_TYPE={COMPUTE_TYPE}, "
            f"BATCH_TIMEOUT={BATCH_TIMEOUT}, MAX_BATCH_SIZE={MAX_BATCH_SIZE}")

if MODEL_TYPE == "whisper":
    logger.info(f"Carregando Whisper '{WHISPER_SIZE}'...")
    model = WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    model = BatchedInferencePipeline(model=model)
    MODEL_LOADED = True
    logger.info("Whisper pronto.")

elif MODEL_TYPE == "parakeet":
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        raise ImportError("NeMo não está instalado. Execute: pip install nemo_toolkit[asr]")
    logger.info(f"Carregando Parakeet do arquivo local: {PARAKEET_MODEL_FILE}")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        restore_path=PARAKEET_MODEL_FILE
    )
    model = model.to(DEVICE)
    model.eval()
    MODEL_LOADED = True
    logger.info("Parakeet pronto.")

else:
    raise ValueError(f"MODEL_TYPE inválido: {MODEL_TYPE}")

# ==================== FILA DE BATCH ====================
request_queue = asyncio.Queue()

async def batch_worker():
    """Worker que processa lotes de áudios (todos já em WAV mono 16kHz)."""
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
        paths = [item[1] for item in batch]   # caminhos dos arquivos WAV

        logger.info(f"Processando lote com {len(paths)} arquivos: {paths}")

        try:
            start_time = time.perf_counter()

            # Log detalhado de cada arquivo do lote (opcional, mas útil)
            for p in paths:
                log_audio_info(p, f"Lote-{id(batch)}")

            if MODEL_TYPE == "whisper":
                results = []
                for path in paths:
                    logger.debug(f"Transcrevendo com Whisper: {path}")
                    segments, _ = model.transcribe(path, language="pt", task="transcribe")
                    results.append(" ".join(seg.text for seg in segments).strip())
            else:  # Parakeet
                logger.debug(f"Transcrevendo lote com Parakeet: {paths}")
                # Parakeet aceita lista de caminhos
                hypotheses = model.transcribe(paths)
                results = [hyp.text for hyp in hypotheses]

            end_time = time.perf_counter()
            per_audio_time = (end_time - start_time) / len(paths)
            logger.info(
                f"Lote de {len(paths)} áudios processado em {end_time - start_time:.3f}s "
                f"(média por áudio: {per_audio_time:.3f}s)"
            )

            for future, text in zip(futures, results):
                future.set_result(text)
                logger.debug(f"Resultado definido: {text[:30]}...")

        except Exception as e:
            logger.exception("Erro na transcrição em lote")
            for future in futures:
                future.set_exception(e)
        finally:
            # Remove os arquivos temporários (já convertidos para WAV)
            for _, tmp_path in batch:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        logger.debug(f"Arquivo temporário removido pelo worker: {tmp_path}")
                except Exception as e:
                    logger.warning(f"Falha ao remover {tmp_path}: {e}")

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
    if MODEL_LOADED:
        return Response(status_code=200, content="ready")
    else:
        return Response(status_code=503, content="loading")

@app.get("/live")
async def live():
    return Response(status_code=200, content="alive")

@app.get("/health")
async def health():
    return Response(status_code=200, content="healthy")

@app.post("/v1/listen")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Recebe um arquivo de áudio (qualquer formato suportado pelo ffmpeg),
    converte para WAV mono 16kHz e enfileira para transcrição.
    """
    logger.info(f"Recebido arquivo: {file.filename}, tamanho: {file.size}, tipo: {file.content_type}")

    original_suffix = os.path.splitext(file.filename)[1].lower() or ".wav"
    content = await file.read()
    logger.debug(f"Conteúdo lido: {len(content)} bytes")

    # Salva o arquivo original em disco (RAM se possível)
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix, dir=TEMP_DIR) as tmp:
        tmp.write(content)
        tmp_original_path = tmp.name
    logger.debug(f"Arquivo original salvo em: {tmp_original_path}")
    log_audio_info(tmp_original_path, "Original")

    # Cria um arquivo WAV de saída (sempre .wav)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR) as wav_tmp:
        wav_path = wav_tmp.name

    # Comando ffmpeg para converter para WAV mono 16kHz
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-i", tmp_original_path,
        "-ac", "1",           # mono
        "-ar", "16000",       # 16 kHz
        "-f", "wav",          # formato WAV
        wav_path
    ]
    logger.debug(f"Executando conversão: {' '.join(cmd)}")
    start_conv = time.perf_counter()

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            logger.error(f"ffmpeg falhou (código {proc.returncode}): {proc.stderr}")
            raise RuntimeError(f"Falha na conversão: {proc.stderr}")
        elapsed = time.perf_counter() - start_conv
        logger.debug(f"Conversão concluída em {elapsed:.3f}s")
    except Exception as e:
        logger.exception("Exceção durante conversão")
        os.unlink(tmp_original_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        raise HTTPException(status_code=400, detail=f"Erro ao converter áudio: {str(e)}")

    # Inspeciona o WAV gerado
    log_audio_info(wav_path, "WAV convertido")

    # Remove o arquivo original (já convertido)
    try:
        os.unlink(tmp_original_path)
        logger.debug(f"Arquivo original removido: {tmp_original_path}")
    except Exception as e:
        logger.warning(f"Não foi possível remover original {tmp_original_path}: {e}")

    # Enfileira para transcrição
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await request_queue.put((future, wav_path))
    logger.info(f"Áudio enfileirado para transcrição: {wav_path}")

    try:
        transcript = await asyncio.wait_for(future, timeout=60.0)
        logger.info(f"Transcrição concluída: '{transcript[:50]}...'")
        response = {
            "results": {
                "channels": [{
                    "alternatives": [{"transcript": transcript, "confidence": 1.0}]
                }]
            }
        }
        return JSONResponse(content=response)
    except asyncio.TimeoutError:
        logger.error(f"Timeout na transcrição para {wav_path}")
        raise HTTPException(status_code=504, detail="Transcrição expirou")
    except Exception as e:
        logger.exception(f"Erro na transcrição para {wav_path}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpeza final (caso o worker não tenha removido por algum motivo)
        for p in [tmp_original_path, wav_path]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                    logger.debug(f"Arquivo removido (finally): {p}")
                except Exception as e:
                    logger.warning(f"Não foi possível remover {p} no finally: {e}")
