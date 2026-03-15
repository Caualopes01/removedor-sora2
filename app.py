import os
import uuid
import tempfile
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np
import torch
from simple_lama_inpainting import SimpleLama

app = FastAPI(title="Watermark Remover API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, coloque o domínio da Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache do modelo
_lama_model = None
JOBS = {}  # job_id -> status/result

OUTPUT_DIR = Path("/tmp/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_lama():
    global _lama_model
    if _lama_model is None:
        _lama_model = SimpleLama()
    return _lama_model


def detect_watermark(frame: np.ndarray) -> np.ndarray:
    """
    Detecta marca d'água automaticamente usando:
    1. Análise de frequência/contraste anormal
    2. Detecção de áreas semi-transparentes consistentes
    Retorna uma máscara binária (255 = área da marca d'água)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- Método 1: Detectar regiões de alto contraste local (logos geralmente têm bordas fortes) ---
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Dilatar as bordas para criar regiões
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # --- Método 2: Detecção de áreas de alta frequência consistentes entre frames ---
    # Procurar regiões com padrão semi-transparente (canal alpha implícito)
    # Converter para float e analisar variação local
    frame_float = frame.astype(np.float32) / 255.0

    # Calcular desvio padrão local em janela pequena
    mean_local = cv2.blur(frame_float, (15, 15))
    diff = np.abs(frame_float - mean_local)
    diff_gray = np.mean(diff, axis=2)

    # Normalizar
    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold para pegar regiões "estranhas"
    _, thresh = cv2.threshold(diff_norm, 30, 255, cv2.THRESH_BINARY)

    # Combinar os dois métodos
    combined = cv2.bitwise_or(dilated, thresh)

    # Encontrar contornos e filtrar por tamanho/posição
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros((h, w), dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > (h * w * 0.3):  # Ignorar muito pequeno ou muito grande
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)

        # Marcas d'água geralmente ficam em cantos ou bordas
        in_corner = (
            (x < w * 0.2 or x + cw > w * 0.8) or
            (y < h * 0.2 or y + ch > h * 0.8)
        )
        if in_corner:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    # Dilatar a máscara final para cobrir bordas da marca d'água
    mask = cv2.dilate(mask, kernel, iterations=4)

    return mask


def remove_watermark_lama(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove a marca d'água usando o modelo LaMa."""
    lama = get_lama()

    # LaMa espera PIL Images
    from PIL import Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask)

    result = lama(frame_pil, mask_pil)
    result_np = np.array(result)
    return cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)


def remove_watermark_inpaint(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fallback: usa inpainting do OpenCV (mais rápido, menor qualidade)."""
    return cv2.inpaint(frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)


def process_video(job_id: str, input_path: str, use_lama: bool = True):
    """Processa o vídeo frame a frame."""
    try:
        JOBS[job_id] = {"status": "processing", "progress": 0}

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Não foi possível abrir o vídeo")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = str(OUTPUT_DIR / f"{job_id}_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Detectar máscara nos primeiros frames para consistência
        masks_sample = []
        for i in range(min(5, total_frames)):
            ret, frame = cap.read()
            if ret:
                mask = detect_watermark(frame)
                masks_sample.append(mask)

        # Máscara consolidada (OR de todos os samples)
        if masks_sample:
            consolidated_mask = masks_sample[0]
            for m in masks_sample[1:]:
                consolidated_mask = cv2.bitwise_or(consolidated_mask, m)
        else:
            consolidated_mask = np.zeros((height, width), dtype=np.uint8)

        # Reiniciar o vídeo
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if np.any(consolidated_mask):
                if use_lama:
                    try:
                        processed = remove_watermark_lama(frame, consolidated_mask)
                    except Exception:
                        processed = remove_watermark_inpaint(frame, consolidated_mask)
                else:
                    processed = remove_watermark_inpaint(frame, consolidated_mask)
            else:
                processed = frame

            out.write(processed)
            frame_idx += 1

            progress = int((frame_idx / total_frames) * 100)
            JOBS[job_id]["progress"] = progress

        cap.release()
        out.release()

        JOBS[job_id] = {
            "status": "done",
            "progress": 100,
            "output_path": output_path,
        }

    except Exception as e:
        JOBS[job_id] = {"status": "error", "message": str(e)}
    finally:
        # Limpar arquivo de entrada
        try:
            os.remove(input_path)
        except Exception:
            pass


@app.get("/")
def root():
    return {"message": "Watermark Remover API", "status": "running"}


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_lama: bool = True,
):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de vídeo válido")

    # Salvar arquivo temporário
    job_id = str(uuid.uuid4())
    input_path = f"/tmp/{job_id}_input{Path(file.filename).suffix}"

    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    JOBS[job_id] = {"status": "queued", "progress": 0}

    # Processar em background
    background_tasks.add_task(process_video, job_id, input_path, use_lama)

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return JOBS[job_id]


@app.get("/download/{job_id}")
def download_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    job = JOBS[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Processamento ainda não finalizado")

    output_path = job["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Arquivo de saída não encontrado")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="video_sem_marca.mp4",
    )
