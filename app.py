import os
import uuid
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np

app = FastAPI(title="Watermark Remover API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS = {}
OUTPUT_DIR = Path("/tmp/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def process_video(job_id: str, input_path: str, regions: list):
    try:
        JOBS[job_id] = {"status": "processing", "progress": 0}

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Não foi possível abrir o vídeo")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = str(OUTPUT_DIR / f"{job_id}_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps
            mask = np.zeros((height, width), dtype=np.uint8)

            for r in regions:
                start_t = float(r.get("start_time", 0))
                end_t   = float(r.get("end_time", 999999))
                if current_time < start_t or current_time > end_t:
                    continue

                rx = int(float(r["x"]) * width)
                ry = int(float(r["y"]) * height)
                rw = int(float(r["w"]) * width)
                rh = int(float(r["h"]) * height)

                rx = max(0, min(rx, width - 1))
                ry = max(0, min(ry, height - 1))
                rw = max(1, min(rw, width - rx))
                rh = max(1, min(rh, height - ry))

                mask[ry:ry+rh, rx:rx+rw] = 255

            if np.any(mask):
                mask  = cv2.dilate(mask, kernel, iterations=2)
                frame = cv2.inpaint(frame, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

            out.write(frame)
            frame_idx += 1
            JOBS[job_id]["progress"] = int((frame_idx / max(total_frames, 1)) * 100)

        cap.release()
        out.release()

        JOBS[job_id] = {"status": "done", "progress": 100, "output_path": output_path}

    except Exception as e:
        JOBS[job_id] = {"status": "error", "message": str(e)}
    finally:
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
    regions: str = Form(default="[]"),
):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de vídeo válido")

    try:
        regions_data = json.loads(regions)
    except Exception:
        regions_data = []

    if not regions_data:
        raise HTTPException(status_code=400, detail="Nenhuma região fornecida")

    job_id     = str(uuid.uuid4())
    suffix     = Path(file.filename or "video.mp4").suffix or ".mp4"
    input_path = f"/tmp/{job_id}_input{suffix}"

    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)

    JOBS[job_id] = {"status": "queued", "progress": 0}
    background_tasks.add_task(process_video, job_id, input_path, regions_data)

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
        raise HTTPException(status_code=400, detail="Processamento não finalizado")
    output_path = job.get("output_path", "")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    return FileResponse(output_path, media_type="video/mp4", filename="video_sem_marca.mp4")
