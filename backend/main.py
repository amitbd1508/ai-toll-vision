from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .config import load_config, project_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

config = load_config()
Path(config["paths"]["videos_dir"]).mkdir(parents=True, exist_ok=True)
Path(config["paths"]["outputs_dir"]).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Toll Booth Vision Analytics (POC)")
app.include_router(router, prefix="")

app.mount("/outputs", StaticFiles(directory=config["paths"]["outputs_dir"]), name="outputs")
app.mount("/videos_static", StaticFiles(directory=config["paths"]["videos_dir"]), name="videos_static")

FRONTEND_DIR = project_root() / "frontend"


@app.get("/")
def dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")
