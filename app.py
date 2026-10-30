"""FloraVision Local Application Server (Phase 7).
Serves Apple-style Glassmorphism UI and unified Computer Vision analysis API.
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Safe stdout handling for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from inference import get_pipeline

app = FastAPI(
    title="FloraVision",
    description="Intelligent Plant Leaf Disease Detection & Segmentation System",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static and templates
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return index_file.read_text(encoding="utf-8")


@app.post("/api/analyze")
async def analyze_uploaded_leaf(file: UploadFile = File(...)):
    """Analyze an uploaded plant leaf image file."""
    try:
        contents = await file.read()
        pipeline = get_pipeline()
        results = pipeline.analyze(contents)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/analyze-sample")
async def analyze_sample_leaf(filename: str = Query(...)):
    """Analyze a predefined curated sample leaf from the dataset."""
    sample_path = SAMPLES_DIR / filename
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample file {filename} not found.")

    try:
        pipeline = get_pipeline()
        results = pipeline.analyze(sample_path)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/samples")
async def list_samples():
    """List all available demo leaf sample files."""
    valid_exts = {".jpg", ".jpeg", ".png"}
    samples = [f.name for f in SAMPLES_DIR.iterdir() if f.suffix.lower() in valid_exts]
    return {"samples": sorted(samples)}


@app.get("/api/health")
async def health_check():
    """Return backend status and pipeline availability."""
    return {
        "status": "online",
        "device": "cpu",
        "version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

