import json
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
LANG = os.environ.get("LANG_CODE", "pt")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
WORK = Path("/tmp/cortes")
WORK.mkdir(exist_ok=True)
LOCK = threading.Lock()

app = FastAPI()
JOBS = {}


class JobIn(BaseModel):
    url: str
    start_min: float = 0
    max_min: float = 20
    clips: int = 5
    clip_sec: int = 45
    cam_x: float = 0.0
    cam_y: float = 0.7
    cam_
