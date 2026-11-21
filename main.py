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
    cam_w: float = 0.3
    cam_h: float = 0.3
    fast: bool = True


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "erro")[-500:])
    return r


def auth():
    return {"Authorization": f"Bearer {GROQ_KEY}"}


def transcribe(audio: Path):
    with open(audio, "rb") as f:
        r = httpx.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=auth(),
            files={"file": ("audio.mp3", f, "audio/mpeg")},
            data={
                "model": "whisper-large-v3-turbo",
                "response_format": "verbose_json",
                "language": LANG,
            },
            timeout=600,
        )
    r.raise_for_status()
    return r.json().get("segments", [])


def ask_llm(prompt):
    base = {
        "model": LLM_MODEL,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
    }
    tentativas = []
    if LLM_MODEL.startswith("openai/gpt-oss"):
        tentativas.append({**base, "reasoning_effort": "low",
                           "response_format": {"type": "json_object"}})
    tentativas.append({**base, "response_format": {"type": "json_object"}})
    tentativas.append(base)
    erro = ""
    for body in tentativas:
        r = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=auth(), json=body, timeout=180,
        )
        if r.status_code == 200:
            texto = r.json()["choices"][0]["message"].get("content") or ""
            i, f = texto.find("{"), texto.rfind("}")
            try:
                return json.loads(texto[i:f + 1])
            except ValueError:
                erro = "resposta sem JSON: " + texto[:150]
                continue
        erro = f"Groq {r.status_code}: {r.text[:250]}"
        if r.status_code not in (400, 422):
            break
    raise RuntimeError("A IA não respondeu direito. " + erro)


def pick_clips(segs, n, sec):
    transcript = "\n".join(
        f"[{int(s['start'])}] {s['text'].strip()}" for s in segs
    )[:20000]
    total = int(segs[-1]["end"]) if segs else 0
    prompt = (
        f"Abaixo está a transcrição de uma live, com o tempo em segundos entre colchetes. "
        f"Escolha os {n} melhores momentos para cortes de TikTok. Critério: "
        + os.environ.get("CRITERIOS", "momentos legais, empolgantes, com muita ação ou que façam rir (risadas, reações exageradas, gritos de empolgação, vitórias, quase morte, fails engraçados). Evite trechos parados, silêncio, leitura de chat e explicações longas.")
        + f" Cada corte deve ter cerca de "
        f"{sec} segundos, começar no início de uma fala e não pode se sobrepor aos outros. "
        f"Duração total disponível: {total}s. Responda SOMENTE em JSON no formato "
        f'{{"clips":[{{"start":0,"end":45,"title":"título curto"}}]}}.\n\n{transcript}'
    )
    data = ask_llm(prompt)
    out = []
    for c in data.get("clips", []):
        try:
            a, b = float(c["start"]), float(c["end"])
        except (KeyError, ValueError, TypeError):
            continue
        if total:
            b = min(b, total)
        if b - a > 90:
            b = a + 90
        if b - a < 10:
            continue
        out.append({"start": a, "end": b, "title": str(c.get("title", "Corte"))[:80]})
    return out[:n]


def fmt(t):
    t = max(t, 0)
    h, rest = divmod(int(t), 3600)
    m, s = divmod(rest, 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segs, a, b, path: Path):
    lines, i = [], 1
    for s in segs:
        if s["end"] <= a or s["start"] >= b:
            continue
        st, en = max(s["start"], a) - a, min(s["end"], b) - a
        text = s["text"].strip().upper()
        if not text:
            continue
        lines.append(f"{i}\n{fmt(st)} --> {fmt(en)}\n{text}\n")
        i += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def cut(video, d, idx, a, b, srt_name, p: JobIn):
    out = f"clip_{idx}.mp4"
    if p.fast:
        run(
            [
                "ffmpeg", "-loglevel", "error", "-nostats", "-y",
                "-ss", str(a), "-i", str(video), "-t", str(round(b - a, 2)),
                "-c", "copy", "-movflags", "+faststart", out,
            ],
            cwd=d,
        )
        return out
    style = "FontName=DejaVu Sans,FontSize=11,Bold=1,Alignment=2,MarginV=130,Outline=2,Shadow=0"
    fc = (
        "[0:v]split=2[a][b];"
        f"[a]crop=iw*{p.cam_w}:ih*{p.cam_h}:iw*{p.cam_x}:ih*{p.cam_y},"
        "scale=1080:640:force_original_aspect_ratio=increase,crop=1080:640[cam];"
        "[b]crop='min(iw,ih*1080/1280)':ih,scale=1080:1280[game];"
        f"[cam][game]vstack=2,subtitles={srt_name}:force_style='{style}'[v]"
    )
    run(
        [
            "ffmpeg", "-loglevel", "error", "-nostats", "-y",
            "-threads", "1", "-filter_complex_threads", "1",
            "-ss", str(a), "-i", str(video), "-t", str(round(b - a, 2)),
            "-filter_complex", fc, "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "25",
            "-x2
