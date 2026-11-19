import json
import os
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
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
WORK = Path("/tmp/cortes")
WORK.mkdir(exist_ok=True)

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


def pick_clips(segs, n, sec):
    transcript = "\n".join(
        f"[{int(s['start'])}] {s['text'].strip()}" for s in segs
    )[:20000]
    total = int(segs[-1]["end"]) if segs else 0
    prompt = (
        f"Abaixo está a transcrição de uma live, com o tempo em segundos entre colchetes. "
        f"Escolha os {n} melhores momentos para cortes de TikTok (engraçados, "
        f"emocionantes, polêmicos ou com reação forte). Cada corte deve ter cerca de "
        f"{sec} segundos, começar no início de uma fala e não pode se sobrepor aos outros. "
        f"Duração total disponível: {total}s. Responda SOMENTE em JSON no formato "
        f'{{"clips":[{{"start":0,"end":45,"title":"título curto"}}]}}.\n\n{transcript}'
    )
    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=auth(),
        json={
            "model": LLM_MODEL,
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    r.raise_for_status()
    data = json.loads(r.json()["choices"][0]["message"]["content"])
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
    style = "FontName=DejaVu Sans,FontSize=11,Bold=1,Alignment=2,MarginV=130,Outline=2,Shadow=0"
    fc = (
        f"[0:v]split=2[a][b];"
        f"[a]crop=iw*{p.cam_w}:ih*{p.cam_h}:iw*{p.cam_x}:ih*{p.cam_y},"
        f"scale=1080:640:force_original_aspect_ratio=increase,crop=1080:640[cam];"
        f"[b]scale=-2:1280,crop=1080:1280[game];"
        f"[cam][game]vstack=2,subtitles={srt_name}:force_style='{style}'[v]"
    )
    run(
        [
            "ffmpeg", "-y", "-ss", str(a), "-to", str(b), "-i", str(video),
            "-filter_complex", fc, "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out,
        ],
        cwd=d,
    )
    return out


def work(job_id, p: JobIn):
    j = JOBS[job_id]
    d = WORK / job_id
    d.mkdir(exist_ok=True)
    try:
        j["msg"] = "Baixando a live..."
        a = int(p.start_min * 60)
        b = a + int(p.max_min * 60)
        run([
            "yt-dlp", "-f", "bv*[height<=720]+ba/b[height<=720]/b",
            "--merge-output-format", "mp4",
            "--download-sections", f"*{a}-{b}",
            "-o", str(d / "video.%(ext)s"), p.url,
        ])
        video = next(d.glob("video.*"))

        j["msg"] = "Extraindo áudio..."
        run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
             "-b:a", "32k", str(d / "audio.mp3")])

        j["msg"] = "Transcrevendo..."
        segs = transcribe(d / "audio.mp3")
        if not segs:
            raise RuntimeError("Não consegui transcrever nada desse trecho.")

        j["msg"] = "IA escolhendo os melhores momentos..."
        picks = pick_clips(segs, p.clips, p.clip_sec)
        if not picks:
            raise RuntimeError("A IA não encontrou trechos bons.")

        for i, c in enumerate(picks, 1):
            j["msg"] = f"Cortando clipe {i} de {len(picks)}..."
            srt = f"sub_{i}.srt"
            write_srt(segs, c["start"], c["end"], d / srt)
            name = cut(video, d, i, c["start"], c["end"], srt, p)
            j["clips"].append({"title": c["title"], "file": f"/files/{job_id}/{name}"})

        video.unlink(missing_ok=True)
        j["status"], j["msg"] = "done", "Pronto!"
    except Exception as e:
        j["status"], j["msg"] = "error", f"Erro: {e}"


@app.post("/jobs")
def create_job(p: JobIn):
    if not GROQ_KEY:
        raise HTTPException(500, "GROQ_API_KEY não configurada")
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"status": "running", "msg": "Na fila...", "clips": []}
    threading.Thread(target=work, args=(job_id, p), daemon=True).start()
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    return JOBS[job_id]


@app.get("/files/{job_id}/{name}")
def get_file(job_id: str, name: str):
    f = (WORK / job_id / name).resolve()
    if WORK.resolve() not in f.parents or not f.exists() or f.suffix != ".mp4":
        raise HTTPException(404)
    return FileResponse(f, media_type="video/mp4", filename=name)


PAGE = """<!doctype html><html lang="pt"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Cortes IA</title>
<style>
body{font-family:system-ui;max-width:560px;margin:auto;padding:16px;background:#111;color:#eee}
input,button{width:100%;padding:12px;margin:4px 0;border-radius:8px;border:0;font-size:16px;box-sizing:border-box}
button{background:#7c3aed;color:#fff;font-weight:600}
label{font-size:13px;opacity:.8}.row{display:flex;gap:8px}
video{width:100%;border-radius:8px}a{color:#a78bfa}
</style></head><body>
<h2>Cortes IA</h2>
<input id="url" placeholder="Link da live (Kick ou YouTube)">
<div class="row"><div><label>Começar no minuto</label><input id="start_min" type="number" value="0"></div>
<div><label>Analisar quantos min</label><input id="max_min" type="number" value="20"></div></div>
<div class="row"><div><label>Nº de clipes</label><input id="clips" type="number" value="5"></div>
<div><label>Duração (s)</label><input id="clip_sec" type="number" value="45"></div></div>
<label>Webcam na live (0 a 1): x, y, largura, altura</label>
<div class="row"><input id="cam_x" value="0"><input id="cam_y" value="0.7">
<input id="cam_w" value="0.3"><input id="cam_h" value="0.3"></div>
<button onclick="go()">Gerar cortes</button>
<p id="msg"></p><div id="out"></div>
<script>
const g=id=>document.getElementById(id).value;
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function go(){
  const body={url:g('url'),start_min:+g('start_min'),max_min:+g('max_min'),clips:+g('clips'),
    clip_sec:+g('clip_sec'),cam_x:+g('cam_x'),cam_y:+g('cam_y'),cam_w:+g('cam_w'),cam_h:+g('cam_h')};
  document.getElementById('out').innerHTML='';
  const r=await fetch('/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){document.getElementById('msg').textContent='Erro ao iniciar';return}
  poll((await r.json()).id);
}
async function poll(id){
  const j=await (await fetch('/jobs/'+id)).json();
  document.getElementById('msg').textContent=j.msg;
  document.getElementById('out').innerHTML=j.clips.map(c=>
    `<p><b>${esc(c.title)}</b></p><video controls playsinline src="${c.file}"></video><a href="${c.file}" download>Baixar</a>`).join('');
  if(j.status==='running') setTimeout(()=>poll(id),4000);
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
