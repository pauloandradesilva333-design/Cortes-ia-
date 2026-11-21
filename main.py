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
            "-x264-params", "rc-lookahead=0:ref=1:bframes=0:threads=1",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out,
        ],
        cwd=d,
    )
    return out


def work(job_id, p: JobIn):
    j = JOBS[job_id]
    d = WORK / job_id
    try:
        d.mkdir(exist_ok=True)
        j["msg"] = "Baixando a live..."
        a = int(p.start_min * 60)
        b = a + int(p.max_min * 60)
        run([
            "yt-dlp", "--no-progress", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
            "--merge-output-format", "mp4",
            "--download-sections", f"*{a}-{b}",
            "-o", str(d / "video.%(ext)s"), p.url,
        ])
        video = next(d.glob("video.*"))

        j["msg"] = "Extraindo áudio..."
        run(["ffmpeg", "-loglevel", "error", "-nostats", "-y", "-threads", "1",
             "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
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
            j["clips"].append({
                "title": c["title"],
                "file": f"/files/{job_id}/{name}",
                "srt": f"/files/{job_id}/{srt}",
            })

        video.unlink(missing_ok=True)
        (d / "audio.mp3").unlink(missing_ok=True)
        j["status"], j["msg"] = "done", "Pronto!"
        notify_make(j["clips"])
    except Exception as e:
        j["status"], j["msg"] = "error", f"Erro: {e}"
    finally:
        LOCK.release()


@app.post("/jobs")
def create_job(p: JobIn):
    if not GROQ_KEY:
        raise HTTPException(500, "GROQ_API_KEY não configurada")
    if not LOCK.acquire(blocking=False):
        raise HTTPException(429, "Já existe um corte em andamento. Espere terminar.")
    try:
        p.max_min = min(max(p.max_min, 1), 20)
        p.clips = min(max(p.clips, 1), 8)
        for old in WORK.iterdir():
            shutil.rmtree(old, ignore_errors=True)
        JOBS.clear()
        job_id = uuid.uuid4().hex[:10]
        JOBS[job_id] = {"status": "running", "msg": "Na fila...", "clips": []}
        threading.Thread(target=work, args=(job_id, p), daemon=True).start()
    except Exception:
        LOCK.release()
        raise
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    return JOBS[job_id]


@app.get("/files/{job_id}/{name}")
def get_file(job_id: str, name: str):
    f = (WORK / job_id / name).resolve()
    if WORK.resolve() not in f.parents or not f.exists() or f.suffix not in (".mp4", ".srt"):
        raise HTTPException(404)
    mt = "video/mp4" if f.suffix == ".mp4" else "text/plain; charset=utf-8"
    return FileResponse(f, media_type=mt, filename=name)


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
<label style="display:flex;gap:8px;align-items:center;font-size:15px;opacity:1"><input id="fast" type="checkbox" checked style="width:auto;margin:0"> Modo rápido (sem layout vertical)</label>
<button id="btn" onclick="go()">Gerar cortes</button>
<p id="msg"></p><div id="out"></div>
<script>
const g=id=>document.getElementById(id).value;
const msg=t=>document.getElementById('msg').textContent=t;
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function go(){
  const body={url:g('url'),start_min:+g('start_min'),max_min:+g('max_min'),clips:+g('clips'),
    clip_sec:+g('clip_sec'),cam_x:+g('cam_x'),cam_y:+g('cam_y'),cam_w:+g('cam_w'),cam_h:+g('cam_h'),fast:document.getElementById('fast').checked};
  document.getElementById('out').innerHTML='';
  msg('Enviando...');
  try{
    const r=await fetch('/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!r.ok){const e=await r.json().catch(()=>({}));msg(e.detail||'Erro ao iniciar');return}
    poll((await r.json()).id);
  }catch(e){msg('Sem conexão com o servidor. Espere um pouco e tente de novo.')}
}
async function poll(id){
  let j;
  try{
    const r=await fetch('/jobs/'+id);
    if(r.status===404){msg('O servidor reiniciou e perdeu o trabalho. Toque em Gerar cortes de novo.');return}
    j=await r.json();
  }catch(e){
    msg('Conexão perdida, tentando de novo...');
    setTimeout(()=>poll(id),5000);return
  }
  msg(j.msg);
  document.getElementById('out').innerHTML=j.clips.map(c=>
    `<p><b>${esc(c.title)}</b></p><video controls playsinline src="${c.file}"></video><a href="${c.file}" download>Baixar</a> &nbsp; <a href="${c.srt}" download>Legenda (.srt)</a>`).join('');
  if(j.status==='running') setTimeout(()=>poll(id),4000);
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.get("/testar")
def testar():
    k = GROQ_KEY
    info = {
        "tamanho_da_chave": len(k),
        "comeca_com_gsk": k.startswith("gsk_"),
        "tem_espaco_sobrando": k != k.strip(),
        "modelo": LLM_MODEL,
    }
    try:
        r = httpx.get("https://api.groq.com/openai/v1/models", headers=auth(), timeout=20)
        info["groq_status"] = r.status_code
        info["groq_resposta"] = r.text[:300]
    except Exception as e:
        info["erro"] = str(e)
    return info


def notify_make(clips):
    url = os.environ.get("MAKE_WEBHOOK_URL", "")
    if not url:
        return
    base = os.environ.get("PUBLIC_URL", "https://cortes-ia-4cgk.onrender.com")
    for c in clips:
        try:
            httpx.post(
                url,
                json={
                    "title": c["title"],
                    "video_url": base + c["file"],
                    "srt_url": base + c["srt"],
                },
                timeout=30,
            )
        except Exception:
            pass


# FIM
