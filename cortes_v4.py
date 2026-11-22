import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
LANG = os.environ.get("LANG_CODE", "pt")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODELS = [m for m in [os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
                             "gemini-3.5-flash", "gemini-3-flash-preview"] if m]
GBASE = "https://generativelanguage.googleapis.com"
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


def loudness(audio: Path):
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostats", "-hide_banner", "-i", str(audio), "-af",
             "asetnsamples=n=16000:p=0,astats=metadata=1:reset=1,"
             "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=300,
        )
    except Exception:
        return []
    out = []
    for line in r.stdout.splitlines():
        if line.startswith("lavfi.astats.Overall.RMS_level="):
            try:
                out.append(max(float(line.split("=", 1)[1]), -90.0))
            except ValueError:
                out.append(-90.0)
    return out


def picos(levels):
    if len(levels) < 10:
        return set()
    ordem = sorted(levels)
    mediana = ordem[len(ordem) // 2]
    limite = max(mediana + 6, ordem[int(len(ordem) * 0.9)])
    return {i for i, v in enumerate(levels) if v >= limite}


def top_picos(levels, quantos=12):
    achados = []
    candidatos = picos(levels)
    for i in sorted(candidatos, key=lambda k: -levels[k]):
        if all(abs(i - j) > 10 for j in achados):
            achados.append(i)
        if len(achados) >= quantos:
            break
    return sorted(achados)


def ask_llm(prompt):
    base = {
        "model": LLM_MODEL,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": prompt}],
    }
    tentativas = []
    if LLM_MODEL.startswith("openai/gpt-oss"):
        tentativas.append({**base, "reasoning_effort": os.environ.get("REASONING", "medium"),
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


def ajustar(a, b, segs, total):
    ini = a
    for s in segs:
        if s["start"] <= a:
            ini = s["start"]
        else:
            break
    ini = max(0.0, ini - 2.0)
    fim = segs[-1]["end"] if segs else b
    for s in segs:
        if s["end"] >= b:
            fim = s["end"]
            break
    fim = fim + 1.5
    if total:
        fim = min(fim, float(total))
    if fim - ini > 90:
        fim = ini + 90
    return round(ini, 2), round(fim, 2)


def pick_clips(segs, n, sec, levels=None):
    altos = picos(levels or [])
    linhas = []
    for s in segs:
        a0, b0 = int(s["start"]), int(s["end"]) + 1
        marca = " [ALTO]" if any(k in altos for k in range(a0, b0)) else ""
        linhas.append(f"[{a0}]{marca} {s['text'].strip()}")
    transcript = "\n".join(linhas)[:20000]
    dica = ""
    if altos:
        dica = (
            " Trechos marcados com [ALTO] tiveram pico de volume (grito, risada, ação) "
            "e merecem prioridade. Picos de volume nos segundos: "
            + ", ".join(str(x) for x in top_picos(levels))
            + "."
        )
    total = int(segs[-1]["end"]) if segs else 0
    pedir = min(n + 3, 10)
    criterio = os.environ.get(
        "CRITERIOS",
        "momentos legais, empolgantes, com muita ação ou que façam rir (risadas, "
        "reações exageradas, gritos de empolgação, vitórias, quase morte, fails "
        "engraçados). Evite trechos parados, silêncio, leitura de chat e explicações longas.",
    )
    prompt = (
        "Abaixo está a transcrição de uma live, com o tempo em segundos entre colchetes. "
        f"Liste os {pedir} melhores candidatos a corte de TikTok. Critério: {criterio} "
        "Cada corte deve contar a cena COMPLETA: um começo com contexto, o pico "
        "(a piada, o susto, a reação ou a ação) e o desfecho logo depois, sem cortar "
        "nenhuma frase no meio. Escolha pelo conteúdo, não pelo tamanho; a duração pode "
        f"variar de 20 a 60 segundos (referência: {sec}s). Os cortes não podem se "
        "sobrepor. Dê a cada um uma nota de 1 a 10 para o potencial de viralizar e "
        "justifique em uma frase curta que cite o que foi dito. Não invente trechos: "
        "use só o que aparece na transcrição. "
        f"Duração total disponível: {total}s.{dica} Responda SOMENTE em JSON no formato "
        '{"clips":[{"start":0,"end":45,"score":8,"title":"título curto","motivo":"frase curta"}]}'
        f".\n\n{transcript}"
    )
    data = ask_llm(prompt)
    return finalizar(data, segs, total, n)


def finalizar(data, segs, total, n):
    cands = []
    for c in data.get("clips", []):
        try:
            a, b = float(c["start"]), float(c["end"])
            nota = float(c.get("score", 5))
        except (KeyError, ValueError, TypeError):
            continue
        if b <= a:
            continue
        a, b = ajustar(a, b, segs, total)
        if b - a < 10:
            continue
        cands.append({"start": a, "end": b, "nota": nota,
                      "title": str(c.get("title", "Corte"))[:80]})
    cands.sort(key=lambda x: -x["nota"])
    escolhidos = []
    for c in cands:
        if all(c["end"] <= e["start"] or c["start"] >= e["end"] for e in escolhidos):
            escolhidos.append(c)
        if len(escolhidos) >= n:
            break
    escolhidos.sort(key=lambda x: x["start"])
    return [{"start": c["start"], "end": c["end"], "title": c["title"]} for c in escolhidos]


def gemini_upload(path: Path):
    size = path.stat().st_size
    h = {"x-goog-api-key": GEMINI_KEY}
    r = httpx.post(
        f"{GBASE}/upload/v1beta/files",
        headers={**h, "X-Goog-Upload-Protocol": "resumable",
                 "X-Goog-Upload-Command": "start",
                 "X-Goog-Upload-Header-Content-Length": str(size),
                 "X-Goog-Upload-Header-Content-Type": "video/mp4",
                 "Content-Type": "application/json"},
        json={"file": {"display_name": path.name}}, timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Gemini envio {r.status_code}: {r.text[:200]}")
    url = r.headers.get("x-goog-upload-url")
    if not url:
        raise RuntimeError("Gemini não devolveu o endereço de envio")

    def partes():
        with open(path, "rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                yield b

    r = httpx.post(
        url,
        headers={"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                 "X-Goog-Upload-Command": "upload, finalize"},
        content=partes(), timeout=900,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Gemini envio {r.status_code}: {r.text[:200]}")
    f = r.json()["file"]
    for _ in range(60):
        if f.get("state") == "ACTIVE":
            return f
        if f.get("state") == "FAILED":
            raise RuntimeError("Gemini não conseguiu processar o vídeo")
        time.sleep(5)
        g = httpx.get(f"{GBASE}/v1beta/{f['name']}", headers=h, timeout=60)
        if g.status_code == 200:
            f = g.json()
    raise RuntimeError("Gemini demorou demais para processar o vídeo")


def gemini_clips(arquivo, prompt):
    erro = ""
    for modelo in GEMINI_MODELS:
        r = httpx.post(
            f"{GBASE}/v1beta/models/{modelo}:generateContent",
            headers={"x-goog-api-key": GEMINI_KEY},
            json={
                "contents": [{"parts": [
                    {"file_data": {"mime_type": "video/mp4", "file_uri": arquivo["uri"]}},
                    {"text": prompt},
                ]}],
                "generationConfig": {"temperature": 0.3,
                                     "responseMimeType": "application/json"},
            },
            timeout=600,
        )
        if r.status_code == 200:
            try:
                partes = r.json()["candidates"][0]["content"]["parts"]
                texto = "".join(p.get("text", "") for p in partes)
            except (KeyError, IndexError, TypeError):
                erro = "resposta vazia"
                continue
            i, f = texto.find("{"), texto.rfind("}")
            try:
                return json.loads(texto[i:f + 1])
            except ValueError:
                erro = "resposta sem JSON"
                continue
        erro = f"Gemini {r.status_code}: {r.text[:200]}"
        if r.status_code in (400, 401, 403):
            break
    raise RuntimeError(erro)


def pick_clips_gemini(video, segs, n, sec):
    total = int(segs[-1]["end"]) if segs else 0
    arquivo = gemini_upload(video)
    pedir = min(n + 3, 10)
    criterio = os.environ.get(
        "CRITERIOS",
        "momentos legais, empolgantes, com muita ação ou que façam rir.",
    )
    prompt = (
        "Você é editor de cortes virais para TikTok. Assista ao vídeo inteiro e ouça o "
        "áudio: é um trecho de uma live de jogo. "
        f"Liste os {pedir} melhores momentos para cortar. Critério: {criterio} "
        "Use o que você VÊ (ação, quedas, batidas, reações, expressões) e o que você OUVE "
        "(gritos, risadas, falas). Cada corte deve contar a cena COMPLETA: começo com "
        "contexto, o pico e o desfecho logo depois, sem cortar frase no meio. A duração "
        f"pode variar de 20 a 60 segundos (referência: {sec}s). Os cortes não podem se "
        "sobrepor. Dê a cada um uma nota de 1 a 10 para o potencial de viralizar. "
        "Os tempos são em SEGUNDOS desde o início deste vídeo (número, não mm:ss). "
        f"Duração total do vídeo: cerca de {total}s. Responda SOMENTE em JSON no formato "
        '{"clips":[{"start":0,"end":45,"score":8,"title":"título curto","motivo":"frase curta"}]}.'
    )
    data = gemini_clips(arquivo, prompt)
    return finalizar(data, segs, total, n)


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

        low = None
        if GEMINI_KEY:
            try:
                j["msg"] = "Baixando versão leve para a IA assistir..."
                run([
                    "yt-dlp", "--no-progress",
                    "-f", "bv*[height<=360]+ba/b[height<=360]/worst",
                    "--merge-output-format", "mp4",
                    "--download-sections", f"*{a}-{b}",
                    "-o", str(d / "low.%(ext)s"), p.url,
                ])
                low = next(d.glob("low.*"))
            except Exception:
                low = None

        j["msg"] = "Extraindo áudio..."
        run(["ffmpeg", "-loglevel", "error", "-nostats", "-y", "-threads", "1",
             "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
             "-b:a", "32k", str(d / "audio.mp3")])

        j["msg"] = "Medindo o volume..."
        levels = loudness(d / "audio.mp3")

        j["msg"] = "Transcrevendo..."
        segs = transcribe(d / "audio.mp3")
        if not segs:
            raise RuntimeError("Não consegui transcrever nada desse trecho.")

        picks, aviso = [], ""
        if low:
            try:
                j["msg"] = "IA assistindo o vídeo..."
                picks = pick_clips_gemini(low, segs, p.clips, p.clip_sec)
            except Exception as e:
                aviso = f" (Gemini falhou: {str(e)[:150]}. Usei a IA de texto.)"
        elif GEMINI_KEY:
            aviso = " (Não consegui baixar a versão leve. Usei a IA de texto.)"
        if not picks:
            j["msg"] = "IA escolhendo os melhores momentos..."
            picks = pick_clips(segs, p.clips, p.clip_sec, levels)
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
        if low:
            low.unlink(missing_ok=True)
        (d / "audio.mp3").unlink(missing_ok=True)
        j["status"], j["msg"] = "done", "Pronto!" + aviso
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
<h2>Cortes IA v4</h2>
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
        "versao": "v4",
        "gemini_chave_configurada": bool(GEMINI_KEY),
    }
    try:
        r = httpx.get("https://api.groq.com/openai/v1/models", headers=auth(), timeout=20)
        info["groq_status"] = r.status_code
        info["groq_resposta"] = r.text[:300]
    except Exception as e:
        info["erro"] = str(e)
    if GEMINI_KEY:
        try:
            g = httpx.get(f"{GBASE}/v1beta/models", headers={"x-goog-api-key": GEMINI_KEY}, timeout=20)
            info["gemini_status"] = g.status_code
            if g.status_code != 200:
                info["gemini_resposta"] = g.text[:200]
        except Exception as e:
            info["gemini_erro"] = str(e)
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
