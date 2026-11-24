import json
import os
import re
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
    scan: bool = True


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


def loudness(audio: Path, rate=16000, limite=5400):
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostats", "-hide_banner", "-i", str(audio), "-vn",
             "-ac", "1", "-ar", str(rate), "-af",
             f"aresample={rate},asetnsamples=n={rate}:p=0,astats=metadata=1:reset=1,"
             "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=limite,
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


def candidatos(levels, k, sep=90):
    n = len(levels)
    if n < 30:
        return []
    pre = [0.0]
    for v in levels:
        pre.append(pre[-1] + v)

    def media(a, b):
        a, b = max(0, a), min(n, b)
        return (pre[b] - pre[a]) / max(1, b - a)

    bloco = 60
    meds = []
    for i in range(0, n, bloco):
        parte = sorted(levels[i:i + bloco])
        meds.append(parte[len(parte) // 2])

    def normal(i):
        c = i // bloco
        viz = sorted(meds[max(0, c - 5):c + 6])
        return viz[len(viz) // 2]

    pontos = [(media(i - 3, i + 8) - normal(i), i) for i in range(n)]
    pontos.sort(reverse=True)
    achados = []
    for sc, i in pontos:
        if all(abs(i - j) >= sep for _, j in achados):
            achados.append((sc, i))
        if len(achados) >= k:
            break
    return achados


def hms(t):
    t = int(t)
    return f"{t // 3600:02}:{t % 3600 // 60:02}:{t % 60:02}"


FMT_LEVE = "bv*[height<=360]+ba/b[height<=360]/worst"
FMT_FULL = "bv*[height<=1080]+ba/b[height<=1080]/b"


def baixar(url, base: Path, formato, a=None, b=None):
    cmd = ["yt-dlp", "--no-progress", "-f", formato, "--merge-output-format", "mp4"]
    if a is not None:
        cmd += ["--download-sections", f"*{int(a)}-{int(b)}"]
    cmd += ["-o", str(base) + ".%(ext)s", url]
    run(cmd)
    return next(base.parent.glob(base.name + ".*"))


def avaliar_janela(arq):
    arquivo = gemini_upload(arq)
    criterio = os.environ.get(
        "CRITERIOS", "momentos legais, empolgantes, com muita ação ou que façam rir.")
    prompt = (
        "Você é editor de cortes virais para TikTok. Assista a este vídeo curto e ouça "
        "o áudio: é um trecho de uma live de jogo. Veja se existe nele UM corte "
        f"excelente. Critério: {criterio} Use o que você vê (ação, quedas, batidas, "
        "reações) e o que ouve (gritos, risadas, falas). O corte deve contar a cena "
        "COMPLETA: contexto, o pico e o desfecho logo depois, sem cortar frase no meio, "
        "com 20 a 60 segundos. Os tempos são em SEGUNDOS desde o início deste vídeo. "
        "Dê uma nota de 1 a 10 (10 = viraliza com certeza; use 3 ou menos se não houver "
        "nada realmente bom). Responda SOMENTE em JSON no formato "
        '{"start":10,"end":45,"score":8,"title":"título curto","motivo":"frase curta"}.'
    )
    for tentativa in range(2):
        try:
            return gemini_clips(arquivo, prompt)
        except RuntimeError as e:
            if "429" in str(e) and tentativa == 0:
                time.sleep(25)
                continue
            raise


def work_scan(job_id, p: JobIn):
    j = JOBS[job_id]
    d = WORK / job_id
    try:
        d.mkdir(exist_ok=True)
        a0 = int(p.start_min * 60)
        total_pedido = int(p.max_min * 60)
        j["msg"] = "Baixando o áudio da live inteira (pode demorar)..."
        if a0 == 0 and p.max_min >= 360:
            tudo = baixar(p.url, d / "tudo", "ba/worst")
        else:
            tudo = baixar(p.url, d / "tudo", "ba/worst", a0, a0 + total_pedido)

        j["msg"] = "Medindo o volume da live inteira..."
        levels = loudness(tudo, rate=8000, limite=7200)
        tudo.unlink(missing_ok=True)
        if len(levels) < 60:
            raise RuntimeError("Não consegui medir o áudio dessa live.")

        k = min(p.clips * 2 + 2, 12)
        cands = candidatos(levels, k)
        if not cands:
            raise RuntimeError("Não achei trechos candidatos.")
        dur = len(levels)

        avaliados, aviso, falhas = [], "", 0
        for n_, (sc, pico) in enumerate(cands, 1):
            ini = max(0, pico - 60)
            fim = min(dur, pico + 30)
            j["msg"] = f"IA assistindo trecho {n_} de {len(cands)}..."
            veredito = None
            if GEMINI_KEY and falhas < 2:
                low = None
                try:
                    low = baixar(p.url, d / f"j{n_}", FMT_LEVE, a0 + ini, a0 + fim)
                    veredito = avaliar_janela(low)
                    falhas = 0
                    time.sleep(3)
                except Exception as e:
                    falhas += 1
                    aviso = f" (Gemini falhou: {str(e)[:120]})"
                finally:
                    if low is not None:
                        low.unlink(missing_ok=True)
            if veredito:
                try:
                    ra = float(veredito["start"])
                    rb = float(veredito["end"])
                    nota = float(veredito.get("score", 5))
                    if rb <= ra:
                        continue
                    cs, ce = ini + max(0.0, ra), ini + rb
                    avaliados.append({
                        "start": cs, "end": min(ce, cs + 90), "nota": nota,
                        "title": str(veredito.get("title", "Corte"))[:80],
                        "motivo": str(veredito.get("motivo", ""))[:140]})
                    continue
                except (KeyError, ValueError, TypeError):
                    pass
            avaliados.append({
                "start": max(0, pico - 25), "end": min(dur, pico + 15),
                "nota": min(5.0, 3 + sc / 4), "title": "Momento de pico de volume",
                "motivo": "escolhido só pelo volume do áudio"})

        avaliados = [c for c in avaliados if c["end"] - c["start"] >= 10]
        avaliados.sort(key=lambda x: -x["nota"])
        escolhidos = []
        for c in avaliados:
            if all(c["end"] <= e["start"] or c["start"] >= e["end"] for e in escolhidos):
                escolhidos.append(c)
            if len(escolhidos) >= p.clips:
                break
        if not escolhidos:
            raise RuntimeError("A IA não achou momentos bons nessa live.")

        for i, c in enumerate(escolhidos, 1):
            j["msg"] = f"Cortando clipe {i} de {len(escolhidos)}..."
            A = max(0, c["start"] - 4)
            B = min(dur, c["end"] + 4)
            sec = baixar(p.url, d / f"s{i}", FMT_FULL, a0 + A, a0 + B)
            ra, rb = c["start"] - A, c["end"] - A
            segs = []
            try:
                audio = d / f"a{i}.mp3"
                run(["ffmpeg", "-loglevel", "error", "-nostats", "-y", "-threads", "1",
                     "-i", str(sec), "-vn", "-ac", "1", "-ar", "16000",
                     "-b:a", "32k", str(audio)])
                segs = transcribe(audio)
                audio.unlink(missing_ok=True)
            except Exception:
                segs = []
            if segs:
                ra, rb = ajustar(ra, rb, segs, int(B - A))
            srt = f"sub_{i}.srt"
            write_srt(segs, ra, rb, d / srt)
            name = cut(sec, d, i, ra, rb, srt, p)
            sec.unlink(missing_ok=True)
            info = f"Minuto {hms(a0 + c['start'])} da live · nota {c['nota']:.0f}"
            if c["motivo"]:
                info += f" · {c['motivo']}"
            j["clips"].append({
                "title": c["title"],
                "info": info,
                "quando": hms(a0 + c["start"]),
                "nota": c["nota"],
                "file": f"/files/{job_id}/{name}",
                "srt": f"/files/{job_id}/{srt}",
            })

        j["status"], j["msg"] = "done", "Pronto!" + aviso
        notify_make(j["clips"])
    except Exception as e:
        j["status"], j["msg"] = "error", f"Erro: {e}"
    finally:
        LOCK.release()


def work(job_id, p: JobIn):
    if p.scan:
        work_scan(job_id, p)
        return
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
        if p.scan:
            p.max_min = 360 if p.max_min <= 0 else min(p.max_min, 480)
        else:
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


class ClipeIn(BaseModel):
    link: str
    titulo: str = "Clipe"


def kick_clips(canal, ordem="view", periodo="month"):
    from curl_cffi import requests as cr
    r = cr.get(
        f"https://kick.com/api/v2/channels/{canal}/clips",
        params={"sort": ordem, "time": periodo},
        impersonate="chrome", timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Kick respondeu {r.status_code}: {r.text[:150]}")
    data = r.json()
    itens = data.get("clips") or data.get("data") or []
    out = []
    for c in itens:
        cid = c.get("id") or c.get("clip_id")
        if not cid:
            continue
        out.append({
            "id": str(cid),
            "titulo": str(c.get("title") or "Clipe")[:100],
            "views": c.get("view_count") or c.get("views") or 0,
            "duracao": c.get("duration") or 0,
            "miniatura": c.get("thumbnail_url") or "",
            "link": f"https://kick.com/{canal}/clips/{cid}",
        })
    return out


@app.get("/clipes")
def clipes(canal: str = "gabepeixe", ordem: str = "view", periodo: str = "month"):
    canal = canal.strip().strip("@/")
    if not re.fullmatch(r"[A-Za-z0-9_\-]{2,40}", canal):
        raise HTTPException(400, "Nome de canal inválido")
    if ordem not in ("view", "date") or periodo not in ("day", "week", "month", "all"):
        raise HTTPException(400, "Filtro inválido")
    try:
        return {"clipes": kick_clips(canal, ordem, periodo)}
    except Exception as e:
        raise HTTPException(502, f"Não consegui listar os clipes: {str(e)[:200]}")


def work_clipe(job_id, p: ClipeIn):
    j = JOBS[job_id]
    d = WORK / job_id
    try:
        d.mkdir(exist_ok=True)
        j["msg"] = "Baixando o clipe..."
        run([
            "yt-dlp", "--no-progress", "-f", "bv*+ba/b",
            "--merge-output-format", "mp4",
            "-o", str(d / "clip_1.%(ext)s"), p.link,
        ])
        arq = next(d.glob("clip_1.*"))
        j["clips"].append({
            "title": p.titulo[:80],
            "file": f"/files/{job_id}/{arq.name}",
            "srt": "",
        })
        j["status"], j["msg"] = "done", "Pronto!"
        notify_make(j["clips"])
    except Exception as e:
        j["status"], j["msg"] = "error", f"Erro: {e}"
    finally:
        LOCK.release()


@app.post("/clipe")
def preparar_clipe(p: ClipeIn):
    if not re.fullmatch(r"https://kick\.com/[A-Za-z0-9_\-]{2,40}/clips/clip_[A-Za-z0-9]+", p.link):
        raise HTTPException(400, "Link de clipe da Kick inválido")
    if not LOCK.acquire(blocking=False):
        raise HTTPException(429, "Já existe um trabalho em andamento. Espere terminar.")
    try:
        for old in WORK.iterdir():
            shutil.rmtree(old, ignore_errors=True)
        JOBS.clear()
        job_id = uuid.uuid4().hex[:10]
        JOBS[job_id] = {"status": "running", "msg": "Na fila...", "clips": []}
        threading.Thread(target=work_clipe, args=(job_id, p), daemon=True).start()
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
<h2>Cortes IA v5</h2>
<input id="url" placeholder="Link da live (Kick ou YouTube)">
<div class="row"><div><label>Começar no minuto</label><input id="start_min" type="number" value="0"></div>
<div><label>Analisar quantos min (0 = tudo)</label><input id="max_min" type="number" value="0"></div></div>
<div class="row"><div><label>Nº de clipes</label><input id="clips" type="number" value="5"></div>
<div><label>Duração (s)</label><input id="clip_sec" type="number" value="45"></div></div>
<label>Webcam na live (0 a 1): x, y, largura, altura</label>
<div class="row"><input id="cam_x" value="0"><input id="cam_y" value="0.7">
<input id="cam_w" value="0.3"><input id="cam_h" value="0.3"></div>
<label style="display:flex;gap:8px;align-items:center;font-size:15px;opacity:1"><input id="scan" type="checkbox" checked style="width:auto;margin:0"> Varrer a live inteira e achar os melhores momentos</label>
<label style="display:flex;gap:8px;align-items:center;font-size:15px;opacity:1"><input id="fast" type="checkbox" checked style="width:auto;margin:0"> Modo rápido (sem layout vertical)</label>
<button id="btn" onclick="go()">Gerar cortes</button>
<p id="msg"></p><div id="out"></div>
<hr style="border-color:#333;margin:24px 0">
<h3>Clipes populares do canal</h3>
<div class="row"><input id="canal" value="gabepeixe">
<select id="periodo" style="width:100%;padding:12px;margin:4px 0;border-radius:8px;border:0;font-size:16px">
<option value="week">Semana</option><option value="month" selected>Mês</option>
<option value="day">Hoje</option><option value="all">Todos</option></select></div>
<button onclick="buscar()">Buscar clipes mais vistos</button>
<p id="msg2"></p><div id="lista"></div>
<script>
const g=id=>document.getElementById(id).value;
const msg=t=>document.getElementById('msg').textContent=t;
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function go(){
  const body={url:g('url'),start_min:+g('start_min'),max_min:+g('max_min'),clips:+g('clips'),
    clip_sec:+g('clip_sec'),cam_x:+g('cam_x'),cam_y:+g('cam_y'),cam_w:+g('cam_w'),cam_h:+g('cam_h'),fast:document.getElementById('fast').checked,scan:document.getElementById('scan').checked};
  document.getElementById('out').innerHTML='';
  msg('Enviando...');
  try{
    const r=await fetch('/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!r.ok){const e=await r.json().catch(()=>({}));msg(e.detail||'Erro ao iniciar');return}
    poll((await r.json()).id);
  }catch(e){msg('Sem conexão com o servidor. Espere um pouco e tente de novo.')}
}
async function buscar(){
  const m=document.getElementById('msg2'); m.textContent='Buscando...';
  document.getElementById('lista').innerHTML='';
  try{
    const r=await fetch('/clipes?canal='+encodeURIComponent(g('canal'))+'&periodo='+g('periodo'));
    const j=await r.json();
    if(!r.ok){m.textContent=j.detail||'Erro ao buscar';return}
    if(!j.clipes.length){m.textContent='Nenhum clipe encontrado.';return}
    m.textContent=j.clipes.length+' clipes, do mais visto para o menos visto:';
    window._cl=j.clipes;
    document.getElementById('lista').innerHTML=j.clipes.map((c,i)=>
      `<div style="margin:12px 0;padding:10px;background:#1b1b1b;border-radius:8px">`+
      (c.miniatura?`<img src="${esc(c.miniatura)}" style="width:100%;border-radius:6px">`:'')+
      `<p><b>${esc(c.titulo)}</b><br><span style="opacity:.7">${c.views} views · ${Math.round(c.duracao)}s</span></p>`+
      `<button onclick="preparar(${i})">Preparar este clipe</button></div>`).join('');
  }catch(e){m.textContent='Sem conexão com o servidor.'}
}
async function preparar(i){
  const c=window._cl[i];
  document.getElementById('out').innerHTML='';
  msg('Enviando...');
  window.scrollTo(0,0);
  try{
    const r=await fetch('/clipe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({link:c.link,titulo:c.titulo})});
    if(!r.ok){const e=await r.json().catch(()=>({}));msg(e.detail||'Erro ao iniciar');return}
    poll((await r.json()).id);
  }catch(e){msg('Sem conexão com o servidor.')}
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
    `<p><b>${esc(c.title)}</b><br><small>${esc(c.info||'')}</small></p><video controls playsinline src="${c.file}"></video><a href="${c.file}" download>Baixar</a>${c.srt?` &nbsp; <a href="${c.srt}" download>Legenda (.srt)</a>`:''}`).join('');
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
        "versao": "v5",
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
                    "srt_url": (base + c["srt"]) if c.get("srt") else "",
                    "quando": c.get("quando", ""),
                    "nota": c.get("nota", ""),
                },
                timeout=30,
            )
        except Exception:
            pass


# FIM
