"""
La Ligne : serveur Hugging Face (Space Docker).

Sert l'application, garde la clé Gemini côté serveur et enregistre les résultats
des appels d'entraînement pour l'onglet Manager.

Secrets à définir dans Settings > Variables and secrets du Space :
  GEMINI_API_KEY    clé Google AI Studio (obligatoire)
  MANAGER_PASSWORD  mot de passe de l'onglet Manager (obligatoire)
  HF_TOKEN          jeton Hugging Face avec droit d'écriture (recommandé, pour la sauvegarde)
  DATASET_REPO      dataset privé de sauvegarde, ex. "ton-pseudo/la-ligne-resultats" (recommandé)
"""
import asyncio
import json
import os
import shutil
import threading
import time
from pathlib import Path

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MANAGER_PASSWORD = os.getenv("MANAGER_PASSWORD", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
DATASET_REPO = os.getenv("DATASET_REPO", "").strip()
API = "https://generativelanguage.googleapis.com/v1beta"
HERE = Path(__file__).parent


# ---------- Stockage des résultats ----------
def pick_data_dir() -> Path:
    for d in [os.getenv("DATA_DIR"), "/data", str(HERE / "data")]:
        if not d:
            continue
        try:
            p = Path(d)
            p.mkdir(parents=True, exist_ok=True)
            (p / ".write_test").write_text("ok")
            (p / ".write_test").unlink()
            return p
        except Exception:
            continue
    raise RuntimeError("Aucun dossier inscriptible pour les données")


DATA_DIR = pick_data_dir()
SESSIONS = DATA_DIR / "sessions.jsonl"
lock = threading.Lock()
scheduler = None

if DATASET_REPO and HF_TOKEN:
    from huggingface_hub import CommitScheduler, hf_hub_download

    if not SESSIONS.exists():
        try:
            path = hf_hub_download(DATASET_REPO, "sessions.jsonl", repo_type="dataset", token=HF_TOKEN)
            shutil.copy(path, SESSIONS)
            print(f"Résultats restaurés depuis {DATASET_REPO}")
        except Exception as e:
            print(f"Pas de sauvegarde à restaurer ({e.__class__.__name__})")
    scheduler = CommitScheduler(
        repo_id=DATASET_REPO, repo_type="dataset", folder_path=DATA_DIR,
        allow_patterns=["sessions.jsonl"], every=2, token=HF_TOKEN, private=True,
    )
    lock = scheduler.lock

FIELDS = {
    "id": str, "date": str, "junior": str, "scenarioId": str, "scenarioTitle": str, "kind": str,
    "level": str, "durationSec": int, "talkShare": int, "rdv": bool, "score": int, "criteria": dict,
    "axe": str, "verdict": str, "erreursSmile": int, "prepFilled": int,
}


def clean(rec: dict) -> dict:
    out = {}
    for k, t in FIELDS.items():
        v = rec.get(k)
        if v is None:
            out[k] = None
            continue
        try:
            if t is str:
                out[k] = str(v)[:600]
            elif t is int:
                out[k] = int(v)
            elif t is bool:
                out[k] = bool(v)
            elif t is dict:
                out[k] = {str(a)[:40]: str(b)[:20] for a, b in dict(v).items()}
        except Exception:
            out[k] = None
    out["receivedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


def read_sessions() -> list:
    if not SESSIONS.exists():
        return []
    rows, seen = [], set()
    with lock:
        lines = SESSIONS.read_text(encoding="utf-8").splitlines()
    for line in lines:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("id") in seen:
            continue
        seen.add(r.get("id"))
        rows.append(r)
    return rows


app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/config")
def config():
    return {
        "serverKey": bool(GEMINI_KEY),
        "managerReady": bool(MANAGER_PASSWORD),
        "backup": scheduler is not None or str(DATA_DIR) == "/data",
    }


@app.post("/api/sessions")
async def save_session(request: Request):
    try:
        rec = clean(await request.json())
    except Exception:
        return JSONResponse({"ok": False, "error": "format"}, status_code=400)
    if not rec.get("id") or not rec.get("junior"):
        return JSONResponse({"ok": False, "error": "incomplet"}, status_code=400)
    with lock:
        with SESSIONS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}


@app.get("/api/sessions")
def list_sessions(request: Request):
    if not MANAGER_PASSWORD:
        return JSONResponse({"ok": False, "error": "Le mot de passe Manager n'est pas configuré dans le Space."}, status_code=503)
    if request.headers.get("x-manager-password", "") != MANAGER_PASSWORD:
        return JSONResponse({"ok": False, "error": "Mot de passe incorrect."}, status_code=401)
    return {"ok": True, "sessions": read_sessions(), "backup": scheduler is not None or str(DATA_DIR) == "/data"}


# ---------- Relais vers Gemini (la clé reste sur le serveur) ----------
@app.get("/api/models")
async def models():
    if not GEMINI_KEY:
        return JSONResponse({"error": {"message": "GEMINI_API_KEY manquante dans les secrets du Space."}}, status_code=503)
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{API}/models", params={"pageSize": 200, "key": GEMINI_KEY})
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/api/generate")
async def generate(request: Request):
    if not GEMINI_KEY:
        return JSONResponse({"error": {"message": "GEMINI_API_KEY manquante dans les secrets du Space."}}, status_code=503)
    body = await request.json()
    model = str(body.get("model", ""))
    if not model.startswith("models/"):
        return JSONResponse({"error": {"message": "Modèle invalide."}}, status_code=400)
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"{API}/{model}:generateContent", params={"key": GEMINI_KEY}, json=body.get("payload", {}))
    return JSONResponse(r.json(), status_code=r.status_code)


def safe_reason(text: str) -> str:
    return (text or "").encode("utf-8")[:120].decode("utf-8", errors="ignore")


@app.websocket("/api/live")
async def live(ws: WebSocket):
    await ws.accept()
    if not GEMINI_KEY:
        await ws.close(code=1011, reason="GEMINI_API_KEY manquante")
        return
    version = ws.query_params.get("v", "v1beta")
    if version not in ("v1beta", "v1alpha"):
        version = "v1beta"
    url = (f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.{version}"
           f".GenerativeService.BidiGenerateContent?key={GEMINI_KEY}")
    code, reason = 1000, ""
    try:
        async with websockets.connect(url, max_size=None, ping_interval=20, open_timeout=20) as up:
            async def client_to_gemini():
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        return
                    data = msg.get("text")
                    if data is None:
                        data = msg.get("bytes")
                    if data is not None:
                        await up.send(data)

            async def gemini_to_client():
                async for m in up:
                    await ws.send_text(m if isinstance(m, str) else m.decode("utf-8"))

            tasks = [asyncio.create_task(client_to_gemini()), asyncio.create_task(gemini_to_client())]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in tasks:
                t.cancel()
            code, reason = up.close_code or 1000, up.close_reason or ""
    except Exception as e:
        code, reason = 1011, f"{e.__class__.__name__}: {e}"
    if code in (1005, 1006, 1015) or not (1000 <= code <= 4999):
        code = 1011
    try:
        await ws.close(code=code, reason=safe_reason(reason))
    except Exception:
        pass
