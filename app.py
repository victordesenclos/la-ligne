"""
La Ligne : serveur Hugging Face (Space Docker).

Sert l'application, garde la clé Gemini côté serveur et enregistre les résultats
des appels d'entraînement pour l'onglet Manager.

Secrets à définir dans Settings > Variables and secrets du Space :
  GEMINI_API_KEY    clé Google AI Studio (obligatoire)
  MANAGER_PASSWORD  mot de passe de l'onglet Manager (obligatoire)
Sauvegarde des résultats (une des deux options) :
  GITHUB_TOKEN + GITHUB_REPO   dépôt GitHub privé dédié aux données, ex. "victordesenclos/la-ligne-data"
  HF_TOKEN + DATASET_REPO      dataset privé Hugging Face, ex. "ton-pseudo/la-ligne-resultats"
"""
import asyncio
import base64
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
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.getenv("GITHUB_REPO", "").strip()
# Tolère les erreurs de saisie courantes : URL complète, ".git" final, barre oblique
GITHUB_REPO = GITHUB_REPO.replace("https://", "").replace("http://", "").replace("github.com/", "").strip("/ ")
if GITHUB_REPO.endswith(".git"):
    GITHUB_REPO = GITHUB_REPO[:-4]
GITHUB_API = os.getenv("GITHUB_API", "https://api.github.com").rstrip("/")
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
META = DATA_DIR / "meta.json"          # date de la dernière remise à zéro
ARCHIVES = DATA_DIR / "archives"       # copies des données avant chaque remise à zéro


def read_meta() -> dict:
    try:
        return json.loads(META.read_text(encoding="utf-8"))
    except Exception:
        return {}
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
    if not META.exists():
        try:
            shutil.copy(hf_hub_download(DATASET_REPO, "meta.json", repo_type="dataset", token=HF_TOKEN), META)
        except Exception:
            pass
    scheduler = CommitScheduler(
        repo_id=DATASET_REPO, repo_type="dataset", folder_path=DATA_DIR,
        allow_patterns=["sessions.jsonl", "meta.json", "archives/*.jsonl"], every=2, token=HF_TOKEN, private=True,
    )
    lock = scheduler.lock

# ---------- Sauvegarde dans un dépôt GitHub privé ----------
def after_reset(rec: dict) -> bool:
    """Ignore les appels antérieurs à la dernière remise à zéro."""
    reset_at = read_meta().get("resetAt")
    return not reset_at or (rec.get("date") or "") >= reset_at


def parse_lines(data: bytes) -> list:
    out = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


class GitHubBackup:
    """Sauvegarde sessions.jsonl dans un dépôt GitHub privé.
    - Au démarrage : FUSIONNE la sauvegarde avec les données locales (rien n'est écrasé).
    - Après chaque appel reçu, toutes les 2 minutes et à l'arrêt : fusionne puis envoie.
      La fusion garantit qu'une restauration ratée au démarrage ne peut jamais effacer la sauvegarde."""
    def __init__(self):
        self.url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/sessions.jsonl"
        self.headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28"}
        self.sha = None
        self.pushed_size = -1
        self.push_lock = threading.Lock()
        self.status = {"mode": "GitHub", "repo": GITHUB_REPO, "lastOk": None, "lastError": None,
                       "lastErrorAt": None, "restored": None}

    def _error(self, msg):
        print(f"Sauvegarde GitHub : {msg}")
        self.status["lastError"] = msg[:300]
        self.status["lastErrorAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def fetch_remote(self):
        """Renvoie (données, sha, ok). ok=False si GitHub n'a pas pu être lu (dans ce cas on n'envoie rien)."""
        r = httpx.get(self.url, headers=self.headers, timeout=30)
        if r.status_code == 404:
            # Fichier absent… ou dépôt inaccessible ? GitHub répond 404 dans les deux cas : on vérifie le dépôt.
            repo = httpx.get(f"{GITHUB_API}/repos/{GITHUB_REPO}", headers=self.headers, timeout=30)
            if repo.status_code != 200:
                raise RuntimeError(f"le dépôt « {GITHUB_REPO} » est introuvable avec ce jeton (code {repo.status_code}) : "
                                   "vérifie GITHUB_REPO (format propriétaire/nom) et que le jeton a accès à ce dépôt, "
                                   "avec la permission Contents en lecture et écriture")
            return b"", None, True
        if r.status_code != 200:
            raise RuntimeError(f"lecture impossible ({r.status_code}) : {r.text[:150]}")
        d = r.json()
        content = base64.b64decode(d.get("content", "")) if d.get("content") else b""
        if not content and d.get("download_url"):
            content = httpx.get(d["download_url"], headers=self.headers, timeout=60).content
        return content, d.get("sha"), True

    def merge_remote(self, remote: bytes) -> int:
        """Ajoute localement les appels présents sur GitHub mais absents d'ici. Renvoie le nombre ajouté."""
        with lock:
            local = SESSIONS.read_bytes() if SESSIONS.exists() else b""
            ids = {r.get("id") for r in parse_lines(local)}
            missing = [r for r in parse_lines(remote) if r.get("id") and r.get("id") not in ids and after_reset(r)]
            if missing:
                with SESSIONS.open("a", encoding="utf-8") as f:
                    for r in missing:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(missing)

    def restore(self):
        try:
            remote, self.sha, _ = self.fetch_remote()
            added = self.merge_remote(remote)
            self.status["restored"] = added
            print(f"Résultats restaurés depuis GitHub ({GITHUB_REPO}) : {added} appel(s)")
        except Exception as e:
            self._error(f"restauration impossible au démarrage ({e})")

    def push(self, force=False, overwrite=False):
        with self.push_lock:
            for attempt in range(2):
                try:
                    if overwrite:
                        g = httpx.get(self.url, headers=self.headers, timeout=30)
                        self.sha = g.json().get("sha") if g.status_code == 200 else None
                        added = 0
                    else:
                        remote, self.sha, _ = self.fetch_remote()
                        added = self.merge_remote(remote)
                    with lock:
                        data = SESSIONS.read_bytes() if SESSIONS.exists() else b"\n"
                    if not force and not overwrite and not added and len(data) == self.pushed_size:
                        return
                    body = {"message": f"Sauvegarde La Ligne {time.strftime('%Y-%m-%d %H:%M')}",
                            "content": base64.b64encode(data or b"\n").decode()}
                    if self.sha:
                        body["sha"] = self.sha
                    r = httpx.put(self.url, headers=self.headers, json=body, timeout=60)
                    if r.status_code in (200, 201):
                        self.sha = r.json()["content"]["sha"]
                        self.pushed_size = len(data)
                        self.status["lastOk"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        self.status["lastError"] = None
                        return
                    if r.status_code in (409, 422) and attempt == 0:
                        continue
                    hint = {401: "jeton invalide ou expiré", 403: "le jeton n'a pas la permission Contents en écriture",
                            404: f"dépôt « {GITHUB_REPO} » introuvable pour ce jeton (nom du dépôt ou accès du jeton)"}.get(r.status_code, r.text[:150])
                    self._error(f"envoi refusé ({r.status_code}) : {hint}")
                    return
                except Exception as e:
                    if attempt == 0:
                        continue
                    self._error(f"envoi impossible ({e})")
                    return

    def put_file(self, name: str, data: bytes, message: str):
        """Envoie un fichier quelconque (meta, archive) dans le dépôt de sauvegarde."""
        url = f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{name}"
        try:
            g = httpx.get(url, headers=self.headers, timeout=30)
            body = {"message": message, "content": base64.b64encode(data).decode()}
            if g.status_code == 200:
                body["sha"] = g.json().get("sha")
            r = httpx.put(url, headers=self.headers, json=body, timeout=60)
            if r.status_code not in (200, 201):
                print(f"Sauvegarde GitHub ({name}) : échec {r.status_code} {r.text[:200]}")
        except Exception as e:
            print(f"Sauvegarde GitHub ({name}) : échec ({e})")

    def restore_meta(self):
        if META.exists():
            return
        try:
            r = httpx.get(f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/meta.json", headers=self.headers, timeout=30)
            if r.status_code == 200 and r.json().get("content"):
                META.write_bytes(base64.b64decode(r.json()["content"]))
        except Exception:
            pass

    def loop(self):
        while True:
            time.sleep(120)
            self.push()


github_backup = None
if GITHUB_TOKEN and GITHUB_REPO and scheduler is None:
    github_backup = GitHubBackup()
    github_backup.restore()
    github_backup.restore_meta()
    threading.Thread(target=github_backup.loop, daemon=True).start()

BACKUP_ON = scheduler is not None or github_backup is not None or str(DATA_DIR) == "/data"


def trigger_backup():
    """Sauvegarde tout de suite après un nouvel appel, sans attendre la sauvegarde périodique."""
    if github_backup is not None:
        threading.Thread(target=github_backup.push, daemon=True).start()
    elif scheduler is not None:
        try:
            scheduler.trigger()
        except Exception:
            pass


def backup_status() -> dict:
    if github_backup is not None:
        return github_backup.status
    if scheduler is not None:
        return {"mode": "Hugging Face", "repo": DATASET_REPO, "lastOk": None, "lastError": None}
    if str(DATA_DIR) == "/data":
        return {"mode": "Disque persistant", "lastOk": None, "lastError": None}
    return {"mode": None}

FIELDS = {
    "id": str, "date": str, "junior": str, "scenarioId": str, "scenarioTitle": str, "kind": str,
    "level": str, "companyId": str, "companyName": str, "durationSec": int, "talkShare": int, "rdv": bool, "score": int, "criteria": dict,
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


@app.on_event("shutdown")
def save_on_shutdown():
    # Render arrête le service quand il se met en veille : on sauvegarde une dernière fois
    if github_backup is not None:
        github_backup.push(force=True)


@app.get("/")
def index():
    return FileResponse(HERE / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/config")
def config():
    return {
        "serverKey": bool(GEMINI_KEY),
        "managerReady": bool(MANAGER_PASSWORD),
        "backup": BACKUP_ON,
        "resetAt": read_meta().get("resetAt"),
    }


@app.post("/api/sessions")
async def save_session(request: Request):
    try:
        rec = clean(await request.json())
    except Exception:
        return JSONResponse({"ok": False, "error": "format"}, status_code=400)
    if not rec.get("id") or not rec.get("junior"):
        return JSONResponse({"ok": False, "error": "incomplet"}, status_code=400)
    if not after_reset(rec) or rec["id"] in known_ids():
        return {"ok": True, "skipped": True}
    with lock:
        with SESSIONS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    trigger_backup()
    return {"ok": True}


def known_ids() -> set:
    return {r.get("id") for r in read_sessions()}


@app.post("/api/sessions/sync")
async def sync_sessions(request: Request):
    """Les juniors renvoient tout leur historique local : le serveur ajoute les appels qu'il n'a pas encore.
    Ainsi, aucun appel n'est perdu, même après un redémarrage du serveur ou une coupure réseau."""
    try:
        body = await request.json()
        recs = [clean(r) for r in (body.get("sessions") or [])[:500] if isinstance(r, dict)]
    except Exception:
        return JSONResponse({"ok": False, "error": "format"}, status_code=400)
    ids = known_ids()
    new = [r for r in recs if r.get("id") and r.get("junior") and r["id"] not in ids and after_reset(r)]
    if new:
        with lock:
            with SESSIONS.open("a", encoding="utf-8") as f:
                for r in new:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        trigger_backup()
    return {"ok": True, "added": len(new)}


@app.get("/api/sessions")
def list_sessions(request: Request):
    if not MANAGER_PASSWORD:
        return JSONResponse({"ok": False, "error": "Le mot de passe Manager n'est pas configuré dans le Space."}, status_code=503)
    if request.headers.get("x-manager-password", "") != MANAGER_PASSWORD:
        return JSONResponse({"ok": False, "error": "Mot de passe incorrect."}, status_code=401)
    return {"ok": True, "sessions": read_sessions(), "backup": BACKUP_ON, "backupStatus": backup_status(),
            "resetAt": read_meta().get("resetAt")}


@app.post("/api/reset")
def reset_sessions(request: Request):
    """Remise à zéro des résultats (onglet Manager). Les données sont archivées avant d'être effacées."""
    if not MANAGER_PASSWORD:
        return JSONResponse({"ok": False, "error": "Le mot de passe Manager n'est pas configuré."}, status_code=503)
    if request.headers.get("x-manager-password", "") != MANAGER_PASSWORD:
        return JSONResponse({"ok": False, "error": "Mot de passe incorrect."}, status_code=401)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    reset_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    archived = 0
    archive_name = f"archives/sessions-{stamp}.jsonl"
    with lock:
        old = SESSIONS.read_bytes() if SESSIONS.exists() else b""
        archived = len([l for l in old.decode("utf-8", errors="ignore").splitlines() if l.strip()])
        if archived:
            ARCHIVES.mkdir(parents=True, exist_ok=True)
            (DATA_DIR / archive_name).write_bytes(old)
        SESSIONS.write_text("\n", encoding="utf-8")
        META.write_text(json.dumps({"resetAt": reset_at}), encoding="utf-8")
    if github_backup is not None:
        if archived:
            github_backup.put_file(archive_name, old, f"Archive avant remise à zéro {stamp}")
        github_backup.push(overwrite=True)
        github_backup.put_file("meta.json", META.read_bytes(), f"Remise à zéro {stamp}")
    return {"ok": True, "archived": archived, "resetAt": reset_at}


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


@app.post("/api/live-token")
async def live_token():
    """Jeton temporaire Gemini : le navigateur se connecte directement à Google pour la voix (plus rapide).
    Usage unique, à utiliser dans les 2 minutes, valable 30 minutes. La vraie clé ne quitte jamais le serveur."""
    if not GEMINI_KEY:
        return JSONResponse({"error": "GEMINI_API_KEY manquante"}, status_code=503)
    now = time.time()
    iso = lambda t: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
    body = {"uses": 1, "expireTime": iso(now + 30 * 60), "newSessionExpireTime": iso(now + 2 * 60)}
    last = ""
    async with httpx.AsyncClient(timeout=20) as c:
        for version in ("v1beta", "v1alpha"):
            r = await c.post(f"https://generativelanguage.googleapis.com/{version}/auth_tokens",
                             headers={"x-goog-api-key": GEMINI_KEY}, json=body)
            if r.status_code == 200 and r.json().get("name"):
                return {"token": r.json()["name"], "expireTime": body["expireTime"]}
            last = f"{r.status_code} {r.text[:200]}"
    print(f"Jeton temporaire refusé : {last}")
    return JSONResponse({"error": f"jeton refusé : {last}"}, status_code=502)


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
