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
import re
import shutil
import hashlib
import threading
import unicodedata
import time
from pathlib import Path
from urllib.parse import unquote

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MANAGER_PASSWORD = os.getenv("MANAGER_PASSWORD", "").strip()   # accès à toutes les agences (CODIR, admin)
# Agences (BU) proposées aux juniors, et mot de passe Manager de chaque agence
FRANCE_PASSWORD = os.getenv("FRANCE_PASSWORD", "").strip()   # espace Direction France (sinon MANAGER_PASSWORD)
BUS = [b.strip() for b in os.getenv("BUS", "Lille,Paris,Lyon,Marseille,Nantes,Bordeaux,Toulouse,Montpellier").split(",") if b.strip()]
DEFAULT_BU = BUS[0] if BUS else "Lille"   # les appels enregistrés avant l'arrivée des agences sont rattachés à la première
def parse_bu_passwords(raw: str):
    """Lit MANAGER_PASSWORDS. Deux formats : simple « Lille:motdepasse, Paris:autre » (recommandé),
    ou JSON {"Lille":"motdepasse"}. Les erreurs courantes du JSON sont tolérées (guillemet oublié, guillemets typographiques)."""
    raw = (raw or "").strip()
    if not raw:
        return {}, False
    txt = raw.translate(str.maketrans({"“": '"', "”": '"', "„": '"', "«": '"', "»": '"', "‘": "'", "’": "'"}))
    try:
        d = json.loads(txt)
        if isinstance(d, dict):
            return {str(k).strip().lower(): str(v).strip() for k, v in d.items() if str(v).strip()}, False
    except Exception:
        pass
    out = {}
    body = txt.strip().strip("{}")
    for part in re.split(r",(?=\s*\"?[^\",:]+\"?\s*:)|\n", body):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k, v = k.strip().strip('"').strip("'").strip(), v.strip().strip('"').strip("'").strip()
        if k and v:
            out[k.lower()] = v
    # Le format simple « Lille:motdepasse, Paris:autre » est un format normal, pas une erreur
    return out, txt.lstrip().startswith("{")


BU_PASSWORDS, BU_PASSWORDS_REPAIRED = parse_bu_passwords(os.getenv("MANAGER_PASSWORDS", ""))
if BU_PASSWORDS_REPAIRED:
    print("MANAGER_PASSWORDS mal formé : lu en mode tolérant pour", ", ".join(sorted(BU_PASSWORDS)) or "aucune agence")
print("Mots de passe manager configurés pour :", ", ".join(sorted(BU_PASSWORDS)) or "aucune agence")
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
ACCOUNTS = DATA_DIR / "accounts.json"  # comptes saisis par les juniors, agence par agence, avec leurs fiches IA


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
def bu_of(rec: dict) -> str:
    return (rec.get("bu") or DEFAULT_BU).strip()


def after_reset(rec: dict) -> bool:
    """Ignore les appels antérieurs à la dernière remise à zéro (générale ou de l'agence du junior)."""
    meta = read_meta()
    limits = [meta.get("resetAt") or "", (meta.get("resetAtBu") or {}).get(bu_of(rec).lower(), "")]
    return (rec.get("date") or "") >= max(limits)


def manager_scope(request) -> str | None:
    """'*' pour l'accès toutes agences, le nom de l'agence pour un manager d'agence, None sinon."""
    # Le navigateur encode le mot de passe (accents et symboles autorisés)
    pw = unquote(request.headers.get("x-manager-password", "")).strip()
    if not pw:
        return None
    if MANAGER_PASSWORD and pw == MANAGER_PASSWORD:
        return "*"
    for bu, p in BU_PASSWORDS.items():
        if pw == p:
            return next((b for b in BUS if b.lower() == bu), bu.capitalize())
    return None


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
                       "lastErrorAt": None, "restored": None, "token": describe_token(GITHUB_TOKEN),
                       "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

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

    def restore_file(self, name: str, path: Path):
        if path.exists():
            return
        try:
            r = httpx.get(f"{GITHUB_API}/repos/{GITHUB_REPO}/contents/{name}", headers=self.headers, timeout=30)
            if r.status_code == 200 and r.json().get("content"):
                path.write_bytes(base64.b64decode(r.json()["content"]))
                print(f"{name} restauré depuis GitHub")
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
    github_backup.restore_file("accounts.json", ACCOUNTS)
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


def describe_token(t: str) -> str:
    """Décrit le jeton reçu sans jamais l'afficher : type, longueur, caractères suspects."""
    raw = os.getenv("GITHUB_TOKEN", "")
    if not t:
        return "aucun jeton reçu"
    kind = ("fine-grained (github_pat_)" if t.startswith("github_pat_") else
            "classique (ghp_)" if t.startswith("ghp_") else "format inconnu : il ne commence ni par github_pat_ ni par ghp_")
    expected = " (attendu : 93)" if t.startswith("github_pat_") else " (attendu : 40)" if t.startswith("ghp_") else ""
    issues = []
    if any(c in t for c in "\"'“”«»"):
        issues.append("contient des guillemets")
    if any(c.isspace() for c in t):
        issues.append("contient une espace ou un retour à la ligne")
    if raw != raw.strip():
        issues.append("espaces autour (retirées automatiquement)")
    return f"jeton {kind}, {len(t)} caractères{expected}" + (" ; " + ", ".join(issues) if issues else "")


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
    "level": str, "bu": str, "companyId": str, "companyName": str, "durationSec": int, "talkShare": int, "rdv": bool, "score": int, "criteria": dict,
    "axe": str, "verdict": str, "erreursSmile": int, "prepFilled": int, "role": str,
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
    if out.get("junior"):
        out["junior"] = canon_name(out["junior"])
    out["receivedAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


# ---------- Une personne = un nom, quelles que soient les majuscules ou les accents ----------
PARTICLES = {"de", "du", "des", "la", "le", "van", "von", "da", "di", "del"}


def canon_name(name) -> str:
    """Mise en forme d'un nom : « victor DESENCLOS » devient « Victor Desenclos »."""
    s = re.sub(r"\s+", " ", str(name or "")).strip()
    out = []
    for i, w in enumerate(s.split(" ") if s else []):
        lw = w.lower()
        if i > 0 and lw in PARTICLES:
            out.append(lw)
            continue
        out.append(re.sub(r"(^|[-'’])(\w)", lambda m: m.group(1) + m.group(2).upper(), lw))
    return " ".join(out)


def name_key(name) -> str:
    """Clé de comparaison : sans majuscules, accents, espaces ni ponctuation."""
    t = unicodedata.normalize("NFD", str(name or "").lower())
    return re.sub(r"[^a-z0-9]", "", re.sub(r"[\u0300-\u036f]", "", t))


def unify_names(rows: list) -> list:
    """Regroupe à la lecture les variantes d'un même nom sous une seule forme (la plus fréquente).
    L'historique enregistré n'est jamais modifié."""
    counts = {}
    for r in rows:
        j = r.get("junior")
        if j:
            c = counts.setdefault(name_key(j), {})
            f = canon_name(j)
            c[f] = c.get(f, 0) + 1
    best = {k: max(v.items(), key=lambda kv: (kv[1], kv[0]))[0] for k, v in counts.items()}
    for r in rows:
        if r.get("junior"):
            r["junior"] = best.get(name_key(r["junior"]), r["junior"])
    return rows


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
    return unify_names(rows)


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
        "managerReady": bool(MANAGER_PASSWORD or BU_PASSWORDS),
        # Diagnostic de connexion manager (noms d'agences seulement, jamais les mots de passe)
        "managerAgencies": [next((b for b in BUS if b.lower() == k), k) for k in sorted(BU_PASSWORDS)],
        "managerAgenciesUnknown": [k for k in sorted(BU_PASSWORDS) if k not in {b.lower() for b in BUS}],
        "managerPasswordsRepaired": BU_PASSWORDS_REPAIRED,
        "directionPassword": bool(MANAGER_PASSWORD),
        "backup": BACKUP_ON,
        "bus": BUS,
        "resetAt": read_meta().get("resetAt"),
        "resetAtBu": read_meta().get("resetAtBu") or {},
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
    if not (MANAGER_PASSWORD or BU_PASSWORDS):
        return JSONResponse({"ok": False, "error": "Aucun mot de passe Manager n'est configuré sur le serveur."}, status_code=503)
    scope = manager_scope(request)
    if not scope:
        want = unquote(request.headers.get("x-manager-bu", "")).strip()
        if want and want.lower() not in BU_PASSWORDS:
            return JSONResponse({"ok": False, "error": f"Aucun mot de passe manager n'est configuré pour l'agence de {want} sur le serveur "
                                 "(variable MANAGER_PASSWORDS dans Render). Seul le mot de passe de direction fonctionne pour elle."}, status_code=401)
        return JSONResponse({"ok": False, "error": f"Mot de passe incorrect{' pour ' + want if want else ''}."}, status_code=401)
    rows = read_sessions()
    for r in rows:
        r["bu"] = bu_of(r)
    if scope != "*":
        rows = [r for r in rows if r["bu"].lower() == scope.lower()]
    return {"ok": True, "scope": scope, "bus": BUS, "sessions": rows, "backup": BACKUP_ON,
            "backupStatus": backup_status(), "resetAt": read_meta().get("resetAt")}


@app.post("/api/reset")
def reset_sessions(request: Request):
    """Remise à zéro (onglet Manager). Un manager d'agence n'efface que son agence ; l'accès général efface tout.
    Les données effacées sont archivées."""
    scope = manager_scope(request)
    if not scope:
        return JSONResponse({"ok": False, "error": "Mot de passe incorrect."}, status_code=401)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    reset_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    suffix = "" if scope == "*" else "-" + "".join(c for c in scope.lower() if c.isalnum())
    archive_name = f"archives/sessions-{stamp}{suffix}.jsonl"
    with lock:
        old = SESSIONS.read_bytes() if SESSIONS.exists() else b""
        rows = parse_lines(old)
        if scope == "*":
            removed, kept = rows, []
        else:
            removed = [r for r in rows if bu_of(r).lower() == scope.lower()]
            kept = [r for r in rows if bu_of(r).lower() != scope.lower()]
        if removed:
            ARCHIVES.mkdir(parents=True, exist_ok=True)
            (DATA_DIR / archive_name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in removed), encoding="utf-8")
        SESSIONS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept) or "\n", encoding="utf-8")
        meta = read_meta()
        if scope == "*":
            meta = {"resetAt": reset_at, "resetAtBu": {}}
        else:
            meta.setdefault("resetAtBu", {})[scope.lower()] = reset_at
        META.write_text(json.dumps(meta), encoding="utf-8")
    if github_backup is not None:
        if removed:
            github_backup.put_file(archive_name, (DATA_DIR / archive_name).read_bytes(), f"Archive avant remise à zéro {stamp}")
        github_backup.push(overwrite=True)
        github_backup.put_file("meta.json", META.read_bytes(), f"Remise à zéro {stamp} ({scope})")
    return {"ok": True, "archived": len(removed), "resetAt": reset_at, "scope": scope}


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


# ---------- Fiche prospect automatique (recherche web) ----------
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "").strip()   # facultatif, ex. "models/gemini-3-flash"
_models_cache = {"t": 0.0, "list": []}
SKIP_MODELS = ("live", "tts", "image", "audio", "embedding", "lite", "learnlm", "robotics", "computer", "aqa", "gemma")


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


async def research_models(c) -> list:
    """Modèles texte capables de chercher sur le web, du plus récent au plus ancien (flash d'abord)."""
    if RESEARCH_MODEL:
        return [RESEARCH_MODEL if RESEARCH_MODEL.startswith("models/") else "models/" + RESEARCH_MODEL]
    if _models_cache["list"] and time.time() - _models_cache["t"] < 3600:
        return _models_cache["list"]
    r = await c.get(f"{API}/models", params={"pageSize": 200, "key": GEMINI_KEY})
    names = []
    for m in (r.json().get("models") or []):
        n = m.get("name", "")
        low = n.lower()
        if "generateContent" not in (m.get("supportedGenerationMethods") or []):
            continue
        if "gemini" not in low or any(x in low for x in SKIP_MODELS):
            continue
        names.append(n)
    names.sort(key=lambda n: (_version(n), "flash" in n.lower(), "preview" not in n.lower()), reverse=True)
    _models_cache.update(t=time.time(), list=names)
    return names


NAMES = {
    "F": ["Claire Morel", "Julie Lambert", "Camille Roussel", "Sophie Bernard", "Laure Petit", "Anne Dubois", "Marion Fontaine", "Céline Garnier"],
    "M": ["Thomas Leroy", "Nicolas Girard", "Antoine Mercier", "Julien Faure", "Pierre Lemoine", "Mathieu Roux", "Romain Chevalier", "Olivier Blanc"],
}


def research_prompt(company: str, brand: str, offline: bool = False) -> str:
    cible = ("un commercial de UX-Republic, cabinet spécialiste de l'expérience utilisateur (UX) du groupe Smile. "
             "L'interlocuteur cible est un décideur UX, design, produit ou expérience client.") if brand == "uxr" else \
            ("un commercial de Smile, entreprise de services numériques spécialisée dans l'open source, le digital, la data et l'IA. "
             "L'interlocuteur cible est un décideur IT, digital, data ou applications métier.")
    regle_hors_ligne = (" ou dont tu es certain ; pour le champ actu, écris une chaîne vide "
                        "si tu ne connais pas d'actualité datée et sûre") if offline else ""
    intro = (f"À partir de tes connaissances (sans accès au web), décris l'entreprise « {company} » (France en priorité) "
             f"pour préparer un appel de prospection de {cible}") if offline else \
            f"Recherche sur le web l'entreprise « {company} » (France en priorité) pour préparer un appel de prospection de {cible}"
    return f"""{intro}

Réponds uniquement avec un objet JSON valide, sans texte autour ni balises Markdown, au format :
{{
  "company": "nom officiel de l'entreprise",
  "desc": "activité en quelques mots, ex. : distributeur d'articles de sport",
  "actu": "une actualité récente, précise et datée, utile pour une accroche (moins de 12 mois si possible)",
  "context": ["4 à 6 faits courts : taille, implantations, organisation, projets digitaux, data ou IT, recrutements tech, enjeux"],
  "besoin": "besoin probable en services numériques, en une phrase",
  "role": "poste de l'interlocuteur le plus pertinent à appeler",
  "gender": "F ou M, au hasard",
  "objection": "l'objection la plus probable de cet interlocuteur, en une phrase entre guillemets français",
  "decision": "qui d'autre intervient dans la décision",
  "relation": "Aucune relation connue avec Smile., sauf si une source publique mentionne un lien avec Smile"
}}

Règles :
- Uniquement des faits trouvés dans tes sources{regle_hors_ligne}. Si tu n'es pas sûr d'un fait, ne l'écris pas.
- Aucun nom de personne réelle : seulement des rôles.
- Tout en français, phrases courtes."""


def extract_json(text: str):
    text = (text or "").replace("```json", "").replace("```", "")
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(text[a:b + 1])
    except Exception:
        return None


def clean_fiche(f: dict, company: str) -> dict:
    s = lambda k, n=400: str(f.get(k) or "").strip()[:n]
    ctx = f.get("context") or []
    if isinstance(ctx, str):
        ctx = [x for x in ctx.split("\n")]
    ctx = [str(x).strip()[:240] for x in ctx if str(x).strip()][:6]
    gender = "F" if str(f.get("gender", "")).upper().startswith("F") else "M"
    pool = NAMES[gender]
    first, last = pool[int(hashlib.md5(company.lower().encode()).hexdigest(), 16) % len(pool)].split(" ", 1)
    return {
        "company": s("company", 80) or company, "desc": s("desc", 120) or "entreprise",
        "actu": s("actu"), "context": ctx, "besoin": s("besoin"), "role": s("role", 120) or "Directeur des systèmes d'information",
        "gender": gender, "first": first, "last": last,
        "objection": s("objection", 200), "decision": s("decision", 200),
        "relation": s("relation", 200) or "Aucune relation connue avec Smile.",
    }


@app.post("/api/research")
async def research(request: Request):
    """Recherche web (Google Search via Gemini) et fiche prospect prête à l'emploi. L'interlocuteur est toujours fictif."""
    if not GEMINI_KEY:
        return JSONResponse({"ok": False, "error": "GEMINI_API_KEY manquante sur le serveur."}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    company = re.sub(r"\s+", " ", str(body.get("company", ""))).strip()[:80]
    if len(company) < 2:
        return JSONResponse({"ok": False, "error": "Nom d'entreprise manquant."}, status_code=400)
    brand = "uxr" if body.get("brand") == "uxr" else "smile"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": research_prompt(company, brand)}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.3},
    }
    last = "Aucun modèle Gemini disponible pour la recherche."
    quota_hit = False

    async def ask(c, m, with_search: bool):
        nonlocal last, quota_hit
        body = dict(payload) if with_search else {
            "contents": [{"role": "user", "parts": [{"text": research_prompt(company, brand, offline=True)}]}],
            "generationConfig": {"temperature": 0.3},
        }
        try:
            r = await c.post(f"{API}/{m}:generateContent", params={"key": GEMINI_KEY}, json=body)
        except Exception as e:
            last = f"{m} : {e}"
            return None
        if r.status_code != 200:
            try:
                msg = str(r.json().get("error", {}).get("message", r.status_code))
            except Exception:
                msg = f"erreur {r.status_code}"
            if r.status_code == 429 or "quota" in msg.lower():
                quota_hit = True
            last = f"{m} : {msg}"
            return None
        cand = (r.json().get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))
        fiche = extract_json(text)
        if not isinstance(fiche, dict):
            last = f"{m} : réponse illisible"
            return None
        sources, seen = [], set()
        for ch in (cand.get("groundingMetadata") or {}).get("groundingChunks") or []:
            w = ch.get("web") or {}
            title = str(w.get("title") or "").strip()
            if title and title not in seen:
                seen.add(title)
                sources.append({"title": title[:120], "uri": str(w.get("uri") or "")})
        return {"ok": True, "model": m, "grounded": with_search, "fiche": clean_fiche(fiche, company), "sources": sources[:8]}

    async with httpx.AsyncClient(timeout=120) as c:
        try:
            models = await research_models(c)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Liste des modèles indisponible : {e}"}, status_code=502)
        # 1) Recherche web : le quota gratuit diffère d'un modèle à l'autre, on les essaie tous
        for m in models[:8]:
            res = await ask(c, m, True)
            if res:
                return res
        # 2) Sans recherche web : fiche tirée des connaissances du modèle, signalée comme à vérifier
        for m in models[:4]:
            res = await ask(c, m, False)
            if res:
                res["fiche"]["actu"] = res["fiche"].get("actu") or ""
                return res
    if quota_hit:
        return JSONResponse({"ok": False, "error": "Quota Gemini dépassé pour le moment. Réessaie plus tard, ou remplis la fiche à la main. "
                             "Pour lever la limite, active la facturation sur le projet Google de la clé."}, status_code=429)
    return JSONResponse({"ok": False, "error": last[:300]}, status_code=502)


# ---------- Comptes de chaque agence (portefeuilles et fiches partagées) ----------
_accounts_timer = None


def akey(t: str) -> str:
    t = re.sub(r"[\u0300-\u036f]", "", __import__("unicodedata").normalize("NFD", str(t or "").lower()))
    return re.sub(r"[^a-z0-9]", "", t)[:60]


def read_accounts() -> dict:
    try:
        return json.loads(ACCOUNTS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_accounts(data: dict):
    global _accounts_timer
    with lock:
        ACCOUNTS.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    if github_backup is not None:
        # Regroupe les modifications rapprochées en une seule sauvegarde
        if _accounts_timer:
            _accounts_timer.cancel()
        _accounts_timer = threading.Timer(15, lambda: github_backup.put_file("accounts.json", ACCOUNTS.read_bytes(), "Comptes des agences"))
        _accounts_timer.daemon = True
        _accounts_timer.start()


def find_bu(bu: str) -> str | None:
    return next((b for b in BUS if b.lower() == str(bu or "").strip().lower()), None)


@app.post("/api/portfolio")
async def save_portfolio(request: Request):
    """Enregistre le portefeuille d'un junior : ses comptes deviennent des comptes de son agence."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "format"}, status_code=400)
    bu = find_bu(body.get("bu"))
    who = canon_name(str(body.get("junior") or ""))[:80]
    first = canon_name(str(body.get("first") or ""))[:40]
    lines = [re.sub(r"\s+", " ", str(x)).strip()[:80] for x in (body.get("lines") or [])[:30]]
    if not bu or not who:
        return JSONResponse({"ok": False, "error": "incomplet"}, status_code=400)
    data = read_accounts()
    agency = data.setdefault(bu.lower(), {})
    wanted = {akey(x): x for x in lines if akey(x)}
    for k in list(agency.keys()):
        acc = agency[k]
        acc["owners"] = [o for o in acc.get("owners", []) if name_key(o.get("id")) != name_key(who)]
        if not acc["owners"]:
            del agency[k]
    for k, name in wanted.items():
        acc = agency.setdefault(k, {"name": name, "owners": []})
        acc["owners"].append({"id": who, "first": first})
    write_accounts(data)
    return {"ok": True}


@app.post("/api/accounts/fiche")
async def save_fiche(request: Request):
    """Partage avec l'agence la fiche IA d'un compte (faits publics et interlocuteur fictif uniquement)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "format"}, status_code=400)
    bu = find_bu(body.get("bu"))
    name = re.sub(r"\s+", " ", str(body.get("name") or "")).strip()[:80]
    f = body.get("fiche") if isinstance(body.get("fiche"), dict) else None
    if not bu or not akey(name) or not f:
        return JSONResponse({"ok": False, "error": "incomplet"}, status_code=400)
    keep = ("desc", "first", "last", "gender", "role", "actu", "context", "besoin", "objection", "decision", "relation", "refs")
    fiche = {k: str(f.get(k) or "")[:1500] for k in keep}
    sources = [{"title": str(x.get("title") or "")[:120], "uri": str(x.get("uri") or "")[:500]}
               for x in (body.get("sources") or [])[:8] if isinstance(x, dict)]
    data = read_accounts()
    acc = data.setdefault(bu.lower(), {}).setdefault(akey(name), {"name": name, "owners": []})
    acc.update(fiche=fiche, sources=sources, verified=bool(body.get("verified")),
               updatedAt=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_accounts(data)
    return {"ok": True}


@app.get("/api/accounts")
def list_accounts(bu: str = ""):
    b = find_bu(bu)
    if not b:
        return {"ok": True, "accounts": []}
    agency = read_accounts().get(b.lower(), {})
    out = [{"key": k, "name": a.get("name", k), "owners": sorted({canon_name(o.get("first")) or "?" for o in a.get("owners", [])}),
            "fiche": a.get("fiche"), "sources": a.get("sources", []), "verified": bool(a.get("verified"))}
           for k, a in agency.items()]
    return {"ok": True, "bu": b, "accounts": sorted(out, key=lambda x: x["name"].lower())}


# ---------- Espace Direction France : analytics nationales, agrégées par agence ----------
CRITERIA_IDS = ["accroche", "raison", "valeur", "decouverte", "ecoute", "objections", "rdv", "conclusion"]


def france_ok(request) -> bool:
    pw = unquote(request.headers.get("x-manager-password", "")).strip()
    if not pw:
        return False
    if FRANCE_PASSWORD:
        return pw == FRANCE_PASSWORD
    return bool(MANAGER_PASSWORD) and pw == MANAGER_PASSWORD


def _avg(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v)) if v else None


def _pct(n, d):
    return round(n / d * 100) if d else None


def block_stats(rows: list) -> dict:
    scored = [r for r in rows if r.get("score") is not None]
    calls = [r for r in scored if r.get("kind") == "appel"]
    actives = {r.get("junior") for r in rows if r.get("junior")}
    return {
        "calls": len(rows),
        "actives": len(actives),
        "perActive": round(len(rows) / len(actives), 1) if actives else 0,
        "avg": _avg([r["score"] for r in scored]),
        "rdvRate": _pct(sum(1 for r in calls if r.get("rdv")), len(calls)),
        "durationMin": round(sum(r.get("durationSec") or 0 for r in rows) / len(rows) / 60, 1) if rows else None,
        "talk": _avg([r.get("talkShare") for r in rows]),
        "prepRate": _pct(sum(1 for r in rows if (r.get("prepFilled") or 0) >= 2), len(rows)),
        "errors": round(sum(r.get("erreursSmile") or 0 for r in scored) / len(scored), 1) if scored else None,
    }


@app.get("/api/france")
def france(request: Request, days: int = 90, bu: str = ""):
    if not (FRANCE_PASSWORD or MANAGER_PASSWORD):
        return JSONResponse({"ok": False, "error": "Aucun mot de passe Direction n'est configuré sur le serveur (FRANCE_PASSWORD ou MANAGER_PASSWORD)."}, status_code=503)
    if not france_ok(request):
        return JSONResponse({"ok": False, "error": "Mot de passe incorrect."}, status_code=401)
    days = max(7, min(int(days or 90), 400))
    now = time.time()
    rows = [r for r in read_sessions() if after_reset(r) and r.get("role") != "manager"]
    for r in rows:
        r["bu"] = bu_of(r)
    since, prev_since = _iso(now - days * 86400), _iso(now - 2 * days * 86400)
    all_cur = [r for r in rows if (r.get("date") or "") >= since]
    all_prev = [r for r in rows if prev_since <= (r.get("date") or "") < since]
    # Filtre agence : indicateurs, graphique, trame, comptes, exercices et alertes ne portent que sur elle
    sel = find_bu(bu) if bu else None
    in_sel = (lambda r: r["bu"].lower() == sel.lower()) if sel else (lambda r: True)
    all_rows = rows
    rows = [r for r in rows if in_sel(r)]
    cur = [r for r in all_cur if in_sel(r)]
    prev = [r for r in all_prev if in_sel(r)]

    # Évolution sur 12 semaines (semaines commençant le lundi)
    import datetime as _dt
    today = _dt.datetime.utcfromtimestamp(now).date()
    monday = today - _dt.timedelta(days=today.weekday())
    weeks = []
    for i in range(11, -1, -1):
        start = monday - _dt.timedelta(weeks=i)
        end = start + _dt.timedelta(weeks=1)
        wr = [r for r in rows if start.isoformat() <= (r.get("date") or "")[:10] < end.isoformat()]
        weeks.append({"week": start.isoformat(), "calls": len(wr), "avg": _avg([r.get("score") for r in wr]),
                      "actives": len({r.get("junior") for r in wr})})

    # Par agence
    agencies = []
    for b in BUS:
        br = [r for r in all_cur if r["bu"].lower() == b.lower()]
        pr = [r for r in all_prev if r["bu"].lower() == b.lower()]
        st = block_stats(br)
        pst = block_stats(pr)
        last = max((r.get("date") or "" for r in all_rows if r["bu"].lower() == b.lower()), default="")
        crit = {}
        appels = [r for r in br if r.get("kind") == "appel" and isinstance(r.get("criteria"), dict)]
        for cid in CRITERIA_IDS:
            vals = [r["criteria"].get(cid) for r in appels if r["criteria"].get(cid) in ("acquis", "en_cours", "a_travailler")]
            crit[cid] = _pct(sum(1 for v in vals if v == "acquis"), len(vals))
        agencies.append({"bu": b, **st, "delta": (st["avg"] - pst["avg"]) if st["avg"] is not None and pst["avg"] is not None else None,
                         "lastActivity": last[:10], "criteria": crit})

    # Maîtrise de la trame au niveau national
    appels = [r for r in cur if r.get("kind") == "appel" and isinstance(r.get("criteria"), dict)]
    national_crit = {}
    for cid in CRITERIA_IDS:
        vals = [r["criteria"].get(cid) for r in appels if r["criteria"].get(cid) in ("acquis", "en_cours", "a_travailler")]
        national_crit[cid] = _pct(sum(1 for v in vals if v == "acquis"), len(vals))

    # Exercices et comptes les plus travaillés
    ex = {}
    for r in cur:
        k = r.get("kind") or "appel"
        key = f"appel-{r.get('level') or '?'}" if k == "appel" else k
        ex[key] = ex.get(key, 0) + 1
    acc = {}
    for r in cur:
        n = (r.get("companyName") or "").strip()
        if n:
            a = acc.setdefault(n, {"name": n, "calls": 0, "scores": [], "bus": set()})
            a["calls"] += 1
            a["bus"].add(r["bu"])
            if r.get("score") is not None:
                a["scores"].append(r["score"])
    top = sorted(acc.values(), key=lambda a: a["calls"], reverse=True)[:10]
    top = [{"name": a["name"], "calls": a["calls"], "avg": _avg(a["scores"]), "agencies": len(a["bus"])} for a in top]

    # Points d'attention
    alerts = []
    cutoff = _iso(now - 14 * 86400)[:10]
    scoped = [a for a in agencies if not sel or a["bu"] == sel]
    for a in scoped:
        if a["lastActivity"] and a["lastActivity"] < cutoff:
            alerts.append(f"{a['bu']} : aucun entraînement depuis le {a['lastActivity'][8:10]}/{a['lastActivity'][5:7]}.")
    active_ag = [a for a in agencies if a["calls"] >= 3 and a["rdvRate"] is not None]
    if sel:
        me = next((a for a in active_ag if a["bu"] == sel), None)
        ranked = sorted(active_ag, key=lambda a: a["rdvRate"], reverse=True)
        if me and len(ranked) >= 2:
            rk = ranked.index(me) + 1
            alerts.append(f"{sel} est {rk}{'re' if rk == 1 else 'e'} sur {len(ranked)} agences actives pour le taux de rendez-vous ({me['rdvRate']} %).")
    elif len(active_ag) >= 2:
        low = min(active_ag, key=lambda a: a["rdvRate"])
        alerts.append(f"{low['bu']} a le taux de rendez-vous le plus bas ({low['rdvRate']} %).")
    weak = [(cid, v) for cid, v in national_crit.items() if v is not None and v < 35]
    for cid, v in weak[:2]:
        alerts.append(f"Critère « {cid} » acquis dans seulement {v} % des appels, au niveau national.")
    never = [a["bu"] for a in scoped if not a["lastActivity"]]
    if never:
        alerts.append("Jamais utilisé : " + ", ".join(never) + ".")

    return {"ok": True, "days": days, "bu": sel, "bus": BUS, "kpi": block_stats(cur), "kpiPrev": block_stats(prev), "weeks": weeks,
            "agencies": agencies, "criteria": national_crit, "exercises": ex, "topAccounts": top, "alerts": alerts,
            "managerCalls": sum(1 for r in read_sessions() if r.get("role") == "manager" and (r.get("date") or "") >= since
                                and (not sel or bu_of(r).lower() == sel.lower()))}


# ---------- Classement des agences (moyennes uniquement) ----------
PERIODS = {"week": 7, "month": 30, "quarter": 90}
WEEKLY_TARGET = 6   # appels par junior et par semaine pour une régularité de 100


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def bu_stats(rows: list, days: int) -> dict:
    out = {}
    for r in rows:
        b = bu_of(r)
        d = out.setdefault(b.lower(), {"bu": b, "juniors": set(), "calls": 0, "scores": [], "appels": 0, "rdv": 0})
        d["juniors"].add(r.get("junior") or "?")
        d["calls"] += 1
        if r.get("score") is not None:
            d["scores"].append(r["score"])
        if r.get("kind") == "appel" and r.get("score") is not None:
            d["appels"] += 1
            d["rdv"] += 1 if r.get("rdv") else 0
    target = WEEKLY_TARGET * days / 7
    for d in out.values():
        n = len(d["juniors"])
        d["avg"] = round(sum(d["scores"]) / len(d["scores"])) if d["scores"] else None
        d["perJunior"] = round(d["calls"] / n, 1) if n else 0
        d["rdvRate"] = round(d["rdv"] / d["appels"] * 100) if d["appels"] else None
        reg = min(100, d["perJunior"] / target * 100) if target else 0
        d["index"] = round(0.6 * d["avg"] + 0.4 * reg) if d["avg"] is not None else None
        d["ranked"] = n >= 2 and d["index"] is not None
    return out


@app.get("/api/leaderboard")
def leaderboard(period: str = "week", bu: str = "", junior: str = ""):
    """Classement des agences. Ne renvoie que des moyennes, jamais de noms ; la place d'un junior dans son agence
    lui est donnée sans révéler les autres."""
    days = PERIODS.get(period, 7)
    now = time.time()
    # Les appels d'entraînement des managers ne comptent pas dans le classement
    rows = [r for r in read_sessions() if after_reset(r) and r.get("role") != "manager"]
    cur = [r for r in rows if (r.get("date") or "") >= _iso(now - days * 86400)]
    prev = [r for r in rows if _iso(now - 2 * days * 86400) <= (r.get("date") or "") < _iso(now - days * 86400)]
    s_cur, s_prev = bu_stats(cur, days), bu_stats(prev, days)
    table = []
    for b in BUS:
        d = s_cur.get(b.lower()) or {"bu": b, "juniors": set(), "calls": 0, "avg": None, "perJunior": 0, "rdvRate": None, "index": None, "ranked": False}
        p = s_prev.get(b.lower())
        delta = d["index"] - p["index"] if d.get("index") is not None and p and p.get("index") is not None else None
        table.append({"bu": b, "index": d["index"], "avg": d["avg"], "perJunior": d["perJunior"], "rdvRate": d["rdvRate"],
                      "juniors": len(d["juniors"]), "calls": d["calls"], "ranked": d["ranked"], "delta": delta})
    ranked = sorted([t for t in table if t["ranked"]], key=lambda t: t["index"], reverse=True)
    for i, t in enumerate(ranked):
        t["rank"] = i + 1
    table = ranked + [t for t in table if not t["ranked"]]
    me = None
    if bu and junior:
        mine = [r for r in cur if bu_of(r).lower() == bu.lower() and r.get("score") is not None]
        per = {}
        for r in mine:
            per.setdefault(r.get("junior"), []).append(r["score"])
        avgs = sorted(((sum(v) / len(v), k) for k, v in per.items()), reverse=True)
        names = [k for _, k in avgs]
        mine_name = next((n for n in names if name_key(n) == name_key(junior)), None)
        if mine_name:
            me = {"rank": names.index(mine_name) + 1, "of": len(names), "avg": round(sum(per[mine_name]) / len(per[mine_name]))}
    return {"ok": True, "period": period, "days": days, "rows": table, "rankedCount": len(ranked), "me": me}


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
