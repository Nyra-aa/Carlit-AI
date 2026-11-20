"""Persistance des conversations dans UN fichier JSON (pas de base de données).

Format : {"version": 1, "conversations": {<id>: {id, title, created_at, updated_at, messages: [...]}}}
- écriture atomique (fichier temporaire + os.replace) : un arrêt brutal ne corrompt pas le fichier existant ;
- fichier illisible : il est renommé `*.corrompu-<date>` (jamais écrasé) et l'historique repart vide, avec un avertissement ;
- verrou : plusieurs sessions de navigateur sur le même serveur Streamlit partagent le même historique (pas d'authentification).
"""
import json, os, threading, uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TITLE = "Nouvelle conversation"

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def make_title(text, width=48):
    t = " ".join(text.split())
    return t if len(t) <= width else t[: width - 1].rstrip() + "…"

class ConversationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.warning = None
        self.data = {"version": 1, "conversations": {}}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            assert isinstance(data.get("conversations"), dict)
            self.data = data
        except Exception:
            bad = self.path.with_name(self.path.name + ".corrompu-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
            os.replace(self.path, bad)
            self.warning = f"Le fichier d'historique était illisible : il a été conservé sous « {bad.name} » et un historique vide a été créé."

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    # ---- lecture ----
    def list(self):
        """Conversations triées de la plus récemment modifiée à la plus ancienne (sans les messages)."""
        with self.lock:
            return sorted(({k: v for k, v in c.items() if k != "messages"} | {"n_messages": len(c["messages"])} for c in self.data["conversations"].values()),
                          key=lambda c: c["updated_at"], reverse=True)

    def get(self, cid):
        with self.lock:
            c = self.data["conversations"].get(cid)
            return json.loads(json.dumps(c)) if c else None      # copie : l'appelant ne modifie pas l'état interne

    # ---- écriture ----
    def create(self, title=DEFAULT_TITLE):
        with self.lock:
            cid, now = uuid.uuid4().hex, _now()
            self.data["conversations"][cid] = dict(id=cid, title=title, created_at=now, updated_at=now, messages=[])
            self._save()
            return cid

    def add_message(self, cid, role, content, **extra):
        with self.lock:
            c = self.data["conversations"][cid]
            msg = dict(role=role, content=content, ts=_now(), **extra)
            c["messages"].append(msg); c["updated_at"] = msg["ts"]
            self._save()
            return msg

    def rename(self, cid, title):
        with self.lock:
            if cid in self.data["conversations"]:
                self.data["conversations"][cid]["title"] = title; self._save()

    def delete(self, cid):
        """Supprime UNE conversation ; les autres sont intactes. Retourne True si elle existait."""
        with self.lock:
            existed = self.data["conversations"].pop(cid, None) is not None
            if existed: self._save()
            return existed
