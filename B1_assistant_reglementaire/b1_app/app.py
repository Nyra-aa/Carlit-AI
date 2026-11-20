"""Sentinelle — assistant réglementaire sourcé (projet B1) : interface Streamlit.

L'interface ne réimplémente RIEN du pipeline : elle appelle `rag_core.RagPipeline` (code des étapes 5/6) avec les paramètres validés lus dans le bundle du modèle final
(k, embedder, méthode, plancher de similarité, prompt RAG, abstention, citation obligatoire). Seul réglage utilisateur : B (sans adaptateur) ou C (avec adaptateur QLoRA).
Lancement : `streamlit run app.py` (GPU NVIDIA requis pour Qwen 4-bit) — voir README.md.
"""
import os, re, sys, json, time, threading, traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
os.environ.setdefault("B1_BUNDLE_DIR", str(APP_DIR / "b1_bundle"))          # bundle du modèle final (voir export_final_bundle.ipynb)
DATA_DIR = Path(os.environ.get("B1_DATA_DIR", APP_DIR / "data"))              # historique des conversations (un fichier JSON)
sys.path.insert(0, str(APP_DIR))
MANIFEST = json.load(open(Path(os.environ["B1_BUNDLE_DIR"]) / "bundle_manifest.json", encoding="utf-8"))
from store import ConversationStore, make_title

TZ = ZoneInfo("Europe/Paris")
APP_NAME = "Sentinelle"
TAGLINE = "Assistant sécurité des données personnelles"
DISCLAIMER = ("Réponse générée automatiquement à partir du guide CNIL « Sécurité des données personnelles ». "
              "Ce n'est ni une position officielle de la CNIL, ni un avis juridique : vérifiez toujours dans le guide.")
SUGGESTIONS = [
    "Quelles règles appliquer pour choisir et gérer les mots de passe ?",
    "Comment sécuriser les sauvegardes des données personnelles ?",
    "Quelles précautions prendre pour les postes de travail et les équipements nomades ?",
    "Que doit contenir un contrat avec un sous-traitant en matière de sécurité ?",
]
STATUS_LABEL = {
    "answered": "Réponse sourcée",
    "abstained_retrieval": "Abstention — aucun passage du guide n'est assez proche de la question (plancher de similarité)",
    "abstained_model": "Abstention — le modèle a estimé que les passages fournis ne contiennent pas l'information",
    "rejected_citation": "Abstention — réponse du modèle écartée car sans citation valide",
}
CONFIG_LABEL = {"C": "Configuration C — Qwen + RAG + adaptateur QLoRA", "B": "Configuration B — Qwen + RAG, sans adaptateur"}

def logo(size=34):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
            '<path d="M32 4 52 12v16c0 14-8.5 25-20 32C20.5 53 12 42 12 28V12z" fill="#0F766E"/>'
            '<path d="M22 31l7 7 14-15" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></svg>')

st.set_page_config(page_title=f"{APP_NAME} — assistant CNIL", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container {max-width: 860px; padding-top: 2rem;}
[data-testid="stSidebar"] .stButton > button {text-align: left; justify-content: flex-start; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.brand {display:flex; align-items:center; gap:.6rem; margin-bottom:.4rem;}
.brand b {font-size:1.25rem;} .brand span {display:block; font-size:.75rem; opacity:.7;}
.group {font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; opacity:.6; margin:.9rem 0 .2rem;}
.hero {text-align:center; margin: 2.5rem 0 1.2rem;} .hero h1 {font-size:2rem; margin:.3rem 0;}
footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- ressources partagées (chargées une seule fois par serveur)
@st.cache_resource
def get_store():
    return ConversationStore(DATA_DIR / "conversations.json")

@st.cache_resource(show_spinner="Chargement de l'index et du modèle (première ouverture : 1 à 3 minutes)…")
def get_system():
    import rag_core as core
    checks = core.startup_checks()
    pipe = None if any(ok is False for _, ok, _ in checks) else core.build_pipeline()
    return dict(core=core, checks=checks, pipe=pipe, lock=threading.Lock())     # un seul GPU : les générations sont mises en file

store = get_store()
for key, default in (("current_id", None), ("config", MANIFEST.get("default_config", "C")), ("details_open", False)):
    st.session_state.setdefault(key, default)

# ----------------------------------------------------------------------------- utilitaires d'affichage
def to_local(iso):
    return datetime.fromisoformat(iso).astimezone(TZ)

def group_label(iso):
    days = (datetime.now(TZ).date() - to_local(iso).date()).days
    return "Aujourd'hui" if days <= 0 else "Hier" if days == 1 else "7 derniers jours" if days < 7 else "30 derniers jours" if days < 30 else "Plus ancien"

def esc(text):
    """Le `$` déclencherait du LaTeX dans le Markdown de Streamlit."""
    return text.replace("$", "\\$")

def format_answer(text, cited):
    text = esc(text)
    return re.sub(r"\[(\d+)\]", lambda m: f"**[{m.group(1)}]**" if int(m.group(1)) in cited else m.group(0), text)

# ----------------------------------------------------------------------------- boîtes de dialogue
@st.dialog("Supprimer la conversation ?")
def confirm_delete(cid):
    conv = store.get(cid)
    if conv is None:
        st.rerun()
    st.write(f"« {conv['title']} » et ses {len(conv['messages'])} message(s) seront définitivement supprimés. Les autres conversations ne sont pas touchées.")
    c1, c2 = st.columns(2)
    if c1.button("Supprimer", type="primary", width="stretch"):
        store.delete(cid)
        if st.session_state.current_id == cid:
            st.session_state.current_id = None
        st.rerun()
    if c2.button("Annuler", width="stretch"):
        st.rerun()

@st.dialog("Paramètres du modèle")
def settings_dialog():
    system = get_system(); core = system["core"]
    st.radio("Configuration utilisée pour répondre", list(CONFIG_LABEL), key="config", format_func=CONFIG_LABEL.get,
             help="Même modèle, même retrieval, même prompt : seul l'adaptateur QLoRA diffère (c'est ce que compare l'étape 6).")
    st.toggle("Ouvrir « Sources et détails » par défaut", key="details_open")
    st.markdown("**Paramètres validés (lecture seule)** — fixés aux étapes 2 à 5, volontairement non modifiables ici :")
    pc, ic = core.pcfg, core.index_cfg
    st.markdown("\n".join(f"- **{k}** : {v}" for k, v in {
        "Modèle": f"{core.MODEL_ID} ({pc['quantization']})",
        "Embeddings": f"{ic['embedder']} · recherche {ic['method']}",
        "Découpage": str(ic["chunking"]),
        "Passages fournis (k)": core.K,
        "Plancher de similarité": "désactivé" if core.FLOOR is None else f"{core.FLOOR:.3f} (cosinus du meilleur passage)",
        "Citation obligatoire": "oui" if core.ENFORCE_CITATION else "non",
        "Décodage": f"glouton, {core.MAX_NEW_TOKENS} jetons max",
        "Empreinte du prompt": pc["prompt_sha256"][:12],
        "Empreinte de l'adaptateur": pc["adapter_sha256"][:12],
        "Bundle exporté le": core.manifest.get("created", "?"),
    }.items()))
    with st.expander("Contrôles de cohérence au démarrage"):
        for name, ok, detail in system["checks"]:
            st.write(("✅" if ok else "⚠️ non vérifié" if ok is None else "❌"), name, f"— {detail}")

# ----------------------------------------------------------------------------- barre latérale
with st.sidebar:
    st.markdown(f'<div class="brand">{logo()}<div><b>{APP_NAME}</b><span>{TAGLINE}</span></div></div>', unsafe_allow_html=True)
    st.button("＋ Nouvelle conversation", width="stretch", type="primary", on_click=lambda: st.session_state.update(current_id=None))
    if store.warning:
        st.warning(store.warning)
    convs = store.list()
    if not convs:
        st.caption("Aucune conversation pour l'instant.")
    last_group = None
    for c in convs:
        g = group_label(c["updated_at"])
        if g != last_group:
            st.markdown(f'<div class="group">{g}</div>', unsafe_allow_html=True); last_group = g
        col_open, col_del = st.columns([0.8, 0.2])
        label = c["title"]
        if col_open.button(label, key=f"open_{c['id']}", width="stretch",
                           type="primary" if c["id"] == st.session_state.current_id else "secondary",
                           help=f"Modifiée le {to_local(c['updated_at']):%d/%m/%Y à %H:%M} · {c['n_messages']} message(s)"):
            st.session_state.current_id = c["id"]; st.rerun()
        if col_del.button("🗑", key=f"del_{c['id']}", help="Supprimer cette conversation"):
            confirm_delete(c["id"])
    st.divider()
    if st.button("⚙️ Paramètres du modèle", width="stretch"):
        settings_dialog()

# ----------------------------------------------------------------------------- système (modèle + index)
try:
    system = get_system()
except Exception as e:     # bundle absent, pas de GPU, etc.
    st.error("Impossible de démarrer le système.")
    st.code(f"{type(e).__name__}: {e}")
    st.info("Vérifier : (1) le dossier `b1_bundle` (voir README), (2) un GPU NVIDIA disponible (Qwen 4-bit via bitsandbytes), (3) les dépendances de `requirements.txt`.")
    st.stop()
core = system["core"]
if system["pipe"] is None:
    st.error("Le bundle du modèle final n'est pas cohérent avec les étapes validées : l'assistant est bloqué pour ne pas répondre avec de mauvais paramètres.")
    for name, ok, detail in system["checks"]:
        st.write(("✅" if ok else "⚠️" if ok is None else "❌"), name, f"— {detail}")
    st.stop()

# ----------------------------------------------------------------------------- génération
def run_pipeline(question, config):
    with system["lock"]:
        res = system["pipe"].answer([question], config)[0]
    for r in res["retrieved"]:                                  # texte des passages récupérés : affichage uniquement
        r["text"] = core.chunk_by_id[r["chunk_id"]]["text"]
    return res

def generate_and_store(cid, question):
    config = st.session_state.config
    try:
        res = run_pipeline(question, config)
        if not res["answer"].strip():
            raise ValueError("réponse vide")
        meta = {k: res[k] for k in ("status", "config", "top1_cosine", "latency_s", "raw_answer", "sources", "retrieved")}
        meta.update(floor=core.FLOOR, k=core.K)
        store.add_message(cid, "assistant", res["answer"], meta=meta)
    except Exception as e:                                       # les interruptions (bouton Stop) ne passent pas ici : elles ne sont pas des `Exception`
        traceback.print_exc()
        if "out of memory" in str(e).lower():
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        msg = "La réponse générée est vide. Vous pouvez relancer la question." if str(e) == "réponse vide" else \
              "Une erreur est survenue pendant la génération. Vous pouvez relancer la question."
        store.add_message(cid, "assistant", msg, error=True, error_detail=f"{type(e).__name__}: {str(e)[:300]}")

# ----------------------------------------------------------------------------- rendu d'un message assistant
def render_assistant(msg, key):
    meta = msg.get("meta") or {}
    if msg.get("error"):
        st.error(msg["content"])
        if msg.get("error_detail"):
            st.caption(f"Détail technique : {msg['error_detail']}")
        return
    status = meta.get("status", "answered")
    sources = meta.get("sources", [])
    if status == "answered":
        st.markdown(format_answer(msg["content"], {s["n"] for s in sources}))
        if sources:
            st.markdown("**Références**\n\n" + "\n".join(f"- **[{s['n']}]** {esc(s['citation'])}" for s in sources))
    else:
        st.markdown(esc(msg["content"]))
        st.caption(STATUS_LABEL.get(status, status))
    st.caption(DISCLAIMER)
    with st.expander("Sources et détails", expanded=st.session_state.details_open):
        tab_cit, tab_ret, tab_diag = st.tabs(["Passages cités", "Passages récupérés", "Diagnostic"])
        with tab_cit:
            if not sources:
                st.write("Aucun passage cité pour cette réponse.")
            for s in sources:
                pages = [p for p in s["pages"] if p >= 1]
                st.markdown(f"**[{s['n']}] {esc(s['citation'])}**")
                st.caption(f"identifiant du chunk : `{s['chunk_id']}` · fiche {s['fiche']} · pages {', '.join(map(str, pages)) if pages else 'non numérotée(s)'}")
                st.markdown("> " + esc(s["text"]).replace("\n", "\n> "))
            if sources:
                st.caption("Les numéros [n] de la réponse renvoient aux passages ci-dessus : ce sont exactement les passages transmis au modèle.")
        with tab_ret:
            ret = meta.get("retrieved", [])
            cited = {s["n"] for s in sources}
            if ret:
                st.dataframe([{"Passage": r["n"], "Cosinus": round(r["cosine"], 3), "Fiche": r["fiche"], "Référence": r["citation"],
                               "Cité": "oui" if r["n"] in cited else "", "chunk": r["chunk_id"]} for r in ret], hide_index=True, width="stretch")
                pick = st.selectbox("Lire le texte d'un passage", [r["n"] for r in ret], format_func=lambda n: f"Passage [{n}]", key=f"pick_{key}")
                st.markdown("> " + esc(ret[pick - 1]["text"]).replace("\n", "\n> "))
            else:
                st.write("Aucun passage n'a été transmis au modèle.")
        with tab_diag:
            st.markdown(f"- **Statut** : {STATUS_LABEL.get(status, status)}\n- **Configuration** : {CONFIG_LABEL.get(meta.get('config'), meta.get('config'))}\n"
                        f"- **Cosinus du meilleur passage** : {meta.get('top1_cosine', float('nan')):.3f} (plancher : "
                        f"{'désactivé' if meta.get('floor') is None else round(meta['floor'], 3)}) · **k** = {meta.get('k')}\n"
                        f"- **Durée** : {meta.get('latency_s', 0):.1f} s")
            st.caption("Une similarité élevée n'est pas une preuve : elle indique qu'un passage ressemble à la question, pas qu'il contient la réponse. "
                       "Seul le modèle, contraint par le prompt, décide si les passages suffisent.")
            if meta.get("raw_answer"):
                st.text_area("Sortie brute du modèle", meta["raw_answer"], height=140, disabled=True, key=f"raw_{key}")

# ----------------------------------------------------------------------------- zone principale
typed = st.chat_input("Posez votre question sur la sécurité des données personnelles…", max_chars=1500)    # Entrée envoie, Maj+Entrée = nouvelle ligne
question = st.session_state.pop("queued_question", None) or typed
if question and question.strip():
    if st.session_state.current_id is None or store.get(st.session_state.current_id) is None:
        st.session_state.current_id = store.create(make_title(question))
    store.add_message(st.session_state.current_id, "user", question.strip())
    st.session_state.generate_now = True                       # consommé (pop) avant la génération : une interruption ne relance pas la génération au rerun suivant

cid = st.session_state.current_id
conv = store.get(cid) if cid else None
if conv is None:
    st.session_state.current_id = None
    st.markdown(f'<div class="hero">{logo(56)}<h1>{APP_NAME}</h1><div>Posez vos questions sur la sécurité des données personnelles : '
                "les réponses s'appuient uniquement sur le guide pratique RGPD de la CNIL et citent leurs sources.</div></div>", unsafe_allow_html=True)
    st.caption("Si le guide ne contient pas l'information, l'assistant le dit au lieu d'inventer. Chaque question est traitée indépendamment (pas de mémoire des échanges précédents). "
               "Prototype académique : ne remplace ni la lecture du guide ni un avis juridique.")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"sug_{i}", width="stretch"):
            st.session_state.queued_question = s; st.rerun()
else:
    st.subheader(conv["title"])
    for i, m in enumerate(conv["messages"]):
        with st.chat_message(m["role"], avatar=None if m["role"] == "user" else "🛡️"):
            if m["role"] == "user":
                st.markdown(esc(m["content"]))
            else:
                render_assistant(m, f"{cid}_{i}")
    last = conv["messages"][-1]
    if last["role"] == "user":
        if st.session_state.pop("generate_now", False):
            with st.chat_message("assistant", avatar="🛡️"):
                with st.spinner("Recherche dans le guide et rédaction de la réponse…"):
                    generate_and_store(cid, last["content"])
            st.rerun()
        else:
            st.warning("La réponse à cette question n'a pas été générée (génération interrompue ou page rechargée).")
            if st.button("Relancer la génération"):
                st.session_state.generate_now = True; st.rerun()
    elif last.get("error") and st.button("Réessayer"):
        # on repose la dernière question utilisateur (le message d'erreur reste dans l'historique)
        prev = next((m for m in reversed(conv["messages"]) if m["role"] == "user"), None)
        if prev:
            st.session_state.queued_question = prev["content"]; st.rerun()
