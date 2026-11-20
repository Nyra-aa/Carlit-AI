"""Cœur RAG du projet B1 (module généré).

Le code des classes (Retriever, build_messages, parse_output, QwenGenerator, RagPipeline) est COPIÉ SANS MODIFICATION des cellules des étapes 5 et 6.
Seul le chargement des chemins/paramètres diffère : tout est lu dans le « bundle » du modèle final (dossier B1_BUNDLE_DIR), jamais codé en dur.
"""
import os, re, json, math, time, gc, hashlib, unicodedata, textwrap
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

BUNDLE_DIR = Path(os.environ.get("B1_BUNDLE_DIR", "b1_bundle"))
STEP2_DIR = BUNDLE_DIR / "step2_index"
STEP3_DIR = BUNDLE_DIR / "step3_dataset"
ADAPTER_DIR = BUNDLE_DIR / "adapter"
PIPELINE_CFG_PATH = BUNDLE_DIR / "step5_pipeline" / "pipeline_config.json"

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

manifest = json.load(open(BUNDLE_DIR / "bundle_manifest.json", encoding="utf-8"))
pcfg = json.load(open(PIPELINE_CFG_PATH, encoding="utf-8"))
index_cfg = json.load(open(STEP2_DIR / "index_config.json", encoding="utf-8"))
rag_prompt = json.load(open(STEP3_DIR / "rag_prompt.json", encoding="utf-8"))
chunks = load_jsonl(STEP2_DIR / "chunks.jsonl")
chunk_by_id = {c["chunk_id"]: c for c in chunks}
K = int(index_cfg["k_final"])
ABSTAIN_MARKER, ABSTAIN_ANSWER = rag_prompt["abstain_marker"], rag_prompt["abstain_answer"]
MODEL_ID = pcfg["model"]
MAX_NEW_TOKENS = int(pcfg["decoding"]["max_new_tokens"])
BATCH_SIZE = int(pcfg["decoding"]["batch_size"])
ENFORCE_CITATION = bool(pcfg["enforce_citation"])
FLOOR = pcfg["similarity_floor"]

# ======================= code repris des étapes 5/6 (inchangé) =======================


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower().replace("’", "'"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[\s  ]+", " ", s).strip()

STOPWORDS = set("le la les l un une des de du d et ou a au aux en dans par pour sur sous avec sans que qui quoi est sont etre ce cet cette ces se sa son ses "
                "leur leurs ne pas plus il ils elle elles on y c s n qu j m t si comme mais donc car faut faire peut doit quel quelle quels quelles".split())
def tokenize(text):
    toks = re.findall(r"[a-z0-9]+", norm(text))
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks if t not in STOPWORDS]

class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs_tokens]
        self.len = np.array([len(d) for d in docs_tokens], dtype=float); self.avg = self.len.mean()
        df = Counter(t for d in self.docs for t in d); n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    def scores(self, q_tokens):
        s = np.zeros(len(self.docs))
        for t in set(q_tokens):
            if t in self.idf:
                tf = np.array([d.get(t, 0) for d in self.docs], dtype=float)
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len / self.avg))
        return s

def rrf_fuse(rank_lists, k=60, topn=10):
    sc = defaultdict(float)
    for ranks in rank_lists:
        for pos, j in enumerate(ranks):
            sc[int(j)] += 1.0 / (k + pos + 1)
    return [j for j, _ in sorted(sc.items(), key=lambda x: -x[1])[:topn]]

def cite(c):
    """Chaîne de citation d'un chunk. IDENTIQUE à l'étape 3 (c'est elle qui apparaît dans les prompts d'entraînement)."""
    p = [x for x in c["pages"] if x >= 1]            # la page de présentation du PDF n'a pas de numéro imprimé
    pages = "page non numérotée" if not p else (f"p. {p[0]}" if len(p) == 1 else f"p. {p[0]}-{p[-1]}")
    where = "Présentation du guide" if c["fiche"] == 0 else f"Fiche {c['fiche']} — {c['fiche_title']}"
    return f"{where} — {c['section_label']}, {pages}"

EMBEDDERS = {
    "e5-base":  {"hf_id": "intfloat/multilingual-e5-base",  "q": "query: ", "p": "passage: "},
    "e5-large": {"hf_id": "intfloat/multilingual-e5-large", "q": "query: ", "p": "passage: "},
    "bge-m3":   {"hf_id": "BAAI/bge-m3",                    "q": "",        "p": ""},
}

class STEmbedder:
    """Enveloppe sentence-transformers (même code qu'à l'étape 2). Vecteurs normalisés : produit scalaire = cosinus."""
    def __init__(self, name):
        import torch
        from sentence_transformers import SentenceTransformer
        self.name, self.cfg = name, EMBEDDERS[name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.cfg["hf_id"], device=self.device)
        self.model.max_seq_length = 512
        if self.device == "cuda":
            self.model.half()
    def _enc(self, texts, prefix):
        return self.model.encode([prefix + t for t in texts], batch_size=32, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype("float32")
    def encode_queries(self, texts): return self._enc(texts, self.cfg["q"])

def make_embedder(name):
    return STEmbedder(name)

class Retriever:
    def __init__(self, out_dir, embedder=None):
        import faiss
        out_dir = Path(out_dir)
        self.cfg = json.load(open(out_dir / "index_config.json", encoding="utf-8"))
        self.chunks = load_jsonl(out_dir / "chunks.jsonl")
        self.index = faiss.read_index(str(out_dir / "faiss.index"))
        assert self.index.ntotal == len(self.chunks), "index FAISS et chunks désynchronisés"
        self.embedder = embedder or make_embedder(self.cfg["embedder"])
        self.bm25 = BM25([tokenize(c["embed_text"]) for c in self.chunks]) if self.cfg["method"] == "hybrid" else None
        self.k = self.cfg["k_final"]
    def search_batch(self, questions, k=None):
        k = k or self.k
        n = min(50, len(self.chunks))
        Q = self.embedder.encode_queries(list(questions))
        scores, ids = self.index.search(Q, n)
        out = []
        for qi, (q, s_row, i_row) in enumerate(zip(questions, scores, ids)):
            if self.bm25 is not None:
                bm_rank = np.argsort(-self.bm25.scores(tokenize(q)))[:n]
                order = rrf_fuse([i_row, bm_rank], k=self.cfg["rrf_k"], topn=k)
            else:
                order = [int(j) for j in i_row[:k]]
            cos = dict(zip(i_row.tolist(), s_row.tolist()))
            res = []
            for rank, j in enumerate(order, 1):
                c = self.chunks[j]
                cj = cos[int(j)] if int(j) in cos else float(Q[qi] @ self.index.reconstruct(int(j)))
                res.append({"rank": rank, "chunk": c, "cosine": cj})
            out.append({"question": q, "top1_cosine": float(s_row[0]), "results": res})
        return out


def build_messages(question, passages):
    """passages : liste de chunks (dict), dans l'ordre d'affichage [1], [2]…"""
    ctx = "\n\n".join(rag_prompt["passage_template"].format(n=i + 1, citation=cite(c), text=c["text"]) for i, c in enumerate(passages))
    return [{"role": "system", "content": rag_prompt["system"]},
            {"role": "user", "content": rag_prompt["user_template"].format(passages=ctx, question=question)}]


class QwenGenerator:
    def __init__(self, model_id=MODEL_ID, adapter_dir=ADAPTER_DIR, max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE):
        import torch, transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        if not torch.cuda.is_available():
            raise RuntimeError("Aucun GPU CUDA : bitsandbytes (4-bit) ne fonctionne que sur NVIDIA. Utiliser Colab/Kaggle, ou une autre chaîne d'inférence (llama.cpp / MLX).")
        self.torch, self.max_new_tokens, self.batch_size = torch, max_new_tokens, batch_size
        compute = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute)
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")      # left padding : requis pour la génération par lots
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        try:
            base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, dtype=compute, device_map={"": 0})
        except TypeError:                                                           # anciennes versions de transformers
            base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, torch_dtype=compute, device_map={"": 0})
        self.model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        self.model.eval()
        self.adapter_sha256 = hashlib.sha256((Path(adapter_dir) / "adapter_model.safetensors").read_bytes()).hexdigest()
        print(f"modèle chargé | dtype de calcul {compute} | VRAM utilisée {torch.cuda.memory_allocated() / 1e9:.1f} Go | adaptateur sha256 {self.adapter_sha256[:12]}…")

    def __call__(self, list_of_messages, adapter, max_new_tokens=None):
        """adapter=True -> configuration C ; adapter=False -> configuration B (LoRA désactivé, poids de base seuls)."""
        outs = []
        for s in range(0, len(list_of_messages), self.batch_size):
            batch = list_of_messages[s:s + self.batch_size]
            prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            kw = dict(max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
            with self.torch.inference_mode():
                if adapter:
                    out = self.model.generate(**enc, **kw)
                else:
                    with self.model.disable_adapter():
                        out = self.model.generate(**enc, **kw)
            outs += [self.tok.decode(g, skip_special_tokens=True).strip() for g in out[:, enc["input_ids"].shape[1]:]]
        return outs


SOURCES_RE = re.compile(r"(?:^|\n)\s*Sources\s*:\s*((?:\[\d+\](?:\s*,\s*)?)+)\s*$")

def parse_output(raw, k):
    """-> dict(abstained, cited (liste d'entiers valides), valid_citation, body)."""
    text = raw.strip()
    if text.upper().startswith(ABSTAIN_MARKER):
        return dict(abstained=True, cited=[], valid_citation=False, body=text)
    m = SOURCES_RE.search(text)
    cited = sorted({int(x) for x in re.findall(r"\d+", m.group(1))}) if m else []
    valid = bool(m) and bool(cited) and all(1 <= c <= k for c in cited)
    body = text[:m.start()].strip() if m else text
    return dict(abstained=False, cited=cited if valid else [], valid_citation=valid, body=body)


class RagPipeline:
    def __init__(self, retriever, generator, floor=None, enforce_citation=ENFORCE_CITATION, k=K):
        self.retriever, self.generator, self.floor, self.enforce, self.k = retriever, generator, floor, enforce_citation, k

    @staticmethod
    def _src(n, c):
        return dict(n=n, chunk_id=c["chunk_id"], fiche=c["fiche"], fiche_title=c.get("fiche_title"), section=c.get("section_label"),
                    pages=c["pages"], citation=cite(c), text=c["text"])

    def answer(self, questions, config="C"):
        assert config in ("B", "C"), "config doit valoir 'B' (RAG, adaptateur désactivé) ou 'C' (RAG + adaptateur)"
        t0 = time.time()
        hits = self.retriever.search_batch(questions, k=self.k)
        results = [None] * len(questions)
        todo = []                                                              # indices à envoyer au LLM
        for i, h in enumerate(hits):
            passages = [r["chunk"] for r in h["results"]]
            base = dict(question=questions[i], config=config, top1_cosine=h["top1_cosine"], raw_answer=None, sources=[],
                        retrieved=[dict(rank=r["rank"], n=r["rank"], chunk_id=r["chunk"]["chunk_id"], cosine=r["cosine"], fiche=r["chunk"]["fiche"],
                                        pages=r["chunk"]["pages"], citation=cite(r["chunk"])) for r in h["results"]])
            if self.floor is not None and h["top1_cosine"] < self.floor:       # [1] garde-fou retrieval : pas d'appel au LLM
                results[i] = dict(base, status="abstained_retrieval", answer=ABSTAIN_ANSWER)
            else:
                results[i] = base; todo.append(i)
        if todo:
            msgs = [build_messages(questions[i], [r["chunk"] for r in hits[i]["results"]]) for i in todo]
            raws = self.generator(msgs, adapter=(config == "C"))
            for i, raw in zip(todo, raws):
                p = parse_output(raw, self.k)
                passages = [r["chunk"] for r in hits[i]["results"]]
                res = dict(results[i], raw_answer=raw)
                if p["abstained"]:                                              # [2] abstention décidée par le LLM
                    res.update(status="abstained_model", answer=ABSTAIN_ANSWER)
                elif not p["valid_citation"] and self.enforce:                  # [3] pas de citation valide : réponse non sourcée
                    res.update(status="rejected_citation", answer=ABSTAIN_ANSWER)
                else:
                    res.update(status="answered", answer=p["body"], sources=[self._src(n, passages[n - 1]) for n in p["cited"]])
                results[i] = res
        dt = (time.time() - t0) / max(len(questions), 1)
        for r in results: r["latency_s"] = dt                                   # moyenne par question du lot (le lot est traité ensemble)
        return results

def show(res, width=420):
    print(f"[{res['config']}] {res['question']}\n  statut={res['status']} | cosinus top-1={res['top1_cosine']:.3f}")
    print("  réponse :", textwrap.shorten(res["answer"].replace("\n", " "), width))
    for s in res["sources"]:
        print(f"  source [{s['n']}] {s['citation']}  ({s['chunk_id']})")
    if res["raw_answer"] and res["status"] in ("abstained_model", "rejected_citation"):
        print("  sortie brute :", textwrap.shorten(res["raw_answer"].replace("\n", " "), 200))


# ======================= fin du code repris =======================
def startup_checks():
    """Contrôles de cohérence du bundle (mêmes garde-fous que l'étape 5/6). Retourne une liste de (nom, ok, détail) ; ok=False est bloquant."""
    out = []
    def add(name, ok, detail=""): out.append((name, ok, str(detail)))
    sha = hashlib.sha256(json.dumps(rag_prompt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    add("prompt RAG = celui de l'étape 5", sha == pcfg["prompt_sha256"], sha[:12])
    add("k du prompt = k_final de l'étape 2", rag_prompt["k_context"] == K, f"k={K}")
    add("embedder / méthode / chunking = étape 5",
        (pcfg["retrieval"]["embedder"], pcfg["retrieval"]["method"], pcfg["retrieval"]["chunking"], pcfg["retrieval"]["k"]) ==
        (index_cfg["embedder"], index_cfg["method"], index_cfg["chunking"], K), f"{index_cfg['embedder']} / {index_cfg['method']} / k={K}")
    add("nombre de chunks = index_config", len(chunks) == index_cfg["n_chunks"], len(chunks))
    val = STEP3_DIR / "val.jsonl"
    if val.exists():
        bad = 0
        for rec in load_jsonl(val):
            q = rec["messages"][1]["content"].split("Question :")[-1].strip()
            if build_messages(q, [chunk_by_id[c] for c in rec["meta"]["context_chunks"]]) != rec["messages"][:2]: bad += 1
        add("prompt reconstruit = prompt d'entraînement (val.jsonl)", bad == 0, f"{bad} écart(s)")
    else:
        add("prompt reconstruit = prompt d'entraînement (val.jsonl)", None, "val.jsonl absent du bundle")
    a = ADAPTER_DIR / "adapter_model.safetensors"
    if a.exists():
        ash = hashlib.sha256(a.read_bytes()).hexdigest()
        add("adaptateur = adaptateur de l'étape 5 (SHA-256)", ash == pcfg["adapter_sha256"], ash[:12])
    else:
        add("adaptateur présent dans le bundle", False, str(a))
    return out

def build_pipeline(generator=None, embedder=None):
    """Assemble retriever + générateur + pipeline avec les paramètres VALIDÉS (plancher, k, citation obligatoire) lus dans la config de l'étape 5."""
    retriever = Retriever(STEP2_DIR, embedder=embedder)
    generator = generator or QwenGenerator()
    assert generator.adapter_sha256 == pcfg["adapter_sha256"], "l'adaptateur chargé n'est pas celui de l'étape 5"
    return RagPipeline(retriever, generator, floor=FLOOR, enforce_citation=ENFORCE_CITATION, k=K)
