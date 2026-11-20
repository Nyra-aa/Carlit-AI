# — interface du chatbot réglementaire B1

Interface Streamlit au-dessus du **vrai pipeline RAG** (étapes 2 à 5). `rag_core.py` reprend sans modification le code des notebooks (retriever, prompt, générateur, citation, abstention) ;
tous les paramètres (k, embedder, méthode, plancher de similarité, prompt, citation obligatoire) sont **lus dans le bundle**, jamais codés en dur, et **vérifiés au démarrage** (empreintes SHA-256).
Si un contrôle échoue, l'application s'arrête au lieu de répondre avec de mauvais paramètres.

## Contenu
| Fichier | Rôle |
|---|---|
| `app.py` | interface (barre latérale, historique, chat, « Sources et détails », paramètres) |
| `rag_core.py` | cœur RAG (généré à partir des cellules validées) + contrôles de démarrage |
| `store.py` | persistance des conversations dans **un fichier JSON** (écriture atomique, fichier corrompu mis de côté) |
| `run_on_colab.ipynb` | lancement sur Colab (GPU T4) avec lien public temporaire |
| `b1_bundle/` | **modèle final** (adaptateur + index + prompt + configuration), produit par `export_final_bundle.ipynb` — à placer à côté de `app.py` |

## Lancer (machine avec GPU NVIDIA ≥ 8 Go de VRAM)
```bash
pip install -r requirements.txt
streamlit run app.py            # variables facultatives : B1_BUNDLE_DIR (défaut ./b1_bundle), B1_DATA_DIR (défaut ./data)
```
Le premier démarrage télécharge Qwen2.5-7B-Instruct (≈ 15 Go). **Sans GPU NVIDIA** (bitsandbytes), le modèle ne se charge pas : l'interface l'indique.

## Fonctionnement
- Une question = une recherche (top-k), puis : plancher de similarité → prompt RAG → génération → citation `[n]` vérifiée → réponse **ou** abstention exacte
  (« INFORMATION ABSENTE : les passages fournis ne contiennent pas cette information. »).
- Les `[n]` renvoient aux passages **réellement transmis au modèle** ; « Sources et détails » donne fiche, section, pages, identifiant de chunk, texte, cosinus et sortie brute.
- Paramètres : seul choix utilisateur = configuration **C** (avec adaptateur QLoRA) ou **B** (sans). Le reste est affiché en lecture seule.

