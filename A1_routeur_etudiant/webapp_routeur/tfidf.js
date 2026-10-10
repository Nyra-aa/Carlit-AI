// tfidf.js — Réplique fidèle de sklearn.feature_extraction.text.TfidfVectorizer
// pour la configuration utilisée à l'entraînement :
//   lowercase=False (minuscule appliquée en amont), token_pattern=(?u)\b\w\w+\b,
//   ngram_range=(1,2), stop_words=<liste spaCy>, sublinear_tf=True,
//   smooth_idf=True, norm='l2'.
//
// Les stopwords sont retirés AVANT la formation des n-grammes (comme sklearn) :
// des tokens adjacents après filtrage forment un bigramme, même s'ils ne sont
// pas adjacents dans le texte original.

let VOCAB = null; // { vocabulary: {term:index}, idf: [...], stopwords: [...], classes: [...], n_features }
let STOPWORDS_SET = null;

async function loadVocab(url = "vocab.json") {
  const res = await fetch(url);
  VOCAB = await res.json();
  STOPWORDS_SET = new Set(VOCAB.stopwords);
  return VOCAB;
}

function setVocab(vocabObject) {
  // Utilisé par les tests Node (pas de fetch() disponible hors navigateur).
  VOCAB = vocabObject;
  STOPWORDS_SET = new Set(VOCAB.stopwords);
}

function tokenize(text) {
  // Equivalent de (?u)\b\w\w+\b : suites maximales de lettres/chiffres/underscore
  // Unicode, de longueur >= 2.
  const matches = text.match(/[\p{L}\p{N}_]+/gu) || [];
  return matches.filter((t) => t.length >= 2);
}

function filterStopwords(tokens) {
  return tokens.filter((t) => !STOPWORDS_SET.has(t));
}

function buildNgrams(tokens) {
  const grams = [];
  for (const t of tokens) grams.push(t); // unigrammes
  for (let i = 0; i < tokens.length - 1; i++) grams.push(tokens[i] + " " + tokens[i + 1]); // bigrammes
  return grams;
}

/**
 * Calcule le vecteur TF-IDF dense (Float32Array) d'un message, dans le même
 * espace de features que le pipeline scikit-learn entraîné.
 */
function vectorize(text) {
  if (!VOCAB) throw new Error("Vocabulaire non chargé — appeler loadVocab() d'abord.");
  const lower = text.toLowerCase();
  const tokens = filterStopwords(tokenize(lower));
  const grams = buildNgrams(tokens);

  const counts = new Map();
  for (const g of grams) {
    const idx = VOCAB.vocabulary[g];
    if (idx === undefined) continue; // hors vocabulaire -> ignoré
    counts.set(idx, (counts.get(idx) || 0) + 1);
  }

  const vec = new Float32Array(VOCAB.n_features);
  let sumSquares = 0;
  for (const [idx, count] of counts.entries()) {
    const tf = 1 + Math.log(count); // sublinear_tf
    const weight = tf * VOCAB.idf[idx];
    vec[idx] = weight;
    sumSquares += weight * weight;
  }
  const norm = Math.sqrt(sumSquares);
  if (norm > 0) {
    for (const idx of counts.keys()) vec[idx] /= norm;
  }
  return vec;
}

function getVocabInfo() {
  return VOCAB;
}

if (typeof module !== "undefined") {
  module.exports = { loadVocab, setVocab, tokenize, filterStopwords, buildNgrams, vectorize, getVocabInfo };
}
