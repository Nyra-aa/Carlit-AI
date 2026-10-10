// bilstm_tokenizer.js — Réplique de tensorflow.keras.preprocessing.text.Tokenizer
// + pad_sequences(padding="post", truncating="post"), pour le cas où le modèle
// sélectionné automatiquement par le pipeline (voir manifest.json) est le BiLSTM
// plutôt que le TF-IDF. Chargé et utilisé uniquement dans ce cas (cf. app.js).

let TOK = null; // { word_index, num_words, oov_token, maxlen, classes }

function setTokenizerData(data) {
  TOK = data;
}

// Filtre par défaut de Keras Tokenizer : ponctuation retirée avant découpage.
const KERAS_FILTER_RE = /[!"#$%&()*+,\-./:;<=>?@[\]^_`{|}~\t\n]/g;

function kerasTextToSequence(text) {
  const cleaned = text.toLowerCase().replace(KERAS_FILTER_RE, " ");
  const words = cleaned.split(/\s+/).filter((w) => w.length > 0);

  const oovIndex = TOK.oov_token ? TOK.word_index[TOK.oov_token] : undefined;
  const seq = [];
  for (const w of words) {
    const idx = TOK.word_index[w];
    if (idx !== undefined && (!TOK.num_words || idx < TOK.num_words)) {
      seq.push(idx);
    } else if (oovIndex !== undefined) {
      seq.push(oovIndex);
    }
    // sinon: mot hors vocabulaire et pas d'oov_token -> ignoré, comme sklearn/keras
  }
  return seq;
}

function padSequence(seq, maxlen) {
  const out = new Float32Array(maxlen); // 0 = padding ; float32 car le modèle
  // ONNX exporté attend ce dtype (cf. export_bilstm_for_web côté Python).
  const truncated = seq.slice(0, maxlen); // truncating="post"
  for (let i = 0; i < truncated.length; i++) out[i] = truncated[i];
  return out;
}

function vectorizeForBilstm(text) {
  if (!TOK) throw new Error("Tokenizer non chargé — appeler setTokenizerData() d'abord.");
  const seq = kerasTextToSequence(text);
  return padSequence(seq, TOK.maxlen);
}

if (typeof module !== "undefined") {
  module.exports = { setTokenizerData, kerasTextToSequence, padSequence, vectorizeForBilstm };
}
