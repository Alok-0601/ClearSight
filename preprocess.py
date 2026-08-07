"""
preprocess.py

Text preprocessing pipeline used by app.py.

IMPORTANT: this must stay IN SYNC with the pipeline used in Save_Model.ipynb
to train tfidf_vectorizer.pkl / linear_svm.pkl. If you change the cleaning
steps here without retraining, predictions will be unreliable.

Pipeline (identical order to training):
    title + text
        -> clean_text        (strip urls / emails / html / newlines / tabs / extra spaces)
        -> lowercase
        -> remove punctuation
        -> tokenize (nltk word_tokenize)
        -> remove stopwords (nltk english stopwords)
        -> lemmatize (nltk WordNetLemmatizer, default POS)
        -> " ".join(tokens)   -> ready for tfidf.transform([...])
"""

import re
import string

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

_lemmatizer = WordNetLemmatizer()
_stop_words = None  # lazy-loaded after nltk resources are confirmed present


def ensure_nltk_resources() -> None:
    """Download any missing NLTK resources. Safe to call every run —
    it's a no-op (fast lookup, no network call) once resources exist."""
    for path, package in _REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words("english"))


# ---------- Cleaning ----------

def _remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def _remove_emails(text: str) -> str:
    return re.sub(r"\S+@\S+", "", text)


def _remove_html(text: str) -> str:
    return re.sub(r"<.*?>", "", text)


def _remove_newlines(text: str) -> str:
    return text.replace("\n", " ")


def _remove_tabs(text: str) -> str:
    return text.replace("\t", " ")


def _remove_extra_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    text = _remove_urls(text)
    text = _remove_emails(text)
    text = _remove_html(text)
    text = _remove_newlines(text)
    text = _remove_tabs(text)
    text = _remove_extra_spaces(text)
    return text


def _remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


# ---------- Public entry point ----------

def preprocess(title: str, text: str) -> str:
    """Take a raw headline + article body and return the final cleaned,
    tokenized, stopword-free, lemmatized string ready for tfidf.transform()."""
    ensure_nltk_resources()

    title = title or ""
    text = text or ""

    content = f"{title} {text}"

    content = clean_text(content)
    content = content.lower()
    content = _remove_punctuation(content)

    tokens = word_tokenize(content)
    tokens = [word for word in tokens if word not in _stop_words]
    tokens = [_lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)
