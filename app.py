"""
app.py — Fake News Detector (Streamlit)

Loads tfidf_vectorizer.pkl + linear_svm.pkl (produced by Save_Model.ipynb),
runs the same preprocessing pipeline used at training time (see preprocess.py),
and classifies a headline + article body as REAL or FAKE.

Run locally:
    streamlit run app.py

Deploy:
    Push this folder (app.py, preprocess.py, requirements.txt, .gitignore,
    tfidf_vectorizer.pkl, linear_svm.pkl) to a GitHub repo, then deploy on
    https://share.streamlit.io pointing at app.py.
"""

import pickle
from pathlib import Path

import numpy as np
import streamlit as st

from preprocess import preprocess, ensure_nltk_resources

APP_DIR = Path(__file__).parent
LABELS = {0: "REAL", 1: "FAKE"}  
# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }

    .app-header {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
    }
    .app-header h1 {
        color: white;
        font-size: 2rem;
        margin: 0 0 0.4rem 0;
        font-weight: 700;
    }
    .app-header p {
        color: rgba(255,255,255,0.88);
        font-size: 0.98rem;
        margin: 0;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.15);
        transition: transform 0.05s ease-in-out;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        border-color: #7C3AED;
        color: #7C3AED;
    }
    div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        font-size: 1.02rem;
        padding: 0.6rem 0;
    }
    div.stFormSubmitButton > button:hover {
        opacity: 0.92;
        color: white;
    }

    .result-card {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 6px solid;
    }
    .result-card.real {
        background: rgba(34, 197, 94, 0.10);
        border-color: #22C55E;
    }
    .result-card.fake {
        background: rgba(239, 68, 68, 0.10);
        border-color: #EF4444;
    }
    .result-icon { font-size: 2.2rem; line-height: 1; }
    .result-label {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .result-card.real .result-label { color: #22C55E; }
    .result-card.fake .result-label { color: #EF4444; }
    .result-sub {
        font-size: 0.92rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Cached setup: nltk resources + model artifacts
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Setting up language resources...")
def setup_nltk():
    ensure_nltk_resources()
    return True


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    vec_path = APP_DIR / "Models" / "tfidf_vectorizer.pkl"
    model_path = APP_DIR / "Models" / "linear_svm.pkl"
    if not vec_path.exists() or not model_path.exists():
        return None, None
    with open(vec_path, "rb") as f:
        tfidf = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return tfidf, model


setup_nltk()
tfidf, model = load_artifacts()

if tfidf is None or model is None:
    st.error(
        "Couldn't find **tfidf_vectorizer.pkl** and/or **linear_svm.pkl** in the `Models/` folder.\n\n"
        "Place both files inside a `Models/` folder next to `app.py` before running Streamlit."
    )
    st.stop()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📰 About")
    st.write(
        "This app classifies news as **Real** or **Fake** using a "
        "TF-IDF + Linear SVM model trained on the WELFake dataset "
        "(~72,000 labeled news articles)."
    )

    st.markdown("### Test-set performance")
    m1, m2 = st.columns(2)
    m1.metric("Accuracy", "95.2%")
    m2.metric("F1 Score", "94.7%")
    st.caption("ROC-AUC: 98.9%")

    with st.expander("How it works"):
        st.markdown(
            "1. Clean text (strip URLs, emails, HTML, extra whitespace)\n"
            "2. Lowercase + remove punctuation\n"
            "3. Tokenize\n"
            "4. Remove stopwords\n"
            "5. Lemmatize\n"
            "6. Vectorize with TF-IDF (unigrams + bigrams)\n"
            "7. Classify with a linear SVM"
        )

    st.divider()
    st.caption("Built with scikit-learn + Streamlit")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>📰 Fake News Detector</h1>
        <p>Paste a headline and article body — the model will judge whether it looks Real or Fake.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Example loader
# --------------------------------------------------------------------------
SAMPLE_REAL_TITLE = "Local Transit Authority Announces Expanded Bus Routes Starting Next Month"
SAMPLE_REAL_TEXT = (
    "The city transit authority confirmed on Tuesday that three new bus routes will "
    "begin service next month, aiming to reduce commute times for residents in the "
    "northern suburbs. Officials said the expansion follows a year-long review of "
    "ridership data and public feedback sessions held earlier this year. Funding for "
    "the new routes comes from a combination of state transportation grants and the "
    "city's existing transit budget. Officials added that updated schedules will be "
    "published on the authority's website closer to the launch date."
)

SAMPLE_FAKE_TITLE = "Scientists 'Confirm' Drinking Coffee Backwards Reverses Aging, Doctors Baffled"
SAMPLE_FAKE_TEXT = (
    "A viral post claims that a secret group of scientists has discovered that "
    "drinking coffee while standing on one's head can reverse the aging process "
    "by up to twenty years. According to the anonymous post, which cites no "
    "actual research institution, thousands of people have supposedly tried the "
    "method with dramatic results overnight. No peer-reviewed study, clinical "
    "trial, or named researcher is mentioned anywhere in the claim, and medical "
    "experts have not been able to verify any part of it."
)

if "headline_input" not in st.session_state:
    st.session_state.headline_input = ""
if "article_input" not in st.session_state:
    st.session_state.article_input = ""

st.write("**Not sure what to try?**")
ex_col1, ex_col2, ex_col3 = st.columns(3)
with ex_col1:
    if st.button("📗 Real-style example", use_container_width=True):
        st.session_state.headline_input = SAMPLE_REAL_TITLE
        st.session_state.article_input = SAMPLE_REAL_TEXT
with ex_col2:
    if st.button("📕 Fake-style example", use_container_width=True):
        st.session_state.headline_input = SAMPLE_FAKE_TITLE
        st.session_state.article_input = SAMPLE_FAKE_TEXT
with ex_col3:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.headline_input = ""
        st.session_state.article_input = ""

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with st.form("news_form"):
    headline = st.text_input(
        "Headline (optional)",
        key="headline_input",
        placeholder="e.g. Local Council Approves New Park Funding",
    )
    article = st.text_area(
        "Article text",
        key="article_input",
        height=220,
        placeholder="Paste the full article text here...",
    )
    submitted = st.form_submit_button("🔍 Analyze", use_container_width=True)

# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
if submitted:
    if not article.strip():
        st.warning("Please paste some article text before analyzing.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = preprocess(headline, article)

            if not cleaned.strip():
                st.warning(
                    "After cleaning, there wasn't any usable text left to analyze "
                    "(e.g. the input was only URLs, punctuation, or stopwords). "
                    "Try pasting a longer excerpt."
                )
                st.stop()

            vec = tfidf.transform([cleaned])
            pred = int(model.predict(vec)[0])
            score = float(model.decision_function(vec)[0])

            # LinearSVC has no predict_proba — this squashes the decision
            # margin into (0, 1) as a rough, uncalibrated confidence signal.
            prob_real = 1.0 / (1.0 + np.exp(-score))
            confidence = (1.0 - prob_real) if pred == 0 else prob_real

        is_real = pred == 0
        label_text = LABELS[pred]
        css_class = "real" if is_real else "fake"
        icon = "✅" if is_real else "🚫"

        st.markdown(
            f"""
            <div class="result-card {css_class}">
                <div class="result-icon">{icon}</div>
                <div>
                    <div class="result-label">{label_text} NEWS</div>
                    <div class="result-sub">Model confidence: {confidence * 100:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(min(max(confidence, 0.0), 1.0))

        st.caption(
            "Confidence is derived from the SVM's decision margin — a useful relative "
            "signal, not a calibrated probability. Always cross-check important claims "
            "against a trusted source."
        )

        with st.expander("See the cleaned text the model actually scored"):
            preview = cleaned if len(cleaned) <= 2000 else cleaned[:2000] + " ..."
            st.code(preview, language=None)

st.divider()
st.caption("TF-IDF + Linear SVM · trained on the WELFake dataset · for educational use, not a substitute for fact-checking.")
