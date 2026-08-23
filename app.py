"""Streamlit client for the ClearSight Verification API.

The Streamlit app is intentionally a client: prediction, URL extraction and
evidence retrieval happen in ``backend/main.py``. This keeps the web UI easy
to deploy on Streamlit Community Cloud while the API remains reusable by any
future frontend.
"""

import os
from typing import Any

import httpx
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


DEFAULT_LOCAL_API_URL = "http://127.0.0.1:8000"


def configured_api_url() -> str:
    """Read the API base URL from Streamlit secrets or a local environment."""
    fallback = os.getenv("BACKEND_API_URL", DEFAULT_LOCAL_API_URL)
    try:
        value = st.secrets.get("BACKEND_API_URL", fallback)
    except StreamlitSecretNotFoundError:
        value = fallback
    return str(value).rstrip("/")


API_URL = configured_api_url()

st.set_page_config(
    page_title="ClearSight News Verifier",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .app-header {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        padding: 2.2rem 2rem; border-radius: 16px; text-align: center;
        margin-bottom: 1.6rem; box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
    }
    .app-header h1 { color: white; font-size: 2rem; margin: 0 0 0.4rem; font-weight: 700; }
    .app-header p { color: rgba(255,255,255,0.88); font-size: 0.98rem; margin: 0; }
    div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white; border: none; font-size: 1.02rem; padding: 0.6rem 0;
    }
    .result-card {
        display: flex; align-items: center; gap: 1rem; padding: 1.4rem 1.6rem;
        border-radius: 14px; margin: 1.2rem 0 0.6rem; border-left: 6px solid;
    }
    .result-card.real { background: rgba(34, 197, 94, 0.10); border-color: #22C55E; }
    .result-card.fake { background: rgba(239, 68, 68, 0.10); border-color: #EF4444; }
    .result-icon { font-size: 2.2rem; line-height: 1; }
    .result-label { font-size: 1.4rem; font-weight: 800; letter-spacing: 0.02em; }
    .result-card.real .result-label { color: #22C55E; }
    .result-card.fake .result-label { color: #EF4444; }
    .result-sub { font-size: 0.92rem; color: rgba(255,255,255,0.7); margin-top: 0.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_error_message(response: httpx.Response) -> str:
    """Return the API's consistent error message when one is available."""
    try:
        body = response.json()
    except ValueError:
        return response.text or f"The API returned HTTP {response.status_code}."
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        return "The submitted data was invalid. Please check the input and try again."
    return f"The API returned HTTP {response.status_code}."


def request_verification(path: str, payload: dict[str, str]) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=httpx.Timeout(65.0, connect=10.0)) as client:
            response = client.post(f"{API_URL}{path}", json=payload)
    except httpx.RequestError as exc:
        st.error(
            "The verification API could not be reached. "
            f"Check the BACKEND_API_URL setting and that the API is running. ({exc})"
        )
        return None
    if response.is_error:
        st.error(api_error_message(response))
        return None
    return response.json()


def render_result(result: dict[str, Any]) -> None:
    prediction = result["prediction"]
    confidence = float(result["confidence"])
    is_real = prediction == "REAL"
    css_class = "real" if is_real else "fake"
    icon = "✅" if is_real else "🚫"

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="result-icon">{icon}</div>
            <div>
                <div class="result-label">{prediction} NEWS</div>
                <div class="result-sub">Model confidence: {confidence * 100:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(confidence, 0.0), 1.0))
    st.caption(result["confidence_note"])

    article = result["article"]
    st.subheader("Article details")
    details = {
        "Title": article.get("title") or "Not available",
        "Publisher": article.get("publisher") or "Not available",
        "Publication date": article.get("published_at") or "Not available",
        "Extraction": article.get("extraction_method", "provided_text"),
    }
    st.json(details, expanded=False)
    if article.get("url"):
        st.link_button("Open submitted article", article["url"])

    st.subheader("Retrieved evidence")
    evidence = result.get("evidence", [])
    if not evidence:
        st.info("No matching evidence was returned. Configure one or both evidence APIs to enable live retrieval.")
    for item in evidence:
        label = f"{item['relation'].replace('_', ' ').title()} · {item['relevance']} relevance"
        with st.expander(f"{item['type'].replace('_', ' ').title()}: {item['title']} — {label}"):
            if item.get("publisher"):
                st.caption(f"Publisher: {item['publisher']}")
            if item.get("published_at"):
                st.caption(f"Published: {item['published_at']}")
            if item.get("rating"):
                st.write(f"Rating: {item['rating']}")
            if item.get("claim"):
                st.write(f"Claim: {item['claim']}")
            if item.get("summary"):
                st.write(item["summary"])
            st.caption(item["relation_reason"])
            st.link_button("Open source", item["url"], key=f"source-{result['id']}-{item['url']}")

    st.subheader("Evidence source status")
    for source in result.get("sources", []):
        status = source["status"]
        if status == "ok":
            st.success(f"{source['source']}: connected")
        elif status == "not_configured":
            st.warning(f"{source['source']}: not configured")
        else:
            st.error(f"{source['source']}: {source.get('detail') or 'request failed'}")

    st.caption(f"Verification ID: {result['id']} · {result['created_at']}")
    st.info("Evidence is retrieved context, not an automatic fact-check verdict. Cross-check important claims with primary sources.")


with st.sidebar:
    st.markdown("## 📰 About ClearSight")
    st.write(
        "ClearSight combines the group's existing TF-IDF + Linear SVM classifier "
        "with optional fact-check and news evidence retrieval."
    )
    st.markdown("### API connection")
    st.code(API_URL, language=None)
    st.caption("Set `BACKEND_API_URL` in Streamlit secrets for the deployed site.")
    st.markdown("### Model")
    m1, m2 = st.columns(2)
    m1.metric("Accuracy", "95.2%")
    m2.metric("F1 Score", "94.7%")
    st.caption("WELFake dataset · confidence is an uncalibrated SVM margin.")


st.markdown(
    """
    <div class="app-header">
        <h1>📰 ClearSight News Verifier</h1>
        <p>Classify an article and retrieve related fact-checking and news evidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

text_tab, url_tab = st.tabs(["Paste article text", "Verify article URL"])

with text_tab:
    with st.form("text_verification_form"):
        title = st.text_input("Headline (optional)", placeholder="e.g. Local Council Approves New Park Funding")
        text = st.text_area(
            "Article text",
            height=220,
            placeholder="Paste at least a short article excerpt (40 characters or more)...",
        )
        verify_text = st.form_submit_button("🔍 Verify article", use_container_width=True)
    if verify_text:
        if len(text.strip()) < 40:
            st.warning("Please provide at least 40 characters of article text.")
        else:
            with st.spinner("Classifying article and retrieving evidence..."):
                result = request_verification("/verify", {"title": title.strip(), "text": text.strip()})
            if result:
                render_result(result)

with url_tab:
    with st.form("url_verification_form"):
        url = st.text_input("Public article URL", placeholder="https://example.com/news/article")
        verify_url = st.form_submit_button("🔗 Extract and verify URL", use_container_width=True)
    if verify_url:
        if not url.strip():
            st.warning("Please paste a public article URL.")
        else:
            with st.spinner("Extracting article, classifying it, and retrieving evidence..."):
                result = request_verification("/verify/url", {"url": url.strip()})
            if result:
                render_result(result)


st.divider()
st.caption("TF-IDF + Linear SVM · optional Google Fact Check Tools + GNews evidence · for educational use")
