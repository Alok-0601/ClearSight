# ClearSight: A Machine Learning Based Fake News Detection System

ClearSight classifies news articles as **Real** or **Fake** and retrieves relevant evidence from optional external fact-check and news sources. The Streamlit interface is a client of the FastAPI verification service, so the same backend can serve the website and a future dedicated frontend.

## Live Demo 
 
https://newsverifierr.streamlit.app/

## Dataset

The model is trained on the [WELFake dataset](https://www.kaggle.com/datasets/vcclab/welfake-dataset), a collection of roughly 72,000 labeled news articles combining data from four different news datasets to reduce overfitting to any single source's writing style. Each row has a title, the article text, and a label marking it as real or fake. The dataset itself isn't included in this repo since it's a fairly large CSV — you can download it from the Kaggle link above if you want to retrain the model yourself.

## Approach

The raw text goes through a cleanup pipeline before it ever reaches the model: the title and article body are combined, URLs/emails/HTML tags and extra whitespace are stripped out, everything is lowercased and stripped of punctuation, then tokenized, run through stopword removal, and lemmatized. What's left is a clean bag of words per article.

For turning that text into numbers, I compared TF-IDF against Word2Vec embeddings, and TF-IDF came out ahead. On the modeling side, I ran hyperparameter tuning and ended up with a linear SVM on top of TF-IDF (unigrams + bigrams, capped at 5000 features) as the best-performing setup — around 95% accuracy and a 94.7% F1 score on the held-out test set, with an ROC-AUC close to 99%.

Everything from raw CSV to trained model lives in `notebooks/`. The app itself doesn't retrain anything — it just loads the saved vectorizer and model and reuses the exact same preprocessing steps at prediction time.

## Tech Stack

Python, pandas, scikit-learn, NLTK for text preprocessing, FastAPI for the verification API, and Streamlit for the interface.


## Running It Locally

Clone the repo, install the dependencies, and make sure `tfidf_vectorizer.pkl` and `linear_svm.pkl` are sitting inside the `Models/` folder (they're already included in this repo, so you shouldn't need to retrain anything unless you want to).


The first run will download a few small NLTK resources (stopwords, tokenizer, lemmatizer data) automatically — that only happens once.

Start the API first, then run the Streamlit client in a second terminal:

```bash
copy .env.example .env
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

```bash
streamlit run app.py
```

The default local API URL is `http://127.0.0.1:8000`. Set `BACKEND_API_URL` if the API runs elsewhere.

## A Few Honest Caveats

The confidence score shown in the app isn't a calibrated probability — linear SVMs don't naturally produce one, so it's derived from how far a prediction sits from the decision boundary. Treat it as a relative signal, not a precise percentage. The model is also only as good as the dataset it learned from, so it reflects the writing patterns and topics present in WELFake at the time it was collected. It's a useful second opinion, not a substitute for actually checking a claim against a reliable source.

## Backend API and evidence pipeline

The `backend/` package is the API/automation contribution. It keeps the existing
TF-IDF + Linear SVM model unchanged and adds a clean FastAPI boundary around it.
It accepts text or an article URL, extracts article metadata for URL submissions,
returns the model result, and optionally retrieves normalized supporting context
from Google Fact Check Tools and GNews.

Evidence is deliberately conservative: related articles remain `related` or
`not_determined`. A result is labelled `supports_claim` or `contradicts_claim`
only when a fact-check rating has high lexical overlap with the submitted article.
This is evidence retrieval, not autonomous fact-checking.

### Run the API

```bash
pip install -r requirements.txt
copy .env.example .env
uvicorn backend.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.
Without API keys, prediction endpoints continue to work and report those sources
as `not_configured`. Add `GOOGLE_FACT_CHECK_API_KEY` and/or `GNEWS_API_KEY` to
`.env` to enable live retrieval.

### Deploy the Streamlit interface with the API

Streamlit Community Cloud runs the Streamlit entry point, while FastAPI must run
as a separate web service. Deploy this repository's API using a Python/ASGI host
such as Render, Railway, or Fly.io with this start command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Then, in Streamlit Community Cloud, open **App settings → Secrets** and add:

```toml
BACKEND_API_URL = "https://your-api-service.example.com"
```

Add `GOOGLE_FACT_CHECK_API_KEY` and `GNEWS_API_KEY` as environment variables on
the API host (not in Streamlit secrets). After the Streamlit app restarts, its
two tabs call `POST /verify` and `POST /verify/url` and display the returned
prediction, extracted metadata, evidence, and source status.

### Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | API availability check |
| `POST` | `/verify` | Verify raw text (`title` optional, `text` required) |
| `POST` | `/verify/url` | Fetch, extract, and verify a public article URL |
| `GET` | `/verification/{id}` | Fetch one in-memory verification result |
| `GET` | `/history` | Fetch recent in-memory verification summaries |

Example request:

```json
POST /verify
{
  "title": "Example headline",
  "text": "At least forty characters of article content are required here."
}
```
