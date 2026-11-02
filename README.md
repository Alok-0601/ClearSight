# ClearSight: A Machine Learning Based Fake News Detection System

ClearSight checks whether a news article looks **real** or **fake**. Paste the
article text or a link, and a TF-IDF + Linear SVM model trained on the WELFake
dataset gives a verdict with a confidence score. Alongside it, ClearSight
searches Google Fact Check Tools and GNews for related fact checks and
coverage, so you can see what reliable sources have said.

## Try it

| | |
| --- | --- |
| **Web app** | https://alok-0601.github.io/ClearSight/ |
| **Telegram bot** | [@clearsight_check_bot](https://t.me/clearsight_check_bot): send a link or paste article text |
| **API docs** | https://clearsight-api.vercel.app/docs |

## How it works

```
article text or link
        │
        ▼
 FastAPI backend (Vercel)
   1. fetch and extract the article (for links)
   2. clean the text           ──▶  preprocess.py
   3. classify                 ──▶  TF-IDF + Linear SVM  ──▶  REAL / FAKE + confidence
   4. find related evidence    ──▶  Google Fact Check Tools, GNews
        │
        ▼
 web app · Streamlit app · Telegram bot
```

All three interfaces call the same API. When the model is less than 65%
confident, the verdict is shown as **UNCERTAIN** instead of being forced into
REAL or FAKE.

## Dataset

The model is trained on the [WELFake dataset](https://www.kaggle.com/datasets/vcclab/welfake-dataset), a collection of roughly 72,000 labeled news articles combining data from four different news datasets to reduce overfitting to any single source's writing style. Each row has a title, the article text, and a label marking it as real or fake. The dataset itself isn't included in this repo since it's a fairly large CSV — you can download it from the Kaggle link above if you want to retrain the model yourself.

## Approach

The raw text goes through a cleanup pipeline before it ever reaches the model: the title and article body are combined, URLs/emails/HTML tags and extra whitespace are stripped out, everything is lowercased and stripped of punctuation, then tokenized, run through stopword removal, and lemmatized. What's left is a clean bag of words per article.

For turning that text into numbers, I compared TF-IDF against Word2Vec embeddings, and TF-IDF came out ahead. On the modeling side, I ran hyperparameter tuning and ended up with a linear SVM on top of TF-IDF (unigrams + bigrams, capped at 5000 features) as the best-performing setup — around 95% accuracy and a 94.7% F1 score on the held-out test set, with an ROC-AUC close to 99%.

Everything from raw CSV to trained model lives in `notebooks/`. The app itself doesn't retrain anything — it just loads the saved vectorizer and model and reuses the exact same preprocessing steps at prediction time.

## Evidence retrieval

Evidence is deliberately conservative. Related news coverage is only ever
labelled `related`. A fact check is labelled `contradicts_claim` or
`supports_claim` only when it closely matches the submitted article and its
rating is clearly false or clearly true; mixed ratings such as "Half true" or
"Unverified" stay `related`. This is evidence retrieval, not automatic
fact-checking.

## Tech stack

- **Model:** Python, pandas, scikit-learn, NLTK
- **API:** FastAPI, hosted on Vercel (a `Dockerfile` is included for other hosts)
- **Interfaces:** a static web app on GitHub Pages, a Streamlit app, and a Telegram bot
- **Evidence:** Google Fact Check Tools API, GNews API

## Project structure

| Path | What it is |
| --- | --- |
| `backend/` | FastAPI app: verification routes, Telegram webhook, evidence services |
| `preprocess.py` | Text cleaning, identical to the training pipeline |
| `Models/` | Saved TF-IDF vectorizer and Linear SVM |
| `nltk_data/` | NLTK resources used by `preprocess.py`, bundled for read-only hosts |
| `notebooks/` | Data exploration and model training |
| `docs/` | Static web app served by GitHub Pages (`frontend/` holds a copy) |
| `app.py` | Streamlit app |
| `scripts/` | One-off helpers, such as registering the Telegram webhook |
| `tests/` | Pytest suite |

## Running it locally

Use **Python 3.12 or 3.13**. The saved model needs scikit-learn 1.6.1, which
has no packages for Python 3.14.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt scikit-learn==1.6.1
copy .env.example .env            # macOS/Linux: cp .env.example .env
```

Start the API, then the Streamlit app in a second terminal:

```bash
uvicorn backend.main:app --reload
```

```bash
streamlit run app.py
```

The API docs are at http://127.0.0.1:8000/docs, and the Streamlit app talks
to `http://127.0.0.1:8000` by default (set `BACKEND_API_URL` to change it).
Without API keys, predictions still work and the evidence sources report
`not_configured`; add `GOOGLE_FACT_CHECK_API_KEY` and `GNEWS_API_KEY` to `.env`
to turn them on. Run the tests with `pytest`.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | API availability check |
| `POST` | `/verify` | Verify raw text (`title` optional, `text` required, at least 40 characters) |
| `POST` | `/verify/url` | Fetch, extract, and verify a public article URL |
| `GET` | `/verification/{id}` | Fetch one recent verification result |
| `GET` | `/history` | Fetch recent verification summaries |
| `POST` | `/telegram/webhook` | Telegram bot updates (only when the bot is configured) |

```json
POST /verify
{
  "title": "Example headline",
  "text": "At least forty characters of article content are required here."
}
```

Each result has a `prediction` (the raw model output, `REAL` or `FAKE`) and a
`verdict`, which is the same as `prediction` unless the confidence is below
65%, in which case it is `UNCERTAIN`. The interfaces show the `verdict`.

History is kept in memory. On a serverless host such as Vercel each instance
has its own memory and instances are recycled, so `/history` and
`/verification/{id}` only show recent results from the instance that answers.

## Deployment

**API on Vercel (free Hobby plan).** The repository is already configured:
`pyproject.toml` points Vercel at `backend.main:app` (the root `app.py` is the
Streamlit app) and lists only the API's dependencies, and `.vercelignore` keeps
everything else out of the bundle.

1. On vercel.com, **Add New → Project** and import this repository, or run
   `vercel deploy --prod` from a local clone.
2. Under **Environment Variables**, add `GOOGLE_FACT_CHECK_API_KEY`,
   `GNEWS_API_KEY` and, for the bot, the Telegram variables below.
3. Deploy, then open `https://<project>.vercel.app/health`.

**API on any Docker host:**

```bash
docker build -t clearsight-api .
docker run -p 8000:8000 --env-file .env clearsight-api
```

**API on other Python hosts** (Render, Railway, Fly.io):

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

**Streamlit app.** On Streamlit Community Cloud, deploy `app.py` and add the
API address under **App settings → Secrets**:

```toml
BACKEND_API_URL = "https://clearsight-api.vercel.app"
```

**Web app.** GitHub Pages serves `docs/index.html` from `main`. It uses the
Vercel API by default, and the address can be changed from its Settings panel.

## Telegram bot

Send [@clearsight_check_bot](https://t.me/clearsight_check_bot) an article link
or at least 40 characters of article text, and it replies with the verdict,
confidence and any related fact checks. In groups, reply to a message with
`/check`. Bot checks are not added to `/history`, because they come from
private chats.

The bot runs inside the API through a webhook, so it needs no extra server. To
set up your own:

1. In Telegram, message [@BotFather](https://t.me/BotFather), send `/newbot`
   and copy the token.
2. Choose a webhook secret, for example
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
3. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_WEBHOOK_SECRET` as environment
   variables on the API host and redeploy.
4. With the same two values in your local `.env`, register the webhook:

   ```bash
   python scripts/set_telegram_webhook.py https://<project>.vercel.app
   ```

   Use the production domain, not a per-deployment preview URL: Vercel protects
   preview URLs with a login, which Telegram cannot pass.

The webhook rejects requests without the secret header and limits each chat to
5 checks per minute.

## Limitations

- **The confidence score isn't a calibrated probability.** Linear SVMs don't
  naturally produce one, so it's derived from how far a prediction sits from
  the decision boundary. Treat it as a relative signal, not a precise
  percentage.
- **The model learned from WELFake**, which is mostly US news, so it reflects
  the writing patterns and topics in that data. It can misjudge news from
  other regions, including Indian news, and it reacts to writing style (such
  as a news-agency dateline) as well as to content. A confident verdict can
  still be wrong.
- **Some sites block automated readers.** For those links, paste the article
  text instead.
- **Evidence matching is by shared keywords**, so a fact check can be about a
  similar but different claim. Read the linked source.

ClearSight is a useful second opinion, not a substitute for checking a claim
against a reliable source.
