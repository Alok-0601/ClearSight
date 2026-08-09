# Fake News Detection

A machine learning project that classifies news articles as **Real** or **Fake** based on their headline and body text. The final model is wrapped in a small Streamlit app so you can paste in an article and get a prediction instantly, instead of just looking at accuracy numbers in a notebook.

## Live Demo

https://newsverifierr.streamlit.app/

## Dataset

The model is trained on the [WELFake dataset](https://www.kaggle.com/datasets/vcclab/welfake-dataset), a collection of roughly 72,000 labeled news articles combining data from four different news datasets to reduce overfitting to any single source's writing style. Each row has a title, the article text, and a label marking it as real or fake. The dataset itself isn't included in this repo since it's a fairly large CSV — you can download it from the Kaggle link above if you want to retrain the model yourself.

## Approach

The raw text goes through a cleanup pipeline before it ever reaches the model: the title and article body are combined, URLs/emails/HTML tags and extra whitespace are stripped out, everything is lowercased and stripped of punctuation, then tokenized, run through stopword removal, and lemmatized. What's left is a clean bag of words per article.

For turning that text into numbers, I compared TF-IDF against Word2Vec embeddings, and TF-IDF came out ahead. On the modeling side, I ran hyperparameter tuning and ended up with a linear SVM on top of TF-IDF (unigrams + bigrams, capped at 5000 features) as the best-performing setup — around 95% accuracy and a 94.7% F1 score on the held-out test set, with an ROC-AUC close to 99%.

Everything from raw CSV to trained model lives in `notebooks/`. The app itself doesn't retrain anything — it just loads the saved vectorizer and model and reuses the exact same preprocessing steps at prediction time.

## Tech Stack

Python, pandas, scikit-learn, NLTK for text preprocessing, and Streamlit for the interface.


## Running It Locally

Clone the repo, install the dependencies, and make sure `tfidf_vectorizer.pkl` and `linear_svm.pkl` are sitting inside the `Models/` folder (they're already included in this repo, so you shouldn't need to retrain anything unless you want to).


The first run will download a few small NLTK resources (stopwords, tokenizer, lemmatizer data) automatically — that only happens once.

## A Few Honest Caveats

The confidence score shown in the app isn't a calibrated probability — linear SVMs don't naturally produce one, so it's derived from how far a prediction sits from the decision boundary. Treat it as a relative signal, not a precise percentage. The model is also only as good as the dataset it learned from, so it reflects the writing patterns and topics present in WELFake at the time it was collected. It's a useful second opinion, not a substitute for actually checking a claim against a reliable source.
