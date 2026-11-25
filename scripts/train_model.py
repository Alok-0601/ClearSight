"""Train the deployed TF-IDF + Linear SVM model from the WELFake CSV.

Same data cleaning, split, features and hyperparameters as
notebooks/Save_Model.ipynb, but every article goes through preprocess(), which
removes source markers (agency datelines, photo credits, outlet names, dates)
before the usual cleaning. Writes Models/tfidf_vectorizer.pkl and
Models/linear_svm.pkl.

Usage: python scripts/train_model.py path/to/WELFake_Dataset.csv
"""

import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess import preprocess  # noqa: E402


def _work(item: tuple[str, str]) -> str:
    return preprocess(*item)


def main() -> None:
    df = pd.read_csv(sys.argv[1])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")
    df = df.drop_duplicates().dropna(subset=["text"])
    df["title"] = df["title"].fillna("")
    df = df[df["text"].str.split().str.len() > 0].reset_index(drop=True)

    with Pool() as pool:
        docs = pool.map(_work, zip(df.title, df.text), chunksize=200)
    X_train, X_test, y_train, y_test = train_test_split(
        docs, df.label.values, test_size=0.2, random_state=42, stratify=df.label.values
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.8)
    model = LinearSVC(C=1, loss="squared_hinge", max_iter=5000, random_state=42)
    model.fit(vectorizer.fit_transform(X_train), y_train)

    X_test_tfidf = vectorizer.transform(X_test)
    pred = model.predict(X_test_tfidf)
    print(f"Test Accuracy  : {accuracy_score(y_test, pred):.4f}")
    print(f"Test Precision : {precision_score(y_test, pred):.4f}")
    print(f"Test Recall    : {recall_score(y_test, pred):.4f}")
    print(f"Test F1 Score  : {f1_score(y_test, pred):.4f}")
    print(f"Test ROC AUC   : {roc_auc_score(y_test, model.decision_function(X_test_tfidf)):.4f}")

    with open(ROOT / "Models" / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(ROOT / "Models" / "linear_svm.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
