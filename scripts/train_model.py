"""Train the deployed TF-IDF + Linear SVM model.

Data: the WELFake CSV (full articles, mostly US news) and, optionally, the
IFND CSV (Indian news headlines; label TRUE or Fake). Every row goes through
preprocess(), which removes source markers and fact-check verdict words
before the usual cleaning. Same TF-IDF features and LinearSVC
hyperparameters as notebooks/Save_Model.ipynb; an 80/20 split stratified by
dataset and label. Writes Models/tfidf_vectorizer.pkl and Models/linear_svm.pkl.

Usage:
    python scripts/train_model.py WELFake_Dataset.csv [--ifnd IFND.csv]
        [--report report.json] [--cache docs.pkl]
"""

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocess import preprocess  # noqa: E402

DATELINE = "WASHINGTON (Reuters) - "


def _work(item: tuple[str, str]) -> str:
    return preprocess(*item)


def load_welfake(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")
    df = df.drop_duplicates().dropna(subset=["text"])
    df["title"] = df["title"].fillna("")
    df = df[df["text"].str.split().str.len() > 0]
    return pd.DataFrame({"title": df.title, "text": df.text, "label": df.label, "source": "welfake"})


def load_ifnd(path: str) -> pd.DataFrame:
    import ftfy  # repairs the file's mixed encodings, which differ between labels

    df = pd.read_csv(path, encoding="cp1252", encoding_errors="replace")
    df = df.dropna(subset=["Statement"]).drop_duplicates("Statement")
    text = df.Statement.astype(str).map(ftfy.fix_text).str.replace("�", "", regex=False)
    # what ftfy cannot repair is Mac-Roman mojibake ("IndiaÕs"), twice as common
    # in TRUE rows as in fake ones, so it would act as a label marker
    text = text.str.replace("Õ", "'", regex=False).str.replace(r"[^\x00-\x7f]+", " ", regex=True)
    label = (df.Label.str.strip().str.lower() == "fake").astype(int)
    return pd.DataFrame({"title": "", "text": text, "label": label, "source": "ifnd"})


def metrics(y, pred, score) -> dict:
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, score)), 4) if len(set(y)) > 1 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("welfake")
    parser.add_argument("--ifnd")
    parser.add_argument("--report")
    parser.add_argument("--cache")
    args = parser.parse_args()

    frames = [load_welfake(args.welfake)] + ([load_ifnd(args.ifnd)] if args.ifnd else [])
    df = pd.concat(frames, ignore_index=True)
    print(df.groupby(["source", "label"]).size().to_string())

    if args.cache and Path(args.cache).exists():
        docs = pd.read_pickle(args.cache)
    else:
        with Pool(4) as pool:  # each worker loads WordNet; more workers can run out of memory
            docs = pd.Series(pool.map(_work, zip(df.title, df.text), chunksize=200))
        if args.cache:
            docs.to_pickle(args.cache)

    idx = np.arange(len(df))
    strata = df.source + "_" + df.label.astype(str)
    train, test = train_test_split(idx, test_size=0.2, random_state=42, stratify=strata)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.8)
    model = LinearSVC(C=1, loss="squared_hinge", max_iter=5000, random_state=42)
    model.fit(vectorizer.fit_transform(docs.iloc[train]), df.label.values[train])

    X_test = vectorizer.transform(docs.iloc[test])
    pred, score, y = model.predict(X_test), model.decision_function(X_test), df.label.values[test]
    report = {"overall": metrics(y, pred, score)}
    for source in df.source.unique():
        mask = (df.source.values[test] == source)
        report[source] = metrics(y[mask], pred[mask], score[mask])
    for name, value in report.items():
        print(f"{name:8} {value}")

    # A news-agency dateline must not turn a correctly caught fake article into REAL.
    fakes = test[(df.source.values[test] == "welfake") & (df.label.values[test] == 1)]
    caught = fakes[model.predict(vectorizer.transform(docs.iloc[fakes])) == 1]
    attacked = [preprocess(t, DATELINE + x) for t, x in zip(df.title.values[caught], df.text.values[caught])]
    flipped = int((model.predict(vectorizer.transform(attacked)) == 0).sum())
    report["dateline_attack"] = {"caught": int(len(caught)), "flipped_to_real": flipped}
    print("dateline attack:", report["dateline_attack"])

    names = vectorizer.get_feature_names_out()
    order = model.coef_[0].argsort()
    report["features"] = {
        "intercept": round(float(model.intercept_[0]), 3),
        "real": [(names[i], round(float(model.coef_[0][i]), 2)) for i in order[:15]],
        "fake": [(names[i], round(float(model.coef_[0][i]), 2)) for i in order[::-1][:15]],
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    with open(ROOT / "Models" / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(ROOT / "Models" / "linear_svm.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
