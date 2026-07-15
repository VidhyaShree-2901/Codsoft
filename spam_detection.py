import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

DATA_FILE = "spam.csv"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    # -------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------
    print("Loading dataset...")
    # encoding="latin-1" because this dataset often has non-UTF8 characters
    df = pd.read_csv(DATA_FILE, encoding="latin-1")

    # Keep only the label and message columns, rename for clarity
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    print(f"  -> {len(df)} messages loaded")

    spam_pct = (df["label"] == "spam").mean() * 100
    print(f"  -> Spam: {spam_pct:.1f}% | Legit (ham): {100 - spam_pct:.1f}%")

    # -------------------------------------------------------------
    # 2. Clean text
    # -------------------------------------------------------------
    df["clean_message"] = df["message"].apply(clean_text)

    # -------------------------------------------------------------
    # 3. Split into train/test
    # -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # -------------------------------------------------------------
    # 4. TF-IDF vectorization
    # -------------------------------------------------------------
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # -------------------------------------------------------------
    # 5. Train and compare models
    # -------------------------------------------------------------
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced"),
    }

    best_model_name, best_model, best_f1 = None, None, 0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)

        print("Confusion matrix ([[TN, FP], [FN, TP]] -- positive class = spam):")
        print(confusion_matrix(y_test, preds, labels=["ham", "spam"]))

        report = classification_report(y_test, preds, output_dict=True)
        spam_f1 = report["spam"]["f1-score"]
        print(classification_report(y_test, preds))

        if spam_f1 > best_f1:
            best_f1, best_model_name, best_model = spam_f1, name, model

    print(f"\n{'='*50}")
    print(f"Best model (by Spam F1-score): {best_model_name} (F1 = {best_f1:.4f})")
    print(f"{'='*50}")

    # -------------------------------------------------------------
    # 6. Try it on a few custom messages
    # -------------------------------------------------------------
    sample_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been suspended. Verify your details immediately.",
    ]
    print("\nSample predictions:")
    for msg in sample_messages:
        cleaned = clean_text(msg)
        vec = vectorizer.transform([cleaned])
        pred = best_model.predict(vec)[0]
        print(f"  [{pred.upper()}] {msg}")


if __name__ == "__main__":
    main()