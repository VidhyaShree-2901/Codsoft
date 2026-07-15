import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

TRAIN_FILE = "train_data.txt"
TEST_FILE = "test_data.txt"
TEST_SOLUTION_FILE = "test_data_solution.txt"


# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
def load_train_data(path):
    """Reads ID ::: TITLE ::: GENRE ::: DESCRIPTION format."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                _id, title, genre, desc = parts
                rows.append((title, genre.strip().lower(), desc))
    return pd.DataFrame(rows, columns=["title", "genre", "description"])


def load_test_data(path):
    """Reads ID ::: TITLE ::: DESCRIPTION format (no genre, for prediction)."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 3:
                _id, title, desc = parts
                rows.append((_id, title, desc))
    return pd.DataFrame(rows, columns=["id", "title", "description"])


# ---------------------------------------------------------------------
# 2. Clean text
# ---------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------
# 3. Main pipeline
# ---------------------------------------------------------------------
def main():
    print("Loading training data...")
    df = load_train_data(TRAIN_FILE)
    print(f"  -> {len(df)} rows, {df['genre'].nunique()} genres")
    df["clean_desc"] = df["description"].apply(clean_text)

    # Split into train/validation so we can measure performance
    X_train, X_val, y_train, y_val = train_test_split(
        df["clean_desc"], df["genre"], test_size=0.2, random_state=42, stratify=df["genre"]
    )

    # TF-IDF vectorization
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # Try a few models and compare
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "Linear SVM": LinearSVC(),
    }

    best_model_name, best_model, best_acc = None, None, 0
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_val_tfidf)
        acc = accuracy_score(y_val, preds)
        print(f"  Validation accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc, best_model_name, best_model = acc, name, model

    print(f"\nBest model: {best_model_name} (accuracy = {best_acc:.4f})")
    preds = best_model.predict(X_val_tfidf)
    print("\nClassification report on validation set:")
    print(classification_report(y_val, preds, zero_division=0))

    # Save model + vectorizer for reuse
    joblib.dump(best_model, "genre_classifier_model.joblib")
    joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
    print("\nSaved model -> genre_classifier_model.joblib")
    print("Saved vectorizer -> tfidf_vectorizer.joblib")

    # -------------------------------------------------------------
    # 4. Predict on the official test set, if present
    # -------------------------------------------------------------
    try:
        test_df = load_test_data(TEST_FILE)
        print(f"\nLoaded test set: {len(test_df)} rows. Predicting genres...")
        test_df["clean_desc"] = test_df["description"].apply(clean_text)
        test_tfidf = vectorizer.transform(test_df["clean_desc"])
        test_df["predicted_genre"] = best_model.predict(test_tfidf)

        out_path = "test_predictions.csv"
        test_df[["id", "title", "predicted_genre"]].to_csv(out_path, index=False)
        print(f"Saved predictions -> {out_path}")

        # If the true labels are available, report test accuracy too
        try:
            sol_df = load_train_data(TEST_SOLUTION_FILE)  # same 4-column format
            merged_acc = accuracy_score(sol_df["genre"], test_df["predicted_genre"])
            print(f"Test set accuracy (vs {TEST_SOLUTION_FILE}): {merged_acc:.4f}")
        except FileNotFoundError:
            pass

    except FileNotFoundError:
        print(f"\nNo '{TEST_FILE}' found — skipping test-set prediction.")
        print("(That's fine, the model is already trained and saved.)")

    # -------------------------------------------------------------
    # 5. Try it on a custom plot summary
    # -------------------------------------------------------------
    sample_plot = (
        "A group of astronauts travel through a wormhole in search of a "
        "new home for humanity as Earth becomes uninhabitable."
    )
    sample_clean = clean_text(sample_plot)
    sample_vec = vectorizer.transform([sample_clean])
    predicted_genre = best_model.predict(sample_vec)[0]
    print(f"\nSample prediction:\n  Plot: {sample_plot}\n  Predicted genre: {predicted_genre}")


if __name__ == "__main__":
    main()