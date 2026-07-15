import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_FILE = "Churn_Modelling.csv"


def main():
    # -------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"  -> {len(df)} customers loaded")

    churn_rate = df["Exited"].mean() * 100
    print(f"  -> Churn rate: {churn_rate:.2f}% of customers left")

    # -------------------------------------------------------------
    # 2. Drop columns that don't help prediction
    # -------------------------------------------------------------
    # RowNumber, CustomerId, Surname are just identifiers -- a customer's
    # name or ID number has no real relationship with whether they churn.
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    # -------------------------------------------------------------
    # 3. Encode categorical columns
    # -------------------------------------------------------------
    # Geography (France/Germany/Spain) and Gender are text -- convert to numbers
    le_geo = LabelEncoder()
    df["Geography"] = le_geo.fit_transform(df["Geography"])

    le_gender = LabelEncoder()
    df["Gender"] = le_gender.fit_transform(df["Gender"])

    # -------------------------------------------------------------
    # 4. Split features / target
    # -------------------------------------------------------------
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric columns (helps Logistic Regression converge properly)
    num_cols = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # -------------------------------------------------------------
    # 5. Train and compare models
    # -------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    best_model_name, best_model, best_f1 = None, None, 0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        print("Confusion matrix ([[TN, FP], [FN, TP]]):")
        print(confusion_matrix(y_test, preds))

        report = classification_report(y_test, preds, target_names=["Stayed", "Churned"], output_dict=True)
        churn_f1 = report["Churned"]["f1-score"]
        print(classification_report(y_test, preds, target_names=["Stayed", "Churned"]))

        if hasattr(model, "predict_proba"):
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            print(f"ROC-AUC score: {auc:.4f}")

        if churn_f1 > best_f1:
            best_f1, best_model_name, best_model = churn_f1, name, model

    print(f"\n{'='*50}")
    print(f"Best model (by Churn F1-score): {best_model_name} (F1 = {best_f1:.4f})")
    print(f"{'='*50}")

    # -------------------------------------------------------------
    # 6. Feature importance (only for tree-based models)
    # -------------------------------------------------------------
    if hasattr(best_model, "feature_importances_"):
        print("\nWhich factors matter most for predicting churn:")
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        print(importances.to_string())


if __name__ == "__main__":
    main()