import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

TRAIN_FILE = "fraudTrain.csv"
TEST_FILE = "fraudTest.csv"


def haversine_distance(lat1, lon1, lat2, lon2):
    """Approx distance in km between customer and merchant location."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def engineer_features(df):
    df = df.copy()

    # --- Time-based features ---
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

    # --- Age of customer at time of transaction ---
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365

    # --- Distance between customer and merchant ---
    df["distance_km"] = haversine_distance(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

    # --- Keep only useful columns ---
    keep_cols = [
        "amt", "category", "gender", "city_pop",
        "trans_hour", "trans_day_of_week", "age", "distance_km",
    ]
    target_col = "is_fraud"
    df = df[keep_cols + [target_col]]
    return df


def main():
    # -------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------
    print("Loading training data...")
    train_df = pd.read_csv("fraudTrain.csv")
    print(f"  -> {len(train_df)} transactions")
    print("Loading test data...")
    test_df = pd.read_csv("fraudTest.csv")
    print(f"  -> {len(test_df)} transactions")

    fraud_pct = train_df["is_fraud"].mean() * 100
    print(f"  -> Fraud rate in training data: {fraud_pct:.3f}%")

    # -------------------------------------------------------------
    # 2. Feature engineering
    # -------------------------------------------------------------
    print("\nEngineering features (age, transaction hour, distance, etc.)...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # Encode categorical columns the same way on both train and test
    for col in ["category", "gender"]:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]], axis=0))
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    X_train = train_df.drop("is_fraud", axis=1)
    y_train = train_df["is_fraud"]
    X_test = test_df.drop("is_fraud", axis=1)
    y_test = test_df["is_fraud"]

    # Scale numeric columns
    scaler = StandardScaler()
    num_cols = ["amt", "city_pop", "age", "distance_km"]
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # -------------------------------------------------------------
    # 3. Train and compare models
    # -------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    best_model_name, best_model, best_f1 = None, None, 0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        print("Confusion matrix ([[TN, FP], [FN, TP]]):")
        print(confusion_matrix(y_test, preds))

        report = classification_report(y_test, preds, target_names=["Legit", "Fraud"], output_dict=True)
        fraud_f1 = report["Fraud"]["f1-score"]
        print(classification_report(y_test, preds, target_names=["Legit", "Fraud"]))

        if hasattr(model, "predict_proba"):
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            print(f"ROC-AUC score: {auc:.4f}")

        if fraud_f1 > best_f1:
            best_f1, best_model_name, best_model = fraud_f1, name, model

    print(f"\n{'='*50}")
    print(f"Best model (by Fraud F1-score): {best_model_name} (F1 = {best_f1:.4f})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()