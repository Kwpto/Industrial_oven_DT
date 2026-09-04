import os

import joblib
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor


def main():
    # ------------------------------------------------------------------
    # 1. Load telemetry
    # ------------------------------------------------------------------
    df = pd.read_csv("baking_oven_telemetry.csv")

    if len(df) != 3000:
        print(f"⚠️  Warning: expected 3000 rows, found {len(df)}")

    # ------------------------------------------------------------------
    # 2. Prepare features and target for quality classifier
    # ------------------------------------------------------------------
    feature_cols = [
        "Zone1_Temp_C",
        "Zone2_Temp_C",
        "Zone3_Temp_C",
        "Humidity_Pct",
        "Belt_Speed_m_min",
        "Gas_Flow_m3_h",
        "Dough_Inlet_Moisture_Pct",
        "Product_Load_kg_min",
        "Exhaust_Damper_Pct",
        "Heat_Recovery_Eff_Pct",
    ]
    X = df[feature_cols]
    y = df["Defect_Flag"]

    # ------------------------------------------------------------------
    # 3. Train/test split (stratified 80/20)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # ------------------------------------------------------------------
    # 4. Train XGBoost classifier
    # ------------------------------------------------------------------
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 5. Evaluate classifier
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("=" * 60)
    print("  INDUSTRIAL OVEN DIGITAL TWIN — XGBOOST QUALITY CLASSIFIER")
    print("=" * 60)
    print(f"Test accuracy: {acc:.4f}")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=["OK", "Defect"]))
    print("=" * 60)

    if acc < 0.98:
        print(f"⚠️  Accuracy {acc:.2%} is below the 98% target.")
    else:
        print(f"✅ Accuracy {acc:.2%} meets the 98% target.")

    # ------------------------------------------------------------------
    # 6. Serialize classifier and SHAP explainer
    # ------------------------------------------------------------------
    clf.save_model("oven_xgboost_model.json")
    print("✅ Quality classifier saved to 'oven_xgboost_model.json'")

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    joblib.dump({"explainer": explainer, "feature_names": feature_cols}, "shap_explainer.joblib")
    print("✅ SHAP explainer saved to 'shap_explainer.joblib'")

    # Save a summary plot for offline inspection
    try:
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols, show=False)
        import matplotlib.pyplot as plt
        plt.savefig("shap_summary.png", bbox_inches="tight")
        plt.close()
        print("✅ SHAP summary plot saved to 'shap_summary.png'")
    except Exception as e:
        print(f"⚠️  Could not save SHAP plot: {e}")

    # ------------------------------------------------------------------
    # 7. Train XGBoost regressor for Zone 2 burner RUL
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PREDICTIVE MAINTENANCE — BURNER RUL REGRESSOR")
    print("=" * 60)

    X_rul = df[feature_cols]
    y_rul = df["RUL_Burner_Hours"]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_rul, y_rul, test_size=0.20, random_state=42
    )

    rul_model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    rul_model.fit(Xr_train, yr_train)

    yr_pred = rul_model.predict(Xr_test)
    mae = mean_absolute_error(yr_test, yr_pred)
    r2 = r2_score(yr_test, yr_pred)

    print(f"RUL MAE: {mae:.2f} hours")
    print(f"RUL R²:  {r2:.4f}")
    rul_model.save_model("oven_rul_model.json")
    print("✅ RUL model saved to 'oven_rul_model.json'")


if __name__ == "__main__":
    main()
