import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Decision Tree Regressor", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "decision_tree_dataset.csv")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def build_model(numeric_features, categorical_features, params: dict):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = DecisionTreeRegressor(
        random_state=42,
        **params,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipe

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

# ---------------- UI ----------------
st.title("Decision Tree Regressor (Streamlit)")

with st.sidebar:
    st.header("Settings")
    target_col = "approved"  # regression target

    uploaded = st.file_uploader("Upload dataset (optional)", type=["csv"], accept_multiple_files=False)
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success("Dataset loaded from upload")
    else:
        df = load_data(DATA_PATH)
        st.info(f"Using dataset: {os.path.basename(DATA_PATH)}")

    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")
        st.stop()

    test_size = st.slider("Test size", 0.10, 0.50, 0.20, 0.05)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    st.subheader("Hyperparameters")
    criterion = st.selectbox("criterion", ["squared_error", "absolute_error", "friedman_mse", "poisson"], index=0)

    max_depth = st.selectbox("max_depth", options=["None", 2, 3, 4, 5, 6, 8, 10, 15, 20], index=0)
    max_depth_val = None if max_depth == "None" else int(max_depth)

    min_samples_split = st.selectbox("min_samples_split", [2, 3, 4, 5, 10, 20], index=0)
    min_samples_leaf = st.selectbox("min_samples_leaf", [1, 2, 3, 5, 10], index=0)

    params = {
        "criterion": criterion,
        "max_depth": max_depth_val,
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf": int(min_samples_leaf),
    }

# ---------------- Data split ----------------
# Identify feature types
X = df.drop(columns=[target_col])
y = pd.to_numeric(df[target_col], errors="coerce")

# If any NaNs were introduced in y, drop those rows
mask = ~y.isna()
X = X.loc[mask].copy()
y = y.loc[mask].copy()

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=float(test_size), random_state=int(random_state)
)

pipe = build_model(numeric_features, categorical_features, params)
pipe.fit(X_train, y_train)

# Predictions
y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)

train_mae, train_mse, train_rmse, train_r2 = compute_metrics(y_train, y_pred_train)
test_mae, test_mse, test_rmse, test_r2 = compute_metrics(y_test, y_pred_test)

# ---------------- Metrics ----------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Train Metrics")
    st.metric("MAE", f"{train_mae:.4f}")
    st.metric("MSE", f"{train_mse:.4f}")
    st.metric("RMSE", f"{train_rmse:.4f}")
    st.metric("R²", f"{train_r2:.4f}")

with col2:
    st.subheader("Test Metrics")
    st.metric("MAE", f"{test_mae:.4f}")
    st.metric("MSE", f"{test_mse:.4f}")
    st.metric("RMSE", f"{test_rmse:.4f}")
    st.metric("R²", f"{test_r2:.4f}")

# ---------------- Plots ----------------
st.subheader("Predicted vs Actual (Test)")
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, y_pred_test, alpha=0.6)
min_v = float(np.nanmin([y_test.min(), y_pred_test.min()]))
max_v = float(np.nanmax([y_test.max(), y_pred_test.max()]))
ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Decision Tree Regressor")
st.pyplot(fig, clear_figure=True)

# Feature importance (only if supported)
# DecisionTreeRegressor exposes feature_importances_ after preprocessing.
try:
    model = pipe.named_steps["model"]
    importances = model.feature_importances_
    # Get output feature names from OneHotEncoder if possible
    preprocess = pipe.named_steps["preprocess"]
    feature_names = []
    for name, transformer, cols in preprocess.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]
            cat_feature_names = ohe.get_feature_names_out(cols)
            feature_names.extend(list(cat_feature_names))

    if len(feature_names) == len(importances):
        st.subheader("Feature Importance")
        fi = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(15)
        )
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.barh(fi["feature"].iloc[::-1], fi["importance"].iloc[::-1])
        ax2.set_xlabel("Importance")
        ax2.set_title("Top 15 Feature Importances")
        st.pyplot(fig2, clear_figure=True)
except Exception:
    pass

# ---------------- Single prediction form ----------------
st.subheader("Make a Prediction")

with st.form("prediction_form"):
    input_row = {}

    for col in numeric_features:
        # Use median as default
        default_val = float(pd.to_numeric(df[col], errors="coerce").median())
        val = st.number_input(f"{col}", value=default_val)
        input_row[col] = val

    for col in categorical_features:
        default_val = df[col].mode(dropna=True)
        default_val = default_val.iloc[0] if len(default_val) else ""
        categories = sorted([str(v) for v in df[col].dropna().unique().tolist()])
        val = st.selectbox(f"{col}", options=categories, index=categories.index(str(default_val)) if str(default_val) in categories else 0)
        input_row[col] = val

    submitted = st.form_submit_button("Predict")

if submitted:
    X_new = pd.DataFrame([input_row])
    pred = float(pipe.predict(X_new)[0])
    st.success(f"Predicted '{target_col}' = {pred:.4f}")

