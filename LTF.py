# LTFU Analyzer — One‑Page Web App (Streamlit)
# -------------------------------------------------------------
# What this does
# - Upload your Excel/CSV
# - Select/confirm the columns for target + features (from your methodology)
# - Trains Logistic Regression + XGBoost with proper preprocessing
# - Shows metrics (Accuracy, Precision, Recall, F1, ROC‑AUC), curves, confusion matrix
# - Interprets models (Permutation Importance + SHAP for XGBoost)
# - Lets you tune the classification threshold and download predictions
#
# How to run (in a terminal):
#   pip install streamlit pandas numpy scikit-learn xgboost shap matplotlib openpyxl
#   streamlit run app.py
# -------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.inspection import permutation_importance

import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="LTFU Analyzer", layout="wide")
st.title("📊 LTFU Analyzer — HIV Care ")
st.caption("Upload your dataset, pick columns, train models, and explore insights.")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===== Sidebar: Upload & Settings =====
st.sidebar.header("1) Upload Data")
upload = st.sidebar.file_uploader("Excel (.xlsx) or CSV", type=["xlsx", "csv"]) 

st.sidebar.header("2) Train / Test Split")
test_size = st.sidebar.slider("Test size", 0.1, 0.3, 0.15, 0.01)
val_size_overall = st.sidebar.slider("Validation size (overall)", 0.1, 0.3, 0.15, 0.01)

st.sidebar.header("3) Threshold")
default_threshold = st.sidebar.slider("Decision threshold (for classification)", 0.1, 0.9, 0.5, 0.01)

st.sidebar.header("4) Advanced")
use_class_weight = st.sidebar.checkbox("Logistic: balanced class_weight", value=True)
use_early_stopping = st.sidebar.checkbox("XGBoost: early stopping", value=True)

st.sidebar.markdown("---")
st.sidebar.info("This app follows my study's methodology (variables, splits, metrics, and interpretability).")

# ===== Helpers =====
def read_df(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

DEFAULT_EXPECTED = {
    "ltfu": "ltfu",
    "durationindays": "durationindays",
    "weight": "weight",
    "cd4": "cd4",
    "counseling": "counseling",
    "disclosure": "disclosure",
    "funds": "funds",
    "mstatus": "mstatus",
    "employmenstat": "employmenstat",
    "education": "education",
    "religion": "religion",
    "age": "age",
}

NUM_DEFAULT = ["durationindays", "weight", "cd4", "age"]
BIN_DEFAULT = ["counseling", "disclosure"]
CAT_DEFAULT = ["funds", "mstatus", "employmenstat", "education", "religion"]

@st.cache_data(show_spinner=False)
def fit_onehot_categories(df, cat_cols):
    vals = {}
    for c in cat_cols:
        if c in df.columns:
            vals[c] = sorted(df[c].dropna().astype(str).unique().tolist())
    return vals

# ===== Main workflow =====
if upload is None:
    st.info("👆 Upload an Excel/CSV file to begin. Expected columns can be mapped after upload.")
    st.stop()

# Load
df = read_df(upload)
df.columns = [c.strip() for c in df.columns]

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

# Column mapping UI
st.subheader("Column Mapping")
col_map = {}
cols = ["(none)"] + list(df.columns)

cm1, cm2, cm3 = st.columns(3)
with cm1:
    for k in ["ltfu", "durationindays", "weight", "cd4"]:
        col_map[k] = st.selectbox(f"Map '{k}'", cols, index=(cols.index(DEFAULT_EXPECTED[k]) if DEFAULT_EXPECTED[k] in df.columns else 0))
with cm2:
    for k in ["age", "counseling", "disclosure"]:
        col_map[k] = st.selectbox(f"Map '{k}'", cols, index=(cols.index(DEFAULT_EXPECTED[k]) if DEFAULT_EXPECTED[k] in df.columns else 0))
with cm3:
    for k in ["funds", "mstatus", "employmenstat", "education", "religion"]:
        col_map[k] = st.selectbox(f"Map '{k}'", cols, index=(cols.index(DEFAULT_EXPECTED[k]) if DEFAULT_EXPECTED[k] in df.columns else 0))

# Build working dataframe
# Invert mapping: incoming -> canonical names
inv = {}
for canonical, incoming in col_map.items():
    if incoming != "(none)":
        inv[incoming] = canonical

work = df.rename(columns=inv).copy()

if "ltfu" not in work.columns:
    st.error("You must map the target column 'ltfu'.")
    st.stop()

# Select features present
num_cols = [c for c in NUM_DEFAULT if c in work.columns]
bin_cols = [c for c in BIN_DEFAULT if c in work.columns]
cat_cols = [c for c in CAT_DEFAULT if c in work.columns]

st.markdown(
    f"**Using features** → Numeric: `{num_cols}` · Binary: `{bin_cols}` · Categorical: `{cat_cols}`"
)

# Clean target - first inspect the data
st.subheader("Target Column Analysis")
target_col = work["ltfu"]
st.write(f"**Target column '{target_col.name}' contains:**")
st.write(f"- Total rows: {len(target_col)}")
st.write(f"- Unique values: {target_col.unique()[:10]}")  # Show first 10 unique values
st.write(f"- Data types: {target_col.dtype}")

# Try to convert to numeric, but handle common cases
y_original = target_col.copy()

# Handle common non-numeric cases
if target_col.dtype == 'object':
    # Try common mappings first
    y_mapped = target_col.map({
        'Yes': 1, 'No': 0, 'yes': 1, 'no': 0,
        '1': 1, '0': 0, 'True': 1, 'False': 0,
        'true': 1, 'false': 0, 'Y': 1, 'N': 0,
        'y': 1, 'n': 0, 'LTFU': 1, 'Not LTFU': 0,
        'ltfu': 1, 'not ltfu': 0
    })
    # Then try numeric conversion
    y = pd.to_numeric(y_mapped, errors="coerce")
else:
    y = pd.to_numeric(target_col, errors="coerce")

X = work.drop(columns=["ltfu"])  # raw; preprocessing in pipelines

# Remove rows with NaN target values
valid_mask = ~y.isna()
invalid_count = (~valid_mask).sum()
if invalid_count > 0:
    st.warning(f"Removing {invalid_count} rows with missing target values out of {len(y)} total rows")
    st.write(f"**Sample of problematic values:** {y_original[~valid_mask].head(10).tolist()}")
    X = X[valid_mask]
    y = y[valid_mask]

# Check if we have enough data
if len(y) < 10:
    st.error(f"Not enough valid data after cleaning. Only {len(y)} rows remain. Please check your dataset.")
    st.stop()

# Check if target has both classes
unique_classes = y.unique()
if len(unique_classes) < 2:
    st.error(f"Target variable must have at least 2 classes. Found only: {unique_classes}. Please check your data.")
    st.stop()

# Display data summary
st.info(f"✅ Data ready: {len(y)} rows, {len(unique_classes)} classes: {unique_classes}")
# Coerce binary Yes/No -> 0/1 if applicable
for b in bin_cols:
    if b in X.columns:
        X[b] = X[b].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).astype("float64")

# Split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
)
# compute val fraction relative to (train+val)
if test_size >= 1.0:
    st.error("Test size must be less than 1.0")
    st.stop()
val_frac = val_size_overall / (1 - test_size)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_frac, stratify=y_trainval, random_state=RANDOM_STATE
)

# Preprocess — Logistic
numeric_lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

binary_lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

categorical_lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

prep_lr = ColumnTransformer(
    transformers=[
        ("num", numeric_lr, [c for c in num_cols if c in X.columns]),
        ("bin", binary_lr, [c for c in bin_cols if c in X.columns]),
        ("cat", categorical_lr, [c for c in cat_cols if c in X.columns]),
    ],
    remainder="drop",
)

logit = Pipeline(steps=[
    ("prep", prep_lr),
    ("clf", LogisticRegression(
        solver="liblinear", penalty="l2", max_iter=2000,
        class_weight=("balanced" if use_class_weight else None),
        random_state=RANDOM_STATE,
    )),
])

# Preprocess — XGB
numeric_xgb = Pipeline(steps=[("pass", "passthrough")])
binary_xgb = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
categorical_xgb = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

prep_xgb = ColumnTransformer(
    transformers=[
        ("num", numeric_xgb, [c for c in num_cols if c in X.columns]),
        ("bin", binary_xgb, [c for c in bin_cols if c in X.columns]),
        ("cat", categorical_xgb, [c for c in cat_cols if c in X.columns]),
    ],
    remainder="drop",
)

# Fit models
with st.spinner("Training models…"):
    # Logistic
    logit.fit(X_train, y_train)

    # XGB with optional early stopping
    Xtr_xgb = prep_xgb.fit_transform(X_train, y_train)
    Xva_xgb = prep_xgb.transform(X_val)
    ytr, yva = y_train.values, y_val.values

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        eval_metric="auc",
    )
    if use_early_stopping:
        xgb_model.fit(Xtr_xgb, ytr, eval_set=[(Xva_xgb, yva)], verbose=False, early_stopping_rounds=50)
    else:
        xgb_model.fit(Xtr_xgb, ytr)

    xgb_clf = Pipeline(steps=[("prep", prep_xgb), ("clf", xgb_model)])

st.success("Models trained.")

# Evaluation helper
def evaluate(model, X_te, y_te, name, threshold):
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall": recall_score(y_te, y_pred, zero_division=0),
        "F1": f1_score(y_te, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_te, y_prob),
    }
    return metrics, y_prob, y_pred

# Compute evals
metrics_lr, prob_lr, pred_lr = evaluate(logit, X_test, y_test, "Logistic", default_threshold)
metrics_xgb, prob_xgb, pred_xgb = evaluate(xgb_clf, X_test, y_test, "XGB", default_threshold)

# Layout
m1, m2 = st.columns(2)
with m1:
    st.subheader("Logistic Regression — Metrics")
    st.table(pd.Series(metrics_lr).round(4))
with m2:
    st.subheader("XGBoost — Metrics")
    st.table(pd.Series(metrics_xgb).round(4))

# Curves
c1, c2 = st.columns(2)
with c1:
    st.markdown("**ROC Curves**")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, prob_lr, name="Logistic", ax=ax)
    RocCurveDisplay.from_predictions(y_test, prob_xgb, name="XGB", ax=ax)
    st.pyplot(fig)
with c2:
    st.markdown("**Precision–Recall Curves**")
    fig2, ax2 = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_test, prob_lr, name="Logistic", ax=ax2)
    PrecisionRecallDisplay.from_predictions(y_test, prob_xgb, name="XGB", ax=ax2)
    st.pyplot(fig2)

# Confusion matrices at selected threshold
cm1, cm2 = st.columns(2)
with cm1:
    st.markdown("**Confusion Matrix — Logistic**")
    cm = confusion_matrix(y_test, pred_lr)
    st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))
with cm2:
    st.markdown("**Confusion Matrix — XGB**")
    cm = confusion_matrix(y_test, pred_xgb)
    st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

# Cross‑validation (optional quick AUC)
with st.expander("Cross‑validation (5‑fold ROC‑AUC)"):
    def cv_auc(pipe, X_df, y_ser, folds=5):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        aucs = []
        for tr, te in skf.split(X_df, y_ser):
            X_tr, X_te = X_df.iloc[tr], X_df.iloc[te]
            y_tr, y_te = y_ser.iloc[tr], y_ser.iloc[te]
            pipe.fit(X_tr, y_tr)
            y_prob = pipe.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, y_prob))
        return float(np.mean(aucs)), float(np.std(aucs))
    colA, colB = st.columns(2)
    with colA:
        mean_lr, std_lr = cv_auc(logit, X_trainval, y_trainval)
        st.metric("LR 5‑fold AUC", f"{mean_lr:.3f}", f"±{std_lr:.3f}")
    with colB:
        mean_xgb, std_xgb = cv_auc(xgb_clf, X_trainval, y_trainval)
        st.metric("XGB 5‑fold AUC", f"{mean_xgb:.3f}", f"±{std_xgb:.3f}")

# Interpretability — Permutation importance
st.subheader("Feature Importance — Permutation (Model‑Agnostic)")
imp_tabs = st.tabs(["Logistic", "XGBoost"]) 

with imp_tabs[0]:
    r = permutation_importance(logit, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc")
    # Try to recover feature names from preprocessor
    try:
        pre = logit.named_steps["prep"]
        fn = None
        if hasattr(pre, "get_feature_names_out"):
            fn = pre.get_feature_names_out()
        else:
            oh = pre.named_transformers_["cat"].named_steps["onehot"]
            cat_feat_names = oh.get_feature_names_out([c for c in cat_cols if c in X.columns])
            num_feat_names = [c for c in num_cols if c in X.columns]
            bin_feat_names = [c for c in bin_cols if c in X.columns]
            fn = np.array(list(num_feat_names) + list(bin_feat_names) + list(cat_feat_names))
        imp_lr = pd.DataFrame({"feature": fn, "importance": r.importances_mean}) \
                    .sort_values("importance", ascending=False).reset_index(drop=True)
        st.dataframe(imp_lr)
    except Exception as e:
        st.warning(f"Could not compute names: {e}")

with imp_tabs[1]:
    r = permutation_importance(xgb_clf, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc")
    try:
        pre = xgb_clf.named_steps["prep"]
        oh = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_feat_names = oh.get_feature_names_out([c for c in cat_cols if c in X.columns])
        num_feat_names = [c for c in num_cols if c in X.columns]
        bin_feat_names = [c for c in bin_cols if c in X.columns]
        fn = np.array(list(num_feat_names) + list(bin_feat_names) + list(cat_feat_names))
        imp_x = pd.DataFrame({"feature": fn, "importance": r.importances_mean}) \
                    .sort_values("importance", ascending=False).reset_index(drop=True)
        st.dataframe(imp_x)
    except Exception as e:
        st.warning(f"Could not compute names: {e}")

# Interpretability — SHAP for XGBoost
st.subheader("SHAP Explanations — XGBoost")
try:
    # Use transformed features
    X_test_proc = xgb_clf.named_steps["prep"].transform(X_test)
    explainer = shap.TreeExplainer(xgb_clf.named_steps["clf"]) 
    shap_values = explainer.shap_values(X_test_proc)

    # Feature names
    pre = xgb_clf.named_steps["prep"]
    oh = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_feat_names = oh.get_feature_names_out([c for c in cat_cols if c in X.columns])
    num_feat_names = [c for c in num_cols if c in X.columns]
    bin_feat_names = [c for c in bin_cols if c in X.columns]
    feature_names = list(num_feat_names) + list(bin_feat_names) + list(cat_feat_names)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Global importance (summary plot)**")
        fig = plt.figure()
        shap.summary_plot(shap_values, features=X_test_proc, feature_names=feature_names, show=False)
        st.pyplot(fig, clear_figure=True)
    with col2:
        st.markdown("**Pick a row for a local explanation**")
        row_idx = st.number_input("Row index (0‑based)", min_value=0, max_value=int(max(0, len(y_test)-1)), value=0, step=1)
        fig2 = plt.figure()
        # Use waterfall plot instead of deprecated force_plot
        shap.waterfall_plot(explainer.expected_value, shap_values[row_idx, :], 
                           feature_names=feature_names, show=False)
        st.pyplot(fig2, clear_figure=True)
except Exception as e:
    st.warning(f"SHAP visualization skipped: {e}")

# Predictions & download
st.subheader("Scored Test Set")
scored = X_test.copy()
scored["y_true"] = y_test.values
scored["p_lr"] = prob_lr
scored["p_xgb"] = prob_xgb
scored["pred_lr"] = pred_lr
scored["pred_xgb"] = pred_xgb
st.dataframe(scored.head(50), use_container_width=True)

csv = scored.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions (CSV)", data=csv, file_name="ltfu_scored.csv", mime="text/csv")

st.markdown("---")
st.caption("Methodology implemented: preprocessing (impute/scale/encode), stratified 70/15/15 split, LR + XGB models, metrics/curves, permutation importance, and SHAP.")
