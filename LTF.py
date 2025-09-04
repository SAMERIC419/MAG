# LTFU Analyzer — Simplified Version for Streamlit Cloud
# This version loads dependencies only when needed to avoid startup issues

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Set page config first
st.set_page_config(page_title="LTFU Analyzer", layout="wide")
st.title("📊 LTFU Analyzer — HIV Care")
st.caption("Upload your dataset, pick columns, train models, and explore insights.")

# Performance notice
st.info("🚀 **Performance Optimized:** Dependencies are loaded only when needed to ensure fast startup.")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===== Helper Functions =====
def check_saved_models():
    """Check if saved models exist and return their info"""
    if os.path.exists("saved_models/metadata.pkl"):
        try:
            with open("saved_models/metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
            return True, metadata
        except:
            return False, None
    return False, None

@st.cache_data(show_spinner=False)
def read_df(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def load_ml_dependencies():
    """Load ML dependencies only when needed"""
    try:
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
        
        return {
            'train_test_split': train_test_split,
            'StratifiedKFold': StratifiedKFold,
            'ColumnTransformer': ColumnTransformer,
            'OneHotEncoder': OneHotEncoder,
            'StandardScaler': StandardScaler,
            'SimpleImputer': SimpleImputer,
            'Pipeline': Pipeline,
            'LogisticRegression': LogisticRegression,
            'accuracy_score': accuracy_score,
            'precision_score': precision_score,
            'recall_score': recall_score,
            'f1_score': f1_score,
            'roc_auc_score': roc_auc_score,
            'confusion_matrix': confusion_matrix,
            'RocCurveDisplay': RocCurveDisplay,
            'PrecisionRecallDisplay': PrecisionRecallDisplay,
            'permutation_importance': permutation_importance,
            'xgb': xgb
        }
    except ImportError as e:
        st.error(f"Error importing ML libraries: {e}")
        st.stop()

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
st.sidebar.header("5) Model Management")

# Check for saved models
models_exist, model_metadata = check_saved_models()

if models_exist:
    st.sidebar.success("✅ **Models Available**")
    st.sidebar.write(f"**Trained:** {model_metadata['timestamp'][:10]}")
    st.sidebar.write(f"**Features:** {len(model_metadata['num_cols']) + len(model_metadata['bin_cols']) + len(model_metadata['cat_cols'])}")
    
    # Option to use saved models or retrain
    use_saved_models = st.sidebar.radio(
        "Choose mode:",
        ["Use Saved Models (Fast)", "Train New Models"],
        index=0
    )
    
    if st.sidebar.button("🗑️ Delete Saved Models"):
        import shutil
        if os.path.exists("saved_models"):
            shutil.rmtree("saved_models")
        st.sidebar.success("Models deleted!")
        st.rerun()
else:
    st.sidebar.warning("⚠️ **No Saved Models**")
    st.sidebar.write("Upload data and train models first")
    use_saved_models = "Train New Models"

st.sidebar.markdown("---")
st.sidebar.info("This app follows my study's methodology (variables, splits, metrics, and interpretability).")

# Default column mappings
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
    "gender": "Gender",
    "whostage": "WHOSTAGE",
    "agecat": "AgeCat",
    "weightcat": "WeighCAt",
}

NUM_DEFAULT = ["durationindays", "weight", "cd4", "age"]
BIN_DEFAULT = ["counseling", "disclosure", "gender"]
CAT_DEFAULT = ["funds", "mstatus", "employmenstat", "education", "religion", "whostage", "agecat", "weightcat"]

# ===== Main workflow =====
if use_saved_models == "Use Saved Models (Fast)":
    st.info("🚀 **Using Saved Models** - Fast prediction mode enabled!")
    
    # Load saved models
    with st.spinner("Loading saved models..."):
        try:
            if not os.path.exists("saved_models"):
                st.error("No saved models found!")
                st.stop()
            
            # Load models
            with open("saved_models/logit_model.pkl", "rb") as f:
                logit = pickle.load(f)
            with open("saved_models/xgb_model.pkl", "rb") as f:
                xgb_clf = pickle.load(f)
            
            # Load metadata
            with open("saved_models/metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
            
            st.success("✅ Models loaded successfully!")
            st.write(f"**Model Info:** Trained on {metadata['timestamp'][:10]} with {len(metadata['num_cols']) + len(metadata['bin_cols']) + len(metadata['cat_cols'])} features")
            
            # Show basic model info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Logistic Regression", "✅ Trained & Ready")
                st.caption(f"Features: {len(metadata['num_cols']) + len(metadata['bin_cols']) + len(metadata['cat_cols'])}")
            with col2:
                st.metric("XGBoost", "✅ Trained & Ready")
                st.caption(f"Trained: {metadata['timestamp'][:10]}")
            
            # Interactive Prediction Form
            st.markdown("---")
            st.subheader("🔮 Interactive LTFU Prediction")
            st.write("Enter patient details below to get real-time LTFU predictions from both models:")
            
            # Create prediction form
            with st.form("prediction_form"):
                st.markdown("### Patient Information")
                
                # Create columns for better layout
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Demographics**")
                    duration_days = st.number_input("Duration in Days", min_value=0, max_value=10000, value=365, step=1)
                    weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=60.0, step=0.1)
                    cd4_count = st.number_input("CD4 Count", min_value=0, max_value=2000, value=500, step=1)
                    age = st.number_input("Age (years)", min_value=0, max_value=120, value=35, step=1)
                    
                    # Binary features
                    counseling = st.selectbox("Counseling", ["Yes", "No"], index=0)
                    disclosure = st.selectbox("Disclosure", ["Yes", "No"], index=0)
                    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
                
                with col2:
                    st.markdown("**Socioeconomic Factors**")
                    funds = st.selectbox("Funding Source", ["Government", "Private", "NGO", "Self", "Other"], index=0)
                    mstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed", "Other"], index=1)
                    employment = st.selectbox("Employment Status", ["Employed", "Unemployed", "Student", "Retired", "Other"], index=0)
                    education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary", "Other"], index=2)
                    religion = st.selectbox("Religion", ["Christian", "Muslim", "Hindu", "Other", "None"], index=0)
                
                with col3:
                    st.markdown("**Clinical Factors**")
                    whostage = st.selectbox("WHO Stage", ["Stage 1", "Stage 2", "Stage 3", "Stage 4"], index=0)
                    weightcat = st.selectbox("Weight Category", ["Underweight", "Normal", "Overweight", "Obese"], index=1)
                
                # Submit button
                submitted = st.form_submit_button("🔮 Predict LTFU Risk", use_container_width=True)
            
            # Process prediction when form is submitted
            if submitted:
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Function to automatically calculate age category from age
                def get_age_category(age_value):
                    if age_value < 25:
                        return "<25"
                    elif age_value < 35:
                        return "25-34"
                    elif age_value < 45:
                        return "35-44"
                    elif age_value < 55:
                        return "45-54"
                    else:
                        return "55+"
                
                # Calculate age category automatically from age input
                agecat = get_age_category(age)
                
                # Show the calculated age category to the user
                st.info(f"📊 **Age Category:** {age} years → {agecat}")
                
                # Create input data
                input_data = {
                    'durationindays': duration_days,
                    'weight': weight,
                    'cd4': cd4_count,
                    'age': age,
                    'counseling': 1 if counseling == "Yes" else 0,
                    'disclosure': 1 if disclosure == "Yes" else 0,
                    'gender': 1 if gender == "Male" else 0,
                    'funds': funds,
                    'mstatus': mstatus,
                    'employmenstat': employment,
                    'education': education,
                    'religion': religion,
                    'whostage': whostage,
                    'agecat': agecat,
                    'weightcat': weightcat
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Ensure all columns are present and in correct order
                for col in metadata['num_cols'] + metadata['bin_cols'] + metadata['cat_cols']:
                    if col not in input_df.columns:
                        if col in metadata['num_cols']:
                            input_df[col] = 0.0
                        elif col in metadata['bin_cols']:
                            input_df[col] = 0
                        elif col in metadata['cat_cols']:
                            input_df[col] = "Unknown"
                
                # Reorder columns to match training data
                input_df = input_df[metadata['num_cols'] + metadata['bin_cols'] + metadata['cat_cols']]
                
                try:
                    # Get predictions from both models
                    with st.spinner("Computing predictions..."):
                        lr_prob = logit.predict_proba(input_df)[0, 1]
                        xgb_prob = xgb_clf.predict_proba(input_df)[0, 1]
                    
                    # Get binary predictions
                    lr_pred = 1 if lr_prob >= default_threshold else 0
                    xgb_pred = 1 if xgb_prob >= default_threshold else 0
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🧠 Logistic Regression")
                        st.metric("LTFU Risk Score", f"{lr_prob:.3f}", help="Probability of being lost to follow-up (0-1)")
                        st.metric("Prediction", "🔴 HIGH RISK" if lr_pred == 1 else "🟢 LOW RISK")
                        
                        # Risk interpretation
                        if lr_prob < 0.3:
                            risk_level = "🟢 Low Risk"
                            recommendation = "Continue current care plan"
                        elif lr_prob < 0.7:
                            risk_level = "🟡 Medium Risk"
                            recommendation = "Consider additional support interventions"
                        else:
                            risk_level = "🔴 High Risk"
                            recommendation = "Implement intensive follow-up protocols"
                        
                        st.info(f"**Risk Level:** {risk_level}")
                        st.info(f"**Recommendation:** {recommendation}")
                    
                    with col2:
                        st.markdown("### 🌳 XGBoost")
                        st.metric("LTFU Risk Score", f"{xgb_prob:.3f}", help="Probability of being lost to follow-up (0-1)")
                        st.metric("Prediction", "🔴 HIGH RISK" if xgb_pred == 1 else "🟢 LOW RISK")
                        
                        # Risk interpretation
                        if xgb_prob < 0.3:
                            risk_level = "🟢 Low Risk"
                            recommendation = "Continue current care plan"
                        elif xgb_prob < 0.7:
                            risk_level = "🟡 Medium Risk"
                            recommendation = "Consider additional support interventions"
                        else:
                            risk_level = "🔴 High Risk"
                            recommendation = "Implement intensive follow-up protocols"
                        
                        st.info(f"**Risk Level:** {risk_level}")
                        st.info(f"**Recommendation:** {recommendation}")
                    
                    # Model agreement
                    st.markdown("### 🤝 Model Agreement")
                    if lr_pred == xgb_pred:
                        st.success("✅ **Models Agree:** Both models predict the same outcome")
                    else:
                        st.warning("⚠️ **Models Disagree:** Consider both predictions and clinical judgment")
                    
                    # Average prediction
                    avg_prob = (lr_prob + xgb_prob) / 2
                    avg_pred = 1 if avg_prob >= default_threshold else 0
                    
                    st.markdown("### 📈 Ensemble Prediction")
                    st.metric("Average Risk Score", f"{avg_prob:.3f}")
                    st.metric("Ensemble Prediction", "🔴 HIGH RISK" if avg_pred == 1 else "🟢 LOW RISK")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.write("Please ensure all required fields are filled correctly.")
            
        except Exception as e:
            st.error(f"❌ Failed to load saved models: {str(e)}")
            st.write("Please train new models or check your saved model files.")
    
    st.stop()

# If we reach here, we need to train new models
if upload is None:
    st.info("👆 Upload an Excel/CSV file to begin. Expected columns can be mapped after upload.")
    st.stop()

# Load ML dependencies only when needed
st.info("🔄 Loading ML libraries...")
ml_libs = load_ml_dependencies()

# Load data
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
    for k in ["age", "counseling", "disclosure", "gender"]:
        col_map[k] = st.selectbox(f"Map '{k}'", cols, index=(cols.index(DEFAULT_EXPECTED[k]) if DEFAULT_EXPECTED[k] in df.columns else 0))
with cm3:
    for k in ["funds", "mstatus", "employmenstat", "education", "religion", "whostage", "agecat", "weightcat"]:
        col_map[k] = st.selectbox(f"Map '{k}'", cols, index=(cols.index(DEFAULT_EXPECTED[k]) if DEFAULT_EXPECTED[k] in df.columns else 0))

# Build working dataframe
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

st.markdown(f"**Using features** → Numeric: `{num_cols}` · Binary: `{bin_cols}` · Categorical: `{cat_cols}`")

# Clean target
st.subheader("Target Column Analysis")
target_col = work["ltfu"]
st.write(f"**Target column '{target_col.name}' contains:**")
st.write(f"- Total rows: {len(target_col)}")
st.write(f"- Unique values: {target_col.unique()[:10]}")
st.write(f"- Data types: {target_col.dtype}")

# Convert target to numeric
y_original = target_col.copy()

if target_col.dtype == 'object':
    mapping_dict = {
        'Yes': 1, 'No': 0, 'yes': 1, 'no': 0,
        '1': 1, '0': 0, 'True': 1, 'False': 0,
        'true': 1, 'false': 0, 'Y': 1, 'N': 0,
        'y': 1, 'n': 0,
        'LTFU': 1, 'ltfu': 1, 'Ltfu': 1,
        'Not LTFU': 0, 'Not-LTFU': 0, 'not ltfu': 0, 'not-ltfu': 0,
        'NOT LTFU': 0, 'NOT-LTFU': 0, 'Not Ltfu': 0, 'Not-ltfu': 0
    }
    
    y_mapped = target_col.map(mapping_dict)
    unique_vals = target_col.unique()
    st.write(f"**Found unique values:** {unique_vals}")
    
    unmapped_mask = y_mapped.isna()
    if unmapped_mask.any():
        unmapped_vals = target_col[unmapped_mask].unique()
        st.warning(f"**Unmapped values found:** {unmapped_vals}")
        st.write("These values will be converted to NaN and removed.")
    
    y = pd.to_numeric(y_mapped, errors="coerce")
else:
    y = pd.to_numeric(target_col, errors="coerce")

X = work.drop(columns=["ltfu"])

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

# Data validation and cleaning
st.subheader("Data Validation")
st.write(f"**Feature matrix shape:** {X.shape}")

# Remove patient ID columns
id_cols = [col for col in X.columns if 'patient' in col.lower() or 'id' in col.lower()]
if id_cols:
    st.info(f"**Removing ID columns:** {id_cols}")
    X = X.drop(columns=id_cols)

# Ensure all numeric columns are actually numeric
for col in num_cols:
    if col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            st.warning(f"Could not convert {col} to numeric, keeping as is")

# Convert all categorical columns to strings
for col in cat_cols:
    if col in X.columns:
        X[col] = X[col].astype(str)
        X[col] = X[col].replace('nan', np.nan)

# Convert all binary columns to strings
for col in bin_cols:
    if col in X.columns:
        X[col] = X[col].astype(str)
        X[col] = X[col].replace('nan', np.nan)

# Check for infinite values
inf_cols = []
for col in X.columns:
    if X[col].dtype in ['float64', 'int64']:
        if np.isinf(X[col]).any():
            inf_cols.append(col)

if inf_cols:
    st.warning(f"**Found infinite values in:** {inf_cols}")
    for col in inf_cols:
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

# Split data
X_trainval, X_test, y_trainval, y_test = ml_libs['train_test_split'](
    X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
)

if test_size >= 1.0:
    st.error("Test size must be less than 1.0")
    st.stop()

val_frac = val_size_overall / (1 - test_size)
X_train, X_val, y_train, y_val = ml_libs['train_test_split'](
    X_trainval, y_trainval, test_size=val_frac, stratify=y_trainval, random_state=RANDOM_STATE
)

# Train models
st.subheader("Model Training")
with st.spinner("Training models..."):
    try:
        # Preprocess — Logistic
        numeric_lr = ml_libs['Pipeline'](steps=[
            ("imputer", ml_libs['SimpleImputer'](strategy="median")),
            ("scaler", ml_libs['StandardScaler']()),
        ])

        binary_lr = ml_libs['Pipeline'](steps=[
            ("imputer", ml_libs['SimpleImputer'](strategy="most_frequent")),
        ])

        categorical_lr = ml_libs['Pipeline'](steps=[
            ("imputer", ml_libs['SimpleImputer'](strategy="most_frequent")),
            ("onehot", ml_libs['OneHotEncoder'](handle_unknown="ignore", sparse_output=False)),
        ])

        prep_lr = ml_libs['ColumnTransformer'](
            transformers=[
                ("num", numeric_lr, [c for c in num_cols if c in X_train.columns]),
                ("bin", binary_lr, [c for c in bin_cols if c in X_train.columns]),
                ("cat", categorical_lr, [c for c in cat_cols if c in X_train.columns]),
            ],
            remainder="drop",
        )

        logit = ml_libs['Pipeline'](steps=[
            ("prep", prep_lr),
            ("clf", ml_libs['LogisticRegression'](
                solver="liblinear", penalty="l2", max_iter=2000,
                class_weight=("balanced" if use_class_weight else None),
                random_state=RANDOM_STATE,
            )),
        ])

        # Preprocess — XGB
        numeric_xgb = ml_libs['Pipeline'](steps=[("pass", "passthrough")])
        binary_xgb = ml_libs['Pipeline'](steps=[("imputer", ml_libs['SimpleImputer'](strategy="most_frequent"))])
        categorical_xgb = ml_libs['Pipeline'](steps=[
            ("imputer", ml_libs['SimpleImputer'](strategy="most_frequent")),
            ("onehot", ml_libs['OneHotEncoder'](handle_unknown="ignore", sparse_output=False)),
        ])

        prep_xgb = ml_libs['ColumnTransformer'](
            transformers=[
                ("num", numeric_xgb, [c for c in num_cols if c in X_train.columns]),
                ("bin", binary_xgb, [c for c in bin_cols if c in X_train.columns]),
                ("cat", categorical_xgb, [c for c in cat_cols if c in X_train.columns]),
            ],
            remainder="drop",
        )

        # Train Logistic Regression
        logit.fit(X_train, y_train)
        
        # Train XGBoost
        Xtr_xgb = prep_xgb.fit_transform(X_train, y_train)
        Xva_xgb = prep_xgb.transform(X_val)
        ytr, yva = y_train.values, y_val.values

        xgb_model = ml_libs['xgb'].XGBClassifier(
            objective="binary:logistic",
            n_estimators=100,
            learning_rate=0.15,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            eval_metric="auc",
            early_stopping_rounds=15 if use_early_stopping else None,
            n_jobs=-1,
        )
        
        if use_early_stopping:
            xgb_model.fit(Xtr_xgb, ytr, eval_set=[(Xva_xgb, yva)], verbose=False)
        else:
            xgb_model.fit(Xtr_xgb, ytr)
        
        xgb_clf = ml_libs['Pipeline'](steps=[("prep", prep_xgb), ("clf", xgb_model)])
        
        st.success("✅ Models trained successfully!")
        
        # Save models for future use
        model_info = {
            "dataset_shape": X.shape,
            "target_distribution": y.value_counts().to_dict(),
            "features_used": len(num_cols) + len(bin_cols) + len(cat_cols),
            "training_date": datetime.now().isoformat()
        }
        
        # Save models
        try:
            os.makedirs("saved_models", exist_ok=True)
            
            with open("saved_models/logit_model.pkl", "wb") as f:
                pickle.dump(logit, f)
            with open("saved_models/xgb_model.pkl", "wb") as f:
                pickle.dump(xgb_clf, f)
            
            metadata = {
                "num_cols": num_cols,
                "bin_cols": bin_cols,
                "cat_cols": cat_cols,
                "model_info": model_info,
                "timestamp": datetime.now().isoformat()
            }
            with open("saved_models/metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)
            
            st.success("💾 **Models saved successfully!** You can now use 'Use Saved Models (Fast)' mode for instant predictions.")
        except Exception as e:
            st.warning(f"⚠️ Models trained but could not be saved: {e}")
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        st.stop()

# Evaluation
def evaluate(model, X_te, y_te, name, threshold):
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "Accuracy": ml_libs['accuracy_score'](y_te, y_pred),
        "Precision": ml_libs['precision_score'](y_te, y_pred, zero_division=0),
        "Recall": ml_libs['recall_score'](y_te, y_pred, zero_division=0),
        "F1": ml_libs['f1_score'](y_te, y_pred, zero_division=0),
        "ROC_AUC": ml_libs['roc_auc_score'](y_te, y_prob),
    }
    return metrics, y_prob, y_pred

# Compute evaluations
metrics_lr, prob_lr, pred_lr = evaluate(logit, X_test, y_test, "Logistic", default_threshold)
metrics_xgb, prob_xgb, pred_xgb = evaluate(xgb_clf, X_test, y_test, "XGB", default_threshold)

# Display results
st.subheader("Model Performance")
m1, m2 = st.columns(2)
with m1:
    st.subheader("Logistic Regression — Metrics")
    st.table(pd.Series(metrics_lr).round(4))
with m2:
    st.subheader("XGBoost — Metrics")
    st.table(pd.Series(metrics_xgb).round(4))

# Confusion matrices
cm1, cm2 = st.columns(2)
with cm1:
    st.markdown("**Confusion Matrix — Logistic**")
    cm = ml_libs['confusion_matrix'](y_test, pred_lr)
    st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))
with cm2:
    st.markdown("**Confusion Matrix — XGB**")
    cm = ml_libs['confusion_matrix'](y_test, pred_xgb)
    st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

# Interactive Prediction Form
st.subheader("🔮 Interactive LTFU Prediction")
st.write("Enter patient details below to get real-time LTFU predictions from both models:")

# Create prediction form
with st.form("prediction_form"):
    st.markdown("### Patient Information")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        duration_days = st.number_input("Duration in Days", min_value=0, max_value=10000, value=365, step=1)
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=60.0, step=0.1)
        cd4_count = st.number_input("CD4 Count", min_value=0, max_value=2000, value=500, step=1)
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=35, step=1)
        
        # Binary features
        counseling = st.selectbox("Counseling", ["Yes", "No"], index=0)
        disclosure = st.selectbox("Disclosure", ["Yes", "No"], index=0)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    
    with col2:
        st.markdown("**Socioeconomic Factors**")
        funds = st.selectbox("Funding Source", ["Government", "Private", "NGO", "Self", "Other"], index=0)
        mstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed", "Other"], index=1)
        employment = st.selectbox("Employment Status", ["Employed", "Unemployed", "Student", "Retired", "Other"], index=0)
        education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary", "Other"], index=2)
        religion = st.selectbox("Religion", ["Christian", "Muslim", "Hindu", "Other", "None"], index=0)
    
    with col3:
        st.markdown("**Clinical Factors**")
        whostage = st.selectbox("WHO Stage", ["Stage 1", "Stage 2", "Stage 3", "Stage 4"], index=0)
        weightcat = st.selectbox("Weight Category", ["Underweight", "Normal", "Overweight", "Obese"], index=1)
    
    # Submit button
    submitted = st.form_submit_button("🔮 Predict LTFU Risk", use_container_width=True)

# Process prediction when form is submitted
if submitted:
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Function to automatically calculate age category from age
    def get_age_category(age_value):
        if age_value < 25:
            return "<25"
        elif age_value < 35:
            return "25-34"
        elif age_value < 45:
            return "35-44"
        elif age_value < 55:
            return "45-54"
        else:
            return "55+"
    
    # Calculate age category automatically from age input
    agecat = get_age_category(age)
    
    # Show the calculated age category to the user
    st.info(f"📊 **Age Category:** {age} years → {agecat}")
    
    # Create input data
    input_data = {
        'durationindays': duration_days,
        'weight': weight,
        'cd4': cd4_count,
        'age': age,
        'counseling': 1 if counseling == "Yes" else 0,
        'disclosure': 1 if disclosure == "Yes" else 0,
        'gender': 1 if gender == "Male" else 0,
        'funds': funds,
        'mstatus': mstatus,
        'employmenstat': employment,
        'education': education,
        'religion': religion,
        'whostage': whostage,
        'agecat': agecat,
        'weightcat': weightcat
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure all columns are present and in correct order
    for col in X.columns:
        if col not in input_df.columns:
            if col in num_cols:
                input_df[col] = 0.0
            elif col in bin_cols:
                input_df[col] = 0
            elif col in cat_cols:
                input_df[col] = "Unknown"
    
    # Only reorder columns that actually exist in both datasets
    common_cols = [col for col in X.columns if col in input_df.columns]
    if len(common_cols) != len(X.columns):
        st.warning(f"Some columns from training data are missing from prediction form. Using {len(common_cols)} out of {len(X.columns)} columns.")
    
    # Reorder columns to match training data (only common columns)
    input_df = input_df[common_cols]
    
    try:
        # Get predictions from both models
        with st.spinner("Computing predictions..."):
            lr_prob = logit.predict_proba(input_df)[0, 1]
            xgb_prob = xgb_clf.predict_proba(input_df)[0, 1]
        
        # Get binary predictions
        lr_pred = 1 if lr_prob >= default_threshold else 0
        xgb_pred = 1 if xgb_prob >= default_threshold else 0
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 Logistic Regression")
            st.metric("LTFU Risk Score", f"{lr_prob:.3f}", help="Probability of being lost to follow-up (0-1)")
            st.metric("Prediction", "🔴 HIGH RISK" if lr_pred == 1 else "🟢 LOW RISK")
            
            # Risk interpretation
            if lr_prob < 0.3:
                risk_level = "🟢 Low Risk"
                recommendation = "Continue current care plan"
            elif lr_prob < 0.7:
                risk_level = "🟡 Medium Risk"
                recommendation = "Consider additional support interventions"
            else:
                risk_level = "🔴 High Risk"
                recommendation = "Implement intensive follow-up protocols"
            
            st.info(f"**Risk Level:** {risk_level}")
            st.info(f"**Recommendation:** {recommendation}")
        
        with col2:
            st.markdown("### 🌳 XGBoost")
            st.metric("LTFU Risk Score", f"{xgb_prob:.3f}", help="Probability of being lost to follow-up (0-1)")
            st.metric("Prediction", "🔴 HIGH RISK" if xgb_pred == 1 else "🟢 LOW RISK")
            
            # Risk interpretation
            if xgb_prob < 0.3:
                risk_level = "🟢 Low Risk"
                recommendation = "Continue current care plan"
            elif xgb_prob < 0.7:
                risk_level = "🟡 Medium Risk"
                recommendation = "Consider additional support interventions"
            else:
                risk_level = "🔴 High Risk"
                recommendation = "Implement intensive follow-up protocols"
            
            st.info(f"**Risk Level:** {risk_level}")
            st.info(f"**Recommendation:** {recommendation}")
        
        # Model agreement
        st.markdown("### 🤝 Model Agreement")
        if lr_pred == xgb_pred:
            st.success("✅ **Models Agree:** Both models predict the same outcome")
        else:
            st.warning("⚠️ **Models Disagree:** Consider both predictions and clinical judgment")
        
        # Average prediction
        avg_prob = (lr_prob + xgb_prob) / 2
        avg_pred = 1 if avg_prob >= default_threshold else 0
        
        st.markdown("### 📈 Ensemble Prediction")
        st.metric("Average Risk Score", f"{avg_prob:.3f}")
        st.metric("Ensemble Prediction", "🔴 HIGH RISK" if avg_pred == 1 else "🟢 LOW RISK")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please ensure all required fields are filled correctly.")

st.markdown("---")

# Predictions & download
st.subheader("📋 Test Set Predictions")
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
st.caption("Methodology implemented: preprocessing (impute/scale/encode), stratified 70/15/15 split, LR + XGB models, metrics/curves, and interpretability.")
