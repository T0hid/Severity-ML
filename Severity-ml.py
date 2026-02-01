"""
================================================================================
GENETIC DISEASE SEVERITY PREDICTION PIPELINE
================================================================================
"""

import os
import sys
import json
import pickle
import warnings
import requests
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Sklearn
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, GroupShuffleSplit, RandomizedSearchCV
)
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)

# Deep Learning & Boosting
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: LightGBM required. Install with: pip install lightgbm")
    sys.exit(1)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: SHAP not installed. Install with: pip install shap")

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

from config import Config

warnings.filterwarnings('ignore')
np.random.seed(Config.RANDOM_SEED)

# ================================================================================
# LOGGING UTILITY
# ================================================================================

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.logs = []
        self.start_time = datetime.now()
        
    def log(self, msg, level="INFO", print_msg=True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] [{level}] {msg}"
        self.logs.append(full_msg)
        if print_msg:
            print(full_msg)
            
    def section(self, title):
        separator = "=" * 70
        self.log(separator)
        self.log(title)
        self.log(separator)

    def subsection(self, title):
        self.log("-" * 50)
        self.log(title)
        
    def save(self):
        elapsed = datetime.now() - self.start_time
        self.log(f"Total runtime: {elapsed}")
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\nSEVERITY PIPELINE LOG\n" + "=" * 80 + "\n\n")
            for log in self.logs:
                f.write(log + "\n")

# Initialize directories and logger
Config.setup_directories()
logger = Logger(Config.OUTPUT_DIR / "pipeline_log.txt")

# ================================================================================
# DATA LOADING
# ================================================================================

class DataContainer:
    """Container for all data, scenarios, and strata"""
    def __init__(self):
        self.scenarios = {}
        self.strata = {}
        self.gene_symbols = None
        self.train_idx = None
        self.test_idx = None
        self.metadata = None
        
    def add_scenario(self, name, X, feature_names):
        self.scenarios[name] = {'X': X, 'features': feature_names}
        
    def add_stratum(self, name, indices):
        self.strata[name] = indices
        
    def get_data(self, scenario, stratum='all'):
        X = self.scenarios[scenario]['X']
        features = self.scenarios[scenario]['features']
        y = self.metadata['Multiclass_Target'].values
        
        if stratum != 'all' and stratum in self.strata:
            idx = self.strata[stratum]
            return X[idx], y[idx], features
        return X, y, features

def validate_files():
    """Ensure required files exist before starting"""
    required = [Config.PRIMARY_DATA, Config.CORE_FEATURES]
    missing = [str(f) for f in required if not os.path.exists(f)]
    if missing:
        logger.log(f"CRITICAL ERROR: Missing input files: {missing}", level="ERROR")
        logger.log("Please place files in the 'data/' directory as specified in config.py")
        sys.exit(1)

def load_mechanistic_features():
    logger.log("Loading mechanistic features...")
    return pd.read_csv(Config.CORE_FEATURES)

def load_variant_features():
    if not os.path.exists(Config.DOMAIN_FEATURES):
        logger.log("Variant features not found, skipping...", level="WARNING")
        return pd.DataFrame()
    return pd.read_csv(Config.DOMAIN_FEATURES).fillna(0)

def extract_phenotype_features(primary_data_path):
    logger.subsection("Extracting Gene-Disease Phenotype Features")
    df = pd.read_csv(primary_data_path, low_memory=False)
    
    col_map = {
        'Gene Name': 'gene_symbol', 'MONDO_ID': 'mondo_id',
        'Severity': 'target_severity', 'HPO_Term': 'hpo_term', 'Disease': 'disease_name'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df = df.dropna(subset=['gene_symbol', 'mondo_id', 'target_severity'])

    top_hpo_terms = df['hpo_term'].value_counts().head(100).index.tolist()
    grouped = df.groupby(['gene_symbol', 'mondo_id', 'disease_name', 'target_severity'])
    
    matrix_rows = []
    for (gene, mondo, disease, severity), group in grouped:
        disease_hpos = set(group['hpo_term'].unique())
        sev_clean = str(severity).strip()
        target_val = Config.TARGET_MAP.get(sev_clean, 0)
        
        row = {
            'gene_symbol': gene, 'mondo_id': mondo, 'disease_name': disease,
            'Multiclass_Target': target_val, 'original_severity': severity
        }
        for hpo in top_hpo_terms:
            clean_name = f"hpo_{hpo.replace(' ', '_').replace('-', '_')[:50]}"
            row[clean_name] = 1 if hpo in disease_hpos else 0
        matrix_rows.append(row)
        
    return pd.DataFrame(matrix_rows)

def preprocess_features(df, feature_cols):
    X = df[feature_cols].copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    
    # Drop high-missing
    missing_pct = X.isnull().mean()
    X = X.drop(columns=missing_pct[missing_pct > 0.5].index.tolist())
    feature_cols = X.columns.tolist()
    
    # Impute and Scale
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, feature_cols

def prepare_data():
    validate_files()
    logger.section("DATA PREPARATION")
    container = DataContainer()
    
    # Load and Merge
    mech_df = load_mechanistic_features()
    pheno_df = extract_phenotype_features(Config.PRIMARY_DATA)
    full_df = pheno_df.merge(mech_df, on='gene_symbol', how='left')
    
    # Ensure mechanistic data exists
    valid_mech_col = [c for c in mech_df.columns if c != 'gene_symbol'][0]
    full_df = full_df.dropna(subset=[valid_mech_col])
    
    # Variant Data
    variant_df = load_variant_features()
    variant_cols = []
    if not variant_df.empty:
        full_df = full_df.merge(variant_df, on='gene_symbol', how='left').fillna(0)
        variant_cols = [c for c in variant_df.columns if c != 'gene_symbol']
        
    container.gene_symbols = full_df['gene_symbol'].values
    
    # Leakage Guard
    leakage_keywords = ['hpo', 'pheno', 'symptom', 'disease', 'onset', 'clinical', 'severity', 'target', 'class']
    safe_keywords = ['depmap', 'gnomad', 'gtex', 'ppi', 'network', 'conservation', 'essential']
    
    potential_mech_cols = [c for c in full_df.columns if c not in variant_cols and not c.startswith('hpo_')]
    cleaned_mech_cols = []
    
    for col in potential_mech_cols:
        col_lower = col.lower()
        if col_lower in ['gene_symbol', 'mondo_id', 'disease_name']: continue
        
        is_leak = any(bad in col_lower for bad in leakage_keywords)
        is_safe = any(safe in col_lower for safe in safe_keywords)
        
        if not is_leak or is_safe:
            cleaned_mech_cols.append(col)

    # Scenarios
    logger.log("Building Scenarios...")
    X_mech, mech_feats = preprocess_features(full_df, cleaned_mech_cols)
    container.add_scenario('S1_Mechanistic', X_mech, mech_feats)
    
    X_pheno, pheno_feats = preprocess_features(full_df, [c for c in full_df.columns if c.startswith('hpo_')])
    container.add_scenario('S5_Phenotype', X_pheno, pheno_feats)
    
    X_mech_pheno = np.hstack([X_mech, X_pheno])
    container.add_scenario('S2_Mech_Pheno', X_mech_pheno, mech_feats + pheno_feats)
    
    if variant_cols:
        X_var, var_feats = preprocess_features(full_df, variant_cols)
        X_mech_var = np.hstack([X_mech, X_var])
        container.add_scenario('S3_Mech_Variant', X_mech_var, mech_feats + var_feats)
        container.add_scenario('S4_Full', np.hstack([X_mech_var, X_pheno]), mech_feats + var_feats + pheno_feats)
    else:
        container.add_scenario('S4_Full', X_mech_pheno, mech_feats + pheno_feats)

    # Metadata & Strata
    container.metadata = full_df[['gene_symbol', 'disease_name', 'Multiclass_Target']].copy()
    # Add Binary Target for legacy compatibility/stratification if needed
    container.metadata['Binary_Target'] = (container.metadata['Multiclass_Target'] >= 2).astype(int)
    
    container.add_stratum('all', np.arange(len(full_df)))
    if 'is_dominant' in full_df.columns:
        container.add_stratum('dominant', np.where(full_df['is_dominant'] == 1)[0])

    # Split
    splitter = GroupShuffleSplit(test_size=Config.TEST_SIZE, n_splits=1, random_state=Config.RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X_mech, container.metadata['Multiclass_Target'], groups=full_df['gene_symbol']))
    container.train_idx = train_idx
    container.test_idx = test_idx
    
    return container

# ================================================================================
# MODELING
# ================================================================================

class SeverityNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

def train_nn(X_train, y_train, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    model = SeverityNet(X_train.shape[1], Config.NUM_CLASSES).to(device)
    
    class_counts = np.bincount(y_train, minlength=Config.NUM_CLASSES)
    weights = torch.FloatTensor(1. / (class_counts + 1e-5)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    
    model.train()
    for _ in range(30):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
    return preds, probs

def train_lgb(X_train, y_train, X_test):
    model = lgb.LGBMClassifier(
        objective='multiclass', num_class=Config.NUM_CLASSES, class_weight='balanced',
        random_state=Config.RANDOM_SEED, verbose=-1, n_jobs=4
    )
    model.fit(X_train, y_train)
    return model.predict(X_test), model.predict_proba(X_test), model

def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        'accuracy': np.mean(y_true == y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except:
        metrics['auc_roc'] = 0.0
    return metrics

# ================================================================================
# LLM & SHAP
# ================================================================================

class LLMExplainer:
    def __init__(self):
        self.api_key = Config.OPENROUTER_API_KEY
        self.enabled = Config.LLM_ENABLED
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def explain(self, gene, prediction, features_text):
        if not self.enabled: return "LLM Disabled"
        
        prompt = f"""
        Gene: {gene}
        Predicted Severity: {prediction}
        Key Features:
        {features_text}
        
        Explain why this genetic condition is predicted as such, contrasting biological risk with phenotype.
        """
        
        try:
            resp = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": Config.LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            return resp.json()['choices'][0]['message']['content'] if resp.status_code == 200 else "API Error"
        except Exception as e:
            return f"Error: {e}"

# ================================================================================
# VISUALIZATION
# ================================================================================

def plot_roc(y_test, y_prob, title, filename):
    plt.figure(figsize=(8, 6))
    classes = range(Config.NUM_CLASSES)
    y_bin = label_binarize(y_test, classes=classes)
    
    for i in classes:
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f'Class {i} (AUC={auc:.2f})')
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title(title)
    plt.savefig(Config.OUTPUT_DIR / 'figures' / filename)
    plt.close()

# ================================================================================
# MAIN
# ================================================================================

def main():
    logger.section("STARTING PIPELINE")
    
    # 1. Prepare Data
    data = prepare_data()
    
    # 2. Train & Evaluate
    results = []
    
    for scenario in data.scenarios:
        logger.subsection(f"Scenario: {scenario}")
        X, y, feats = data.get_data(scenario, 'all')
        
        X_train, X_test = X[data.train_idx], X[data.test_idx]
        y_train, y_test = y[data.train_idx], y[data.test_idx]
        
        # LightGBM
        pred, prob, model = train_lgb(X_train, y_train, X_test)
        metrics = compute_metrics(y_test, pred, prob)
        results.append({'Scenario': scenario, 'Model': 'LightGBM', **metrics})
        plot_roc(y_test, prob, f"{scenario} ROC", f"roc_{scenario}.png")
        
        # PyTorch
        pred_nn, prob_nn = train_nn(X_train, y_train, X_test, y_test)
        metrics_nn = compute_metrics(y_test, pred_nn, prob_nn)
        results.append({'Scenario': scenario, 'Model': 'DNN', **metrics_nn})
        
        # SHAP (For Best Model only)
        if scenario == 'S4_Full' and HAS_SHAP:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
            # Save raw SHAP data
            with open(Config.OUTPUT_DIR / "results" / "shap_values.pkl", "wb") as f:
                pickle.dump(shap_vals, f)
    
    # 3. Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(Config.OUTPUT_DIR / "results" / "metrics.csv", index=False)
    logger.log("\nMetrics Summary:")
    print(res_df)
    
    # 4. LLM Explanations (Sample)
    if Config.LLM_ENABLED:
        logger.section("Generating Explanations")
        llm = LLMExplainer()
        # Mock example - logic to extract top features from SHAP would go here
        exp = llm.explain("TEST_GENE", "Severe", "- High conservation score\n- Associated with seizures")
        with open(Config.OUTPUT_DIR / "explanations" / "sample.txt", "w") as f:
            f.write(exp)
            
    logger.save()

if __name__ == "__main__":
    main()