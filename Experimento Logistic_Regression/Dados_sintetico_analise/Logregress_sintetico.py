# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Código do TCC: Engenharia de Atributos Não Supervisionada para Detecção de Anomalias.
Avaliação usando Regressão Logística com Repeated Stratified Hold-Out (30 repetições).

Cenários avaliados:
  1) LR - Baseline (sem ajuste de peso)
  2) LR - SMOTE
  3) LR - Cost-sensitive (class_weight="balanced")
  4) LR - Engenharia de atributos não supervisionada (Cenário Proposto)
"""
import warnings
warnings.filterwarnings('ignore')

import os
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from numpy import mean, std
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_rel

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    f1_score, average_precision_score, precision_score, recall_score,
    brier_score_loss, precision_recall_curve, roc_auc_score
)
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as Pipe
from imblearn.datasets import fetch_datasets

# --- Configurações globais de plotagem ---
plt.rcParams.update({
    'font.family':         'Arial',
    'font.size':           11,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.spines.left':    True,
    'axes.spines.bottom':  True,
    'axes.grid':           False,
    'axes.facecolor':      'white',
    'figure.facecolor':    'white',
    'savefig.facecolor':   'white',
    'figure.dpi':          120,
    'savefig.dpi':         400,
})

print("Iniciando pipeline de avaliação - Regressão Logística (4 Cenários)")
print("-" * 60)

# --- 1. Carregamento e Particionamento dos Dados ---

N_REPS    = 30
BASE_SEED = 42
SEEDS     = [BASE_SEED + i * 13 for i in range(N_REPS)]

df = pd.read_csv("dataset_tcc_v3_0.88_2.csv",sep=',')
df.columns = [f"Feature{i}" for i in range(len(df.columns[:-1]))] + ['Target']

feature_cols = df.columns[:-1].tolist()
target_col   = df.columns[-1]

X = df[feature_cols].values
y = LabelEncoder().fit_transform(df[target_col].values)

# Separando 15% exclusivamente para o tuning (evita data leakage nos testes)
X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X, y, test_size=0.85, random_state=BASE_SEED, stratify=y
)

print(f"Amostras para tuning (X_dev): {X_dev.shape[0]}")
print(f"Amostras para avaliação (X_holdout): {X_holdout.shape[0]}")

# Gerando os splits do hold-out
all_splits = []
for seed in SEEDS:
    Xtr, Xte, ytr, yte = train_test_split(
        X_holdout, y_holdout, test_size=0.3, random_state=seed, stratify=y_holdout
    )
    all_splits.append((Xtr, Xte, ytr, yte))

# Split de referência usado mais tarde para plotagem das curvas
X_train, X_test, y_train, y_test = all_splits[0]

# --- 2. Transformer Não Supervisionado ---

class UnsupervisedFeaturizer(BaseEstimator, TransformerMixin):
    """
    Gera novas features baseadas em algoritmos não supervisionados.
    Permite ativar/desativar métodos para facilitar o tuning.
    """
    def __init__(self, k_list=(2, 4), pca_comp=5, gmm_list=(2, 5, 7), 
                 lof_n_neighbors=20, return_orig=True, use_kmeans=True, 
                 use_pca=True, use_gmm=True, use_lof=True):
        self.k_list           = k_list
        self.pca_comp         = pca_comp
        self.gmm_list         = gmm_list
        self.lof_n_neighbors  = lof_n_neighbors
        self.return_orig      = return_orig
        self.use_kmeans       = use_kmeans
        self.use_pca          = use_pca
        self.use_gmm          = use_gmm
        self.use_lof          = use_lof

    def fit(self, X, y=None):
        if self.use_kmeans:
            self.kmeans_ = {}
            for k in self.k_list:
                self.kmeans_[k] = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)

        if self.use_pca:
            self.pca_ = PCA(n_components=self.pca_comp, random_state=42).fit(X)

        if self.use_gmm:
            self.gmm_ = {}
            for n in self.gmm_list:
                self.gmm_[n] = GaussianMixture(n_components=n, covariance_type='full', random_state=42).fit(X)

        if self.use_lof:
            self.lof_ = LocalOutlierFactor(n_neighbors=self.lof_n_neighbors, contamination='auto', novelty=True).fit(X)

        return self

    def transform(self, X):
        parts = []
        if self.return_orig:
            parts.append(X)

        if self.use_kmeans:
            for k, km in self.kmeans_.items():
                parts.append(km.transform(X).min(axis=1).reshape(-1, 1))
                parts.append(km.predict(X).reshape(-1, 1))

        if self.use_pca:
            trans = self.pca_.transform(X)
            recon = self.pca_.inverse_transform(trans)
            parts.append(np.mean((X - recon) ** 2, axis=1).reshape(-1, 1))
            parts.append(trans)

        if self.use_gmm:
            for n, gmm in self.gmm_.items():
                parts.append(gmm.score_samples(X).reshape(-1, 1))

        if self.use_lof:
            # Invertendo o sinal do LOF para que valores altos indiquem maior anormalidade
            parts.append(-self.lof_.score_samples(X).reshape(-1, 1))

        if not parts:
            raise ValueError("Nenhuma técnica ativada no UnsupervisedFeaturizer.")

        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        names = []
        if self.use_kmeans:
            for k in self.k_list:
                names += [f'km_k{k}_dist', f'km_k{k}_clust']
        if self.use_pca:
            names.append(f'pca_n{self.pca_comp}_error')
            names += [f'pca_n{self.pca_comp}_pc{i+1}' for i in range(self.pca_comp)]
        if self.use_gmm:
            for n in self.gmm_list:
                names.append(f'gmm_n{n}_loglik')
        if self.use_lof:
            names.append('lof_novelty_score')
        return names

# --- 3. Funções Auxiliares e Pipelines ---

def fisher_j1(probs: np.ndarray, y: np.ndarray) -> float:
    """Calcula o critério Fisher J1 para avaliar a separabilidade."""
    p0, p1 = probs[y == 0], probs[y == 1]
    num  = (p1.mean() - p0.mean()) ** 2
    den  = p0.var() + p1.var() + 1e-10
    return float(num / den)

def make_pipe_orig(model):
    return Pipe([('scaler', PowerTransformer()), ('model', model)])

def make_pipe_smote(model):
    return Pipe([('scaler', PowerTransformer()), ('smote', SMOTE(random_state=42)), ('model', model)])

def make_pipe_tunable(model):
    return Pipe([
        ('scaler',  PowerTransformer()),
        ('unsup',   UnsupervisedFeaturizer(return_orig=True)),
        ('scaler2', PowerTransformer()),
        ('model',   model),
    ])

def make_pipe_full(model):
    # Versão preliminar, será reescrita após o tuning
    return Pipe([
        ('scaler',  PowerTransformer()),
        ('unsup',   UnsupervisedFeaturizer(return_orig=True)),
        ('scaler2', PowerTransformer()),
        ('model',   model),
    ])

# --- 4. Tuning de Hiperparâmetros (Cenário 4) ---

lr_none_params     = dict(solver='lbfgs', max_iter=1000, random_state=42, class_weight=None)
lr_balanced_params = dict(solver='lbfgs', max_iter=1000, random_state=42, class_weight='balanced')

param_distributions = {
    'unsup__pca_comp':         [3, 4, 5],
    'unsup__k_list':           [(2, 4), (3, 5), (3, 5, 8), (2, 5, 10)],
    'unsup__gmm_list':         [(2, 5, 7)],
    'unsup__lof_n_neighbors':  [20, 50, 100],
}

print("\nExecutando RandomizedSearch no cenário 4...")
_model_ft = LogisticRegression(**lr_none_params)
_pipe_ft  = make_pipe_tunable(_model_ft)

random_search = RandomizedSearchCV(
    _pipe_ft,
    param_distributions=param_distributions,
    n_iter=20,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='average_precision',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

random_search.fit(X_dev, y_dev)

_best_unsup_params = {
    k.replace('unsup__', ''): v
    for k, v in random_search.best_params_.items()
    if k.startswith('unsup__')
}

_uf_tmp   = UnsupervisedFeaturizer(**_best_unsup_params)
N_FEAT_C4 = len(feature_cols) + len(_uf_tmp.get_feature_names_out())

# Atualiza o pipeline completo com os melhores parâmetros encontrados
def make_pipe_full(model):
    return Pipe([
        ('scaler',  PowerTransformer()),
        ('unsup',   UnsupervisedFeaturizer(return_orig=True, **_best_unsup_params)),
        ('scaler2', PowerTransformer()),
        ('model',   model),
    ])

# --- 5. Definição dos Cenários ---

SCENARIO_LABELS = ['1-Orig (cw=None)', '2-Orig+SMOTE', '3-Orig (cw=balanced)', '4-Orig+Unsup']

SCENARIOS = [
    ('LR (cw=None)',        make_pipe_orig,   lr_none_params,     '1-Orig (cw=None)'),
    ('LR (cw=None)+SMOTE',  make_pipe_smote,  lr_none_params,     '2-Orig+SMOTE'),
    ('LR (balanced)',       make_pipe_orig,   lr_balanced_params, '3-Orig (cw=balanced)'),
    ('LR+Unsup',            make_pipe_full,   lr_none_params,     '4-Orig+Unsup'),
]

FEAT_TO_SEP = {
    '1-Orig (cw=None)':     'C1 — Dados',
    '2-Orig+SMOTE':         "C2 — Dados com 'SMOTE'",
    '3-Orig (cw=balanced)': "C3 — Dados com 'cost-sensitive'",
    '4-Orig+Unsup':         'C4 — Dados com engenharia de atributos',
}

SEP_SCENARIO_INFO = [
    ('C1 — Dados',                             '1-Orig (cw=None)',     6),
    ("C2 — Dados com 'SMOTE'",                 '2-Orig+SMOTE',         6),
    ("C3 — Dados com 'cost-sensitive'",        '3-Orig (cw=balanced)', 6),
    ('C4 — Dados com engenharia de atributos', '4-Orig+Unsup',         N_FEAT_C4),
]

# --- 7. Avaliação (Repeated Stratified Hold-Out) ---

metrics_keys = ['F1_score', 'PR_AUC', 'ROC_AUC', 'Precision', 'Recall']
sep_keys     = ['Fisher J1', 'Brier Score']

rep_metrics = {lbl: {k: [] for k in metrics_keys} for lbl in SCENARIO_LABELS}
rep_sep     = {sn:  {k: [] for k in sep_keys}     for sn in [info[0] for info in SEP_SCENARIO_INFO]}

fitted_pipes = {}
probs_sep    = {}

print(f"Rodando validação cruzada ({N_REPS} repetições)...")

for rep_idx, (X_tr, X_te, y_tr, y_te) in enumerate(all_splits):
    is_ref = (rep_idx == 0)

    for model_name, pipe_fn, lr_params_dict, feat_label in SCENARIOS:
        model = LogisticRegression(**lr_params_dict)
        pipe  = pipe_fn(model)
        pipe.fit(X_tr, y_tr)

        y_pred  = pipe.predict(X_te)
        y_proba = pipe.predict_proba(X_te)[:, 1]

        # Métricas de desempenho
        rep_metrics[feat_label]['F1_score'].append(f1_score(y_te, y_pred))
        rep_metrics[feat_label]['PR_AUC'].append(average_precision_score(y_te, y_proba))
        rep_metrics[feat_label]['ROC_AUC'].append(roc_auc_score(y_te, y_proba))
        rep_metrics[feat_label]['Precision'].append(precision_score(y_te, y_pred))
        rep_metrics[feat_label]['Recall'].append(recall_score(y_te, y_pred))

        # Métricas de separabilidade
        sep_name = FEAT_TO_SEP[feat_label]
        rep_sep[sep_name]['Fisher J1'].append(fisher_j1(y_proba, y_te))
        rep_sep[sep_name]['Brier Score'].append(brier_score_loss(y_te, y_proba))

        if is_ref:
            fitted_pipes[feat_label]           = pipe
            probs_sep[FEAT_TO_SEP[feat_label]] = y_proba

# Consolidando resultados
final_rows = []
for feat_label in SCENARIO_LABELS:
    m = rep_metrics[feat_label]
    final_rows.append({
        'Features':       feat_label,
        'F1_score':       np.mean(m['F1_score']),
        'PR_AUC':         np.mean(m['PR_AUC']),
        'ROC_AUC':        np.mean(m['ROC_AUC']),
        'Precision':      np.mean(m['Precision']),
        'Recall':         np.mean(m['Recall']),
        'F1_std':         np.std(m['F1_score']),
        'PR_AUC_std':     np.std(m['PR_AUC']),
        'ROC_AUC_std':    np.std(m['ROC_AUC']),
        'Precision_std':  np.std(m['Precision']),
        'Recall_std':     np.std(m['Recall']),
    })
results_df_test = pd.DataFrame(final_rows).sort_values(by=['Features'])

sep_records = []
for sep_name, feat_label, n_feat in SEP_SCENARIO_INFO:
    s = rep_sep[sep_name]
    sep_records.append({
        'Cenário':          sep_name,
        'Features':         feat_label,
        '# Features':       n_feat,
        'Fisher J1':        np.mean(s['Fisher J1']),
        'Fisher J1_std':    np.std(s['Fisher J1']),
        'Brier Score':      np.mean(s['Brier Score']),
        'Brier Score_std':  np.std(s['Brier Score']),
    })
sep_df = pd.DataFrame(sep_records)

# --- 8. Exportação dos Dados ---

combined_holdout_df = results_df_test.merge(
    sep_df[['Features', 'Cenário', '# Features', 'Fisher J1', 'Fisher J1_std', 'Brier Score', 'Brier Score_std']],
    on='Features', how='inner'
)

col_order = [
    'Cenário', 'Features', '# Features',
    'PR_AUC', 'PR_AUC_std', 'ROC_AUC', 'ROC_AUC_std',
    'F1_score', 'F1_std', 'Precision', 'Precision_std', 'Recall', 'Recall_std',
    'Fisher J1', 'Fisher J1_std', 'Brier Score', 'Brier Score_std',
]
combined_holdout_df = combined_holdout_df[col_order]
combined_holdout_df.to_excel('figures_all_v3/results_holdout_completo_v5_lr.xlsx', index=False)

# --- 9. Feature Importance (Cenário 4) ---

def _build_fi(pipe_fn, label):
    model  = LogisticRegression(**lr_none_params)
    pipe   = pipe_fn(model)
    pipe.fit(X_train, y_train)
    fitted = pipe.named_steps['model']
    feats  = pipe.named_steps['unsup']
    names  = feature_cols + list(feats.get_feature_names_out())
    
    n_coef = fitted.coef_.shape[1]
    if len(names) != n_coef:
        names = [f'feature_{i}' for i in range(n_coef)]
        
    raw  = np.abs(fitted.coef_[0])
    norm = raw / raw.sum()
    
    return pd.DataFrame({
        'Feature':    names,
        'Importance': norm,
        'Coef_Raw':   fitted.coef_[0],
    }).sort_values('Importance', ascending=False)

def _group_importance(fi_df):
    return {
        'K-Means':   fi_df[fi_df['Feature'].str.contains('km_')]['Importance'].sum(),
        'PCA':       fi_df[fi_df['Feature'].str.contains('pca_')]['Importance'].sum(),
        'GMM':       fi_df[fi_df['Feature'].str.contains('gmm_')]['Importance'].sum(),
        'LOF':       fi_df[fi_df['Feature'].str.contains('lof_')]['Importance'].sum(),
        'Originais': fi_df[fi_df['Feature'].isin(feature_cols)]['Importance'].sum(),
    }

fi_c4    = _build_fi(make_pipe_full, 'C4')
gi_c4    = _group_importance(fi_c4)
st_df_c4 = (pd.DataFrame(list(gi_c4.items()), columns=['Tipo', 'Importância'])
              .sort_values('Importância', ascending=False))

fi_c4.to_excel('figures_all_v3/feature_importance_c4_v5_lr.xlsx', index=False)
st_df_c4.to_excel('figures_all_v3/feature_importance_by_group_c4_v5_lr.xlsx', index=False)

# --- 10. Visualizações Exploratórias ---

PALETTE_6 = {
    '1-Orig (cw=None)':     '#373B42',
    '2-Orig+SMOTE':         '#5C7A99',
    '3-Orig (cw=balanced)': '#9EA1A7',
    '4-Orig+Unsup':         '#A3BACF',
}
SEP_COLORS = ['#373B42', '#373B42', '#373B42', '#373B42']

def _color_fi(f):
    if f in feature_cols: return '#373B42'
    elif 'km_' in f: return '#5C7A99'
    elif 'pca_' in f: return '#A3BACF'
    elif 'gmm_' in f: return '#9EA1A7'
    elif 'lof_' in f: return '#C46D61'
    return '#5C7A99'

# 10.1 Feature Importance Top 20
fig, ax = plt.subplots(1, 1, figsize=(12, 9))
top20_tmp  = fi_c4.head(20)
colors_tmp = [_color_fi(f) for f in top20_tmp['Feature']]
ax.barh(range(len(top20_tmp)), top20_tmp['Importance'], color=colors_tmp, alpha=0.88, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(top20_tmp)))
ax.set_yticklabels(top20_tmp['Feature'], fontsize=9.5)
ax.set_xlabel('|Coeficiente LR| (normalizado, soma=1)', fontsize=10)
ax.set_title('C4: Dados com engenharia de atributos')
ax.invert_yaxis()

_fi_group_patches = [
    mpatches.Patch(facecolor='#373B42', label='Originais'),
    mpatches.Patch(facecolor='#5C7A99', label='K-Means'),
    mpatches.Patch(facecolor='#A3BACF', label='PCA'),
    mpatches.Patch(facecolor='#9EA1A7', label='GMM'),
    mpatches.Patch(facecolor='#C46D61', label='LOF'),
]
fig.legend(handles=_fi_group_patches, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06))
plt.tight_layout()
plt.savefig('figures_all_v3/feature_importance_v5_lr.png', dpi=300, bbox_inches='tight')
plt.close()

# 10.2 Importância por grupo
group_palette = {k: '#373B42' for k in ['Originais', 'K-Means', 'PCA', 'GMM', 'LOF']}
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
st_sorted      = st_df_c4.sort_values('Importância', ascending=True)
bar_colors_grp = [group_palette.get(t, '#95a5a6') for t in st_sorted['Tipo']]
bars = ax.barh(st_sorted['Tipo'], st_sorted['Importância'], color=bar_colors_grp, alpha=0.88)
total_imp = st_sorted['Importância'].sum()

for bar, val in zip(bars, st_sorted['Importância']):
    pct = val / total_imp * 100
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2.,
            f'{pct:.1f}%', va='center', ha='left', fontsize=9.5, fontweight='bold')
ax.set_xlabel('Importância Agregada', fontsize=10)
plt.tight_layout()
plt.savefig('figures_all_v3/feature_importance_bar_v5_lr.png', dpi=300, bbox_inches='tight')
plt.close()

# 10.3 Coeficientes com sinal
fig, ax = plt.subplots(1, 1, figsize=(12, 9))
top20_tmp   = fi_c4.head(20).copy()
coef_colors = ['#A3BACF' if c > 0 else '#C46D61' for c in top20_tmp['Coef_Raw']]
ax.barh(range(len(top20_tmp)), top20_tmp['Coef_Raw'], color=coef_colors, alpha=0.88)
ax.set_yticks(range(len(top20_tmp)))
ax.set_yticklabels(top20_tmp['Feature'], fontsize=9.5)
ax.set_xlabel('Coeficiente LR (com sinal)', fontsize=10)
ax.set_title('C4: Dados com engenharia de atributos')
ax.invert_yaxis()
ax.axvline(0, color='#2c3e50', linewidth=0.9, linestyle='--')
plt.tight_layout()
plt.savefig('figures_all_v3/feature_coef_signed_v5_lr.png', dpi=300, bbox_inches='tight')
plt.close()

# 10.4 Boxplot/Stripplot
_V_SHORT           = ['C1', 'C2', 'C3', 'C4']
_SEP_NAMES_ORDERED = [info[0] for info in SEP_SCENARIO_INFO]

fig = plt.figure(figsize=(22, 24))
gs  = GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.32)

_SUBPLOT_CFG = [
    (fig.add_subplot(gs[0, :2]),  'perf', 'PR_AUC',      'PR-AUC',       3),
    (fig.add_subplot(gs[0, 2:]),  'perf', 'F1_score',    'F1',           3),
    (fig.add_subplot(gs[1, :2]),  'perf', 'ROC_AUC',     'ROC-AUC',      3),
    (fig.add_subplot(gs[1, 2:]),  'sep',  'Brier Score', 'Brier Score',  4),
    (fig.add_subplot(gs[2, 0:2]), 'sep',  'Fisher J1',   'Fisher J1',    4),
]

for ax, src, key, ylabel, dec in _SUBPLOT_CFG:
    if src == 'perf':
        data_violin = [rep_metrics[lbl][key] for lbl in SCENARIO_LABELS]
        n_items     = len(SCENARIO_LABELS)
    else:
        data_violin = [rep_sep[sn][key] for sn in _SEP_NAMES_ORDERED]
        n_items     = len(_SEP_NAMES_ORDERED)

    plot_df = pd.DataFrame({_V_SHORT[i]: data_violin[i] for i in range(n_items)}).melt(var_name='Cenário', value_name='Valor')

    sns.boxplot(data=plot_df, x='Cenário', y='Valor', ax=ax, palette=SEP_COLORS, width=0.4, showfliers=False,
                boxprops=dict(alpha=0.6, edgecolor='white', linewidth=1.5), medianprops=dict(color='#373B42', linewidth=2))
    sns.stripplot(data=plot_df, x='Cenário', y='Valor', ax=ax, palette=SEP_COLORS, size=6, alpha=0.7, jitter=0.08,
                  edgecolor='white', linewidth=0.6)

    fmt = f'{{:.{dec}f}}'
    for i, vals in enumerate(data_violin):
        mu, sig = round(np.mean(vals),3), round(np.std(vals),3)
        ax.text(i + 0.35, mu, f'{fmt.format(mu)}\n±{fmt.format(sig)}', ha='left', va='center', fontsize=8, color=SEP_COLORS[i],
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=SEP_COLORS[i], alpha=0.9, linewidth=1))

    ax.set(xlabel='', ylabel='', xlim=(-0.5, n_items - 0.1))
    ax.set_xticklabels(_V_SHORT, fontsize=8)
    ax.set_title(ylabel, fontsize=11, loc='center', fontweight='bold', color='black')

plt.savefig('figures_all_v3/boxplot_stripplot_metrics_repeated_ho_v5_lr.png', dpi=300, bbox_inches='tight')
plt.close()

# --- 11. Curvas Precision-Recall ---

PR_CURVE_SCENARIOS = [
    ('C1: Dados',                              'C1', '1-Orig (cw=None)',     '-'),
    ("C2: Dados com 'SMOTE'",                  'C2', '2-Orig+SMOTE',         '--'),
    ("C3: Dados com 'cost-sensitive'",         'C3', '3-Orig (cw=balanced)', '-.'),
    ('C4: Dados com engenharia de atributos',  'C4', '4-Orig+Unsup',         ':'),
]

pr_results = []
for label_long, label_short, feat_label, ls in PR_CURVE_SCENARIOS:
    pipe = fitted_pipes.get(feat_label)
    if not pipe: continue
    y_proba = pipe.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, y_proba)
    pr_results.append(dict(
        label_long=label_long, label_short=label_short, feat_label=feat_label,
        prec=prec, rec=rec, thr=thr, pr_auc_mean=np.mean(rep_metrics[feat_label]['PR_AUC']),
        pr_auc_std=np.std(rep_metrics[feat_label]['PR_AUC']), ls=ls
    ))

baseline_rate = y_test.sum() / len(y_test)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for r, ax in zip(pr_results, axes.flatten()):
    ax.plot(r['rec'], r['prec'], color='#373B42', linestyle='solid', linewidth=2.2)
    ax.fill_between(r['rec'], r['prec'], baseline_rate, alpha=0.02, color='#373B42')
    ax.set_title(f"{r['label_short']}\nPR-AUC = {r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}", fontsize=9.5, fontweight='bold', pad=6)
    ax.set(xlabel='Revocação', ylabel='Precisão', xlim=(-0.02, 1.02), ylim=(0.0, 1.05))

plt.tight_layout()
plt.savefig('figures_all_v3/pr_curve_matrix_v5_lr.png', dpi=300, bbox_inches='tight')
plt.close()

# --- 12. Tabela Visual Resumo ---

feat_label_map = {info[0]: info[1] for info in SEP_SCENARIO_INFO}
perf_lookup    = results_df_test.set_index('Features').to_dict('index')

SEP_NAME_TO_SHORT = {
    'C1 — Dados':                             'C1',
    "C2 — Dados com 'SMOTE'":                 'C2',
    "C3 — Dados com 'cost-sensitive'":        'C3',
    'C4 — Dados com engenharia de atributos': 'C4',
}

col_labels = ['Cenário', '# Feat.', 'PR_AUC', 'ROC_AUC', 'Precision', 'Recall', 'F1', 'Fisher J1 ↑', 'Brier Score ↓']
col_widths = [0.11, 0.05, 0.09, 0.09, 0.09, 0.09, 0.09, 0.13, 0.13, 0.13]
col_x      = [sum(col_widths[:i]) for i in range(len(col_widths))]
row_h, header_y = 0.135, 0.88
row_ys = [header_y - (i + 1) * row_h for i in range(len(sep_records))]

fig, ax = plt.subplots(figsize=(28, 4.5))
ax.axis('off')
ax.set(xlim=(0, 1), ylim=(0, 1))

def draw_cell_s4(ax, x, y, w, h, bg, text, fg='black', fontsize=9.5, bold=False):
    ax.add_patch(plt.Rectangle((x, y - h), w, h, transform=ax.transAxes, clip_on=False, facecolor=bg, edgecolor='white', linewidth=1.8, zorder=2))
    ax.text(x + w / 2., y - h / 2., text, transform=ax.transAxes, ha='center', va='center', fontsize=fontsize, fontweight='bold' if bold else 'normal', color=fg, clip_on=False, zorder=3)

for cx, cw, cl in zip(col_x, col_widths, col_labels):
    draw_cell_s4(ax, cx, header_y, cw, row_h, '#373B42', cl, fg='white', fontsize=9.0, bold=True)

for ri, (row, ry) in enumerate(zip(sep_records, row_ys)):
    row_bg     = SEP_COLORS[ri]
    short_name = SEP_NAME_TO_SHORT.get(row['Cenário'], row['Cenário'])
    fl         = feat_label_map[row['Cenário']]

    draw_cell_s4(ax, col_x[0], ry, col_widths[0], row_h, row_bg, short_name, fg='white', fontsize=9.5, bold=True)
    draw_cell_s4(ax, col_x[1], ry, col_widths[1], row_h, '#ecf0f1', str(row['# Features']), fg='#1c2833', fontsize=9.5)

    metrics_vals = [(perf_lookup[fl][k], perf_lookup[fl][f'{k}_std']) if k != 'F1_score' else (perf_lookup[fl][k], perf_lookup[fl]['F1_std']) for k in ['PR_AUC', 'ROC_AUC', 'Precision', 'Recall', 'F1_score']]
    for (val, sdv), cx, cw in zip(metrics_vals, col_x[2:7], col_widths[2:7]):
        draw_cell_s4(ax, cx, ry, cw, row_h, '#ecf0f1', f'{val:.4f}\n±{sdv:.4f}', fg='#1c2833', fontsize=8.0)

    for mk, sk, cx, cw in zip(['Fisher J1', 'Brier Score'], ['Fisher J1_std', 'Brier Score_std'], col_x[7:], col_widths[7:]):
        draw_cell_s4(ax, cx, ry, cw, row_h, '#ecf0f1', f'{row[mk]:.4f}\n±{row[sk]:.4f}', fg='#1c2833', fontsize=8.0)

plt.tight_layout()
plt.savefig('figures_all_v3/sep_S4_summary_table_holdout.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# --- 13. Análise de Threshold ---

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (r, ax) in enumerate(zip(pr_results, axes.flatten())):
    ax.plot(r['thr'], r['prec'][:-1], color='#5C7A99', linestyle='-',  linewidth=2.0, label='Precisão')
    ax.plot(r['thr'], r['rec'][:-1],  color='#C46D61', linestyle='--', linewidth=2.0, label='Revocação')
    ax.set_title(f"{r['label_short']}", fontsize=11, fontweight='bold', pad=8)
    ax.set(xlabel='Limiar', xlim=(0.0, 1.0), ylim=(0.0, 1.05))
    if idx % 2 == 0: ax.set_ylabel('Precisão, Revocação', fontsize=9.5)
    ax.legend(fontsize=8.5, framealpha=0.7, loc='best')

plt.tight_layout()
plt.savefig('figures_all_v3/precision_recall_threshold_matrix_v5_lr.png', dpi=300, bbox_inches='tight')
plt.close()

# --- 14. Análise de Significância Estatística ---

_METRICS_SIG  = {'F1_score': 'F1', 'PR_AUC': 'PR-AUC', 'ROC_AUC': 'ROC-AUC'}
_PAIRS_SIG    = [(3, 0), (3, 1), (3, 2)]
_PAIR_LABELS  = ['C4 vs C1', 'C4 vs C2', 'C4 vs C3']
_N_TESTS      = len(_PAIRS_SIG) * len(_METRICS_SIG)
_ALPHA_BONF   = 0.05 / _N_TESTS

_data_sig = {mkey: {lbl: np.array(rep_metrics[lbl][mkey]) for lbl in SCENARIO_LABELS} for mkey in _METRICS_SIG}

_results_sig, _sig_records = {}, []

def _sig_label(p, alpha_bonf):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    elif p < alpha_bonf: return '†'
    return 'ns'

for (_i, _j), _plabel in zip(_PAIRS_SIG, _PAIR_LABELS):
    _results_sig[_plabel] = {}
    for mkey, mlabel in _METRICS_SIG.items():
        va, vb = _data_sig[mkey][SCENARIO_LABELS[_i]], _data_sig[mkey][SCENARIO_LABELS[_j]]
        stat, pval = ttest_rel(va, vb) if len(va) == len(vb) else (0.0, 1.0)
        
        diff = va - vb
        diff_mean = diff.mean()
        diff_ci = 1.96 * (diff.std(ddof=1) / np.sqrt(len(diff)))
        
        _results_sig[_plabel][mkey] = {'stat': stat, 'pval': pval, 'diff_mean': diff_mean, 'diff_ci': diff_ci}
        _sig_records.append({
            'Cenário comparativo': _plabel, 'Métrica': mlabel, 't stat': round(stat, 4), 'p-valor': round(pval, 4),
            'p_bonf_corr': round(min(pval * _N_TESTS, 1.0), 4), 'Sig (α=0.05)': 'sim' if pval < 0.05 else 'não',
            'Sig (Bonferroni)': 'sim' if pval < _ALPHA_BONF else 'não', 'Diferença média': round(diff_mean, 4), 'IC 95% (±)': round(diff_ci, 4),
        })

pd.DataFrame(_sig_records).to_excel('figures_all_v3/stat_sig_results_v5_lr.xlsx', index=False)

# Tabela de p-valores
_fig14a, _ax14a = plt.subplots(figsize=(11, 3.2))
_ax14a.axis('off')
_COL_LABELS, _COL_W = ['Cenário comparativo', 'F1 (p-value)', 'PR-AUC (p-value)', 'ROC-AUC (p-value)'], [0.30, 0.233, 0.233, 0.234]
_COL_X = [sum(_COL_W[:k]) for k in range(len(_COL_W))]

def _draw_tbl_cell(ax, x, y, w, h, bg, text, fg='black', fontsize=10, bold=False):
    ax.add_patch(plt.Rectangle((x, y - h), w, h, transform=ax.transAxes, clip_on=False, facecolor=bg, edgecolor='#BDBDBD', linewidth=0.8, zorder=2))
    ax.text(x + w / 2., y - h / 2., text, transform=ax.transAxes, ha='center', va='center', fontsize=fontsize, fontweight='bold' if bold else 'normal', color=fg, clip_on=False, zorder=3)

for cx, cw, cl in zip(_COL_X, _COL_W, _COL_LABELS): _draw_tbl_cell(_ax14a, cx, 0.88, cw, 0.175, '#404040', cl, fg='white', bold=True)

for _ri, (_plabel, _) in enumerate(zip(_PAIR_LABELS, _PAIRS_SIG)):
    _ry, _rbg = 0.88 - (_ri + 1) * 0.175, ['#FFFFFF', '#F2F2F2'][_ri % 2]
    _draw_tbl_cell(_ax14a, _COL_X[0], _ry, _COL_W[0], 0.175, _rbg, _plabel)
    for _ci, _mk in enumerate(['F1_score', 'PR_AUC', 'ROC_AUC'], start=1):
        _p_txt = f"{_results_sig[_plabel][_mk]['pval']:.4f}".replace('.', ',')
        _draw_tbl_cell(_ax14a, _COL_X[_ci], _ry, _COL_W[_ci], 0.175, _rbg, _p_txt)

_fig14a.suptitle('Significância Estatística — t de Student Pareado + Bonferroni', fontsize=11, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig('figures_all_v3/stat_sig_table_v5_lr.png', dpi=300, bbox_inches='tight')
plt.close(_fig14a)

# Forest Plot
_fig14b, _axes14b = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
_y_pos = np.arange(len(_PAIR_LABELS))

for _ax, _mk in zip(_axes14b, ['F1_score', 'PR_AUC', 'ROC_AUC']):
    _color = {'F1_score': '#1a5276', 'PR_AUC': '#1e8449', 'ROC_AUC': '#784212'}[_mk]
    _means, _cis, _pvals = (list(t) for t in zip(*[(_results_sig[pl][_mk]['diff_mean'], _results_sig[pl][_mk]['diff_ci'], _results_sig[pl][_mk]['pval']) for pl in _PAIR_LABELS]))
    
    _ax.axvline(0, color='#7F8C8D', linewidth=1.2, linestyle='--', zorder=1)
    _ax.errorbar(_means, _y_pos, xerr=_cis, fmt='o', color=_color, ecolor=_color, elinewidth=2, capsize=5, capthick=2, markersize=9, markerfacecolor='white', markeredgecolor=_color, markeredgewidth=2.2, zorder=3)
    
    for _yi in _y_pos: _ax.axhspan(_yi - 0.35, _yi + 0.35, color='#EAF2FF' if _yi % 2 == 0 else 'white', alpha=0.5, zorder=0)

    _xlim_max = max(abs(m) + ci for m, ci in zip(_means, _cis)) * 1.55
    for _yi, (_p, _sl) in enumerate(zip(_pvals, [_sig_label(p, _ALPHA_BONF) for p in _pvals])):
        _ax.text(_xlim_max * 0.98, _yi, f'p = {f"{_p:.4f}".replace(".", ",")}  {_sl}', va='center', ha='right', fontsize=8.5, fontweight='bold' if _sl != 'ns' else 'normal')

    _ax.set(xlim=(-_xlim_max, _xlim_max), yticks=_y_pos, xlabel='Diferença média (C4 − Cx)')
    _ax.set_yticklabels(_PAIR_LABELS, fontsize=10, fontweight='bold')
    _ax.set_title({'F1_score': 'F1-Score', 'PR_AUC': 'PR-AUC', 'ROC_AUC': 'ROC-AUC'}[_mk], fontsize=12, fontweight='bold', pad=10, color=_color)
    _ax.spines['top'].set_visible(False)
    _ax.spines['right'].set_visible(False)

_fig14b.suptitle('Forest Plot — Diferença Média C4 vs Baseline', fontsize=12, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig('figures_all_v3/stat_sig_forest_v5_lr.png', dpi=300, bbox_inches='tight')
plt.close(_fig14b)

print("Processamento concluído. Todos os arquivos foram gerados na pasta figures_all_v3/")