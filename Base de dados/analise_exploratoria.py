"""
=============================================================================
TCC - Engenharia de Atributos Não Supervisionada em Dados Desbalanceados
Script de Análise Exploratória de Dados (EDA)
Autor: Uilson Barbosa da Silva
=============================================================================
Este script gera os 3 painéis visuais fundamentais apresentados no TCC:
  1. Painel Comparativo de Desbalanceamento de Classes (Proporções e Razão)
  2. Painel de Densidade Condicional e Assimetria (Violin Plot com Quartis)
  3. Projeção 2D de Sobreposição Espacial e Detectabilidade (PCA)

Requisitos:
  pip install pandas numpy matplotlib seaborn scikit-learn pillow
=============================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image

# ---------------------------------------------------------------------------
# 1. CONFIGURAÇÕES GERAIS E PALETA DE CORES
# ---------------------------------------------------------------------------
# Configurações estéticas globais (Padrão para publicação científica / 300 DPI)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#BDC3C7',
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'figure.dpi': 300,
})

# Paleta conceitual consistente com o TCC
COR_NORMAL = '#373B42'    # Cinza escuro/antracite: Classe Majoritária (Normal, -1)
COR_ANOMALIA = '#C46D61'  # Coral/Terracota: Classe Minoritária (Anomalia, +1)
COR_DESTAQUE = '#5C7A99'  # Azul acinzentado para destaques complementares

# ---------------------------------------------------------------------------
# 2. DIRETÓRIOS E CARREGAMENTO DOS DADOS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / 'datasets'
OUTPUT_DIR = BASE_DIR / 'figuras_analise_exploratoria'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATH_SINTETICO = DATASETS_DIR / 'dataset_sintetico.csv'
PATH_MAMOGRAFIA = DATASETS_DIR / 'mamography.csv'

print(f"[+] Carregando bases de dados de: {DATASETS_DIR}")
df_syn = pd.read_csv(PATH_SINTETICO)
df_mam = pd.read_csv(PATH_MAMOGRAFIA, sep=';')

# Definição das colunas de atributos
COLS_SYN = [
    'info_feat_1_lognorm', 'info_feat_2_beta', 'info_feat_3_fisk',
    'info_feat_4', 'noise_feat_1', 'noise_feat_2'
]
LABELS_SYN = ['Info 1\n(Lognorm)', 'Info 2\n(Beta)', 'Info 3\n(Fisk)', 'Info 4', 'Ruído 1', 'Ruído 2']

COLS_MAM = ['atributo_0', 'atributo_1', 'atributo_2', 'atributo_3', 'atributo_4', 'atributo_5']
LABELS_MAM = [f'Atributo {i}' for i in range(6)]

print(f"    - Base Sintética: {df_syn.shape[0]:,} amostras, {df_syn.shape[1]} colunas.")
print(f"    - Base Mamografia: {df_mam.shape[0]:,} amostras, {df_mam.shape[1]} colunas.")


# ===========================================================================
# GRÁFICO 1: PAINEL COMPARATIVO DE DESBALANCEAMENTO DE CLASSES
# ===========================================================================
def gerar_grafico_1_desbalanceamento():
    """
    Gera barras horizontais 100% empilhadas ilustrando o desbalanceamento severo
    nas bases sintética e real, justificando o uso de PR-AUC e F1-Score.
    """
    print("\n[1/3] Gerando Gráfico 1: Painel Comparativo de Desbalanceamento...")
    
    # Contagens e percentuais - Sintético
    n_norm_syn = (df_syn['target'] == -1).sum()
    n_anom_syn = (df_syn['target'] == 1).sum()
    tot_syn = len(df_syn)
    pct_norm_syn = (n_norm_syn / tot_syn) * 100
    pct_anom_syn = (n_anom_syn / tot_syn) * 100
    
    # Contagens e percentuais - Mamografia
    n_norm_mam = (df_mam['classe'] == -1).sum()
    n_anom_mam = (df_mam['classe'] == 1).sum()
    tot_mam = len(df_mam)
    pct_norm_mam = (n_norm_mam / tot_mam) * 100
    pct_anom_mam = (n_anom_mam / tot_mam) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # --- Subplot A: Dados Sintéticos ---
    axes[0].barh(0, pct_norm_syn, color=COR_NORMAL, height=0.6, label='Normal (Classe -1)')
    axes[0].barh(0, pct_anom_syn, left=pct_norm_syn, color=COR_ANOMALIA, height=0.6, label='Anomalia (Classe +1)')
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(-0.5, 0.7)
    axes[0].set_title(
        f"Dados Sintéticos\nTotal: {tot_syn:,} amostras (Razão ~{round(n_norm_syn/n_anom_syn)}:1)",
        fontsize=12, fontweight='bold', pad=14, color='#1B2631'
    )
    axes[0].set_xlabel('Percentual (%)', fontsize=11)
    axes[0].set_yticks([])
    axes[0].text(
        pct_norm_syn / 2, 0, f"{n_norm_syn:,} amostras\n({pct_norm_syn:.2f}%)",
        ha='center', va='center', color='white', fontweight='bold', fontsize=11
    )
    axes[0].annotate(
        f"{n_anom_syn:,} anomalias\n({pct_anom_syn:.2f}%)",
        xy=(pct_norm_syn + pct_anom_syn / 2, 0.3),
        xytext=(pct_norm_syn + pct_anom_syn / 2, 0.52),
        arrowprops=dict(arrowstyle='->', color=COR_ANOMALIA, lw=1.5),
        ha='center', va='bottom', color=COR_ANOMALIA, fontweight='bold', fontsize=10
    )

    # --- Subplot B: Dados Reais (Mamografia) ---
    axes[1].barh(0, pct_norm_mam, color=COR_NORMAL, height=0.6)
    axes[1].barh(0, pct_anom_mam, left=pct_norm_mam, color=COR_ANOMALIA, height=0.6)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(-0.5, 0.7)
    axes[1].set_title(
        f"Dados Reais (Mamografia)\nTotal: {tot_mam:,} amostras (Razão ~{round(n_norm_mam/n_anom_mam)}:1)",
        fontsize=12, fontweight='bold', pad=14, color='#1B2631'
    )
    axes[1].set_xlabel('Percentual (%)', fontsize=11)
    axes[1].set_yticks([])
    axes[1].text(
        pct_norm_mam / 2, 0, f"{n_norm_mam:,} amostras\n({pct_norm_mam:.2f}%)",
        ha='center', va='center', color='white', fontweight='bold', fontsize=11
    )
    axes[1].annotate(
        f"{n_anom_mam:,} anomalias\n({pct_anom_mam:.2f}%)",
        xy=(pct_norm_mam + pct_anom_mam / 2, 0.3),
        xytext=(pct_norm_mam + pct_anom_mam / 2, 0.52),
        arrowprops=dict(arrowstyle='->', color=COR_ANOMALIA, lw=1.5),
        ha='center', va='bottom', color=COR_ANOMALIA, fontweight='bold', fontsize=10
    )

    # Legenda compartilhada inferior
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.12),
        frameon=True, fontsize=11
    )

    plt.subplots_adjust(wspace=0.25, bottom=0.22)
    
    # Salvar em alta resolução (PNG e JPG)
    out_png = OUTPUT_DIR / 'eda_fig1_desbalanceamento.png'
    out_jpg = OUTPUT_DIR / 'eda_fig1_desbalanceamento.jpg'
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    Image.open(out_png).convert('RGB').save(out_jpg, quality=95)
    plt.close(fig)
    print(f"    -> Salvo com sucesso: {out_jpg}")


# ===========================================================================
# GRÁFICO 2: PAINEL DE DENSIDADE CONDICIONAL E ASSIMETRIA (VIOLIN PLOT)
# ===========================================================================
def gerar_grafico_2_violin_densidade():
    """
    Gera violin plots bivariados (divididos por classe) com as variáveis padronizadas (Z-Score).
    Exibe quartis internos e assimetrias severas, justificando o uso do PowerTransformer.
    """
    print("\n[2/3] Gerando Gráfico 2: Painel de Densidade Condicional e Assimetria...")
    
    # Padronização (Z-score) para viabilizar a visualização em escala comum
    scaler = StandardScaler()
    syn_z = pd.DataFrame(scaler.fit_transform(df_syn[COLS_SYN]), columns=LABELS_SYN)
    syn_z['Classe'] = df_syn['target'].values
    
    mam_z = pd.DataFrame(scaler.fit_transform(df_mam[COLS_MAM]), columns=LABELS_MAM)
    mam_z['Classe'] = df_mam['classe'].values
    
    # Formato longo (melt) para o Seaborn
    syn_melt = syn_z.melt(id_vars='Classe', var_name='Variável', value_name='Escore Z')
    mam_melt = mam_z.melt(id_vars='Classe', var_name='Variável', value_name='Escore Z')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    palette = {-1: COR_NORMAL, 1: COR_ANOMALIA}
    
    # Marcadores explícitos da legenda para garantir preenchimento exato das cores
    legend_elements = [
        mpatches.Patch(facecolor=COR_NORMAL, label='Normal (−1)'),
        mpatches.Patch(facecolor=COR_ANOMALIA, label='Anomalia (+1)')
    ]
    
    # --- Painel A: Dados Sintéticos ---
    sns.violinplot(
        data=syn_melt, x='Variável', y='Escore Z', hue='Classe',
        split=True, inner='quartile', palette=palette, ax=axes[0],
        density_norm='width', linewidth=0.8, cut=0
    )
    axes[0].set_title(
        '(A) Dados Sintéticos\n(Assimetria: Info 1 = 4,11 | Info 3 = 3,22)',
        fontsize=12, fontweight='bold', pad=14, color='#1B2631'
    )
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Escore Z (padronizado)', fontsize=11)
    axes[0].axhline(y=0, color='#BDC3C7', linestyle='--', linewidth=0.6, alpha=0.7)
    axes[0].legend(
        handles=legend_elements, title='Classe',
        loc='upper right', frameon=True, fontsize=9, title_fontsize=9
    )
    axes[0].tick_params(axis='x', rotation=0)
    
    # --- Painel B: Dados Reais (Mamografia) ---
    sns.violinplot(
        data=mam_melt, x='Variável', y='Escore Z', hue='Classe',
        split=True, inner='quartile', palette=palette, ax=axes[1],
        density_norm='width', linewidth=0.8, cut=0
    )
    axes[1].set_title(
        '(B) Dados Reais (Mamografia)\n(Assimetria: Attr 0 = 7,21 | Attr 2 = 8,89)',
        fontsize=12, fontweight='bold', pad=14, color='#1B2631'
    )
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Escore Z (padronizado)', fontsize=11)
    axes[1].axhline(y=0, color='#BDC3C7', linestyle='--', linewidth=0.6, alpha=0.7)
    axes[1].legend(
        handles=legend_elements, title='Classe',
        loc='upper right', frameon=True, fontsize=9, title_fontsize=9
    )
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.subplots_adjust(wspace=0.28)
    
    out_png = OUTPUT_DIR / 'eda_fig2_violin_densidade.png'
    out_jpg = OUTPUT_DIR / 'eda_fig2_violin_densidade.jpg'
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    Image.open(out_png).convert('RGB').save(out_jpg, quality=95)
    plt.close(fig)
    print(f"    -> Salvo com sucesso: {out_jpg}")


# ===========================================================================
# GRÁFICO 3 (OU 4): PROJEÇÃO 2D DE SOBREPOSIÇÃO ESPACIAL (PCA)
# ===========================================================================
def gerar_grafico_3_separabilidade_pca():
    """
    Gera a projeção 2D dos dados nos dois primeiros componentes principais (PCA),
    destacando a Zona Detectável (outliers) vs. Zona Difícil (sobreposição com normais).
    """
    print("\n[3/3] Gerando Gráfico 3: Projeção 2D de Sobreposição Espacial (PCA)...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # --- Painel A: PCA Sintético ---
    pca_syn = PCA(n_components=2, random_state=42)
    coords_syn = pca_syn.fit_transform(StandardScaler().fit_transform(df_syn[COLS_SYN]))
    var_syn = pca_syn.explained_variance_ratio_ * 100
    
    mask_norm_syn = (df_syn['target'] == -1)
    mask_anom_syn = (df_syn['target'] == 1)
    
    axes[0].scatter(
        coords_syn[mask_norm_syn, 0], coords_syn[mask_norm_syn, 1],
        c=COR_NORMAL, alpha=0.15, s=12, label='Normal (-1)', edgecolor='none'
    )
    axes[0].scatter(
        coords_syn[mask_anom_syn, 0], coords_syn[mask_anom_syn, 1],
        c=COR_ANOMALIA, alpha=0.85, s=28, label='Anomalia (+1)', edgecolor='white', linewidth=0.5
    )
    axes[0].set_title(
        f'(A) Dados Sintéticos — Projeção PCA\n(PC1: {var_syn[0]:.1f}% | PC2: {var_syn[1]:.1f}% Variância Explicada)',
        fontsize=12, fontweight='bold', pad=14
    )
    axes[0].set_xlabel('Componente Principal 1 (PC1)', fontsize=11)
    axes[0].set_ylabel('Componente Principal 2 (PC2)', fontsize=11)
    axes[0].legend(loc='lower left', frameon=True)
    axes[0].grid(True, linestyle=':', alpha=0.4)
    
    axes[0].annotate(
        'Zona Detectável\n(Anomalias isoladas)', xy=(4.0, 3.2), xytext=(5.5, 4.2),
        arrowprops=dict(arrowstyle='->', color='#922B21', lw=1.5),
        fontsize=10, fontweight='bold', color='#922B21',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='none', alpha=0.9)
    )
    axes[0].annotate(
        'Zona Difícil\n(Sobreposição com normais)', xy=(0.0, -0.2), xytext=(-5.2, -4.0),
        arrowprops=dict(arrowstyle='->', color='#1B4F72', lw=1.5),
        fontsize=10, fontweight='bold', color='#1B4F72',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#D4E6F1', edgecolor='none', alpha=0.9)
    )
    
    # --- Painel B: PCA Mamografia ---
    pca_mam = PCA(n_components=2, random_state=42)
    coords_mam = pca_mam.fit_transform(StandardScaler().fit_transform(df_mam[COLS_MAM]))
    var_mam = pca_mam.explained_variance_ratio_ * 100
    
    mask_norm_mam = (df_mam['classe'] == -1)
    mask_anom_mam = (df_mam['classe'] == 1)
    
    axes[1].scatter(
        coords_mam[mask_norm_mam, 0], coords_mam[mask_norm_mam, 1],
        c=COR_NORMAL, alpha=0.15, s=12, label='Normal (-1)', edgecolor='none'
    )
    axes[1].scatter(
        coords_mam[mask_anom_mam, 0], coords_mam[mask_anom_mam, 1],
        c=COR_ANOMALIA, alpha=0.85, s=32, label='Anomalia (+1)', edgecolor='white', linewidth=0.5
    )
    axes[1].set_title(
        f'(B) Dados Reais (Mamografia) — Projeção PCA\n(PC1: {var_mam[0]:.1f}% | PC2: {var_mam[1]:.1f}% Variância Explicada)',
        fontsize=12, fontweight='bold', pad=14
    )
    axes[1].set_xlabel('Componente Principal 1 (PC1)', fontsize=11)
    axes[1].set_ylabel('Componente Principal 2 (PC2)', fontsize=11)
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].grid(True, linestyle=':', alpha=0.4)
    
    axes[1].annotate(
        'Alta Sobreposição\n(Região Central Indiscernível)', xy=(-0.5, 0.0), xytext=(-2.0, 8.5),
        arrowprops=dict(arrowstyle='->', color='#1B4F72', lw=1.5),
        fontsize=10, fontweight='bold', color='#1B4F72',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#D4E6F1', edgecolor='none', alpha=0.9)
    )
    axes[1].annotate(
        'Casos Periféricos\n(Outliers Radiológicos)', xy=(7.0, -1.0), xytext=(4.5, -6.0),
        arrowprops=dict(arrowstyle='->', color='#922B21', lw=1.5),
        fontsize=10, fontweight='bold', color='#922B21',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='none', alpha=0.9)
    )
    
    plt.subplots_adjust(wspace=0.30)
    
    out_png = OUTPUT_DIR / 'eda_fig4_separabilidade_pca.png'
    out_jpg = OUTPUT_DIR / 'eda_fig4_separabilidade_pca.jpg'
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    Image.open(out_png).convert('RGB').save(out_jpg, quality=95)
    plt.close(fig)
    print(f"    -> Salvo com sucesso: {out_jpg}")


# ===========================================================================
# EXECUÇÃO PRINCIPAL
# ===========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("INICIANDO GERAÇÃO DOS GRÁFICOS DE ANÁLISE EXPLORATÓRIA (TCC)")
    print("=" * 70)
    
    gerar_grafico_1_desbalanceamento()
    gerar_grafico_2_violin_densidade()
    gerar_grafico_3_separabilidade_pca()
    
    print("\n" + "=" * 70)
    print("TODOS OS GRÁFICOS FORAM GERADOS COM SUCESSO EM 300 DPI!")
    print(f"Diretório de saída: {OUTPUT_DIR}")
    print("=" * 70)
