# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib.lines import Line2D
import matplotlib as mpl

# ==============================================================================
# CONFIGURAÇÃO DE FONTES
# ==============================================================================
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10

# DEFINIÇÃO DA PALETA EXTRAÍDA DA IMAGEM
# Cor 1 (Majoritária): Verde Água / Ciano
# Cor 2 (Minoritária): Rosa / Magenta
cor_majoritaria = '#5C7A99' 
cor_minoritaria = '#C46D61'

# 1. Carregando os dados
df = pd.read_csv('mamography.csv', sep=';')

# 2. Ordenação para sobreposição (Minoritária por cima)
df_sorted = df.sort_values(by='classe', ascending=True)
features = df_sorted.drop(columns=['classe'])
classe = df_sorted['classe']
colors = classe.map({-1: cor_majoritaria, 1: cor_minoritaria})

# ==============================================================================
# GRÁFICO 1: SCATTER PLOT MATRIX
# ==============================================================================
axes = scatter_matrix(
    features, 
    diagonal='kde', 
    color=colors, 
    alpha=0.8,
    s=15, 
    figsize=(12, 12),
    grid=False,
    density_kwds={'color': '#333333', 'linewidth': 1.5} 
)

for ax_row in axes:
    for ax in ax_row:
        # Fontes e Ticks
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)
        ax.tick_params(axis='both', labelsize=9, width=1.5, color='black')
        
        # Eixos Sólidos Pretos (1.5 pt) - Sem Spines Top/Right
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_linewidth(1.5)
        # ax.spines['bottom'].set_color('black')
        # ax.spines['left'].set_linewidth(1.5)
        # ax.spines['left'].set_color('black')
        
        # Fundo Transparente/Sem preenchimento
        ax.set_facecolor('none')

# Legenda customizada com as cores da paleta (Sem borda)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Classe -1', 
           markerfacecolor=cor_majoritaria, markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Classe 1', 
           markerfacecolor=cor_minoritaria, markersize=8)
]
fig1 = plt.gcf()
fig1.legend(handles=legend_elements, loc='upper center', fontsize=10, 
            bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('scatter_matrix_paleta.png', dpi=300, bbox_inches='tight')

# ==============================================================================
# GRÁFICO 2: DISTRIBUIÇÃO DO classe
# ==============================================================================
counts = df['classe'].value_counts()
percentages = (counts / len(df)) * 100

plt.figure(figsize=(6, 5))
# Plotando com as cores da paleta
ax2 = counts.plot(kind='bar', color=[cor_majoritaria, cor_minoritaria], 
                  edgecolor='black', linewidth=1.2,alpha=0.8)

# Rótulos (Absoluto + Porcentagem)
for p, pct in zip(ax2.patches, percentages):
    ax2.annotate(f'{int(p.get_height()):,}\n({pct:.2f}%)', 
                 (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', 
                 fontsize=10, fontweight='bold')

# Formatação Rigorosa dos Eixos
ax2.grid(False)
ax2.set_facecolor('none')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
# ax2.spines['bottom'].set_linewidth(1.5)
# ax2.spines['bottom'].set_color('black')
# ax2.spines['left'].set_linewidth(1.5)
# ax2.spines['left'].set_color('black')

plt.xticks(ticks=[0, 1], labels=['Classe -1', 'Classe 1'], rotation=0, fontsize=10)
plt.ylabel('Frequência', fontsize=11)
# plt.xlabel('Classe (classe)', fontsize=11)
plt.ylim(0, counts.max() * 1.2)

plt.tight_layout()
plt.savefig('classe_distribution_paleta.png', dpi=300, bbox_inches='tight')