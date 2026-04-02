# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class StateOfTheArtImbalancedGenerator:
    """
    Gerador de dataset sintético v3 — Dois Regimes de Detectabilidade.

    Motivação
    ---------
    Em dados reais de eventos raros, a classe minoritária NÃO é homogênea:
    uma fração possui "assinatura" estatística suficientemente distinta para
    que modelos lineares atribuam probabilidades acima do limiar basal imposto
    pela prior. Outra fração é quase indistinguível da maioria — zonas de 
    alta incerteza irredutível.

    Parâmetros
    ----------
    n_samples          : total de amostras
    minority_ratio     : proporcao da classe minoritaria (~1.4%)
    random_state       : semente para reprodutibilidade
    overlap_strength   : sobreposicao nas zonas DIFÍCEIS (0-1)
    detectable_fraction: fracao da minoria com sinal detectavel (0-1)
                         -> essas amostras recebem separacao moderada e SEM
                            perturbacao XOR, permitindo P > limiar basal.
                         -> o restante vai para a zona dura: alto overlap + XOR.
    """

    def __init__(self,
                 n_samples=20000,
                 minority_ratio=0.014,
                 random_state=42,
                 overlap_strength=0.70,
                 detectable_fraction=0.40):

        self.n_samples = n_samples
        self.minority_ratio = minority_ratio
        self.random_state = random_state
        self.overlap_strength = overlap_strength
        self.detectable_fraction = detectable_fraction

        self.n_minority = int(n_samples * minority_ratio)
        self.n_majority = n_samples - self.n_minority

        self.n_detectable = int(self.n_minority * detectable_fraction)
        self.n_hard = self.n_minority - self.n_detectable

        np.random.seed(random_state)

        # Zona DETECTAVEL: separacao ~2.5 sigma — modelo consegue P > limiar basal
        self._center_detectable = [2.5, -1.8, 2.0, 2.2]

        # Zona DIFÍCIL: separacao baixa ~1.4 sigma + XOR — modelo nunca passa P ~ 0.2
        self._centers_hard = [
            [-1.4,  1.6, -0.9, -1.1],  # Cluster B
            [ 0.6,  2.1,  1.5, -0.5],  # Cluster C
        ]

    # -------------------------------------------------------------------------
    # TRANSFORMACOES ESTATÍSTICAS
    # -------------------------------------------------------------------------

    def _apply_gaussian_copula(self, X):
        """
        Copula Gaussiana — transforma 3 features preservando correlacao:
          Feature 1 (idx 0) -> Log-Normal(s=0.9, scale=e^2)
          Feature 2 (idx 1) -> Beta(alpha=1.5, beta=4.0)
          Feature 3 (idx 2) -> Fisk / Log-Logística (c=3.0)
                               Substitui a Pareto para garantir variância finita,
                               mantendo a cauda pesada característica de distribuições
                               de riqueza/valores extremos sem distorcer cálculos de
                               distância em algoritmos não supervisionados.
        """
        out = X.copy()

        # Feature 1 -> Log-Normal
        z0 = (out[:, 0] - np.mean(out[:, 0])) / (np.std(out[:, 0]) + 1e-9)
        u0 = np.clip(stats.norm.cdf(z0), 1e-6, 1 - 1e-6)
        out[:, 0] = stats.lognorm.ppf(u0, s=0.9, scale=np.exp(2.0))

        # Feature 2 -> Beta
        z1 = (out[:, 1] - np.mean(out[:, 1])) / (np.std(out[:, 1]) + 1e-9)
        u1 = np.clip(stats.norm.cdf(z1), 1e-6, 1 - 1e-6)
        out[:, 1] = stats.beta.ppf(u1, a=1.5, b=4.0)

        # Feature 3 -> Fisk / Log-Logística
        # c=3.0 garante cauda pesada com variância finita e bem comportada.
        z2 = (out[:, 2] - np.mean(out[:, 2])) / (np.std(out[:, 2]) + 1e-9)
        u2 = np.clip(stats.norm.cdf(z2), 1e-6, 1 - 1e-6)
        out[:, 2] = stats.fisk.ppf(u2, c=3.0)

        return out

    def _add_xor_perturbation(self, X, y, mask):
        """
        Perturbacao XOR aplicada SOMENTE nas amostras da mascara (zona difícil).
        Cria interacao nao-monotonica f3 x f4 que modelos lineares nao capturam.
        """
        X_out = X.copy()

        pos_idx = np.where(mask & (y == 1))[0]
        neg_idx = np.where(mask & (y == -1))[0]

        if len(pos_idx) > 1:
            flip = np.random.choice(pos_idx, size=len(pos_idx) // 2, replace=False)
            X_out[flip, 2] = -X_out[flip, 2]

        if len(neg_idx) > 1:
            flip = np.random.choice(neg_idx, size=len(neg_idx) // 3, replace=False)
            X_out[flip, 3] = -X_out[flip, 3]

        # Ruido heteroscedastico: maior na zona difícil
        noise_scale = np.where(mask, 0.45, 0.15)
        X_out += np.random.randn(*X_out.shape) * noise_scale[:, None]

        return X_out

    def _generate_noise_features(self, n):
        n1 = np.random.normal(0, 2.5, (n, 1))
        n2 = np.random.uniform(-4, 4, (n, 1))
        return np.hstack((n1, n2))

    # -------------------------------------------------------------------------
    # GERACAO DAS CLASSES
    # -------------------------------------------------------------------------

    def _generate_majority(self):
        cov = np.array([
            [ 1.8,  0.9,  0.4, -0.3],
            [ 0.9,  1.4,  0.6,  0.2],
            [ 0.4,  0.6,  1.3,  0.3],
            [-0.3,  0.2,  0.3,  1.0]
        ])
        return np.random.multivariate_normal([0, 0, 0, 0], cov, self.n_majority)

    def _generate_detectable_minority(self):
        """
        Zona DETECTAVEL: separacao moderada, covariancia apertada, sem XOR.
        """
        cov = np.array([
            [ 0.5, -0.3,  0.1,  0.1],
            [-0.3,  0.4, -0.2,  0.0],
            [ 0.1, -0.2,  0.4,  0.1],
            [ 0.1,  0.0,  0.1,  0.3]
        ])
        return np.random.multivariate_normal(
            self._center_detectable, cov, self.n_detectable
        )

    def _generate_hard_minority(self):
        """
        Zona DIFÍCIL: separacao baixa, covariancia larga, recebera XOR.
        """
        cov = np.array([
            [ 0.8, -0.4,  0.2,  0.1],
            [-0.4,  0.7, -0.3,  0.1],
            [ 0.2, -0.3,  0.7,  0.2],
            [ 0.1,  0.1,  0.2,  0.6]
        ])
        n_B = int(self.n_hard * 0.60)
        n_C = self.n_hard - n_B
        B = np.random.multivariate_normal(self._centers_hard[0], cov, n_B)
        C = np.random.multivariate_normal(self._centers_hard[1], cov, n_C)
        return np.vstack((B, C))

    # -------------------------------------------------------------------------
    # ANOMALIAS ESTRUTURAIS
    # -------------------------------------------------------------------------

    def _inject_contamination(self):
        """Majoritarias dentro da zona detectavel."""
        n = int(self.n_detectable * 0.12 * self.overlap_strength)
        cov = np.eye(4) * (0.4 + 0.3 * self.overlap_strength)
        feat = np.random.multivariate_normal(self._center_detectable, cov, n)
        return feat, np.full(n, -1)

    def _inject_boundary_outliers(self):
        """Amostras mistas na fronteira."""
        n = int(self.n_minority * 0.18)
        center = [c * 0.6 for c in self._center_detectable]
        feat = np.random.multivariate_normal(center, np.eye(4) * 0.6, n * 2)
        labs = np.concatenate([np.full(n, -1), np.full(n, 1)])
        return feat, labs

    def _inject_hard_zone_invasion(self):
        """Maioria invade Cluster C."""
        n = int(self.n_hard * 0.15)
        feat = np.random.multivariate_normal(
            self._centers_hard[1], np.eye(4) * 0.55, n
        )
        return feat, np.full(n, -1)

    def _inject_isolated_outliers(self):
        """Outliers excêntricos mistos."""
        n_min = int(self.n_minority * 0.06)
        n_maj = int(n_min * 0.4)
        center = [-3.5, -3.5, -3.5, -3.5]
        f_min = np.random.multivariate_normal(center, np.eye(4) * 0.35, n_min)
        f_maj = np.random.multivariate_normal(center, np.eye(4) * 0.60, n_maj)
        feat = np.vstack((f_min, f_maj))
        labs = np.concatenate([np.full(n_min, 1), np.full(n_maj, -1)])
        return feat, labs

    # -------------------------------------------------------------------------
    # PIPELINE PRINCIPAL
    # -------------------------------------------------------------------------

    def generate_complete_dataset(self):
        print("=" * 68)
        print("  GERADOR v3 — DOIS REGIMES DE DETECTABILIDADE")
        print(f"  detectable_fraction = {self.detectable_fraction:.0%}  |  "
              f"overlap_strength = {self.overlap_strength}")
        print("=" * 68)

        # 1. Classes base
        maj      = self._generate_majority()
        min_det  = self._generate_detectable_minority()
        min_hard = self._generate_hard_minority()

        # 2. Anomalias
        cont_f, cont_l   = self._inject_contamination()
        bound_f, bound_l = self._inject_boundary_outliers()
        inv_f, inv_l     = self._inject_hard_zone_invasion()
        iso_f, iso_l     = self._inject_isolated_outliers()

        # 3. Empilhamento
        X_info = np.vstack((maj, min_det, min_hard,
                            cont_f, bound_f, inv_f, iso_f))
        y = np.concatenate((
            np.full(len(maj),      -1),
            np.full(len(min_det),   1),
            np.full(len(min_hard),  1),
            cont_l, bound_l, inv_l, iso_l
        ))

        # 4. Mascara da zona DIFÍCIL (recebe XOR)
        n_safe = len(maj) + len(min_det)
        hard_mask = np.zeros(len(y), dtype=bool)
        hard_mask[n_safe:] = True

        # 5. Copula Gaussiana (toda a base)
        X_cop = self._apply_gaussian_copula(X_info)

        # 6. XOR somente na zona difícil
        X_perturbed = self._add_xor_perturbation(X_cop, y, hard_mask)

        # 7. Features de ruído
        noise = self._generate_noise_features(len(X_perturbed))
        X = np.hstack((X_perturbed, noise))

        # 8. Embaralhamento
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == -1)
        print(f"  Total             : {len(X):,}")
        print(f"  Classe -1         : {n_neg:,}  ({n_neg/len(y)*100:.2f}%)")
        print(f"  Classe +1         : {n_pos:,}  ({n_pos/len(y)*100:.2f}%)")
        print(f"  Minoria detectavel: ~{self.n_detectable} amostras")
        print(f"  Minoria difícil   : ~{self.n_hard} amostras")
        print("=" * 68)

        # NOME DA COLUNA ATUALIZADO PARA FISK
        feat_names = ['info_feat_1_lognorm', 'info_feat_2_beta',
                      'info_feat_3_fisk', 'info_feat_4',
                      'noise_feat_1', 'noise_feat_2']
        
        df = pd.DataFrame(X, columns=feat_names)
        df['target'] = y
        return df, X, y

    # -------------------------------------------------------------------------
    # VISUALIZACAO
    # -------------------------------------------------------------------------

    def visualize_dataset(self, df):
        import seaborn as sns

        COLOR_MAJ = '#6A8FB5'   # azul-acinzentado (Classe -1)
        COLOR_MIN = '#C97B6B'   # terracota (Classe +1)
        palette   = {-1: COLOR_MAJ, 1: COLOR_MIN}

        maj  = df[df['target'] == -1]
        min_ = df[df['target'] ==  1]

        # ------------------------------------------------------------------
        # FIGURA 1 — Visualização original (histogramas + dispersão)
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        # fig.suptitle(
        #     f"Dataset v3  |  detectable={self.detectable_fraction:.0%}  "
        #     f"overlap={self.overlap_strength}",
        #     fontsize=13, fontweight='bold'
        # )

        axes[0].hist(maj['info_feat_1_lognorm'],  bins=60, alpha=0.5,
                      color=COLOR_MAJ, label='Majoritaria (-1)', density=True)
        axes[0].hist(min_['info_feat_1_lognorm'], bins=60, alpha=0.6,
                      color=COLOR_MIN, label='Minoritaria (+1)', density=True)
        axes[0].set_title('Feature 1: Log-Normal')
        axes[0].legend(fontsize=8)

        axes[1].hist(maj['info_feat_2_beta'],  bins=50, alpha=0.5,
                      color=COLOR_MAJ, density=True)
        axes[1].hist(min_['info_feat_2_beta'], bins=50, alpha=0.6,
                      color=COLOR_MIN, density=True)
        axes[1].set_title('Feature 2: Beta [0,1]')

        axes[2].scatter(maj['info_feat_1_lognorm'],  maj['info_feat_2_beta'],
                        c=COLOR_MAJ, alpha=0.12, s=7,  label='Majoritaria')
        axes[2].scatter(min_['info_feat_1_lognorm'], min_['info_feat_2_beta'],
                        c=COLOR_MIN, alpha=0.55, s=16, label='Minoritaria')
        axes[2].set_title('Dispersao: F1 x F2')
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig('dataset_v3_visualizacao.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        # ------------------------------------------------------------------
        # FIGURA 2 — Distribuição de classes (barplot)
        # ------------------------------------------------------------------
        n_neg = len(maj)
        n_pos = len(min_)
        total = n_neg + n_pos

        fig2, ax2 = plt.subplots(figsize=(7, 6))
        bars = ax2.bar(
            ['Classe -1', 'Classe 1'],
            [n_neg, n_pos],
            color=[COLOR_MAJ, COLOR_MIN],
            width=0.5,
            edgecolor='white'
        )
        # Rótulos sobre as barras
        for bar, count in zip(bars, [n_neg, n_pos]):
            pct = count / total * 100
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f'{count:,}\n({pct:.2f}%)',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )

        ax2.set_xlabel('classe', fontsize=12)
        ax2.set_ylabel('Frequência', fontsize=12)
        ax2.set_title(
            f'Distribuição de Classes — Dataset v3\n'
            f'detectable={self.detectable_fraction:.0%}  |  '
            f'overlap={self.overlap_strength}',
            fontsize=12, fontweight='bold'
        )
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.set_ylim(0, n_neg * 1.12)
        plt.tight_layout()
        plt.savefig('classe_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        # ------------------------------------------------------------------
        # FIGURA 3 — Scatter matrix / pairplot
        # ------------------------------------------------------------------
        COLOR_MAJ_SC = '#3333CC'   # azul vivo (Classe -1) — como na referência
        COLOR_MIN_SC = '#EE2222'   # vermelho vivo (Classe +1) — como na referência
        COLOR_KDE    = '#2AABB0'   # teal/ciano da diagonal — como na referência

        feat_cols = [c for c in df.columns if c != 'target']
        n_feats   = len(feat_cols)

        df_maj = df[df['target'] == -1][feat_cols]
        df_min = df[df['target'] ==  1][feat_cols]

        # Amostra para não sobrecarregar (mantém proporção)
        n_maj_plot = min(3000, len(df_maj))
        n_min_plot = min(len(df_min), int(n_maj_plot * len(df_min) / len(df_maj)) + 1)
        df_maj_s = df_maj.sample(n=n_maj_plot, random_state=42)
        df_min_s = df_min.sample(n=n_min_plot, random_state=42)

        fig3, axes = plt.subplots(n_feats, n_feats,
                                  figsize=(2.2 * n_feats, 2.2 * n_feats))

        for i in range(n_feats):
            for j in range(n_feats):
                ax = axes[i, j]
                if i == j:
                    # Diagonal: KDE marginal única (todos os dados), cor teal
                    all_vals = df[feat_cols[i]]
                    sns.kdeplot(x=all_vals, ax=ax,
                                color=COLOR_KDE, linewidth=1.8, fill=False)
                    ax.set_yticks([])
                else:
                    # Off-diagonal: majoritária bem transparente, minoritária opaca
                    ax.scatter(df_maj_s.iloc[:, j], df_maj_s.iloc[:, i],
                               c=COLOR_MAJ_SC, alpha=0.08, s=5,  linewidths=0)
                    ax.scatter(df_min_s.iloc[:, j], df_min_s.iloc[:, i],
                               c=COLOR_MIN_SC, alpha=0.55, s=12, linewidths=0)

                # Eixos: apenas índice numérico (como na referência)
                if j == 0:
                    ax.set_ylabel(str(i), fontsize=9, rotation=0, labelpad=10)
                else:
                    ax.set_ylabel('')
                ax.set_xlabel('')
                ax.tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.6)
                    spine.set_color('black')

        fig3.suptitle(
            f'Scatter Matrix — Dataset v3  |  '
            f'detectable={self.detectable_fraction:.0%}  '
            f'overlap={self.overlap_strength}',
            y=1.01, fontsize=12, fontweight='bold'
        )
        fig3.tight_layout(h_pad=0.4, w_pad=0.4)
        fig3.savefig('scatter_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig3)


# =============================================================================
# EXECUCAO
# =============================================================================
if __name__ == "__main__":
    gen = StateOfTheArtImbalancedGenerator(
        n_samples=18000,
        minority_ratio=0.02,
        overlap_strength=0.70,
        detectable_fraction=0.88,
        random_state=42
    )

    df, X, y = gen.generate_complete_dataset()
    gen.visualize_dataset(df)

    out = "dataset_sintetico.csv"
    df.to_csv(out, index=False)
    print(f"Salvo como: {out}")