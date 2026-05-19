import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare
import networkx as nx
import math
import operator
import warnings
from multiprocessing import Pool, cpu_count

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class StandardSignificance:
    @staticmethod
    def compute_wilcoxon_holm(df_perf, alpha=0.05):
        classifiers = sorted(df_perf['subject'].unique())
        pivot = df_perf.pivot(index='observation',
                              columns='subject',
                              values='value')

        data_for_friedman = [
            df_perf[df_perf['subject'] == c]['value'].values
            for c in classifiers
        ]

        _, p_friedman = friedmanchisquare(*data_for_friedman)

        if p_friedman >= alpha:
            return None, None, p_friedman

        p_values = []
        m = len(classifiers)

        for i in range(m - 1):
            for j in range(i + 1, m):
                c1, c2 = classifiers[i], classifiers[j]
                _, p_val = wilcoxon(pivot[c1], pivot[c2], zero_method='pratt')
                p_values.append([c1, c2, p_val, False])

        p_values.sort(key=operator.itemgetter(2))

        for i in range(len(p_values)):
            adj_alpha = alpha / (len(p_values) - i)
            if p_values[i][2] <= adj_alpha:
                p_values[i][3] = True
            else:
                break

        avg_ranks = pivot.rank(ascending=False, axis=1).mean().sort_values(ascending=False)

        return p_values, avg_ranks, p_friedman


def _permutation_worker(args):
    data_matrix, seed = args
    rng = np.random.default_rng(seed)
    shuffled = np.array([rng.permutation(row) for row in data_matrix])
    n, k = shuffled.shape
    ranks = np.apply_along_axis(
        lambda r: (-r).argsort().argsort() + 1.0, axis=1, arr=shuffled
    ).astype(float)
    col_rank_sums = ranks.sum(axis=0)
    stat = (12.0 / (n * k * (k + 1))) * np.sum(col_rank_sums ** 2) - 3 * n * (k + 1)
    return stat


def permutation_global_test(df_perf, n_permutations=10_000, alpha=0.05, n_jobs=None):
    if n_jobs is None:
        n_jobs = cpu_count()

    classifiers = sorted(df_perf['subject'].unique())
    pivot = df_perf.pivot(index='observation',
                          columns='subject',
                          values='value')
    data_matrix = pivot[classifiers].values

    n, k = data_matrix.shape

    ranks_obs = np.apply_along_axis(
        lambda r: (-r).argsort().argsort() + 1.0, axis=1, arr=data_matrix
    ).astype(float)
    col_rank_sums_obs = ranks_obs.sum(axis=0)
    stat_obs = (12.0 / (n * k * (k + 1))) * np.sum(col_rank_sums_obs ** 2) - 3 * n * (k + 1)

    seeds = np.random.SeedSequence().generate_state(n_permutations)
    args = [(data_matrix, int(s)) for s in seeds]

    with Pool(processes=n_jobs) as pool:
        perm_stats = pool.map(_permutation_worker, args)

    perm_stats = np.array(perm_stats)
    p_perm = float(np.mean(perm_stats >= stat_obs))

    return p_perm, stat_obs, p_perm < alpha


def global_significance(df_perf, alpha=0.05):
    classifiers = sorted(df_perf['subject'].unique())
    data_for_friedman = [
        df_perf[df_perf['subject'] == c]['value'].values
        for c in classifiers
    ]
    _, p_friedman = friedmanchisquare(*data_for_friedman)
    return p_friedman < alpha, p_friedman


class MARSEngine:

    @staticmethod
    def compute_weighted_rank_matrix(data_matrix):
        n_ds, n_clfs = data_matrix.shape
        weighted_ranks = np.zeros_like(data_matrix, dtype=float)

        for i in range(n_ds):
            row = data_matrix[i, :]
            mx, mn = np.max(row), np.min(row)

            temp = (-row).argsort()
            ranks = np.empty_like(temp, dtype=float)
            ranks[temp] = np.arange(n_clfs) + 1.0

            if mx == mn:
                weighted_ranks[i, :] = ranks
                continue

            not_min_mask = row > mn
            weights = np.zeros(n_clfs, dtype=float)

            weights[not_min_mask] = (mx - mn) / (row[not_min_mask] - mn)

            w_vals = np.sort(weights[not_min_mask])
            if len(w_vals) > 1:
                delta = np.max(np.diff(w_vals))
            elif len(w_vals) > 0:
                delta = w_vals[0]
            else:
                delta = 1.0

            weights[~not_min_mask] = (w_vals[-1] + delta) if len(w_vals) > 0 else 1.0

            weighted_ranks[i, :] = ranks * weights

        return weighted_ranks

    @staticmethod
    def compute_mars_with_standard_sig(df_perf, alpha=0.05):
        p_vals, _, p_friedman = StandardSignificance.compute_wilcoxon_holm(df_perf, alpha)

        if p_vals is None:
            return None, None, p_friedman

        pivot = df_perf.pivot(index='observation',
                              columns='subject',
                              values='value')

        X = pivot.values
        wr_mat = MARSEngine.compute_weighted_rank_matrix(X)

        mars_ranks = pd.Series(
            wr_mat.mean(axis=0),
            index=pivot.columns
        ).sort_values(ascending=False)

        return p_vals, mars_ranks, p_friedman


def compute_cd_mars(wr_mat, n_ds):
    k = wr_mat.shape[1]
    theo_std = math.sqrt((k**2 - 1) / 12)
    obs_std = wr_mat.std()
    cd_base = 2.8 * math.sqrt(k * (k + 1) / (6 * n_ds))
    cd_mars = cd_base * (obs_std / theo_std)
    return cd_mars


class UnifiedPlotter:

    @staticmethod
    def graph_ranks(avranks, names, p_values=None, cd=None,
                    title="", name="", color_mode='color'):

        plt.style.use('default')

        nnames = list(names)
        sums = np.array(avranks)
        k = len(sums)

        if color_mode == 'bw':
            colors = ['black'] * k
        else:
            import matplotlib
            colormap = matplotlib.colormaps['Dark2']
            colors = [colormap(i / k) for i in range(k)]

        lowv = min(1, int(math.floor(min(sums))))
        highv = max(k, int(math.ceil(max(sums))))

        title_gap   = 0.07
        cline       = 0.32
        step_height = 0.16
        bottom_pad  = 0.12
        mid = math.ceil(k / 2)

        g = nx.Graph()
        g.add_nodes_from(nnames)

        if cd is not None:
            for i in range(k):
                for j in range(i + 1, k):
                    if abs(sums[i] - sums[j]) < cd:
                        g.add_edge(nnames[i], nnames[j])
        elif p_values is not None:
            for p in p_values:
                if not p[3]:
                    g.add_edge(p[0], p[1])

        cliques = sorted(
            list(nx.find_cliques(g)),
            key=lambda x: np.min([sums[nnames.index(n)] for n in x])
        )
        active_cliques = [c for c in cliques if len(c) > 1]
        clique_count   = len(active_cliques)

        if title == "Standard":
            display_title = f"{name} - Standard" if name else "Standard"
        else:
            display_title = f"{name} - MARS" if name else "MARS"

        curr_clq_y = cline + 0.08
        for _ in active_cliques:
            curr_clq_y += 0.09

        sorted_idx    = np.argsort(sums)
        label_elbow_y = curr_clq_y + 0.02

        label_ys = []
        for i in range(k):
            if i < mid:
                ly = label_elbow_y + i * step_height
            else:
                ly = label_elbow_y + (k - i - 1) * step_height
            label_ys.append(ly)

        max_label_y = max(label_ys)
        true_max_y  = max_label_y + bottom_pad

        width      = 12
        fig_height = true_max_y * 3.5

        fig = plt.figure(figsize=(width, fig_height))
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(true_max_y, 0)

        textspace  = 3.2
        scalewidth = width - 2 * textspace

        def xpos(rank):
            rel = (rank - lowv) / (highv - lowv) if highv > lowv else 0
            return (textspace + scalewidth * (1 - rel)) / width

        ax.text(0.5, title_gap, display_title,
                ha='center', va='top',
                fontsize=24, fontweight='bold')

        ax.plot([textspace / width, (width - textspace) / width],
                [cline, cline], color='black', lw=3)

        midv = (lowv + highv) / 2.0
        for val, label in [(lowv, str(lowv)),
                           (midv, f'{midv:.1f}' if midv != int(midv) else str(int(midv))),
                           (highv, str(highv))]:
            p = xpos(val)
            ax.plot([p, p], [cline - 0.03, cline], color='black', lw=2)
            ax.text(p, cline - 0.04, label,
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

        curr_clq_y = cline + 0.08
        for clq in active_cliques:
            r_vals = [sums[nnames.index(n)] for n in clq]
            p1, p2 = xpos(min(r_vals)), xpos(max(r_vals))
            ax.plot([p1, p2], [curr_clq_y, curr_clq_y],
                    color='black', lw=9, solid_capstyle='butt', zorder=5)
            curr_clq_y += 0.09

        for i, idx in enumerate(sorted_idx):
            rank, name_i = sums[idx], nnames[idx]
            m_col      = colors[i % len(colors)]
            side_right = i < mid

            px = xpos(rank)
            lx = (width - textspace + 0.5) / width if side_right \
                 else (textspace - 0.5) / width
            ly = label_ys[i]

            ax.plot([px, px, lx], [cline, ly, ly],
                    color=m_col, lw=3.5, solid_capstyle='round', zorder=3)

            ha = 'left' if side_right else 'right'
            
            ax.text(lx, ly - 0.010,
                    name_i,
                    color=m_col,
                    ha=ha, va='bottom',
                    fontsize=18, fontweight='bold')
            
            rank_label = f"Rank Score: {rank:.2f}" if title != "Standard" else f"Rank: {rank:.2f}"
            ax.text(lx, ly + 0.015,
                    rank_label,
                    color=m_col,
                    ha=ha, va='top',
                    fontsize=11, fontweight='normal', alpha=0.85)

        return fig  

def write_csv_report(path, mode, ranks, p_values, p_friedman,
                     p_perm=None, stat_obs=None, n_permutations=None, alpha=0.05):
    rows = []

    rows.append(["## GLOBAL SIGNIFICANCE", "", "", ""])
    rows.append(["Test", "Statistic / Info", "p-value", "Significant"])
    rows.append([
        "Friedman (chi-square)", "—",
        f"{p_friedman:.6f}",
        "Yes" if p_friedman < alpha else "No"
    ])

    if p_perm is not None:
        rows.append([
            f"Permutation ({n_permutations:,} perms, all threads)",
            f"Friedman stat = {stat_obs:.4f}",
            f"{p_perm:.6f}",
            "Yes" if p_perm < alpha else "No"
        ])

    rows.append(["", "", "", ""])

    rank_label = "Avg Rank" if mode == "standard" else "MARS Weighted Rank"
    rows.append([f"## CLASSIFIER RANKINGS ({mode.upper()})", "", "", ""])
    rows.append(["Rank", "Classifier", rank_label, ""])

    if ranks is not None:
        for position, (clf, r) in enumerate(ranks.items(), start=1):
            rows.append([str(position), clf, f"{r:.4f}", ""])
    else:
        rows.append(["—", "No significant result", "—", ""])

    rows.append(["", "", "", ""])

    rows.append(["## PAIRWISE TESTS (WILCOXON-HOLM)", "", "", ""])
    rows.append(["Classifier A", "Classifier B", "p-value (raw)", "Significant (after Holm)"])

    if p_values:
        for entry in p_values:
            rows.append([entry[0], entry[1], f"{entry[2]:.6f}", "Yes" if entry[3] else "No"])
    else:
        rows.append(["—", "—", "—", "Not computed (Friedman not significant)"])

    pd.DataFrame(rows).to_csv(path, index=False, header=False)


def main():
    parser = argparse.ArgumentParser(description="MARS Software Suite")
    parser.add_argument("--input", required=True)
    parser.add_argument("--standard", action="store_true")
    parser.add_argument("--mars", action="store_true")
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--bw", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--perms", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=None)

    args = parser.parse_args()

    if not (args.standard or args.mars):
        args.mars = True

    color_mode = 'color' if args.color else 'bw'

    df     = pd.read_csv(args.input)
    prefix = os.path.splitext(args.input)[0]
    n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()

    p_perm, stat_obs, perm_sig = permutation_global_test(
        df, n_permutations=args.perms, alpha=args.alpha, n_jobs=n_jobs
    )

    if args.standard:
        p_vals, ranks, p_friedman = StandardSignificance.compute_wilcoxon_holm(df, args.alpha)

        if ranks is not None:
            fig = UnifiedPlotter.graph_ranks(
                ranks.values, ranks.index,
                p_values=p_vals,
                title="Standard",
                name=args.name,
                color_mode=color_mode
            )
            fig.savefig(f"{prefix}_std.pdf", bbox_inches='tight')
            plt.close(fig)

        write_csv_report(
            path=f"{prefix}_std_report.csv",
            mode="standard",
            ranks=ranks,
            p_values=p_vals,
            p_friedman=p_friedman,
            p_perm=p_perm,
            stat_obs=stat_obs,
            n_permutations=args.perms,
            alpha=args.alpha
        )

    if args.mars:
        p_vals, _, p_friedman = StandardSignificance.compute_wilcoxon_holm(df, args.alpha)

        if p_vals is None:
            write_csv_report(
                path=f"{prefix}_mars_report.csv",
                mode="mars",
                ranks=None,
                p_values=None,
                p_friedman=p_friedman,
                p_perm=p_perm,
                stat_obs=stat_obs,
                n_permutations=args.perms,
                alpha=args.alpha
            )
        else:
            pivot = df.pivot(index='observation',
                             columns='subject',
                             values='value')

            X      = pivot.values
            wr_mat = MARSEngine.compute_weighted_rank_matrix(X)

            mars_ranks = pd.Series(
                wr_mat.mean(axis=0),
                index=pivot.columns
            ).sort_values(ascending=False)

            fig = UnifiedPlotter.graph_ranks(
                mars_ranks.values,
                mars_ranks.index,
                p_values=p_vals,
                title="MARS",
                name=args.name,
                color_mode=color_mode
            )

            fig.savefig(f"{prefix}_mars.pdf", bbox_inches='tight')
            plt.close(fig)

            write_csv_report(
                path=f"{prefix}_mars_report.csv",
                mode="mars",
                ranks=mars_ranks,
                p_values=p_vals,
                p_friedman=p_friedman,
                p_perm=p_perm,
                stat_obs=stat_obs,
                n_permutations=args.perms,
                alpha=args.alpha
            )


if __name__ == "__main__":
    main()
