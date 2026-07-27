# graph_seed,mod_seed,n_nodes,difficulty,mod,k,n_edges,f1,f1_err,shd_score,shd_count,sid_score,sid_count,parent_aid_score,parent_aid_count,oset_aid_score,oset_aid_count

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from configs.config import DataConfig, PathsConfig

# read the CSV file
df = pd.read_csv(PathsConfig.out_csv)

print(df.head())

# k_ratio: edge mods → k/n_edges; node mods → k/n_nodes
edge_mods = set(DataConfig.edge_modifications)
df["k_ratio"] = df.apply(
    lambda x: x["k"] / x["n_edges"] if x["mod"] in edge_mods else x["k"] / x["n_nodes"],
    axis=1,
)

df = df.drop(columns=["k", "n_edges", "shd_count", "sid_count", "parent_aid_count", "oset_aid_count"])

# Add binned k_ratio for violin plots
df["x_bin"] = pd.cut(df["k_ratio"], np.arange(0, 1.01, 0.1))

id_vars = ["graph_seed", "mod_seed", "n_nodes", "difficulty", "mod", "k_ratio", "x_bin"]
# 0-is-better metrics only (omit raw f1; use f1_err)
value_vars = ["f1_err", "shd_score", "sid_score", "parent_aid_score", "oset_aid_score"]
df = df.melt(
    id_vars=id_vars,
    value_vars=value_vars,
    var_name="metric",
    value_name="value",
)

print(df.head())

out_dir = PathsConfig.outputs_dir
out_dir.mkdir(parents=True, exist_ok=True)

# Stable ordering
difficulties = [d for d in DataConfig.difficulty if d in set(df["difficulty"])]
mods = [m for m in (DataConfig.edge_modifications + DataConfig.node_modifications) if m in set(df["mod"])]
unique_n_nodes = sorted(df["n_nodes"].unique())
palette = sns.color_palette("Set2", len(unique_n_nodes))
analysis_metrics = list(value_vars)


def _style_xticklabels(ax, rotation=30):
    ax.tick_params(axis="both", which="major", labelsize=10)
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")


# ===== PRIMARY OVERVIEW (one figure per difficulty) =====
print("Generating primary overview...")
for diff in difficulties:
    df_diff = df[df["difficulty"] == diff]
    g = sns.relplot(
        kind="line",
        data=df_diff,
        x="k_ratio",
        y="value",
        hue="n_nodes",
        row="mod",
        col="metric",
        errorbar=("ci", 95),
        height=4,
        aspect=1.5,
        facet_kws={"margin_titles": True},
        palette=palette,
    )
    g.set_axis_labels("k ratio (per mod type)", "Score (0 = perfect)")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.figure.subplots_adjust(top=0.94)
    g.figure.suptitle(
        f"Metric scores vs k ratio — difficulty={diff}",
        fontsize=16,
        y=0.995,
    )
    if g.legend is not None:
        g.legend.set_title("n_nodes")
    for ax in g.axes.flat:
        _style_xticklabels(ax)
    g.savefig(out_dir / f"k_ratio_vs_score_{diff}.png", bbox_inches="tight")
    plt.close("all")


# ======== VIOLIN PLOTS (one figure per metric × difficulty) ========
# Median trend lines skip empty bins in the data but still connect across gaps.
print("Generating violin plots...")
for diff in difficulties:
    df_diff = df[df["difficulty"] == diff]
    for metric in value_vars:
        fig, axes = plt.subplots(
            nrows=len(mods),
            ncols=1,
            figsize=(14, 4 * max(len(mods), 1)),
            sharex=True,
        )
        axes = np.atleast_1d(axes).ravel()
        for i, mod in enumerate(mods):
            ax = axes[i]
            df_sub = df_diff[(df_diff["metric"] == metric) & (df_diff["mod"] == mod)]
            if df_sub.empty:
                ax.set_visible(False)
                continue
            sns.violinplot(
                data=df_sub,
                x="x_bin",
                y="value",
                hue="n_nodes",
                split=False,
                dodge=True,
                density_norm="width",
                inner=None,
                ax=ax,
                palette=palette,
                cut=0,
            )
            # Transparency on violin bodies only (before overlaying median lines)
            for artist in ax.collections:
                artist.set_alpha(0.6)

            # Continuous median line per n_nodes: connect occupied bins across gaps
            categories = list(df_sub["x_bin"].cat.categories)
            cat_to_x = {cat: xi for xi, cat in enumerate(categories)}
            n_hue = len(unique_n_nodes)
            for j, n in enumerate(unique_n_nodes):
                sub_n = df_sub[df_sub["n_nodes"] == n]
                if sub_n.empty:
                    continue
                med = sub_n.groupby("x_bin", observed=True)["value"].median().dropna()
                if med.empty:
                    continue
                xs = np.array([cat_to_x[idx] for idx in med.index], dtype=float)
                if n_hue > 1:
                    # Match seaborn categorical dodge for hue
                    xs = xs + (j - (n_hue - 1) / 2) * (0.8 / n_hue)
                ax.plot(
                    xs,
                    med.to_numpy(),
                    marker="d",
                    color=palette[j],
                    linestyle="-",
                    linewidth=1.5,
                    zorder=3,
                    label="_nolegend_",
                )

            ax.set_title(f"{mod} — {metric} (difficulty={diff})")
            ax.legend(title="n_nodes", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.set_xlabel("k ratio bin")
            ax.set_ylabel(metric)
            labels = [str(cat) for cat in categories]
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=40, ha="right")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_dir / f"k_ratio_vs_score_violin_{metric}_{diff}.png", bbox_inches="tight")
        plt.close(fig)


# ======= CORRELATION HEATMAPS (one figure per difficulty) ========
wide = df.pivot_table(
    index=["graph_seed", "mod_seed", "n_nodes", "difficulty", "mod", "k_ratio"],
    columns="metric",
    values="value",
).reset_index()

n_mods = len(mods)
n_cols = 3
n_rows = int(np.ceil(n_mods / n_cols)) if n_mods else 1

print("Generating correlation heatmaps...")
for diff in difficulties:
    wide_diff = wide[wide["difficulty"] == diff]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for i, mod in enumerate(mods):
        ax = axes[i]
        corr = wide_diff.loc[wide_diff["mod"] == mod, analysis_metrics].corr(method="spearman")
        sns.heatmap(
            corr,
            ax=ax,
            annot=True,
            fmt=".2f",
            vmin=-1,
            vmax=1,
            cmap="vlag",
            square=True,
            cbar=(i == n_mods - 1),
        )
        ax.set_title(mod)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    for j in range(len(mods), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Metric agreement (Spearman) — difficulty={diff}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"metric_agreement_spearman_{diff}.png", bbox_inches="tight")
    plt.close(fig)


# ======= SENSITIVITY-SLOPE HEATMAPS (one figure per difficulty; panels = n_nodes) =======
# how fast does this score get worse as you corrupt more of the graph?
print("Generating sensitivity-slope heatmaps...")
slope_rows = []
for (mod, metric, n_nodes, diff), group in df.groupby(["mod", "metric", "n_nodes", "difficulty"]):
    if metric not in analysis_metrics:
        continue
    if group["k_ratio"].nunique() < 2:
        slope = np.nan
    else:
        slope = float(np.polyfit(group["k_ratio"].to_numpy(), group["value"].to_numpy(), 1)[0])
    slope_rows.append(
        {"mod": mod, "metric": metric, "n_nodes": n_nodes, "difficulty": diff, "slope": slope}
    )

slope_df = pd.DataFrame(slope_rows)

for diff in difficulties:
    slope_diff = slope_df[slope_df["difficulty"] == diff]
    n_node_vals = sorted(slope_diff["n_nodes"].unique())
    if not n_node_vals:
        continue
    fig, axes = plt.subplots(
        1,
        len(n_node_vals),
        figsize=(6 * len(n_node_vals), max(4, 0.5 * len(mods))),
        squeeze=False,
    )
    for i, n in enumerate(n_node_vals):
        ax = axes[0, i]
        mat = (
            slope_diff[slope_diff["n_nodes"] == n]
            .pivot(index="mod", columns="metric", values="slope")
            .reindex(index=mods, columns=analysis_metrics)
        )
        sns.heatmap(mat, ax=ax, annot=True, fmt=".2f", cmap="mako", cbar=True)
        ax.set_title(f"n_nodes = {n}")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        ax.set_ylabel("modification" if i == 0 else "")

    fig.suptitle(f"Sensitivity slope (d score / d k_ratio) — difficulty={diff}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"sensitivity_slope_heatmap_{diff}.png", bbox_inches="tight")
    plt.close(fig)

print(f"Wrote figures to {out_dir} (one set per difficulty: {difficulties})")
