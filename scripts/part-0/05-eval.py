# graph_seed,mod_seed,n_nodes,difficulty,mod,k,n_edges,f1,f1_err,shd_score,shd_count,sid_score,sid_count,parent_aid_score,parent_aid_count,oset_aid_score,oset_aid_count

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read the CSV file
df = pd.read_csv("outputs/part-0/results_test.csv")

# print the first 5 rows
print(df.head())

# compute k ratio using k/n_edges if mod has edge in it, otherwise using k/n_nodes if mod has the word "node" in it
df["k_ratio"] = df.apply(lambda x: x["k"] / x["n_edges"] if "edge" in x["mod"] else x["k"] / x["n_nodes"], axis=1)
print(df["k_ratio"])

# drop following rows: k, n_edges, shd_count, sid_count, parent_aid_count, oset_aid_count
df = df.drop(columns=["k", "n_edges", "shd_count", "sid_count", "parent_aid_count", "oset_aid_count"])
print(df.head())

# Add binned k_ratio to df for plotting
df["x_bin"] = pd.cut(df["k_ratio"], np.arange(0, 1.01, 0.1))

id_vars = ["graph_seed", "mod_seed", "n_nodes", "difficulty", "mod", "k_ratio", "x_bin"]
value_vars = ["f1", "f1_err", "shd_score", "sid_score", "parent_aid_score", "oset_aid_score"]
df = df.melt(
    id_vars=id_vars, 
    value_vars=value_vars,
    var_name="metric",
    value_name="value"
    )

print(df.head())

# PRIMARY OVERVIEW
# Create a FacetGrid line plot with enhanced styling, titles, and labels
g = sns.relplot(
    kind="line", 
    data=df, 
    x="k_ratio", 
    y="value", 
    hue="n_nodes", 
    row="mod", 
    col="metric", 
    errorbar=("ci", 95),
    height=4, 
    aspect=1.5,
    facet_kws={'margin_titles': True}
)

# Set axis labels and a global figure title
g.set_axis_labels("k ratio (per mod type)", "Score")
g.set_titles(row_template='{row_name}', col_template='{col_name}')  # show metric and mod in faceted axes
g.figure.subplots_adjust(top=0.88)
g.figure.suptitle("Metric Scores by k Ratio, n_nodes, Modification Type", fontsize=16, y=0.99)

# Improve legend
g._legend.set_title("n_nodes")

# Make ticks a bit larger and rotate x labels for clarity
for ax in g.axes.flat:
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')

# Save and close
g.savefig("outputs/part-0/k_ratio_vs_score.png", bbox_inches="tight")
plt.close("all")


# ======== VIOLIN PLOTS ========
# k_ratio vs score, violin plot per n for each mod

# overlay sns.pointplot (mean per bin, dodge matched) and connect means as the trend line. Three violins (N=4/16/64) per ratio bin. Save one figure per metric (5 figures).
# For each metric, plot violin+pointplot (one figure per metric, saved)

unique_n_nodes = sorted(df["n_nodes"].unique())
palette = sns.color_palette("Set2", len(unique_n_nodes))
# dodge needs >= 2 hue levels; with a single n_nodes value it divides by (n-1)=0
point_dodge = 0.4 if len(unique_n_nodes) > 1 else False

for metric in value_vars:
    fig, axes = plt.subplots(
        nrows=len(df["mod"].unique()), 
        ncols=1, 
        figsize=(14, 4 * len(df["mod"].unique())),
        sharex=True,
    )
    if len(df["mod"].unique()) == 1:
        axes = [axes]
    for i, mod in enumerate(df["mod"].unique()):
        ax = axes[i]
        # Filter for this mod+metric
        df_sub = df[(df["metric"] == metric) & (df["mod"] == mod)]
        # violinplot (distribution in each bin by n_nodes)
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
        # Overlay pointplot (means per bin / n_nodes, trend line)
        sns.pointplot(
            data=df_sub,
            x="x_bin",
            y="value",
            hue="n_nodes",
            dodge=point_dodge,  # align with violins when multiple N; False if only one
            errorbar=("ci", 95),
            ax=ax,
            palette=palette,
            legend=False,
            markers="d",
            linestyles="-",
        )
        ax.set_title(f"{mod} - {metric}")
        ax.legend(title="n_nodes", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel("k/n_edges or k/n_nodes ratio bin")
        ax.set_ylabel(metric)
        labels = [str(cat) for cat in df_sub["x_bin"].cat.categories]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha='right')

    fig.tight_layout()
    fig.savefig(f"outputs/part-0/k_ratio_vs_score_violin_{metric}.png")
    plt.close(fig)


# ======= CORRELATION HEATMAPS ========
# Pivot tidy back so the metrics are columns; for each mod, compute Spearman corr.
# Answers: "do the scores rank the same corruptions the same way?"
# Use 0-is-better metrics only (drop raw f1 — it's perfectly anti-correlated with f1_err).
analysis_metrics = [m for m in value_vars if m != "f1"]

wide = df.pivot_table(
    index=["graph_seed", "mod_seed", "n_nodes", "difficulty", "mod", "k_ratio"],
    columns="metric",
    values="value",
).reset_index()

mods = list(df["mod"].unique())
n_mods = len(mods)
n_cols = 3
n_rows = int(np.ceil(n_mods / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
axes = np.atleast_1d(axes).ravel()

for i, mod in enumerate(mods):
    ax = axes[i]
    corr = wide.loc[wide["mod"] == mod, analysis_metrics].corr(method="spearman")
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

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Metric agreement (Spearman) per modification", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("outputs/part-0/metric_agreement_spearman.png", bbox_inches="tight")
plt.close(fig)


# ======= sensitivity-slope heatmap ======
# For each (mod, metric, n_nodes) fit slope of value vs k_ratio.
# Heatmap: rows = mod, cols = metric, one panel per N.
# Shows which metric is most responsive to each modification.

slope_rows = []
for (mod, metric, n_nodes), group in df.groupby(["mod", "metric", "n_nodes"]):
    if metric not in analysis_metrics:
        continue
    if group["k_ratio"].nunique() < 2:
        slope = np.nan
    else:
        slope = float(np.polyfit(group["k_ratio"].to_numpy(), group["value"].to_numpy(), 1)[0])
    slope_rows.append({"mod": mod, "metric": metric, "n_nodes": n_nodes, "slope": slope})

slope_df = pd.DataFrame(slope_rows)
n_node_vals = sorted(slope_df["n_nodes"].unique())
fig, axes = plt.subplots(
    1,
    len(n_node_vals),
    figsize=(6 * len(n_node_vals), max(4, 0.5 * len(mods))),
    squeeze=False,
)

for i, n in enumerate(n_node_vals):
    ax = axes[0, i]
    mat = (
        slope_df[slope_df["n_nodes"] == n]
        .pivot(index="mod", columns="metric", values="slope")
        .reindex(index=mods, columns=analysis_metrics)
    )
    sns.heatmap(
        mat,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="mako",
        cbar=True,
    )
    ax.set_title(f"n_nodes = {n}")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    ax.set_ylabel("modification" if i == 0 else "")

fig.suptitle("Sensitivity slope (d score / d k_ratio)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig("outputs/part-0/sensitivity_slope_heatmap.png", bbox_inches="tight")
plt.close(fig)
