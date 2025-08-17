import pickle
import numpy as np
from matplotlib import pyplot as plt
from causal_graph_comparison import OUTPUTS_DIR

seed = 1
# experiment_name = f"{model}-modes_{modes}-diff_{difficulty}-seed_{seed}"

results_pkl_path = OUTPUTS_DIR / f"evaluation.pkl"
# Load evaluation results
with open(results_pkl_path, "rb") as f:
    output_dict = pickle.load(f)

results_pkl_path = OUTPUTS_DIR / f"evaluation_savar_gt.pkl"
# Load evaluation results
with open(results_pkl_path, "rb") as f:
    savar_gt_dict = pickle.load(f)

print(savar_gt_dict.keys())

def get_marker_and_color(model):
    if model == "mlp":
        marker = "X"  # filled X
        color = "red"
    elif model == "cnn":
        marker = "o"  # filled circle
        color = "blue"
    elif model == "lstm":
        marker = "s"  # filled square
        color = "green"
    elif model == "vae":
        marker = "^"  # filled triangle
        color = "magenta"
    return marker, color

diff_map = {
    "easy": "e",
    "med_easy": "me",
    "med_hard": "mh",
    "hard": "h",
}


# =============================
#         R2 VALUES
# =============================

# looking for r2 > 0.6
# metric_x = "intervention_rmse"
# metric_y = "crl_parent_aid" # strong correlation in mode 4
# metric_y = "crl_oset_aid" # strong correlation in mode 4
# metric_y = "crl_ancestor_aid" # correlation in mode 4 & 16?
# metric_y = "crl_shd" # no correlation
# metric_y = "crl_f1" # no correlation
# metric_y = "cd_parent_aid" # weak correlation in mode 4
# metric_y = "cd_oset_aid" # weak correlation in mode 4
# metric_y = "cd_ancestor_aid" # weak correlation in mode 4
# metric_y = "cd_shd" # medium correlation in mode 4
metric_y = "cd_f1" # some correlation in mode 4?

metric_x = "lsd"
# metric_y = "crl_parent_aid" # strong correlation in mode 4 except for hard
# metric_y = "crl_oset_aid" # strong correlation in mode 4 except for hard
# metric_y = "crl_ancestor_aid" # strong correlation in mode 4 except for hard
# metric_y = "crl_shd" # no correlation in mode 4 <-- lots of 0s because found correct graph
metric_y = "crl_f1" # questionable correlation in mode 4
# metric_y = "cd_parent_aid" # only strong correlation in mode 4 for easy
# metric_y = "cd_oset_aid" # moderate correlation in mode 4
# metric_y = "cd_ancestor_aid" # moderate correlation in mode 4
# metric_y = "cd_shd" # kind of strong correlation in mode 4
# metric_y = "cd_f1" # some strong correlation in mode 4?

metric_x = "next_step_rmse"
# metric_y = "crl_parent_aid" # extraordinarily strong correlation in mode 4 but e and me have 0s because found correct graph
# metric_y = "crl_oset_aid" # strong correlation in mode 4 but e and me have 0s because found correct graph
# metric_y = "crl_ancestor_aid" # strong correlation in mode 4, same as above
# metric_y = "crl_shd" # no correlation
# metric_y = "crl_f1" # no correlation
# metric_y = "cd_parent_aid" # correlation on easy and hard in mode 4, absolutely no correlation in mode 16 / 64 meaning cd found roughly same val for each diff across models
# metric_y = "cd_oset_aid" # nothing really to be honest
# metric_y = "cd_ancestor_aid" #nothing
# metric_y = "cd_shd" # strong correlation for me and e in mode 4
metric_y = "cd_f1" # strong correlation for me and e in mode 4 (easiest ones to learn right)

metric_x = "next_step_r2"
# metric_y = "crl_parent_aid" # extraordinarily strong correlation negative in mode 4
# metric_y = "crl_oset_aid"  # extraordinarily strong correlation negative in mode 4 
# metric_y = "crl_ancestor_aid" # extraordinarily strong correlation negative in mode 4 
# metric_y = "crl_shd" # no correlation
# metric_y = "crl_f1" # no correlation
# metric_y = "cd_parent_aid" # correlation on easy and hard in mode 4, absolutely no correlation in mode 16 / 64 meaning cd found roughly same val for each diff across models
# metric_y = "cd_oset_aid" # correlation on mh and me
# metric_y = "cd_ancestor_aid" #correlation on mh
# metric_y = "cd_shd" # strong correlation for me and e in mode 4
# metric_y = "cd_f1" # strong correlation for me and e in mode 4 (easiest ones to learn right)

# strong r2 in mode 4 but not for hard

# report r2 between parent_aid/oset_aid, parent_aid/ancestor_aid, parent_aid/shd, parent_aid/f1, parent_aid/sid, ancestor_aid/oset_aid
# report r2 between crl_parent_aid/cd_parent_aid, crl_oset_aid/cd_oset_aid, crl_ancestor_aid/cd_ancestor_aid, crl_shd/cd_shd, crl_f1/cd_f1


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
axes = {
    "4": ax1,
    "16": ax2, 
    "64": ax3
}

md = {}
for mode in ["4", "16", "64"]:
    for difficulty in ["easy", "med_easy", "med_hard", "hard"]:
        md[f"{mode}-{difficulty}"] = []


for k, v in output_dict.items():
    identifier = k.split("-")
    model = identifier[0]
    mode = identifier[1].split("_")[1]
    difficulty = identifier[2].split("_")[1]

    if len(identifier[2].split("_")) > 2:
        difficulty = identifier[2].split("_")[1] + "_" + identifier[2].split("_")[2]

    if mode not in ["4", "16", "64"]:
        continue

    y_metric = v[metric_y]
    if (isinstance(y_metric, list) or isinstance(y_metric, tuple)) and len(y_metric) > 1:
        y_metric = y_metric[0]


    x_metric = v[metric_x]
    if (isinstance(x_metric, list) or isinstance(x_metric, tuple)) and len(x_metric) > 1:
        x_metric = x_metric[0]

    md[f"{mode}-{difficulty}"].append((x_metric, y_metric))

    marker, color = get_marker_and_color(model)
    axes[mode].scatter(x_metric, y_metric, marker=marker, color=color)
    axes[mode].annotate(f"{diff_map[difficulty]}", (x_metric, y_metric))

# Add model legend markers and labels to each subplot
for mode, ax in axes.items():
    for model in ["mlp", "cnn", "lstm", "vae"]:
        marker, color = get_marker_and_color(model)
        ax.scatter([], [], marker=marker, color=color, label=model)
    
    ax.legend()
    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    ax.set_title(f"{metric_x} vs {metric_y}\n{mode} modes")
    # ax.set_xscale('log')  # Set x-axis to log scale
    # ax.set_xlim(0.8, 1.1)  # Set x-axis limits between 0 and 1

# calculate r2 for each mode-difficulty
for k, v in md.items():
    mode, difficulty = k.split("-")
    xd = [x[0] for x in v]
    yd = [x[1] for x in v]

    # determine best fit line
    par = np.polyfit(xd, yd, 1, full=True)

    slope=par[0][0]
    intercept=par[0][1]
    xl = [min(xd), max(xd)]
    yl = [slope*xx + intercept  for xx in xl]

    # coefficient of determination, plot text
    variance = np.var(yd)
    residuals = np.var([(slope*xx + intercept - yy)  for xx,yy in zip(xd,yd)])
    Rsqr = np.round(1-residuals/variance, decimals=2)
    x_pos = .5*max(xd)+.5*min(xd)
    y_pos = .5*max(yd)+.5*min(yd)
    axes[mode].text(x_pos, y_pos, diff_map[difficulty]+', $R^2 = %0.2f$'% Rsqr, fontsize=14)
    axes[mode].plot(xl, yl, color="black", linewidth=2)

plt.tight_layout()
plt.show()


# Plot RMSE vs (structural, causal) metrics for all models
# one plot per comparisonshow that causal metric has best R^2 or something like that
# ------> this is done

# Plot iRMSE vs (structural, causal) metrics for all models
# show that causal metric has best R^2
# ------> this is done

# Plot LSD vs (structural, causal) metrics for all models
# one plot per comparison, show that causal metric has best R^2 or something like that
# ------> this is done

# Plot PSD (fft coefficients) for 4 modes over 20 timesteps compared to targets for all models
# Create subplots grid for different modes and difficulties
modes = ["4", "16", "64"]
difficulties = ["easy", "med_easy", "med_hard", "hard"]
models = ["mlp", "cnn", "lstm", "vae"]

fig, axes = plt.subplots(len(modes), len(difficulties), figsize=(20, 15))
fig.suptitle("FFT Coefficients Comparison Across Models", fontsize=16)

# Add row and column labels
for i, mode in enumerate(modes):
    axes[i,0].set_ylabel(f"{mode} modes", fontsize=12)
for j, diff in enumerate(difficulties):
    axes[0,j].set_title(f"{diff} difficulty", fontsize=12)

for i, mode in enumerate(modes):
    for j, difficulty in enumerate(difficulties):
        ax = axes[i,j]
        
        # Plot ground truth/savar line first
        try:
            # Use CNN as reference for savar (ground truth) data
            experiment_name = f"cnn-modes_{mode}-diff_{difficulty}-seed_{seed}"
            data = output_dict[experiment_name]
            x = np.arange(data["fft_coeffs_savar"].shape[0])
            ax.plot(x, data["fft_coeffs_savar"][:,0], 'k-', label="Ground Truth", linewidth=2)
            
            # Plot each model
            for model in models:
                experiment_name = f"{model}-modes_{mode}-diff_{difficulty}-seed_{seed}"
                if experiment_name in output_dict:
                    data = output_dict[experiment_name]
                    marker, color = get_marker_and_color(model)
                    ax.plot(x, data["fft_coeffs_rollouts"][:,0], color=color, 
                           label=model.upper(), alpha=0.7)
            
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Coefficient Value")
            
            # Only show legend for first subplot
            if i == 0 and j == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
        except KeyError as e:
            print(f"Missing data for mode {mode}, difficulty {difficulty}: {e}")
            ax.text(0.5, 0.5, 'No Data Available', 
                   horizontalalignment='center', verticalalignment='center')

plt.tight_layout()
plt.show()
plt.close()


# dict_keys(['crl_parent_aid', 'crl_oset_aid', 'crl_ancestor_aid', 'crl_sid', 'crl_shd', 'crl_f1', 'crl_precision',
# 'crl_recall', 'cd_parent_aid', 'cd_oset_aid', 'cd_ancestor_aid', 'cd_sid', 'cd_shd', 'cd_f1', 'cd_precision', 'cd_recall',

# prev 5 time steps + 1 next time step for all models vs gt?
# actual data of savar
# show what data looks like for savar 100 time steps

# Ground truth graph vs learned graph <-- a few examples
# use tigramite plotting tool for 1 example
# also plot adjacency matrices to show progresive difficulty + modes

# =============================
#         SAVAR GT
# =============================

gt_dict = {}
for mode in ["4", "16", "64"]:
    for difficulty in ["easy", "med_easy", "med_hard", "hard"]:
        gt_dict["crl_true_positives"] = []
        gt_dict["cd_true_positives"] = []
        gt_dict["crl_parent_aid"] = []
        gt_dict["crl_shd"] = []
        gt_dict["crl_f1"] = []
        gt_dict["cd_parent_aid"] = []
        gt_dict["cd_shd"] = []
        gt_dict["cd_f1"] = []

idx = 0
for i, mode in enumerate(modes):
    for j, difficulty in enumerate(difficulties):
        data_name = f"savar-modes_{mode}-diff_{difficulty}-seed_{seed}"
        data = savar_gt_dict[data_name]
        # print(data.keys())
        gt_dict["crl_true_positives"].append(data["crl_true_positives"])
        gt_dict["cd_true_positives"].append(data["cd_true_positives"])
        gt_dict["crl_parent_aid"].append((data["crl_parent_aid"][0], f"{mode}-{difficulty}", idx))
        gt_dict["crl_shd"].append((data["crl_shd"], f"{mode}-{difficulty}", idx))
        gt_dict["crl_f1"].append((data["crl_f1"], f"{mode}-{difficulty}", idx))
        gt_dict["cd_parent_aid"].append((data["cd_parent_aid"][0], f"{mode}-{difficulty}", idx))
        gt_dict["cd_shd"].append((data["cd_shd"], f"{mode}-{difficulty}", idx))
        gt_dict["cd_f1"].append((data["cd_f1"], f"{mode}-{difficulty}", idx))
        idx += 1

        print(data_name)
        print(f"crl_parent_aid: {data['crl_parent_aid']}")
        print(f"crl_shd: {data['crl_shd']}")
        print(f"crl_f1: {data['crl_f1']}")
        print(f"cd_parent_aid: {data['cd_parent_aid']}")
        print(f"cd_shd: {data['cd_shd']}")
        print(f"cd_f1: {data['cd_f1']}")
        print("")


# calculate accuracy for crl_true_positives, cd_true_positives
crl_accuracy = np.mean(gt_dict["crl_true_positives"])
cd_accuracy = np.mean(gt_dict["cd_true_positives"])
# count the number of 1s in crl_true_positives, cd_true_positives
crl_true_positives_count = np.sum(gt_dict["crl_true_positives"])
cd_true_positives_count = np.sum(gt_dict["cd_true_positives"])
print(f"CRL Accuracy: {crl_accuracy} ({crl_true_positives_count} / {len(gt_dict['crl_true_positives'])})")
print(f"CD Accuracy: {cd_accuracy} ({cd_true_positives_count} / {len(gt_dict['cd_true_positives'])})")

def plot_metric_comparison(gt_dict, metric_name):
    """Plot comparison between CRL and CD for a given metric.
    
    Args:
        gt_dict: Dictionary containing the metrics data
        metric_name: String indicating which metric to plot ('f1', 'shd', or 'parent_aid')
    """
    metric_info = {
        'f1': {
            'title': 'CRL F1 vs CD F1',
            'ylabel': 'F1 Score',
            'log_scale': False
        },
        'shd': {
            'title': 'CRL SHD vs CD SHD', 
            'ylabel': 'SHD Score',
            'log_scale': True
        },
        'parent_aid': {
            'title': 'CRL Parent Aid vs CD Parent Aid',
            'ylabel': 'Parent Aid Score',
            'log_scale': False
        }
    }

    # Get CRL data
    x = [x[2] for x in gt_dict[f"crl_{metric_name}"]]
    y = [x[0] for x in gt_dict[f"crl_{metric_name}"]]
    plt.scatter(x, y, label=f"CRL {metric_name.upper()}")

    # Add labels for CRL points
    for i in range(len(x)):
        plt.annotate(gt_dict[f"crl_{metric_name}"][i][1], 
                    (x[i], y[i]), 
                    xytext=(5, 5), 
                    textcoords='offset points')

    # Get CD data
    x = [x[2] for x in gt_dict[f"cd_{metric_name}"]]
    y = [x[0] for x in gt_dict[f"cd_{metric_name}"]]
    plt.scatter(x, y, label=f"CD {metric_name.upper()}")

    # Add labels for CD points
    for i in range(len(x)):
        plt.annotate(gt_dict[f"cd_{metric_name}"][i][1], 
                    (x[i], y[i]), 
                    xytext=(5, -5), 
                    textcoords='offset points')

    plt.xlabel("Increasing modes & difficulties")
    plt.ylabel(metric_info[metric_name]['ylabel'])
    if metric_info[metric_name]['log_scale']:
        plt.yscale('log')
    plt.title(metric_info[metric_name]['title'])
    plt.legend()
    plt.show()

# Plot all three metrics
metrics = ['parent_aid', 'shd', 'f1']
for metric in metrics:
    plot_metric_comparison(gt_dict, metric)



# report the crl & cd average shd, f1, parent_aid across modes

# diagram of method

# Tables
# Hyperparameters?

# 9) Plotting...
# explore-mlp-output.ipynb
# causal-discovery-mlp.ipynb
# causal-discovery-groundtruth.ipynb
# causal-discovery-mlp-1000samples.ipynb

# TODO make this into ipynb 
# plot gt vs learned graph

# Plot time series rollout for 4 modes over 20 timesteps compared to targets for all models
# this is just illustrative example for methods illustration

# Confirm that RMSE correlates to other statistical metrics
# get r2 val?

# intervened 5 time stesp