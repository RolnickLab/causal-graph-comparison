import pickle
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path

CURRENT_DIR = Path(__file__).parent
seed = 1

results_pkl_path = CURRENT_DIR / f"evaluation_final.pkl"
# Load evaluation results
with open(results_pkl_path, "rb") as f:
    output_dict = pickle.load(f)

results_pkl_path = CURRENT_DIR / f"evaluation_savar_gt_final.pkl"
# Load evaluation results
with open(results_pkl_path, "rb") as f:
    savar_gt_dict = pickle.load(f)

print(savar_gt_dict.keys())

def get_marker_and_color(model):
    if model == "mlp":
        marker = "X"  # filled X
        color = "#94D0FF"
    elif model == "cnn":
        marker = "o"  # filled circle
        color = "#FF6AD5"
    elif model == "lstm":
        marker = "s"  # filled square
        color = "#966BFF"
    elif model == "vae":
        marker = "^"  # filled triangle
        color = "#FFA58B"
    return marker, color

corr_lines = {
    "lin": {
        "color": "#666666",
        "linestyle": ":",
        "linewidth": 2
    },
    "quad": {
        "color": "#999999",
        "linestyle": "-",
        "linewidth": 2
    }
}

diff_map = {
    "easy": "e",
    "med_easy": "me",
    "med_hard": "mh",
    "hard": "h",
}


md = {} # md for mode-difficulty
for mode in ["4", "16"]:
    for difficulty in ["easy", "med_easy", "med_hard", "hard"]:
        md[f"{mode}-{difficulty}"] = []

plotting = [
    ("next_step_rmse", "intervention_rmse"), 
    ("next_step_rmse", "lsd"), 
    ("intervention_rmse", "lsd"), 
    ("crl_parent_aid", "crl_oset_aid"), 
    ("cd_parent_aid", "cd_oset_aid"),
    ("crl_parent_aid", "crl_f1"),
    ("cd_parent_aid", "cd_f1"),
    ("crl_parent_aid", "crl_shd"),
    ("cd_parent_aid", "cd_shd"),
    ("next_step_rmse", "crl_parent_aid"),
    ("next_step_rmse", "cd_parent_aid"),
    ("next_step_rmse", "crl_f1"),
    ("next_step_rmse", "cd_f1"),
    ("next_step_rmse", "crl_shd"),
    ("next_step_rmse", "cd_shd"),
    ("intervention_rmse", "crl_parent_aid"),
    ("intervention_rmse", "cd_parent_aid"),
    ("intervention_rmse", "crl_f1"),
    ("intervention_rmse", "cd_f1"),
    ("intervention_rmse", "crl_shd"),
    ("intervention_rmse", "cd_shd"),
    ("lsd", "crl_parent_aid"),
    ("lsd", "cd_parent_aid"),
    ("lsd", "crl_f1"),
    ("lsd", "cd_f1"),
    ("lsd", "crl_shd"),
    ("lsd", "cd_shd")
]

allcaps = ["lsd", "shd", "aid", "rmse", "f1", "cd", "picabu"]

def plot_metrics_comparison(metric_pair):
    print(f"Plotting {metric_pair}...")
    metric_x, metric_y = metric_pair
    xy_pairs = {"4": [], "16": []}
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes = {
        "4": axes[0],
        "16": axes[1], 
    }   

    to_plot = {"4": [], "16": []}

    for k, v in output_dict.items():
        identifier = k.split("-")
        model = identifier[0]
        mode = str(identifier[1].split("_")[1])
        difficulty = identifier[2].split("_")[1]

        if len(identifier[2].split("_")) > 2:
            difficulty = identifier[2].split("_")[1] + "_" + identifier[2].split("_")[2]

        # Skip VAE med_easy and med_hard for 4 modes when using CRL metrics
        if model == "vae" and difficulty in ["med_easy", "hard"] and mode == "4" and ("crl" in metric_x or "crl" in metric_y):
            continue

        y_metric = v[metric_y]
        if (isinstance(y_metric, list) or isinstance(y_metric, tuple)) and len(y_metric) > 1:
            y_metric = y_metric[0]

        x_metric = v[metric_x]
        if (isinstance(x_metric, list) or isinstance(x_metric, tuple)) and len(x_metric) > 1:
            x_metric = x_metric[0]

        # md for mode-difficulty
        md[f"{mode}-{difficulty}"].append((x_metric, y_metric))
        if not (model == "mlp" and mode == "16"):
            xy_pairs[mode].append((x_metric, y_metric, f"{model}-{mode}-{difficulty}"))

        marker, color = get_marker_and_color(model)
        to_plot[mode].append((x_metric, y_metric, marker, color, diff_map[difficulty]))

    # Create lines for legend outside plots
    legend_lines = []
    legend_labels = []
     
    # Add model markers for legend
    for model in ["mlp", "cnn", "lstm", "vae"]:
        marker, color = get_marker_and_color(model)
        line = plt.Line2D([], [], marker=marker, color=color, label=model.upper(), linestyle='None', markersize=12)
        legend_lines.append(line)
        legend_labels.append(model.upper())

    # Add fit lines for legend
    legend_lines.extend([
        plt.Line2D([], [], **corr_lines["quad"]),
        plt.Line2D([], [], **corr_lines["lin"])
    ])
    legend_labels.extend(['Quadratic Fit', 'Linear Fit'])


    def format_title(metric):
        x = metric.replace("crl", "picabu")
        x = ' '.join(word.upper() if word in allcaps else word.capitalize() 
                        for word in x.split('_'))
        x = x.replace("CD", "Avg-PCMCI")
        return x

    x_title = format_title(metric_x)
    y_title = format_title(metric_y)


    for mode, ax in axes.items():
        # Format x and y labels using same convention as title
        ax.set_xlabel(x_title, fontsize=16, fontweight='bold')
        ax.set_ylabel(y_title, fontsize=16, fontweight='bold')

        ax.set_title(f"{mode} Modes", fontsize=16, fontweight='bold')

        # Calculate R-squared values before setting axis limits
        xd = [x[0] for x in xy_pairs[mode]]
        yd = [x[1] for x in xy_pairs[mode]]

        # determine best fit line quadratic
        coeffs = np.polyfit(xd, yd, 2)
        p = np.poly1d(coeffs)

        # determine best fit line linear
        coeffs_linear = np.polyfit(xd, yd, 1)
        p_linear = np.poly1d(coeffs_linear)

        # Calculate R-squared for quadratic
        y_mean = np.mean(yd)
        ss_tot = sum((float(y) - y_mean) ** 2 for y in yd)
        ss_res = sum((float(y) - p(x)) ** 2 for x, y in zip(xd, yd))
        Rsqr_quad = np.round(1 - (ss_res / ss_tot), decimals=2)

        # Calculate R-squared for linear
        ss_res_linear = sum((float(y) - p_linear(x)) ** 2 for x, y in zip(xd, yd))
        Rsqr_linear = np.round(1 - (ss_res_linear / ss_tot), decimals=2)

        # Remove outliers by setting axis limits to exclude extreme values
        if len(xd) > 0 and len(yd) > 0:
            x_q1, x_q3 = np.percentile(xd, [25, 75])
            y_q1, y_q3 = np.percentile(yd, [25, 75])
            x_iqr = x_q3 - x_q1
            y_iqr = y_q3 - y_q1
            

            buffer = 1.6
            x_lower = x_q1 - buffer * x_iqr
            x_upper = x_q3 + buffer * x_iqr
            y_lower = y_q1 - buffer * y_iqr
            y_upper = y_q3 + buffer * y_iqr
            
            ax.set_xlim(x_lower, x_upper)
            ax.set_ylim(y_lower, y_upper)

            ax.tick_params(axis='both', which='major', labelsize=16)
            # ax.tick_params(axis='both', which='minor', labelsize=8)

            x_buf = 0.015 * (x_upper - x_lower)
            y_buf = 0.025 * (y_upper - y_lower)
        
            for x_metric, y_metric, marker, color, diff in to_plot[mode]:
                axes[mode].scatter(x_metric, y_metric, marker=marker, color=color, s=120)
                axes[mode].annotate(diff, (x_metric - x_buf, y_metric - y_buf))

            # Generate points for smooth curves using the constrained x range
            xl = np.linspace(x_lower, x_upper, 100)
            yl_quad = p(xl)
            yl_linear = p_linear(xl)

            # Plot the fit lines
            ax.plot(xl, yl_quad, **corr_lines["quad"])
            ax.plot(xl, yl_linear, **corr_lines["lin"])

            # Add R-squared text within the constrained axes
            x_pos = 0.35 * (x_upper - x_lower) + x_lower  # 5% from left
            # y_pos = 0.95 * (y_upper - y_lower) + y_lower  # 95% from bottom
            y_pos = 0.12 * (y_upper - y_lower) + y_lower  # 5% from bottom
            y_offset = (y_upper - y_lower) * 0.05  # 5% of y-axis range
            ax.text(x_pos, y_pos-y_offset, 'Quad $R^2 = %0.2f$'% Rsqr_quad, fontsize=16)
            ax.text(x_pos, y_pos-2*y_offset, 'Linear $R^2 = %0.2f$'% Rsqr_linear, fontsize=16)

    # Add single legend to the right of both plots
    fig.legend(legend_lines, legend_labels, loc='center right', bbox_to_anchor=(1.16, 0.5), fontsize=16)
    
    plt.suptitle(f"{x_title} vs {y_title} ", fontsize=20)
    plt.tight_layout()
    plt.savefig(f'plots/metrics_comparison_{metric_x}_vs_{metric_y}.png', bbox_inches='tight')
    plt.close()

# PLOT ALL METRICS
for metric_pair in plotting:
    plot_metrics_comparison(metric_pair)


# Plot PSD (fft coefficients) for 4 modes over 20 timesteps compared to targets for all models
# Create subplots grid for different modes and difficulties
# take absolute value
modes = ["4", "16"]
difficulties = ["easy", "med_easy", "med_hard", "hard"]
models = ["mlp", "cnn", "lstm", "vae"]

fig = plt.figure(figsize=(15, 6))
fig.suptitle("FFT Coefficients Comparison Across Models (Absolute Value)", fontsize=16, y=1.05)
subfigs = fig.subfigures(2, 4)


axes = []
for row in range(2):
    row_axs = []
    for col in range(4):
        subfig = subfigs[row, col]
        if row == 1 and col == 3:  # Last plot (8th position) with broken axis
            ax = subfig.subplots(2, 1, sharex=True)
        else:        
            ax = subfig.subplots(1, 1)
        row_axs.append(ax)
    axes.append(row_axs)

d = 0.5  # proportion of vertical to horizontal extent of the slanted line
spine_args = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)


# Add row and column labels
# for i, mode in enumerate(modes):
#     axes[i,0].set_ylabel(f"{mode} modes", fontsize=12)
for j, diff in enumerate(difficulties):
    diff = diff.replace("med_", "Med-")
    diff = diff.replace("easy", "Easy")
    diff = diff.replace("hard", "Hard")
    axes[0][j].set_title(f"{diff} difficulty", fontsize=14)

for i, mode in enumerate(modes):
    for j, difficulty in enumerate(difficulties):
        ax = axes[i][j]
        # if i == 1 and j == 3:
        #     ax = ax[1]
        
        # Plot ground truth/savar line first
        try:
            # Use CNN as reference for savar (ground truth) data
            experiment_name = f"cnn-modes_{mode}-diff_{difficulty}-seed_{seed}"
            data = output_dict[experiment_name]
            x = np.arange(data["fft_coeffs_savar"].shape[0])
            if i == 1 and j == 3:
                ax[1].plot(x, np.abs(data["fft_coeffs_savar"][:,0]), 'k-', label="Ground Truth", linewidth=2)
            else:
                ax.plot(x, np.abs(data["fft_coeffs_savar"][:,0]), 'k-', label="Ground Truth", linewidth=2)
            
            # Plot each model
            for model in models:
                experiment_name = f"{model}-modes_{mode}-diff_{difficulty}-seed_{seed}"
                if experiment_name in output_dict:
                    data = output_dict[experiment_name]
                    marker, color = get_marker_and_color(model)
                    if i == 1 and j == 3:
                        if model == "mlp":
                            ax[0].plot(x, np.abs(data["fft_coeffs_rollouts"][:,0]), color=color, 
                            label=model.upper(), alpha=0.7)
                        else:
                            ax[1].plot(x, np.abs(data["fft_coeffs_rollouts"][:,0]), color=color, 
                            label=model.upper(), alpha=0.7)
                    else:
                        ax.plot(x, np.abs(data["fft_coeffs_rollouts"][:,0]), color=color, 
                            label=model.upper(), alpha=0.7)
            
            if i == 1:
                if j == 3:
                    ax[1].set_xlabel("Frequency", fontsize=14, fontweight='bold')
                    ax[1].spines.top.set_visible(False)
                    ax[0].spines.bottom.set_visible(False)
                    
                    ax[0].xaxis.tick_top()
                    ax[0].tick_params(labeltop=False)  # don't put tick labels at the top
                    ax[1].xaxis.tick_bottom()
                    
                    # Add slanted break lines
                    ax[0].plot([0, 1], [0, 0], transform=ax[0].transAxes, **spine_args)
                    ax[1].plot([0, 1], [1, 1], transform=ax[1].transAxes, **spine_args)
                else:
                    ax.set_xlabel("Frequency", fontsize=14, fontweight='bold')
                # ax.set_xlabel("Frequency", fontsize=14, fontweight='bold')
            if j == 0:
                
                ax.set_ylabel("Coefficient Value", fontsize=14, fontweight='bold')
            
            if i == 0:
                ax.set_ylim(0, 1.3)
            else:
                if j < 3:
                    ax.set_ylim(0, 1.0)
                else:
                    # ax[1].set_ylim(0, 0.44)
                    ax[1].set_ylim(0, 0.44)

            if j == 3:
                if i==0:
                    ax.text(1.05, 1, f'{mode} Modes', transform=ax.transAxes, fontsize=14, verticalalignment='top', fontweight='bold')
                else:
                    ax[0].text(1.05, 1, f'{mode} Modes', transform=ax[0].transAxes, fontsize=14, verticalalignment='top', fontweight='bold')
            
            # Add mode label to each subplot
            # ax.text(0.02, 0.98, f'Mode {mode}', transform=ax.transAxes,
            #        verticalalignment='top', fontsize=10)
            
            # Only show legend for last subplot
            if i == 0 and j == len(difficulties) - 1:
                ax.legend(bbox_to_anchor=(1.05, -0.02), loc='lower left', fontsize=14)
                
        except KeyError as e:
            print(f"Missing data for mode {mode}, difficulty {difficulty}: {e}")
            ax.text(0.5, 0.5, 'No Data Available', 
                   horizontalalignment='center', verticalalignment='center')

# plt.subplots_adjust(bottom=0.1, left=0.5, right=0.2, top=0.9)
# plt.tight_layout(pad=1000.0)
plt.gcf().subplots_adjust(right=0.9)
# plt.tight_layout(pad=3.0)
plt.savefig('plots/fft_coefficients_comparison.png', bbox_inches='tight')
plt.close()
# plt.show()


# =============================
#         SAVAR GT
# =============================

gt_dict = {}
for mode in ["4", "16"]:
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

# Calculate averages per number of modes
modes = ["4", "16"]
metrics = ["parent_aid", "shd", "f1"]
difficulties = ["easy", "med_easy", "med_hard", "hard"]

print("\nAverages per number of modes:")
print("-" * 50)
for mode in modes:
    print(f"\nMode {mode}:")
    
    # Calculate CRL averages
    for metric in metrics:
        values = [x[0] for x in gt_dict[f"crl_{metric}"] 
                 if x[1].startswith(mode)]
        avg = np.mean(values)
        print(f"CRL {metric:10}: {avg:.3f}")
    
    # Calculate CD averages  
    for metric in metrics:
        values = [x[0] for x in gt_dict[f"cd_{metric}"]
                 if x[1].startswith(mode)]
        avg = np.mean(values)
        print(f"CD {metric:10}: {avg:.3f}")


def plot_crl_cd_comparison(gt_dict, metric_name):
    """Plot comparison between CRL and CD for a given metric.
    
    Args:
        gt_dict: Dictionary containing the metrics data
        metric_name: String indicating which metric to plot ('f1', 'shd', or 'parent_aid')
    """
    print(f"Plotting {metric_name} comparison...")
    metric_info = {
        'f1': {
            'title': 'F1 Scores comparing GT to graphs learned using PICABU and Avg-PCMCI',
            'ylabel': 'F1 Score',
            'log_scale': False,
            'y_pos': 0.7,
            'legend_loc': 'center right',
            'picabu_first': True
        },
        'shd': {
            'title': 'SHD Scores comparing GT to graphs learned using PICABU and Avg-PCMCI', 
            'ylabel': 'SHD Score',
            'log_scale': True,
            'y_pos': 500,
            'legend_loc': 'upper left',
            'picabu_first': False
        },
        'parent_aid': {
            'title': 'Parent-Aid Scores comparing GT to graphs learned using PICABU and Avg-PCMCI',
            'ylabel': 'Parent Aid Score',
            'log_scale': False,
            'y_pos': 0.35,
            'legend_loc': 'upper left',
            'picabu_first': False
        }
    }

    # Define x-axis labels
    x_labels = ['Easy', 'Med-Easy', 'Med-Hard', 'Hard']*2
    x_positions = range(len(x_labels))

    # Get CRL data 
    y_crl = [x[0] for x in gt_dict[f"crl_{metric_name}"]]
    plt.figure(figsize=(12, 8))
    plt.axvspan(-0.5, 3.5, color='#999999', alpha=0.3)
    # Plot CRL points and add value labels
    
    
    y_cd = [x[0] for x in gt_dict[f"cd_{metric_name}"]]
    
    def draw_picabu():
        plt.scatter(x_positions, y_crl, marker='o', s=150, label=f"PICABU {metric_name.upper()}", color='#FF6AD5')
        for i, val in enumerate(y_crl):
            plt.annotate(f'{val:.2f}', 
                        (x_positions[i], val),
                        xytext=(0, -20), 
                        textcoords='offset points',
                        fontsize=12,
                        ha='center')

    def draw_pcmci():
        plt.scatter(x_positions, y_cd, marker='^', s=150, label=f"Avg-PCMCI {metric_name.upper()}", color='#966BFF')
        for i, val in enumerate(y_cd):
            plt.annotate(f'{val:.2f}',
                        (x_positions[i], val),
                        xytext=(0, -20),
                        textcoords='offset points', 
                        fontsize=12,
                        ha='center')
    
    if metric_info[metric_name]['picabu_first']:
        draw_picabu()
        draw_pcmci()
    else:
        draw_pcmci()
        draw_picabu()

    # Get CD data and add value labels


    # plt.xticks(x_positions, x_labels, rotation=45)
    plt.xticks(x_positions, x_labels, fontsize=16)
    plt.xlabel("Modes & Difficulties", fontsize=16, fontweight='bold')
    plt.text(1.5, metric_info[metric_name]['y_pos'], "4 Modes", fontsize=14, ha='center', fontweight='bold')
    plt.text(5.5, metric_info[metric_name]['y_pos'], "16 Modes", fontsize=14, ha='center', fontweight='bold')
    plt.ylabel(metric_info[metric_name]['ylabel'], fontsize=16, fontweight='bold')
    plt.title(metric_info[metric_name]['title'], fontsize=18)
    plt.legend(loc=metric_info[metric_name]['legend_loc'], fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    plt.savefig(f'plots/{metric_name}_comparison.png')
    plt.close()

# Plot all three metrics
metrics = ['parent_aid', 'shd', 'f1']
for metric in metrics:
    plot_crl_cd_comparison(gt_dict, metric)

