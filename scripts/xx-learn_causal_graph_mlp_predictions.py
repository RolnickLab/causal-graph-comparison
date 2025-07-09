import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite import plotting as tp
from tigramite import data_processing as pp
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from causal_graph_comparison.mlp import Net
from causal_graph_comparison import OUTPUTS_DIR, MODELS_DIR

def load_mlp_predictions(timestamp="2025_07_01_19_03_57"):
    """
    Load MLP predictions from the test results
    """
    # Load the test results
    results_path = OUTPUTS_DIR / f"test_results-{timestamp}.npz"
    data = np.load(results_path)
    
    # Extract outputs (predictions)
    outputs = data['outputs']  # Shape: (995, 8000) - 995 predictions, each predicting 5 time steps * 40*40 spatial grid
    
    print(f"Loaded predictions shape: {outputs.shape}")
    print(f"Number of predictions: {outputs.shape[0]}")
    print(f"Output size per prediction: {outputs.shape[1]}")
    
    return outputs

def reshape_predictions_to_temporal(outputs, future_timesteps=5, lat=40, lon=40):
    """
    Reshape predictions from flattened to temporal format
    """
    # Reshape from (995, 8000) to (995, 5, 40, 40)
    # Each prediction contains 5 time steps of 40x40 spatial fields
    reshaped = outputs.reshape(outputs.shape[0], future_timesteps, lat, lon)
    
    print(f"Reshaped predictions shape: {reshaped.shape}")
    print(f"Temporal dimension: {reshaped.shape[1]}")
    print(f"Spatial dimensions: {reshaped.shape[2]}x{reshaped.shape[3]}")
    
    return reshaped

def reduce_spatial_dimensions(predictions, n_modes=4):
    """
    Reduce spatial dimensions to n_modes using PCA
    """
    # Flatten spatial dimensions for PCA
    # Shape: (995, 5, 40, 40) -> (995*5, 1600)
    n_samples, n_timesteps, lat, lon = predictions.shape
    flattened = predictions.reshape(n_samples * n_timesteps, lat * lon)
    
    # Standardize the data
    scaler = StandardScaler()
    flattened_scaled = scaler.fit_transform(flattened)
    
    # Apply PCA to reduce to n_modes
    pca = PCA(n_components=n_modes)
    reduced = pca.fit_transform(flattened_scaled)
    
    # Reshape back to temporal format: (995, 5, 4)
    reduced_temporal = reduced.reshape(n_samples, n_timesteps, n_modes)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")
    print(f"Reduced data shape: {reduced_temporal.shape}")
    
    return reduced_temporal, pca, scaler

def learn_causal_graph_pcmci(data, tau_max=3, pc_alpha=0.05, save_results=True, output_dir=None):
    """
    Learn causal graph using PCMCI+ on the reduced data
    Handles multiple independent sequences (samples) properly
    """
    print(f"\nLearning causal graph with PCMCI+")
    print(f"Data shape: {data.shape}")
    print(f"Number of sequences: {data.shape[0]}")
    print(f"Time steps per sequence: {data.shape[1]}")
    print(f"Number of modes: {data.shape[2]}")
    print(f"tau_max: {tau_max}")
    print(f"pc_alpha: {pc_alpha}")
    
    # Create variable names for the modes
    var_names = [f"mode_{i}" for i in range(data.shape[2])]
    
    # For multiple sequences, we need to concatenate them into a single time series
    # but preserve the sequence boundaries for proper analysis
    n_sequences, n_timesteps, n_modes = data.shape
    
    # Reshape to (n_sequences * n_timesteps, n_modes) for PCMCI+
    # This treats each sequence as a separate realization
    data_reshaped = data.reshape(-1, n_modes)
    
    print(f"Reshaped data for PCMCI+: {data_reshaped.shape}")
    print(f"Total time points: {data_reshaped.shape[0]}")
    
    # Create a proper tigramite DataFrame with the correct shape (T, N)
    dataframe = pp.DataFrame(data_reshaped, var_names=var_names)
    
    # Initialize PCMCI+ with the dataframe
    pcmci = PCMCI(
        dataframe=dataframe,  # Pass the dataframe, not None
        cond_ind_test=ParCorr(),
        verbosity=1
    )
    
    # Run PCMCI+ on the concatenated data
    # PCMCI+ will automatically handle the multiple sequences
    results = pcmci.run_pcmci(
        tau_max=tau_max,
        pc_alpha=pc_alpha
    )
    
    # Extract results
    graph = results['graph']
    val_matrix = results['val_matrix']
    p_matrix = results['p_matrix']
    
    print(f"\nPCMCI+ Results:")
    print(f"Graph shape: {graph.shape}")
    print(f"Value matrix shape: {val_matrix.shape}")
    print(f"P-value matrix shape: {p_matrix.shape}")
    
    # Print the discovered links
    print(f"\nDiscovered causal links:")
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            for tau in range(tau_max):
                if graph[i, j, tau] != "":
                    print(f"{var_names[i]} <-{graph[i, j, tau]}- {var_names[j]} (lag {tau+1})")
    
    # Save results if requested
    if save_results and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the results
        np.save(output_dir / "pcmci_graph.npy", graph)
        np.save(output_dir / "pcmci_val_matrix.npy", val_matrix)
        np.save(output_dir / "pcmci_p_matrix.npy", p_matrix)
        
        # Save as CSV for easier inspection
        links_data = []
        for i in range(len(var_names)):
            for j in range(len(var_names)):
                for tau in range(tau_max):
                    if graph[i, j, tau] != "":
                        links_data.append({
                            'target': var_names[i],
                            'source': var_names[j],
                            'lag': tau + 1,
                            'link_type': graph[i, j, tau],
                            'value': val_matrix[i, j, tau],
                            'p_value': p_matrix[i, j, tau]
                        })
        
        if links_data:
            links_df = pd.DataFrame(links_data)
            links_df.to_csv(output_dir / "pcmci_links.csv", index=False)
            print(f"Saved results to {output_dir}")
    
    return results

def plot_causal_graph(results, var_names, tau_max, output_dir=None):
    """
    Plot the discovered causal graph
    """
    # Fix PCMCI+ results for plotting
    fixed_results = fix_pcmci_results_for_plotting(results, tau_max)
    
    graph = fixed_results['graph']
    val_matrix = fixed_results['val_matrix']
    
    print(f"Final graph shape for plotting: {graph.shape}")
    print(f"Final value matrix shape for plotting: {val_matrix.shape}")
    
    # Create the plot
    fig, axes = plt.subplots(1, graph.shape[2], figsize=(5*graph.shape[2], 5))
    
    # Handle single subplot case
    if graph.shape[2] == 1:
        axes = [axes]
    
    # Plot graph for each lag
    for lag in range(graph.shape[2]):
        lag_label = "Instantaneous" if lag == 0 else f"Lag {lag}"
        tp.plot_graph(
            graph=graph[:, :, lag],
            val_matrix=val_matrix[:, :, lag],
            var_names=var_names,
            fig_ax=(fig, axes[lag]),  # Pass as tuple (fig, ax)
            title=f'Causal Graph ({lag_label})'
        )
    
    plt.tight_layout()
    
    if output_dir is not None:
        plt.savefig(Path(output_dir) / "causal_graph.png", dpi=300, bbox_inches='tight')
        print(f"Saved causal graph plot to {output_dir}/causal_graph.png")
    
    plt.show()

def analyze_mode_relationships(data, results, var_names):
    """
    Analyze relationships between modes
    """
    print(f"\nMode Analysis:")
    print(f"Data shape: {data.shape}")
    
    # Calculate correlations between modes
    correlations = np.corrcoef(data.reshape(-1, data.shape[2]).T)
    
    print(f"\nCorrelation matrix between modes:")
    print(pd.DataFrame(correlations, index=var_names, columns=var_names))
    
    # Count causal links per mode
    graph = results['graph']
    n_modes = len(var_names)
    
    print(f"\nCausal link counts per mode:")
    for i, mode in enumerate(var_names):
        incoming = np.sum(graph[i, :, :] != "")
        outgoing = np.sum(graph[:, i, :] != "")
        print(f"{mode}: {incoming} incoming, {outgoing} outgoing links")

def fix_pcmci_results_for_plotting(results, tau_max):
    """
    Fix PCMCI+ results to work with tigramite plotting
    PCMCI+ returns (N, N, tau_max) but tigramite expects (N, N, tau_max+1)
    """
    graph = results['graph']
    val_matrix = results['val_matrix']
    
    print(f"Original graph shape: {graph.shape}")
    print(f"Original value matrix shape: {val_matrix.shape}")
    print(f"tau_max: {tau_max}")
    
    # Check if we need to add instantaneous lag
    if graph.shape[2] == tau_max:
        print("Adding instantaneous lag (lag=0)...")
        
        n_modes = graph.shape[0]
        graph_fixed = np.zeros((n_modes, n_modes, tau_max + 1), dtype=object)
        val_matrix_fixed = np.zeros((n_modes, n_modes, tau_max + 1))
        
        # Set instantaneous lag to empty
        graph_fixed[:, :, 0] = ''
        val_matrix_fixed[:, :, 0] = 0.0
        
        # Copy lagged relationships
        graph_fixed[:, :, 1:] = graph
        val_matrix_fixed[:, :, 1:] = val_matrix
        
        print(f"Fixed graph shape: {graph_fixed.shape}")
        print(f"Fixed value matrix shape: {val_matrix_fixed.shape}")
        
        return {'graph': graph_fixed, 'val_matrix': val_matrix_fixed}
    else:
        print("Graph shape already correct")
        return results

def main():
    parser = argparse.ArgumentParser(description='Learn causal graph on MLP predictions using PCMCI+')
    parser.add_argument('--timestamp', type=str, default="2025_07_01_19_03_57", 
                       help='Timestamp of the test results to load')
    parser.add_argument('--n_modes', type=int, default=4, 
                       help='Number of modes for dimensionality reduction')
    parser.add_argument('--tau_max', type=int, default=3, 
                       help='Maximum time lag for PCMCI+')
    parser.add_argument('--pc_alpha', type=float, default=0.05, 
                       help='Significance level for PCMCI+')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Directory to save results')
    parser.add_argument('--plot', action='store_true', 
                       help='Plot the causal graph')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = OUTPUTS_DIR / f"causal_analysis_{args.timestamp}"
    
    print(f"Loading MLP predictions from timestamp: {args.timestamp}")
    print(f"Output directory: {args.output_dir}")
    
    # Step 1: Load MLP predictions
    outputs = load_mlp_predictions(args.timestamp)
    
    # Step 2: Reshape to temporal format
    predictions = reshape_predictions_to_temporal(outputs, future_timesteps=5, lat=40, lon=40)
    
    # Step 3: Reduce spatial dimensions to modes
    reduced_data, pca, scaler = reduce_spatial_dimensions(predictions, n_modes=args.n_modes)
    
    # Step 4: Learn causal graph using PCMCI+
    results = learn_causal_graph_pcmci(
        reduced_data, 
        tau_max=args.tau_max, 
        pc_alpha=args.pc_alpha,
        save_results=True,
        output_dir=args.output_dir
    )
    
    # Step 5: Analyze mode relationships
    var_names = [f"mode_{i}" for i in range(args.n_modes)]
    analyze_mode_relationships(reduced_data, results, var_names)
    
    # Step 6: Plot causal graph if requested
    if args.plot:
        plot_causal_graph(results, var_names, args.tau_max, args.output_dir)
    
    # Save additional information
    output_dir = Path(args.output_dir)
    np.save(output_dir / "reduced_data.npy", reduced_data)
    np.save(output_dir / "pca_components.npy", pca.components_)
    np.save(output_dir / "pca_explained_variance.npy", pca.explained_variance_ratio_)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    print(f"Key files:")
    print(f"  - pcmci_graph.npy: Causal graph structure")
    print(f"  - pcmci_links.csv: Detailed link information")
    print(f"  - reduced_data.npy: 4-mode time series data")
    print(f"  - causal_graph.png: Visual representation (if --plot used)")

if __name__ == "__main__":
    main() 