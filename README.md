# Causal graph comparison 

## Installation

```python
pip install -e . # Install the package in editable mode
```

## Usage instructions

## TODO
- [x] Make MLP into reusable module
- [ ] Replace all vars with config

- [x] subsampling middle px of modes
- [x] figure out if test loader is sequential 
- [ ] implement VAE
- [x] implement LSTM
- [x] implement MLP
- [x] run inference on PICABU (rolloutbf.py)
- [ ] run models with multiple seeds
- [x] show that we can recover ground truth graph using pcmci on 1000 time steps
- [ ] show that we can recover ground truth graph using pcmci on 50 time steps

- [ ] create inference script to run teacher forcing so we learn graph on input/ouptut pairs
 
- [x] run PCMCI+ from tigris package on mlp output
- [x] run RMSE on mlp output vs ground truth output
- [ ] run causal graph comparison on mlp output vs ground truth output SAVAR

- [ ] Get targets  for picabu outputs (so i have something to plot against)

- [x] Make sure you set tau_min=1 in PCMCI
- [x] Run PCMCI “hack” on the savar data (GT) to make sure you recover the correct causal graph
- [x] Instead of setting threshold on p-values to 0.05, order p-values of MCI tests and take the K best where K is the GT number of edges

Set alpha = 0.0001 in PCMCI 
Set n_gt_connections to what the GT number of connections is  
After running PCMCI, run:

    graph = results["graph"]
    graph[
        results["val_matrix"]
        < np.abs(results["val_matrix"].flatten()[results["val_matrix"].flatten().argsort()[::-1][n_gt_connections - 1]])
    ] = ""

### Run PCMCI + “hack PCMCI” on MLP/LSTM/PICABU
- [ ] Run on both the autoregressive rollout and on “GT inputs / model output” pairs

### PICABU:
- [ ] Run an autoregressive rollout without the Bayesian filter. You would have to code the autoregressive rollout similarly to MLP / LSTM
- [ ] Run the bayesian filter and make sure you can recover the corresponding target time series (as initial conditions are taken at random rn)

### Non-causal PICABU:
- [ ] Run PICABU after setting the following parameters in the json file: “sparsity_upper_threshold”=1; “ortho_h_threshold”=1000, “ortho_mu_init”=1e-5, “ortho_mu_mult_factor”=1. This set of param will basically deactivate any causal constraint in PICABU

### Uncertainty analysis (lower priority):
- [ ] Run each model with multiple initial conditions --> Are the causal graphs constant across initial conditions?
- [ ] Run each model with multiple random seeds --> Are the causal graphs constant across random seeds?
- [ ] Need to find a way to quantify the variability among causal graphs (maybe it’s better here to use the distance between results["val_matrix"] i.e. the corresponding p-values rather than the resulting graph with alpha = 0.05

