import os
import warnings
from climatem.config import dataParams, expParams, gtParams, rolloutParams, trainParams, modelParams, optimParams, plotParams, savarParams
from causal_graph_comparison import CONFIGS_DIR, SCRATCH_DIR, PROJECT_ROOT
from causal_graph_comparison.utils import get_json_config

def assert_args(
    experiment_params,
    data_params,
    gt_params,
    optim_params,
):
    """Raise errors or warnings if some args should not take some combination of values."""
    # raise errors if some args should not take some combination of values
    if gt_params.no_gt and (
        gt_params.debug_gt_graph or gt_params.debug_gt_z or gt_params.debug_gt_w
    ):
        raise ValueError(
            "Since no_gt==True, all other args should not use ground-truth values"
        )

    if experiment_params.latent and (
        experiment_params.d_z is None
        or experiment_params.d_x is None
        or experiment_params.d_z <= 0
        or experiment_params.d_x <= 0
    ):
        raise ValueError(
            "When using latent model, you need to define d_z and d_x with integer values greater than 0"
        )

    # string input with limited possible values
    supported_dataformat = ["numpy", "hdf5"]
    if data_params.data_format not in supported_dataformat:
        raise ValueError(
            f"This file format ({data_params.data_format}) is not \
                         supported. Supported types are: {supported_dataformat}"
        )
    supported_optimizer = ["sgd", "rmsprop"]
    if optim_params.optimizer not in supported_optimizer:
        raise ValueError(
            f"This optimizer type ({optim_params.optimizer}) is not \
                         supported. Supported types are: {supported_optimizer}"
        )

    # warnings, strange choice of args combination
    if not experiment_params.latent and gt_params.debug_gt_z:
        warnings.warn(
            "Are you sure you want to use gt_z even if you don't have latents"
        )
    if experiment_params.latent and (experiment_params.d_z > experiment_params.d_x):
        warnings.warn(
            "Are you sure you want to have a higher dimension for d_z than d_x"
        )

    return


def load_picabu_config():

    """Load and validate PICABU configuration parameters.

    Returns:
        Tuple containing experiment, data, ground truth, training, model, optimization, 
        plotting, SAVAR and rollout parameters loaded from config file.
    """

    params = get_json_config(CONFIGS_DIR / "savar-picabu.json")

    # get user's scratch directory:
    params["data_params"]["data_dir"] = params["data_params"]["data_dir"].replace("$SCRATCH", str(SCRATCH_DIR))
    print("new data path:", params["data_params"]["data_dir"])

    params["exp_params"]["exp_path"] = params["exp_params"]["exp_path"].replace("$SCRATCH", str(SCRATCH_DIR))
    print("new exp path:", params["exp_params"]["exp_path"])

    # get directory of project via current file (aka .../climatem/scripts/main_picabu.py)
    params["data_params"]["icosahedral_coordinates_path"] = params["data_params"][
        "icosahedral_coordinates_path"
    ].replace("$CLIMATEMDIR", str(PROJECT_ROOT / "climatem"))
    print("new icosahedron path:", params["data_params"]["icosahedral_coordinates_path"])

    params["data_params"]["climateset_data"] = params["data_params"][
        "climateset_data"
    ].replace("$SCRATCH", str(SCRATCH_DIR))
    print("new climateset data path:", params["data_params"]["climateset_data"])

    experiment_params = expParams(**params["exp_params"])
    data_params = dataParams(**params["data_params"])
    gt_params = gtParams(**params["gt_params"])
    train_params = trainParams(**params["train_params"])
    model_params = modelParams(**params["model_params"])
    optim_params = optimParams(**params["optim_params"])
    plot_params = plotParams(**params["plot_params"])
    savar_params = savarParams(**params["savar_params"])
    rollout_params = rolloutParams(**params["rollout_params"])

    # Handle SAVAR-specific parameters
    if "savar" in data_params.in_var_ids:
        experiment_params.lat = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.lon = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.d_x = int(experiment_params.lat * experiment_params.lon)
        plot_params.savar = True
    else:
        plot_params.savar = False

    # Save configuration to folder where data is generated
    data_dir = params["data_params"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    assert_args(experiment_params, data_params, gt_params, optim_params)

    # print ("all params:")
    # print(data_params.__dict__)
    # print(gt_params.__dict__)
    # print(train_params.__dict__)
    # print(model_params.__dict__)
    # print(optim_params.__dict__)
    # print(plot_params.__dict__)
    # print(savar_params.__dict__)

    return experiment_params, data_params, gt_params, train_params, model_params, optim_params, plot_params, savar_params, rollout_params

if __name__ == "__main__":
    load_picabu_config()