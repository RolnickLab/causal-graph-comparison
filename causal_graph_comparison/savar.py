import os
from pathlib import Path
import pickle
import shutil

from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from causal_graph_comparison import CONFIGS_DIR


def generate_savar_data(experiment_params, data_params, savar_params, train_params):

    # Create data directory if it doesn't exist
    os.makedirs(data_params.data_dir, exist_ok=True)
    print(f"Data directory: {data_params.data_dir}")

    savar_name = f"modes_{experiment_params.d_z}-difficulty_{savar_params.difficulty}-seed_{experiment_params.random_seed}"
    #  check if pickle exists, if so, load datamodule
    pickle_path = Path(data_params.data_dir) / Path(f"{savar_name}.pkl")
    if pickle_path.exists():
        print("SAVAR data already exists, skipping generation")
        datamodule = pickle.load(open(pickle_path, "rb"))
    else:
        print("SAVAR data does not exist, generating...")

        # Create SAVAR datamodule
        datamodule = CausalClimateDataModule(
            tau=experiment_params.tau,
            future_timesteps=experiment_params.future_timesteps,
            num_months_aggregated=data_params.num_months_aggregated,
            train_val_interval_length=data_params.train_val_interval_length,
            in_var_ids=data_params.in_var_ids,
            out_var_ids=data_params.out_var_ids,
            train_years=data_params.train_years,
            train_historical_years=data_params.train_historical_years,
            test_years=data_params.test_years,  
            val_split=1 - train_params.ratio_train,  # fraction of testing to split for valdation
            seq_to_seq=data_params.seq_to_seq,  # if true maps from T->T else from T->1
            channels_last=data_params.channels_last,  # wheather variables come last our after sequence lenght
            train_scenarios=data_params.train_scenarios,
            test_scenarios=data_params.test_scenarios,
            train_models=data_params.train_models,
            batch_size=data_params.batch_size,
            eval_batch_size=data_params.eval_batch_size,
            num_workers=experiment_params.num_workers,
            pin_memory=experiment_params.pin_memory,
            load_train_into_mem=data_params.load_train_into_mem,
            load_test_into_mem=data_params.load_test_into_mem,
            verbose=experiment_params.verbose,
            seed=experiment_params.random_seed,
            seq_len=data_params.seq_len,
            data_dir=data_params.climateset_data,
            output_save_dir=data_params.data_dir,
            num_ensembles=data_params.num_ensembles,  # 1 for first ensemble, -1 for all
            lon=experiment_params.lon,
            lat=experiment_params.lat,
            num_levels=data_params.num_levels,
            global_normalization=data_params.global_normalization,
            seasonality_removal=data_params.seasonality_removal,
            reload_climate_set_data=data_params.reload_climate_set_data,
            icosahedral_coordinates_path=data_params.icosahedral_coordinates_path,
            # Below SAVAR data arguments
            time_len=savar_params.time_len,
            comp_size=savar_params.comp_size,
            noise_val=savar_params.noise_val,
            n_per_col=savar_params.n_per_col,
            difficulty=savar_params.difficulty,
            seasonality=savar_params.seasonality,
            overlap=savar_params.overlap,
            is_forced=savar_params.is_forced,
            plot_original_data=savar_params.plot_original_data,
        )
        datamodule.setup()

        # save config file to data directory
        savar_name = datamodule.train_val_input4mips.savar_name
        shutil.copy2(CONFIGS_DIR / "savar-picabu.json", os.path.join(data_params.data_dir, f"{savar_name}_config.json"))

        # save datamodule to pickle
        with open(pickle_path, "wb") as f:
            pickle.dump(datamodule, f)

    return datamodule