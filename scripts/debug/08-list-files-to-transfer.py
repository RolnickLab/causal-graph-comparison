from pathlib import Path

path = Path("~/scratch/cgc/outputs").expanduser()
from shutil import copy2


# models = ["mlp", "lstm", "cnn", "vae"]
models = ["vae"]
difficulties = ["easy", "med_easy", "med_hard", "hard"]
modes = ["4", "16"]
files = [
    "flat_graph-binary-cd", # lin/nonlin doesn't matter, but it can
    "flat_graph-binary-crl", # has to say lin/nonlin
    "samples_1000-rollouts_20steps",
    "samples_1000-rollouts_1steps",
    "interventions",
]


for model in models:
    for mode in modes:
        for difficulty in difficulties:
            print(f"model: {model} difficulty: {difficulty} mode: {mode}")
            for file in files:
                # mlp-modes_4-diff_hard-seed_1-samples_1000-rollouts_1steps.npz
                # for f in path.glob(f"{model}-modes_{mode}-diff_{difficulty}-seed_1*{file}*.npz"):
                #      print(f)
                #     print(f"Created: {f.stat().st_ctime}")
                # Get list of files matching pattern
                file_list = list(path.glob(f"{model}-modes_{mode}-diff_{difficulty}-seed_1*{file}*.npz"))
                
                # Find file with latest creation time
                if file_list:
                    print("files found:", len(file_list))
                    latest_file = max(file_list, key=lambda f: f.stat().st_ctime)
                    print(f"Latest file: {latest_file}\n")
                    # print(f"Created: {latest_file.stat().st_ctime}")

                # Create eval subdirectory if it doesn't exist
                eval_dir = path / "eval-vae"
                # eval_dir.mkdir(exist_ok=True)

                # Copy latest file to eval subdirectory if it exists
                if file_list:
                    copy2(latest_file, eval_dir / latest_file.name)

            print("=============\n")
