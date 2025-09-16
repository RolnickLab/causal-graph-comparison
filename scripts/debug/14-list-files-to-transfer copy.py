from pathlib import Path

path = Path("~/scratch/cgc/outputs").expanduser()
from shutil import copy2


models = ["savar"]
difficulties = ["easy", "med_easy", "med_hard", "hard"]
modes = ["4", "16"]

files = [
    "flat_graph-binary-cd", # never says anything
    "flat_graph-binary-crl", # 
]


for model in models:
    for mode in modes:
        for difficulty in difficulties:
            prefer_linear = True
            if difficulty in ["med_easy", "med_hard"] and mode == "4":
                prefer_linear = False
                
            print(f"model: {model} difficulty: {difficulty} mode: {mode}")
            for file in files:
                # mlp-modes_4-diff_hard-seed_1-samples_1000-rollouts_1steps.npz
                file_list = list(path.glob(f"{model}-modes_{mode}-diff_{difficulty}-seed_1*{file}*.npz"))
                
                print("files found:", len(file_list))
                file_list_copy = list(file_list)
                if "crl" in file:
                    if prefer_linear:
                        file_list = [f for f in file_list if "linear" in f.name]
                    else:
                        file_list = [f for f in file_list if "linear" not in f.name]
                try:
                    latest_file = file_list[0]
                    print(f"Latest file: {latest_file}\n")
                except:
                    print ("No files matching, old file list:", file_list_copy)
                
                

                # Create eval subdirectory if it doesn't exist
                eval_dir = path / "eval"
                # eval_dir.mkdir(exist_ok=True)

                # Copy latest file to eval subdirectory if it exists
                if file_list:
                    print(f"Copying file...")
                    copy2(latest_file, eval_dir / latest_file.name)

            print("=============\n")
