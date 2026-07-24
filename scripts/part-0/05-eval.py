# graph_seed,mod_seed,n_nodes,difficulty,mod,k,n_edges,f1,f1_err,shd_score,shd_count,sid_score,sid_count,parent_aid_score,parent_aid_count,oset_aid_score,oset_aid_count

import pandas as pd

# read the CSV file
df = pd.read_csv("outputs/part-0/results_test.csv")

# print the first 5 rows
print(df.head())

# print the last 5 rows
print(df.tail())

# compute k ratio using k/n_edges if mod has edge in it, otherwise using k/n_nodes if mod has the word "node" in it
df["k_ratio"] = df.apply(lambda x: x["k"] / x["n_edges"] if "edge" in x["mod"] else x["k"] / x["n_nodes"], axis=1)
print(df["k_ratio"])

# k_ratio vs score, violin plot per n for each mod
