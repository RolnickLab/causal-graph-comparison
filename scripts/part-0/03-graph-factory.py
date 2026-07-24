import pandas as pd
import numpy as np
from pathlib import Path

from causal_graph_comparison.part_0 import SyntheticGraphFactory, GraphModifier, score_pair
from causal_graph_comparison.graph_utils import binarize_array
from configs.config import PathsConfig, DataConfig

PathsConfig.outputs_dir.mkdir(parents=True, exist_ok=True)


modifier = GraphModifier()
rows = []  # collect in memory; write CSV once at the end
incomplete = 0

for n_nodes in DataConfig.number_of_nodes:
    for diff in DataConfig.difficulty:
        for graph_seed in range(DataConfig.num_graph_seeds):
            print(("--------------------"))
            print(f"Generating graph for n_nodes={n_nodes}, difficulty={diff}, graph_seed={graph_seed}")
            factory = SyntheticGraphFactory(n_nodes=n_nodes, difficulty=diff, max_time_steps=5)
            gt = factory.generate(seed_graph=graph_seed)
            n_edges = np.count_nonzero(gt)
            print(f"n_edges: {n_edges}")
            binarized_gt = binarize_array(gt)
            print(f"binarized_gt: {binarized_gt}")

            ks_edges = list({max(1, int(round(val * n_edges))) for val in [1/n_edges, 2/n_edges, 3/n_edges, 1/2, 1/4, 1/8, 1/16, 1/64]})
            ks_nodes = list({max(1, int(round(val * n_nodes))) for val in [1/n_nodes, 2/n_nodes, 3/n_nodes, 1/2, 1/4, 1/8, 1/16, 1/64]})
            print(f"ks_edges: {ks_edges}")
            print(f"ks_nodes: {ks_nodes}")

            # (mod_list, k_list) pairs — one loop body for edges and nodes
            for mods, ks in (
                (DataConfig.edge_modifications, ks_edges),
                (DataConfig.node_modifications, ks_nodes),
            ):
                for mod in mods:
                    for k in ks:
                        for mod_seed in range(DataConfig.num_mod_seeds):
                            print(f"Applying {mod} operation with k={k} and seed={mod_seed}")
                            try:
                                mod_graph, gt_graph = modifier.apply(binarized_gt, operation=mod, k=k, seed=mod_seed, mod_val=-1)
                            except ValueError as e:
                                print(f"Error applying {mod} operation with k={k} and seed={mod_seed}: {e}")
                                incomplete += 1
                                continue
                            print(f"mod_graph: {mod_graph}")
                            f1, shd_score, sid_score, parent_aid_score, oset_aid_score = score_pair(
                                gt_graph, mod_graph
                            )
                            f1_inv = 1 - f1
                            print(f"f1: {f1}, parent_aid_score: {parent_aid_score}")
                            print("")

                            rows.append(
                                {
                                    "graph_seed": graph_seed,
                                    "mod_seed": mod_seed,
                                    "n_nodes": n_nodes,
                                    "difficulty": diff,
                                    "mod": mod,
                                    "k": k,
                                    "n_edges": n_edges,
                                    "f1": f1,
                                    "f1_err": f1_inv,
                                    "shd_score": shd_score[0],
                                    "shd_count": shd_score[1],
                                    "sid_score": sid_score[0],
                                    "sid_count": sid_score[1],
                                    "parent_aid_score": parent_aid_score[0],
                                    "parent_aid_count": parent_aid_score[1],
                                    "oset_aid_score": oset_aid_score[0],
                                    "oset_aid_count": oset_aid_score[1],
                                }
                            )
                        print("-----------")
            
pd.DataFrame(rows).to_csv(PathsConfig.out_csv, index=False)
print(f"Wrote {len(rows)} rows to {PathsConfig.out_csv}")
print(f"Incomplete: {incomplete}")