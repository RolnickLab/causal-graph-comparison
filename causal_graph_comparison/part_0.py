import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random

from climatem.synthetic_data.generate_savar_datasets import create_links_coeffs
from climatem.synthetic_data.savar import dict_to_matrix
from climatem.synthetic_data.utils import check_stability
from causal_graph_comparison.graph_utils import binarize_array, flatten_temporal_adjacency_graph2

class SyntheticGraphFactory:
    """
    Factory for generating synthetic graphs.

    Args:
        n_nodes: number of nodes in the graph
        difficulty: difficulty of the graph
        max_time_steps: number of time steps in the time series
    """
    def __init__(self, n_nodes, difficulty, max_time_steps):
        self.n_nodes = n_nodes
        self.difficulty = difficulty
        self.max_time_steps = max_time_steps
        self.denom = 2

    def generate_links_coeffs(self, seed_graph: int) -> np.ndarray:
        np.random.seed(seed_graph)

        if self.n_nodes <= 4:
            prob = 1 # set prob = 1 if N <= 4, meaning every node is connected to every other node at one time step

        # This is the probabiliity of having a link between latent k and j, with k different from j. latents always have one link with themselves at a previous time.
        if self.difficulty == "easy":
            prob = 0 #N / N^2*tau
        elif self.difficulty == "med_easy":
            prob = 1 / (self.n_nodes - 1) #2N / N^2*tau
        elif self.difficulty == "med_hard":
            prob = 2 / (self.n_nodes - 1) #3N / N^2*tau
        elif self.difficulty == "hard":
            prob = 1 / self.denom #N + N*(N-1)/2 / N^2*tau

        links_coeffs = create_links_coeffs(n_modes=self.n_nodes, tau=self.max_time_steps, prob_edge=prob, difficulty=self.difficulty)
        return links_coeffs
        
    def ensure_stability(self, links_coeffs: dict) -> dict:
        # Check for stationarity of \PHI
        while True:
            try:
                check_stability(links_coeffs)
                break
            except AssertionError:
                # Regenerate the graph if stability fails
                links_coeffs = create_links_coeffs(n_modes=self.n_nodes, tau=self.max_time_steps, prob_edge=None, difficulty=self.difficulty)
        return links_coeffs

    def generate(self, seed_graph: int) -> np.ndarray:
        '''
        Generate a temporal adjacency matrix of shape [time, child, parent]
        '''
        links_coeffs = self.generate_links_coeffs(seed_graph)
        links_coeffs = self.ensure_stability(links_coeffs)
        graph = dict_to_matrix(links_coeffs) # [child (N), parent (N), lag-1(tau)]

        # rearrange matrix to [time, child, parent]
        graph = graph.transpose(2, 0, 1)

        return graph


class GraphModifier:
    def __init__(self):
        pass

    def apply(self, gt_graph: np.ndarray, operation: str, k: int, seed: int, mod_val = None) -> np.ndarray:
        '''
        Apply a random modification to the graph.

        Args:
            gt_graph: The binary ground truth graph to modify. Shape: [time, child, parent]
            operation: The operation to apply. Choices are
                - "delete_edges": delete k edges from the graph
                - "insert_edges": insert k edges to the graph
                - "delete_nodes": delete k nodes from the graph
                - "insert_nodes": insert k nodes to the graph
                - "modify_lag": modify the lag of k nodes
                - "shift_lag": shift the lag of all the nodes

                - TODO: "modify_weights": modify the weights of k edges 
                - TODO:"shift_weights": shift the weights of all the edges
                - TODO: when adding edges, add with appropriate weights within distribution of existing edge weights

            k: The number of edges to delete or insert. Cannot be greater than the number of edges in the graph.
            seed: The seed for the random number generator.

        Returns:
            The modified binary graph.
        '''
        rng = np.random.default_rng(seed)
        graph = gt_graph.copy()

        edges = self.get_edges(graph)
        non_edges = self.get_non_edges(graph)

        max_time_steps = self.count_time_steps(graph)
        num_nodes = self.count_nodes(graph)

        if operation == "delete_edges":
            # delete k edges from the graph
            graph = self.delete_edges(graph, k, rng, edges)

        elif operation == "insert_edges":
            # insert k edges to the graph
            graph = self.insert_edges(graph, k, rng, non_edges)

        elif operation == "delete_nodes":
            # delete k nodes from the graph
            graph = self.delete_nodes(graph, k, num_nodes, rng)

        elif operation == "insert_nodes":
            # insert k nodes to the graph
            graph, gt_graph = self.insert_nodes(graph, k, num_nodes, max_time_steps, rng)

        elif operation == "modify_lag":
            # modify the lag of k nodes
            if mod_val is None:
                raise ValueError("mod_val must be provided for modify_lag operation (modify lag by mod_val time steps)")
            graph = self.modify_lag(graph, k, edges, max_time_steps, rng, mod_val)

        elif operation == "randomly_modify_lag":
            # randomly modify the lag of k nodes
            graph = self.randomly_modify_lag(graph, k, edges, max_time_steps, rng)

        elif operation == "modify_weights":
            # modify the weights of k edges
            raise NotImplementedError("modify_weights is not implemented yet")
        elif operation == "shift_weights":
            # shift the weights of all the edges
            raise NotImplementedError("shift_weights is not implemented yet")
        else:
            raise ValueError(f"Invalid operation: {operation}")

        return graph, gt_graph

    def delete_edges(self, graph: np.ndarray, k: int, rng: np.random.Generator, edges: list[tuple[int, int, int]]) -> np.ndarray:
        '''
        Delete k edges from the graph.
        '''
        self.check_k(k, len(edges), "edges")

        idxs_to_delete = rng.choice(len(edges), size=k, replace=False)
        edges_to_delete = [edges[i] for i in idxs_to_delete]

        for edge in edges_to_delete:
            graph[edge[0], edge[1], edge[2]] = 0
        return graph

    def insert_edges(self, graph: np.ndarray, k: int, rng: np.random.Generator, non_edges: list[tuple[int, int, int]]) -> np.ndarray:
        '''
        Insert k edges to the graph.
        '''
        self.check_k(k, len(non_edges), "non-edges")

        idxs_to_insert = rng.choice(len(non_edges), size=k, replace=False)
        edges_to_insert = [non_edges[i] for i in idxs_to_insert]

        for edge in edges_to_insert:
            graph[edge[0], edge[1], edge[2]] = 1
        return graph

    def delete_nodes(self, graph: np.ndarray, k: int, num_nodes: int, rng: np.random.Generator) -> np.ndarray:
        '''
        Delete k nodes from the graph by zeroing out all rows and columns corresponding to the nodes
        '''
        self.check_k(k, num_nodes, "nodes")
        
        nodes_to_delete = rng.choice(num_nodes, size=k, replace=False)
        for node in nodes_to_delete:
            graph[:, node, :] = 0
            graph[:, :, node] = 0
        return graph
        
    def insert_nodes(self, graph: np.ndarray, k: int, num_nodes: int, max_time_steps: int, rng: np.random.Generator) -> np.ndarray:
        '''
        Insert k nodes to the graph. 
        Return the new graph with the inserted nodes & adjust ground truth graph accordingly.
        '''

        new_graph = graph.copy()

        for _ in range(k):
            n = new_graph.shape[1] 
            temp_graph = np.zeros((new_graph.shape[0], n + 1, n + 1))
            temp_graph[:, :n, :n] = new_graph

            # eligible edges are in the new column & row added to the graph
            eligible_edges = [(i, n) for i in range(n)] + [(n, i) for i in range(n)]
            eligible_edges = list(set(eligible_edges))

            random_time_step = rng.integers(0, max_time_steps)
            new_child, new_parent = eligible_edges[rng.choice(len(eligible_edges))]

            temp_graph[random_time_step, new_child, new_parent] = 1
            new_graph = temp_graph.copy()

        new_gt = np.zeros_like(new_graph)
        new_gt[:, :num_nodes, :num_nodes] = graph
        
        return new_graph, new_gt

    
    def modify_lag(self, graph: np.ndarray, k: int, edges: list[tuple[int, int, int]], max_time_steps: int, rng: np.random.Generator, mod_val: int) -> np.ndarray:
        '''
        Modify the lag of k nodes by mod_val number of time steps uniformly.
        '''
        self.check_k(k, len(edges), "edges")
        
        idxs_to_modify = rng.choice(len(edges), size=k, replace=False)
        edges_to_modify = [edges[i] for i in idxs_to_modify]

        for edge in edges_to_modify:
            new_lag = max(0, min(edge[0] + mod_val, max_time_steps - 1))

            # make sure new_lag is not already taken
            if graph[new_lag, edge[1], edge[2]] != 0:
                raise ValueError(f"Cannot modify lag of edge {edge} to time step{new_lag} because it is already taken")
            else:
                graph[new_lag, edge[1], edge[2]] = 1

            graph[edge[0], edge[1], edge[2]] = 0
        
        return graph

    def randomly_modify_lag(self, graph: np.ndarray, k: int, edges: list[tuple[int, int, int]], max_time_steps: int, rng: np.random.Generator) -> np.ndarray:
        '''
        Randomly modify the lag of k edges.
        '''
        self.check_k(k, len(edges), "edges")
        idxs_to_modify = rng.choice(len(edges), size=k, replace=False)
        edges_to_modify = [edges[i] for i in idxs_to_modify]

        for edge in edges_to_modify:
            # sample eligible new lags for each edge
            eligible = []

            for i in range(max_time_steps):
                if graph[i, edge[1], edge[2]] == 0:
                    eligible.append(i)
            
            if len(eligible) == 0:
                raise ValueError(f"Cannot modify lag of edge {edge} because all other time steps are already taken")
            
            new_lag = rng.choice(eligible)

            graph[edge[0], edge[1], edge[2]] = 0
            graph[new_lag, edge[1], edge[2]] = 1

        return graph


    def get_edges(self, graph: np.ndarray) -> list[tuple[int, int, int]]:
        # get the edges of the graph
        edges = []
        time_idx, child_idx, parent_idx = np.where(graph != 0)
        edges.extend(list(zip(time_idx, child_idx, parent_idx)))
                   
        return edges

    def get_non_edges(self, graph: np.ndarray) -> list[tuple[int, int, int]]:
        # get the non-edges of the graph
        non_edges = []
        time_idx, child_idx, parent_idx = np.where(graph == 0)
        non_edges.extend(list(zip(time_idx, child_idx, parent_idx)))
        return non_edges

    def count_nodes(self, graph: np.ndarray) -> int:
        # count the number of nodes in the graph
        return graph.shape[1]

    def count_time_steps(self, graph: np.ndarray) -> int:
        # count the number of time steps in the graph
        return graph.shape[0]

    def check_k(self, k: int, max_val: int, operation: str) -> bool:
        '''
        Check if k is valid for the graph.
        '''
        if k < 1:
            raise ValueError(f"Must apply at least one modification")
        if k > max_val:
            raise ValueError(f"Cannot apply {k} modifications to a graph with {max_val} {operation}")
        return True

# graph = binarize_array(graph)
# graph = flatten_temporal_adjacency_graph2(shape="time_child_parent", graph=graph)


    
# def generate_dataset(self, n_graphs: int, seed_graphs: list[int]) -> list[np.ndarray]:
#     graphs = []
#     for seed_graph in seed_graphs:
#         graph = self.generate(seed_graph)
#         graphs.append(graph)
#     return graphs



    

    
