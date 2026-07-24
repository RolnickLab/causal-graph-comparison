"""
Demonstration that `flatten_temporal_adjacency_graph2` behaves as intended.

Convention
----------
`flatten_temporal_adjacency_graph2` expects an ABSOLUTE-TIME indexed temporal graph:
axis-0 index t == absolute time t, and index 0 is the contemporaneous / lag-0 slice.

Our synthetic (savar) graphs are LAG-1 indexed: [time, child, parent] where index 0 == lag 1
(dict_to_matrix stores graph[..., abs(tau) - 1]). Before flattening we therefore prepend an
empty lag-0 slice -- exactly what `score_pair()` does. The helper `flatten_lag1()` below
mirrors that prepend + flatten so this demo exercises the real function under the real
calling convention.

A temporal edge "parent --lag L--> child" then unrolls, in the flattened static graph over
(node, absolute-time) pairs, into edges  x{parent}@t{s} -> x{child}@t{s - L}  for every window
position where both endpoints are in range. Two invariants must hold:
  (1) Causality: every flat edge goes strictly backward in lag, i.e. time(row) > time(col).
  (2) Faithfulness: a lag-L temporal edge parent->child produces EXACTLY (T + 1 - L) flat edges
      (T = number of lags), all with time(row) - time(col) == L, and every flat edge maps back
      to exactly one such temporal edge.

Run with:  /Users/cisaicu/miniconda3/envs/climenv/bin/python scripts/part-0/06-demo-flatten.py
"""

import numpy as np

from causal_graph_comparison.graph_utils import flatten_temporal_adjacency_graph2


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def flatten_lag1(temporal_lag1: np.ndarray) -> np.ndarray:
    """Flatten a LAG-1 indexed graph the way score_pair does: prepend an empty lag-0 slice."""
    time_steps, _, num_nodes = temporal_lag1.shape
    lag0 = np.zeros((1, num_nodes, num_nodes), dtype=temporal_lag1.dtype)
    padded = np.concatenate([lag0, temporal_lag1], axis=0)
    return flatten_temporal_adjacency_graph2(shape="time_child_parent", graph=padded)


def decode(idx: int, n_times: int) -> tuple[int, int]:
    """Map a flat-graph index to (node, absolute_time)."""
    return idx // n_times, idx % n_times


def pretty_print(flat: np.ndarray, num_nodes: int) -> None:
    """Print the flattened adjacency matrix with x{node}t{time} labels."""
    n_times = flat.shape[0] // num_nodes
    labels = [f"x{node}t{t}" for node in range(num_nodes) for t in range(n_times)]
    header = "         " + " ".join(f"{lab:>5}" for lab in labels)
    print(header)
    for i, row in enumerate(flat):
        cells = " ".join(f"{int(v):>5}" for v in row)
        print(f"{labels[i]:>7} | {cells}")
    print()


def verify(temporal_lag1: np.ndarray, flat: np.ndarray) -> None:
    """Assert the two invariants hold for `flat` given the LAG-1 temporal graph."""
    time_steps, _, num_nodes = temporal_lag1.shape
    n_times = time_steps + 1  # one extra absolute-time slice for the prepended lag-0

    assert flat.shape == (num_nodes * n_times, num_nodes * n_times), (
        f"expected shape {(num_nodes * n_times,) * 2}, got {flat.shape}"
    )

    # (1) Causality + every flat edge maps back to a real temporal edge.
    rows, cols = np.where(flat != 0)
    for r, c in zip(rows, cols):
        parent, p_time = decode(r, n_times)
        child, c_time = decode(c, n_times)
        lag = p_time - c_time
        assert lag >= 1, f"non-causal flat edge {r}->{c} (lag={lag})"
        assert temporal_lag1[lag - 1, child, parent] == 1, (
            f"flat edge x{parent}t{p_time}->x{child}t{c_time} has no source temporal edge"
        )

    # (2) Every temporal edge produces exactly (n_times - lag) flat edges.
    for t_lag, child, parent in zip(*np.where(temporal_lag1 != 0)):
        lag = t_lag + 1
        expected = n_times - lag
        count = 0
        for p_time in range(lag, n_times):
            c_time = p_time - lag
            if flat[parent * n_times + p_time, child * n_times + c_time]:
                count += 1
        assert count == expected, (
            f"temporal edge lag={lag} x{parent}->x{child}: found {count} flat edges, expected {expected}"
        )


def line(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Convention: why the lag-0 prepend is required.
# --------------------------------------------------------------------------- #
line("Convention: v2 is absolute-time indexed; a raw lag-1 graph loses its lag-1 edges")
g_conv = np.zeros((2, 1, 1), dtype=int)
g_conv[0, 0, 0] = 1  # lag-1 self-loop
raw = flatten_temporal_adjacency_graph2(shape="time_child_parent", graph=g_conv)
print(f"v2 on the RAW lag-1 graph        -> {int(raw.sum())} edges (lag-1 edge dropped: index 0 read as lag 0)")
print(f"flatten_lag1 (prepend + v2)      -> {int(flatten_lag1(g_conv).sum())} edges (correct)")
print("=> score_pair prepends the empty lag-0 slice for exactly this reason.")

# --------------------------------------------------------------------------- #
# Case A: a single node with a lag-1 self-loop (the edge that used to vanish).
# --------------------------------------------------------------------------- #
line("Case A: 1 node, lag-1 self-loop, T=2  ->  should NOT be dropped")
gA = np.zeros((2, 1, 1), dtype=int)
gA[0, 0, 0] = 1  # lag index 0 == lag 1: x0 --lag1--> x0
fA = flatten_lag1(gA)
print("temporal_graph[lag_index, child, parent]:")
print(gA, "\n")
print("flattened (rows=source/parent, cols=target/child):")
pretty_print(fA, num_nodes=1)
print("Expected: x0t1->x0t0 and x0t2->x0t1 (the stationary lag-1 dependence unrolled).")
verify(gA, fA)
print("Invariants OK.")

# --------------------------------------------------------------------------- #
# Case B: a cross-node lag-1 edge x1 -> x0.
# --------------------------------------------------------------------------- #
line("Case B: 2 nodes, x1 --lag1--> x0, T=2")
gB = np.zeros((2, 2, 2), dtype=int)
gB[0, 0, 1] = 1  # time_child_parent: child=0, parent=1, lag index 0 == lag 1
fB = flatten_lag1(gB)
print("flattened:")
pretty_print(fB, num_nodes=2)
print("Expected: x1t1->x0t0 and x1t2->x0t1 only.")
verify(gB, fB)
print("Invariants OK.")

# --------------------------------------------------------------------------- #
# Case C: a lag-2 self-loop, to show time-difference == lag.
# --------------------------------------------------------------------------- #
line("Case C: 1 node, lag-2 self-loop, T=3  ->  edges span exactly 2 time steps")
gC = np.zeros((3, 1, 1), dtype=int)
gC[1, 0, 0] = 1  # lag index 1 == lag 2
fC = flatten_lag1(gC)
pretty_print(fC, num_nodes=1)
print("Expected: x0t2->x0t0 and x0t3->x0t1 (time gap of exactly 2).")
verify(gC, fC)
print("Invariants OK.")

# --------------------------------------------------------------------------- #
# Case D: your real "easy" 4-node ground truth (3 lag-1 self-loops + 1 lag-5).
# --------------------------------------------------------------------------- #
line("Case D: easy 4-node GT (nodes 1,2,3 lag-1 self-loops; node 0 lag-5 self-loop)")
gD = np.zeros((5, 4, 4), dtype=int)
gD[0, 1, 1] = 1
gD[0, 2, 2] = 1
gD[0, 3, 3] = 1
gD[4, 0, 0] = 1
fD = flatten_lag1(gD)
print(f"total flat edges: {int(fD.sum())}  (expected 16 = 3 lag-1 loops * 5 + 1 lag-5 loop * 1)")
n_times_D = 6
for node, lag in [(1, 1), (2, 1), (3, 1), (0, 5)]:
    edges = [
        (decode(r, n_times_D), decode(c, n_times_D))
        for r, c in zip(*np.where(fD != 0))
        if r // n_times_D == node and c // n_times_D == node
    ]
    print(f"  node {node} (lag {lag}) -> {len(edges)} flat edges: {edges}")
verify(gD, fD)
print("Invariants OK.")

# --------------------------------------------------------------------------- #
# Case E: randomized stress test of the invariants.
# --------------------------------------------------------------------------- #
line("Case E: randomized stress test (100 random lag-1 temporal graphs)")
rng = np.random.default_rng(0)
for _ in range(100):
    T = int(rng.integers(1, 6))
    N = int(rng.integers(1, 5))
    g = (rng.random((T, N, N)) < 0.3).astype(int)
    f = flatten_lag1(g)
    verify(g, f)
print("All 100 random graphs satisfy both invariants. OK.")

print("\nAll demonstrations passed.")
