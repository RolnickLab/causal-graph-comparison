import numpy as np

def process_temporal_adjacency(A_temporal, num_nodes, tau):
    """
    Processes a temporal adjacency matrix to propagate connections and apply constraints.

    Args:
        A_temporal (np.array): The initial temporal adjacency matrix of shape 
                               (tau, num_nodes, num_nodes). A_temporal[t, i, j] 
                               represents the connection from node i at time lag t+1 
                               to node j at time 0.
        num_nodes (int): The number of nodes in the graph.
        tau (int): The number of past time steps considered.

    Returns:
        np.array: The final processed and flattened adjacency matrix.
        np.array: The initial flattened adjacency matrix before processing.
    """
    timesteps = tau + 1
    size = num_nodes * timesteps

    # ==========================================================================
    # Part 1: Initialize the Flattened Adjacency Matrix
    # ==========================================================================
    # Create an empty flattened matrix of size (N*T, N*T).
    # N = num_nodes, T = timesteps
    A_flat = np.zeros((size, size))

    # Populate the flattened matrix with the initial connections.
    # According to the problem, these are connections from a node `i` at a past
    # time `t_lag` to a node `j` at the current time `t=0`.
    # `A_temporal[t, i, j]` corresponds to a lag of `t+1`.
    for t_lag_idx, i, j in np.argwhere(A_temporal > 0):
        val = A_temporal[t_lag_idx, i, j]
        
        # The source time is the lag (t_lag_idx + 1)
        time_src = t_lag_idx + 1
        node_src = i
        
        # The destination is always time 0 initially
        time_dst = 0
        node_dst = j

        # Calculate the flat indices
        # row_idx = source, col_idx = destination
        row_idx = node_src * timesteps + time_src
        col_idx = node_dst * timesteps + time_dst
        
        A_flat[row_idx, col_idx] = val

    # Keep a copy of the initial state for comparison
    initial_A_flat = np.copy(A_flat)

    # ==========================================================================
    # Part 2: Propagate Connections Backwards in Time
    # ==========================================================================
    # This step propagates the connections while preserving the time lag.
    # If x_i(t_s) -> x_j(t_d), we add connections x_i(t_s+k) -> x_j(t_d+k).
    
    # Iterate over all possible source nodes and times
    for node_src in range(num_nodes):
        # Original connections are from t > 0
        for time_src in range(1, timesteps):
            # Iterate over all possible destination nodes
            for node_dst in range(num_nodes):
                # Check for an initial connection to time=0
                time_dst_initial = 0
                row_initial = node_src * timesteps + time_src
                col_initial = node_dst * timesteps + time_dst_initial
                
                value = initial_A_flat[row_initial, col_initial]
                
                if value > 0:
                    # Propagate this connection forward in time
                    for k in range(1, timesteps):
                        new_time_src = time_src + k
                        new_time_dst = time_dst_initial + k
                        
                        # Stop if either the new source or dest time is out of bounds
                        if new_time_src >= timesteps or new_time_dst >= timesteps:
                            break
                        
                        # Calculate new indices and set the value
                        row_new = node_src * timesteps + new_time_src
                        col_new = node_dst * timesteps + new_time_dst
                        A_flat[row_new, col_new] = value

    # ==========================================================================
    # Part 3: Apply Causality Constraints
    # ==========================================================================
    # To enforce causality, a connection can only go from a later time point
    # to an earlier one (e.g., t_src > t_dst). We must zero out any connection
    # where the source time is less than or equal to the destination time.
    # This prevents instantaneous connections (t_src = t_dst) and connections
    # that go forward in time (t_src < t_dst).

    # Create a vector representing the time step for each row/column
    time_indices = np.arange(size) % timesteps
    
    # Create a boolean mask where connections are invalid (t_src <= t_dst)
    # We use broadcasting to compare every source time with every dest time.
    invalid_mask = time_indices[:, np.newaxis] <= time_indices[np.newaxis, :]
    
    # Apply the mask to zero out all invalid connections
    A_flat[invalid_mask] = 0
                
    return A_flat, initial_A_flat


def print_matrix_with_labels(matrix, num_nodes, tau, title):
    """Helper function to print the matrix with readable labels."""
    timesteps = tau + 1
    labels = [f"x{n}t{t}" for n in range(num_nodes) for t in range(timesteps)]
    
    print(f"\n===== {title} =====")
    # Print header
    header = "      " + " ".join(f"{l:<4}" for l in labels)
    print(header)
    print("    -" + "-" * (len(labels) * 5))
    
    # Print rows
    for i, row in enumerate(matrix):
        row_str = " ".join(f"{int(val):<4}" for val in row)
        print(f"{labels[i]:<5}| {row_str}")


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    # Define the parameters of your temporal graph
    NUM_NODES = 4
    TAU = 5 # 5 previous time steps

    # Create a sample temporal adjacency matrix of shape (tau, num_nodes, num_nodes)
    # This is the input you would provide.
    # For this example, let's create some specific connections:
    # 1. Node 1 -> Node 0 with a lag of 1 (from t=1 to t=0)
    # 2. Node 2 -> Node 3 with a lag of 3 (from t=3 to t=0)
    # 3. Node 0 -> Node 2 with a lag of 5 (from t=5 to t=0)
    A_temporal_input = np.zeros((TAU, NUM_NODES, NUM_NODES))
    A_temporal_input[0, 1, 0] = 1  # Lag 1: x1(t-1) -> x0(t) is a connection from t=1 to t=0
    A_temporal_input[2, 2, 3] = 1  # Lag 3: x2(t-3) -> x3(t) is a connection from t=3 to t=0
    A_temporal_input[4, 0, 2] = 1  # Lag 5: x0(t-5) -> x2(t) is a connection from t=5 to t=0

    # Process the matrix
    final_matrix, initial_matrix = process_temporal_adjacency(A_temporal_input, NUM_NODES, TAU)

    # Print the results for verification
    print_matrix_with_labels(initial_matrix, NUM_NODES, TAU, "Initial Flattened Matrix (Connections to t=0 only)")
    print_matrix_with_labels(final_matrix, NUM_NODES, TAU, "Final Processed Matrix (Propagated and Constrained)")

