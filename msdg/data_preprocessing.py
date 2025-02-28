import numpy as np
import torch
from torch_geometric.data import Data


def create_causal_time_window(data, window_size, num_scales):
    """
    Transform raw MVTS into multiscale dynamic graphs using a causal time window.

    Args:
        data (np.ndarray): MVTS with shape (num_timesteps, num_variables)
        window_size (int): Size of the causal time window
        num_scales (int): Number of scales for multiscale representation

    Returns:
        List of torch_geometric.data.Data: Multiscale dynamic graphs
    """
    num_timesteps, num_variables = data.shape
    multiscale_graphs = []

    for scale in range(1, num_scales + 1):
        # Adjust window size for each scale
        scaled_window = window_size * scale
        edge_index = []
        edge_attr = []
        node_features = []

        # Create graph for each time step
        for t in range(scaled_window, num_timesteps):
            # Nodes represent variables at each timestep
            window_data = data[t - scaled_window:t]
            node_features.append(window_data)

            # Edges represent potential lagged dependencies
            for i in range(num_variables):
                for j in range(num_variables):
                    if i != j:  # Exclude self-loops
                        edge_index.append([i, j])
                        # Edge weight could be based on correlation (simplified)
                        edge_attr.append(np.corrcoef(window_data[:, i], window_data[:, j])[0, 1])

        node_features = torch.tensor(np.array(node_features), dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        multiscale_graphs.append(graph)

    return multiscale_graphs


def preprocess_data(data_path, window_size=5, num_scales=3):
    """
    Load and preprocess MVTS data.

    Args:
        data_path (str): Path to MVTS data
        window_size (int): Size of causal time window
        num_scales (int): Number of scales

    Returns:
        List of torch_geometric.data.Data: Preprocessed multiscale graphs
    """
    # Load data (assuming CSV with shape: timesteps x variables)
    data = np.genfromtxt(data_path, delimiter=',')
    return create_causal_time_window(data, window_size, num_scales)