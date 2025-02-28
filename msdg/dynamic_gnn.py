import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DynamicGNN(torch.nn.Module):
    """
    Dynamic Graph Neural Network for modeling multiscale causality.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(DynamicGNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, output_dim))

    def forward(self, graph):
        """
        Forward pass for a single graph.

        Args:
            graph (torch_geometric.data.Data): Input graph

        Returns:
            torch.Tensor: Node-level predictions
        """
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight=edge_attr)
            x = F.relu(x)
        x = self.layers[-1](x, edge_index, edge_weight=edge_attr)
        return x


def train_dynamic_gnn(model, graphs, num_epochs=100, lr=0.01):
    """
    Train the Dynamic GNN on multiscale graphs.

    Args:
        model (DynamicGNN): GNN model
        graphs (List[Data]): List of multiscale graphs
        num_epochs (int): Number of training epochs
        lr (float): Learning rate

    Returns:
        List[torch.Tensor]: Predictions for each scale
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    predictions = []

    for scale, graph in enumerate(graphs):
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = model(graph)
            # Simplified loss (e.g., reconstruction loss)
            loss = F.mse_loss(out, graph.x)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Scale {scale}, Epoch {epoch}, Loss: {loss.item()}")
        model.eval()
        with torch.no_grad():
            pred = model(graph)
            predictions.append(pred)
    return predictions