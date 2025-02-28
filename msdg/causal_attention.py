import torch
import torch.nn.functional as F


class CausalAttention(torch.nn.Module):
    """
    Causal Attention Mechanism for integrating predicted causal strengths.
    """

    def __init__(self, input_dim, attention_dim):
        super(CausalAttention, self).__init__()
        self.query = torch.nn.Linear(input_dim, attention_dim)
        self.key = torch.nn.Linear(input_dim, attention_dim)
        self.value = torch.nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.tensor(attention_dim, dtype=torch.float))

    def forward(self, predictions, potential_causality):
        """
        Apply causal attention to integrate predictions.

        Args:
            predictions (torch.Tensor): Predicted causal strengths
            potential_causality (torch.Tensor): Potential causality scores

        Returns:
            torch.Tensor: Attention-weighted predictions
        """
        Q = self.query(predictions)
        K = self.key(potential_causality)
        V = self.value(predictions)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        return out


def iterative_causal_inference(predictions, potential_causality, num_iterations=5):
    """
    Iteratively apply causal attention for inference.

    Args:
        predictions (List[torch.Tensor]): Predictions from each scale
        potential_causality (torch.Tensor): Initial potential causality
        num_iterations (int): Number of iterations

    Returns:
        torch.Tensor: Final causal strengths
    """
    attention = CausalAttention(input_dim=predictions[0].shape[-1], attention_dim=64)
    causal_strengths = torch.stack(predictions).mean(dim=0)  # Initial aggregation

    for _ in range(num_iterations):
        causal_strengths = attention(causal_strengths, potential_causality)
    return causal_strengths