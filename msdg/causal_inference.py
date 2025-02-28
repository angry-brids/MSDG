import torch


def compute_granger_causality(causal_strengths, threshold=0.1):
    """
    Compute Granger causality from causal strengths.

    Args:
        causal_strengths (torch.Tensor): Integrated causal strengths
        threshold (float): Threshold for determining causality

    Returns:
        torch.Tensor: Causal relationships (binary adjacency matrix)
    """
    # Normalize causal strengths
    causal_strengths = torch.sigmoid(causal_strengths)
    # Apply threshold to determine causality
    causal_matrix = (causal_strengths > threshold).float()
    return causal_matrix


def infer_causality(predictions, potential_causality):
    """
    Infer Granger causality using the MSDG method.

    Args:
        predictions (List[torch.Tensor]): Predictions from each scale
        potential_causality (torch.Tensor): Potential causality scores

    Returns:
        torch.Tensor: Granger causality matrix
    """
    causal_strengths = iterative_causal_inference(predictions, potential_causality)
    causal_matrix = compute_granger_causality(causal_strengths)
    return causal_matrix