import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.linear_model import LogisticRegression


def evaluate_causality(causal_matrix, ground_truth):
    """
    Evaluate the inferred causality against ground truth.

    Args:
        causal_matrix (torch.Tensor): Inferred Granger causality matrix
        ground_truth (torch.Tensor): Ground truth causality matrix

    Returns:
        dict: Evaluation metrics
    """
    causal_matrix = causal_matrix.cpu().numpy().flatten()
    ground_truth = ground_truth.cpu().numpy().flatten()

    metrics = {
        "accuracy": accuracy_score(ground_truth, causal_matrix),
        "f1_score": f1_score(ground_truth, causal_matrix),
        "kappa": cohen_kappa_score(ground_truth, causal_matrix)
    }
    return metrics


def downstream_classification(data, causal_features, labels):
    """
    Use MSDG-extracted causal features for MVTS classification.

    Args:
        data (np.ndarray): Original MVTS data
        causal_features (torch.Tensor): Extracted causal features
        labels (np.ndarray): Classification labels

    Returns:
        dict: Classification metrics
    """
    # Flatten features for classification
    causal_features = causal_features.cpu().numpy().reshape(len(data), -1)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(causal_features, labels)
    predictions = clf.predict(causal_features)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_score": f1_score(labels, predictions, average='weighted'),
        "kappa": cohen_kappa_score(labels, predictions)
    }
    return metrics