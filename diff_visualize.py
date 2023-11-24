import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Optional
from sklearn.decomposition import PCA

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--control-path", type=str)
    parser.add_argument("-t", "--test-path", type=str)
    return parser.parse_args()

def import_embeds(path: str, idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)
    toxic_embeds = []
    clean_embeds = []
    for e in data['toxic_embeds']:
        vec = e[0]
        if idx is not None:
            vec = vec[idx]
        toxic_embeds += [vec]
    for e in data['clean_embeds']:
        vec = e[0]
        if idx is not None:
            vec = vec[idx]
        clean_embeds += [vec]
    return np.array(toxic_embeds), np.array(clean_embeds)

def get_diffs(control_embeds: np.ndarray, test_embeds: np.ndarray) -> np.ndarray:
    assert test_embeds.shape == control_embeds.shape, "test embeds must be the same shape as control embeds"
    return test_embeds - control_embeds

def get_avg_correlation(correlation_mat: np.ndarray) -> float:
    np.fill_diagonal(correlation_mat, 0)
    return np.sum(correlation_mat) / ((correlation_mat.shape[0] * correlation_mat.shape[1]) - correlation_mat.shape[0])

if __name__ == "__main__":
    args = parse_args()

    control_toxic, control_clean = import_embeds(args.control_path)
    test_toxic, test_clean = import_embeds(args.test_path)

    diffs = get_diffs(control_toxic, test_toxic)
    norms = np.linalg.norm(diffs, axis=-1)
    normalized_diffs = diffs / norms[:, np.newaxis]
    diff_correlation = diffs @ diffs.T
    norm_diff_correlation = normalized_diffs @ normalized_diffs.T

    # standard_norms = []
    # for i, c in enumerate(control_toxic):
    #     for j in range(i + 1, control_toxic.shape[0]):
    #         standard_norms += [np.linalg.norm(control_toxic[j] - c)]

    avg_diff = np.mean(diffs, axis=0)
    print(f"L2-Norm of Average Difference Vector: {np.linalg.norm(avg_diff)}")
    print(f"Average of Difference Vector's L2-Norm: {np.mean(norms)}")
    # print(f"Average Norm of Differences in Control Toxic Set: {np.mean(standard_norms)}")

    avg_correlation = get_avg_correlation(diff_correlation)
    norm_avg_correlation = get_avg_correlation(norm_diff_correlation)
    print(f"Average Correlation: {avg_correlation}")
    print(f"Average Normalized Correlation (Cosine Sim): {norm_avg_correlation}")

    plt.imshow(diff_correlation)
    plt.title("Inner Product Correlation Matrix")
    plt.show()

    plt.imshow(norm_diff_correlation)
    plt.title("Normalized Inner Product Correlation Matrix (Cosine Sim)")
    plt.show()

    reduced_diffs = PCA(n_components=2).fit_transform(diffs)
    plt.scatter(reduced_diffs[:, 0], reduced_diffs[:, 1])
    plt.title("Reduced-Dim Representation of GCG Difference Vectors")
    plt.show()