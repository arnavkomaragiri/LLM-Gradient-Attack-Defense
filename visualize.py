import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Optional, Union
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def import_embeds(path: str, idx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)
    toxic_embeds = []
    clean_embeds = []
    for e in data['toxic_embeds']:
        toxic_embeds += [e[0][idx]]
    for e in data['clean_embeds']:
        clean_embeds += [e[0][idx]]
    return np.array(toxic_embeds), np.array(clean_embeds)

def get_reducer(vecs: np.ndarray, mode: str = "pca", n_components: Optional[int] = None) -> Union[PCA, UMAP]:
    match mode:
        case "pca":
            return PCA(n_components=n_components).fit(vecs)
        case "umap":
            return UMAP().fit(vecs)
        case _:
            raise ValueError(f"unrecognized dimensionality reduction mode {mode}")

def dim_reduce(vecs: np.ndarray, mode: str = "pca") -> np.ndarray:
    match mode:
        case "pca":
            return PCA(n_components=2).fit_transform(vecs)
        case "umap":
            return UMAP().fit_transform(vecs)
        case _:
            raise ValueError(f"unrecognized dimensionality reduction mode {mode}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-d", "--dimensionality-reduction", type=str, choices=['pca', 'umap'], default="pca")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    toxic_embeds, clean_embeds = import_embeds(args.path)
    embeds = np.concatenate((toxic_embeds, clean_embeds), axis=0)
    labels = np.concatenate((np.ones(toxic_embeds.shape[0]), np.zeros(clean_embeds.shape[0])), axis=0)
    toxic_mask, clean_mask = (labels == 1), (labels == 0)
    # reduced_embeds = dim_reduce(embeds, args.dimensionality_reduction)
    reducer = get_reducer(embeds, mode=args.dimensionality_reduction)
    reduced_embeds = reducer.transform(embeds)[:, :2]

    plt.title("Low-Dimension Representation of AdvBench vs Dolly Embeddings")
    plt.scatter(reduced_embeds[toxic_mask, 0], reduced_embeds[toxic_mask, 1], label="advbench")
    plt.scatter(reduced_embeds[clean_mask, 0], reduced_embeds[clean_mask, 1], label="clean")
    plt.legend()
    plt.show()

    if args.dimensionality_reduction == "pca":
        variance_ratio = reducer.explained_variance_ratio_
        target_variance = 0.95
        variance_dim = (np.cumsum(variance_ratio) >= target_variance).nonzero()[0].min()
        print(f"Variance Ratio={target_variance * 100:.2f}% Dimension: {variance_dim}")

        var_embeds = reducer.transform(embeds)[:, :variance_dim]

        cluster = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(var_embeds)
        toxic_labels, clean_labels = cluster.labels_[labels == 1], cluster.labels_[labels == 0]

        toxic_unique, toxic_counts = np.unique(toxic_labels, return_counts=True)
        clean_unique, clean_counts = np.unique(clean_labels, return_counts=True)

        majority_toxic = toxic_unique[toxic_counts.argmax()]
        majority_clean = clean_unique[clean_counts.argmax()]

        true_toxic = (toxic_labels == majority_toxic).sum()
        true_clean = (clean_labels == majority_clean).sum()
        false_toxic = (toxic_labels != majority_toxic).sum()
        false_clean = (clean_labels != majority_clean).sum()

        conf_mat = np.array([[true_clean, false_clean], [false_toxic, true_toxic]])
        iou = (false_toxic + false_clean) / np.sum(conf_mat)
        print(f"IOU Score: {iou}")
        plt.matshow(conf_mat)

        # taken from https://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib
        for (y, x), value in np.ndenumerate(conf_mat):
            plt.text(x, y, f"{value:.2f}", va="center", ha="center")
        plt.xlabel("Actual Toxic")
        plt.ylabel("Predicted Toxic")
        plt.title("Toxicity Cluster Confusion Matrix")
        plt.show()