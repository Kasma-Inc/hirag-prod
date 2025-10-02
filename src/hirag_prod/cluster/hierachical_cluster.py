from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


class HierarchicalClustering:
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        linkage_method: str = "ward",
    ):
        """
        Initialize the HierarchicalCluster object.

        Args:
            n_clusters (int, optional): Number of clusters to find.
            distance_threshold (float, optional): Distance threshold for cluster merging.
            linkage_method (str): Linkage method to use for clustering.
        """
        if n_clusters is None and distance_threshold is None:
            raise ValueError(
                "Either n_clusters or distance_threshold must be provided."
            )
        if n_clusters is not None and distance_threshold is not None:
            raise ValueError(
                "Only one of n_clusters or distance_threshold should be set."
            )

        # Validate linkage method
        valid_linkage_methods = ["ward", "complete", "average", "single"]
        if linkage_method not in valid_linkage_methods:
            raise ValueError(f"linkage_method must be one of {valid_linkage_methods}")

        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage_method = linkage_method
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            distance_threshold=self.distance_threshold,
            linkage=self.linkage_method,
        )
        self.cluster_labels = None
        self.silhouette_score = None
        self.linkage_matrix = None

    def fit(self, feature_matrix):
        """
        Fit the hierarchical clustering model to the feature matrix.

        Args:
            feature_matrix (array-like): The feature matrix to cluster, shape (n_samples, n_features).

        Returns:
            tuple: (cluster_labels, model, silhouette_score, linkage_matrix)
                - cluster_labels: Array of cluster labels for each sample
                - model: Fitted AgglomerativeClustering model
                - silhouette_score: Silhouette coefficient for the clustering
                - linkage_matrix: Linkage matrix for dendrogram plotting

        Raises:
            ValueError: If feature_matrix is invalid or clustering fails.
        """
        if feature_matrix is None or len(feature_matrix) == 0:
            raise ValueError("feature_matrix cannot be None or empty")

        try:
            self.cluster_labels = self.model.fit_predict(feature_matrix)

            # Only compute silhouette score if we have more than one cluster
            n_unique_labels = len(np.unique(self.cluster_labels))
            if n_unique_labels > 1 and n_unique_labels < len(feature_matrix):
                self.silhouette_score = silhouette_score(
                    feature_matrix, self.cluster_labels
                )
            else:
                self.silhouette_score = None  # Cannot compute for single cluster or each point is its own cluster
        except Exception as e:
            raise ValueError(f"Clustering failed: {str(e)}")

        # Compute linkage matrix for dendrogram
        try:
            if self.linkage_method == "ward":
                self.linkage_matrix = linkage(feature_matrix, method="ward")
            else:
                # For non-ward methods, we can use feature matrix directly
                self.linkage_matrix = linkage(
                    feature_matrix, method=self.linkage_method, metric="euclidean"
                )
        except Exception as e:
            print(f"Warning: Could not compute linkage matrix: {str(e)}")
            self.linkage_matrix = None

        return (
            self.cluster_labels,
            self.model,
            self.silhouette_score,
            self.linkage_matrix,
        )

    def plot_dendrogram(
        self, figsize=(10, 7), title="Hierarchical Clustering Dendrogram"
    ):
        """
        Plot the dendrogram for the hierarchical clustering.

        Args:
            figsize (tuple): Figure size for the plot. Default is (10, 7).
            title (str): Title for the dendrogram plot.

        Returns:
            matplotlib.pyplot: The pyplot object for further customization.

        Raises:
            ValueError: If linkage matrix is not computed or fit() hasn't been called.
        """
        if self.linkage_matrix is None:
            raise ValueError(
                "Linkage matrix not computed. Call fit() first or linkage computation failed."
            )

        plt.figure(figsize=figsize)
        dendrogram(self.linkage_matrix)
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()
        return plt

    def get_cluster_info(self):
        """
        Get information about the clustering results.

        Returns:
            dict: Dictionary containing clustering information including:
                - n_clusters: Number of clusters found
                - silhouette_score: Silhouette coefficient (if computed)
                - cluster_labels: Array of cluster labels

        Raises:
            ValueError: If clustering hasn't been performed yet.
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering not performed yet. Call fit() first.")

        return {
            "n_clusters": len(np.unique(self.cluster_labels)),
            "silhouette_score": self.silhouette_score,
            "cluster_labels": (
                self.cluster_labels.copy() if self.cluster_labels is not None else None
            ),
        }
