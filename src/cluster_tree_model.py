import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score as ts_silhouette_score
import random
from tslearn.metrics import dtw, cdist_dtw
from itertools import product
import json
import sys
import os
sys.path.append(os.getcwd()+'/src') 
from data_process import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(parent_dir, "config.json")
 
def load_config():
    """
    Load configuration from the config.json file.

    Returns
    -------
    dict
        Configuration dictionary loaded from the JSON file.
    """
    with open(config_path, "r") as file:
        return json.load(file)
config = load_config()

class ClusterTreeModel():
    """
    Hierarchical clustering model for fleet charging demand profiles.

    This class builds a two-root hierarchical cluster tree using KMeans for
    coarse partitioning and DTW-based time-series KMeans for recursive
    subdivision. It exposes training, saving and prediction utilities that
    interact with a provided `dataloader` for persistence.
    """
    def __init__(self, dataloader, AC_model=True, DC_model=False, experiment_name=None, threshold_size=21):
        """
        Initialize the Cluster Tree Model for hierarchical time series clustering.

        Parameters
        ----------
        dataloader : Dataloader
            Dataloader instance for loading and saving data and models.
        AC_model : bool, optional
            Flag to indicate if this is an AC charging model. Default is True.
        DC_model : bool, optional
            Flag to indicate if this is a DC charging model. Default is False.
        experiment_name : str, optional
            Name of the experiment for organizing results. Default is None.
        threshold_size : int, optional
            Minimum cluster size threshold for stopping tree subdivision. Default is 21.
        """
        self.dataloader = dataloader
        self.final_clusters = []
        self.AC_model = AC_model
        self.DC_model = DC_model
        self.experiment_name = experiment_name
        self.threshold_size = threshold_size

    def set_seed(self, seed):
        """
        Set random seed for reproducibility.

        Parameters
        ----------
        seed : int
            Random seed value.
        """
        random.seed(seed)
        np.random.seed(seed)
    
    def _decide_per_voting(self, db_scores, ch_scores, silhouette_scores):
        """
        Determine optimal number of clusters by voting across multiple metrics.

        Uses Davies-Bouldin (lower is better), Calinski-Harabasz (higher is better),
        and Silhouette (higher is better) scores to vote on the best k value.

        Parameters
        ----------
        db_scores : list
            Davies-Bouldin scores for different k values.
        ch_scores : list
            Calinski-Harabasz scores for different k values.
        silhouette_scores : list
            Silhouette scores for different k values.

        Returns
        -------
        int
            Optimal number of clusters (k+2 where k is the winning index).
        """
        db_index = np.argmin(db_scores)
        ch_index = np.argmax(ch_scores)
        silhouette_index = np.argmax(silhouette_scores)
        votes = np.zeros(len(db_scores))
        votes[db_index] += 1
        votes[ch_index] += 1
        votes[silhouette_index] += 1
        return np.argmax(votes)+2

    def train(self):
        """
        Train the hierarchical cluster tree model.

        This method performs a two-stage clustering:
        1. Initial K-means clustering on weekday/weekend average features
        2. Recursive time-series clustering (DTW-based) on weekly averages

        The training creates a hierarchical tree structure that subdivides clusters
        until reaching the threshold size.

        Returns
        -------
        None
            Results are saved to disk via the dataloader.
        """
        in_array, csv_file_names = self.dataloader.get_in_array('weekday_weekend_average', AC_model=self.AC_model, DC_model=self.DC_model)
        # Sort the input array and file names alphabetically by file name
        sorted_indices = np.argsort(csv_file_names)
        in_array = in_array[sorted_indices]
        csv_file_names = np.array(csv_file_names)[sorted_indices]

        dim_kmean = 0
        if dim_kmean == 1:
            kmeans_array = np.empty((1, 1)) 
            for index in range(0, in_array.shape[0]):
                values_to_append = np.array([[np.max(in_array[index, 24:72])/np.max(in_array[index, 0:24])]])
                kmeans_array = np.concatenate((kmeans_array, values_to_append), axis=0)
        else:
            kmeans_array = np.empty((1, 2))  # Initialize an empty array with the right number of rows
            for index in range(0, in_array.shape[0]):
                values_to_append = np.array([[np.max(in_array[index, 0:24]), np.max(in_array[index, 24:72])]])

                
                # Append along axis 1
                kmeans_array = np.concatenate((kmeans_array, values_to_append), axis=0)
        kmeans_array = kmeans_array[1::]
        clusters, _, _, _, _, _ = self._cluster(k=2, kmeans_array=kmeans_array)

        # Filter the list to include only files with "kmeans" in the title
        in_array, csv_file_names = self.dataloader.get_in_array('weekly_average', AC_model=self.AC_model, DC_model=self.DC_model)
        sorted_indices = np.argsort(csv_file_names)
        in_array = in_array[sorted_indices]
        csv_file_names = np.array(csv_file_names)[sorted_indices]

        csv_file_names_1 = np.array(csv_file_names)[clusters==0]
        in_array_1 = in_array[clusters==0]
        self._cluster_tree(csv_file_names_1, in_array_1, 'weekly_average', csv_file_names_1, self.threshold_size, id='0')

        csv_file_names_2 = np.array(csv_file_names)[clusters==1]
        in_array_2 = in_array[clusters==1]
        self._cluster_tree(csv_file_names_2, in_array_2, 'weekly_average', csv_file_names_2, self.threshold_size, id='1')

        self._build_tree()
    
    def train_basemodel(self):
        id = 'base'
        in_array, csv_file_names = self.dataloader.get_in_array('weekday_weekend_average', AC_model=self.AC_model, DC_model=self.DC_model)
        # Sort the input array and file names alphabetically by file name
        sorted_indices = np.argsort(csv_file_names)
        in_array = in_array[sorted_indices]
        csv_file_names = np.array(csv_file_names)[sorted_indices]

        _, db_scores, ch_scores, silhouette_scores, _, _ = self._cluster_shape(k=None, in_array=in_array, kmax=10, filename='base')

        mean_index = self._decide_per_voting(db_scores=db_scores, ch_scores=ch_scores, silhouette_scores=silhouette_scores)

        clusters, _, _, _, variances, distances = self._cluster_shape(mean_index, in_array=in_array, kmax=None, filename='base')

        ids = [f'{id}{str(i)}' for i in np.unique(clusters)]
        self.dataloader.save_csv_files(clusters, csv_file_names, ids, variances, distances)


    def save_model(self, filename):
        """
        Save the trained cluster tree model to disk.

        Parameters
        ----------
        filename : str
            Name of the file to save the model to.

        Returns
        -------
        None
        """
        self.dataloader.save_model(self, 'cluster_models', filename)

    def _cluster(self, k, kmeans_array, kmax=None, scale=False):
        """
        Perform K-means clustering and compute evaluation metrics.

        Parameters
        ----------
        k : int or None
            Number of clusters. If None, will search from 2 to kmax.
        kmeans_array : np.ndarray
            Feature array for K-means clustering.
        kmax : int, optional
            Maximum number of clusters to try. Default is None.
        scale : bool, optional
            Whether to scale features by max value. Default is False.

        Returns
        -------
        clusters : np.ndarray
            Cluster assignments for each data point.
        db_scores : list
            Davies-Bouldin scores.
        ch_scores : list
            Calinski-Harabasz scores.
        silhouette_scores : list
            Silhouette scores.
        elbow_scores : list
            Within-cluster sum of squares (inertia) values.
        variances : list
            Within-cluster variances for each cluster.
        distances : list
            Average DTW distances within each cluster.
        """
        self.set_seed(42)
        db_scores =[]
        ch_scores = []
        silhouette_scores = []
        variances = []
        distances = []
        if scale:
            kmeans_array = kmeans_array / np.max(kmeans_array, axis=1, keepdims=True)
        if k is not None:
            kmin = k
            kmax = k+1
        else:
            kmin=2
        for k in range(kmin, kmax):
            model = KMeans(n_clusters=k)
            clusters = model.fit_predict(kmeans_array)
            self.dataloader.save_submodel(model, 'cluster_models', 'root_model.pkl')

            idx = np.argsort(clusters)
            sorted_kmeans_array = kmeans_array[idx,:]
            sorted_clusters = clusters[idx]

            db_scores.append(davies_bouldin_score(sorted_kmeans_array, sorted_clusters))
            ch_scores.append(calinski_harabasz_score(sorted_kmeans_array, sorted_clusters))
            silhouette_scores.append(silhouette_score(sorted_kmeans_array, sorted_clusters))
            variance_tuple = []
            distances_tuple = []
            for cluster_id in np.unique(clusters):
                cluster_data = sorted_kmeans_array[sorted_clusters == cluster_id]
                variance_tuple.append(np.mean(np.var(cluster_data, axis=0)))
                distances_tuple.append(np.mean(cdist_dtw(cluster_data, global_constraint="sakoe_chiba", sakoe_chiba_radius=1)))
            variances.append(variance_tuple)
            distances.append(distances_tuple)

        return clusters, db_scores, ch_scores, silhouette_scores, variances, distances
    
    def _davies_bouldin_dtw(self, X, labels, centers, sakoe_chiba_radius=1):
        """
        Compute the Davies-Bouldin Index for clustering results using Dynamic Time Warping (DTW) as the distance metric.
        The Davies-Bouldin Index is a metric for evaluating clustering algorithms, where a lower value indicates better clustering.
        This implementation uses DTW with a Sakoe-Chiba global constraint to measure distances between time series.
        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples, ...) containing the data points (e.g., time series).
        labels : np.ndarray
            Array of shape (n_samples,) containing the cluster labels assigned to each data point.
        centers : np.ndarray
            Array of shape (n_clusters, ...) containing the cluster centers (e.g., centroid time series).
        sakoe_chiba_radius : int, optional (default=1)
            The radius parameter for the Sakoe-Chiba global constraint in DTW.
        Returns
        -------
        float
            The Davies-Bouldin Index for the clustering result. Lower values indicate better clustering.
        Notes
        -----
        - Requires a `dtw` function compatible with the signature used in the code.
        - Assumes that `X`, `labels`, and `centers` are all compatible in terms of shape and content.
        """

        k = len(centers)
        S = []
        
        # Compute intra-cluster dispersion S_i for each cluster
        for i in range(k):
            cluster_points = X[labels == i]
            center = centers[i]
            # Average DTW distance from all points in cluster i to its center
            S_i = np.mean([
                dtw(
                    x.ravel(), center.ravel(),
                    global_constraint="sakoe_chiba",   # use Sakoe-Chiba constraint
                    sakoe_chiba_radius=sakoe_chiba_radius
                )
                for x in cluster_points
            ])
            S.append(S_i)

        R = []
        # Compute pairwise cluster similarity ratio R_ij
        for i in range(k):
            Ri = []
            for j in range(k):
                if i != j:
                    # DTW distance between cluster centers i and j
                    Mij = dtw(
                        centers[i].ravel(), centers[j].ravel(),
                        global_constraint="sakoe_chiba",
                        sakoe_chiba_radius=sakoe_chiba_radius
                    )
                    # Ratio of within-cluster dispersion to between-cluster distance
                    Rij = (S[i] + S[j]) / Mij
                    Ri.append(Rij)
            # Take the maximum R_ij for cluster i
            R.append(max(Ri))

        # Davies-Bouldin Index = average of R_i across all clusters
        return np.mean(R)
    
    def _calinski_harabasz_dtw(self, X, labels, centers, sakoe_chiba_radius=1):
        """
        Compute the Calinski-Harabasz index for clustering results using DTW.

        The Calinski-Harabasz index compares inter-cluster dispersion to
        intra-cluster dispersion. This implementation measures distances with
        DTW using an optional Sakoe-Chiba constraint.

        Parameters
        ----------
        X : np.ndarray
            Array of input time series data, shape (n_samples, ...).
        labels : np.ndarray
            Array of cluster labels for each sample, shape (n_samples,).
        centers : np.ndarray
            Array of cluster centers, shape (n_clusters, ...).
        sakoe_chiba_radius : int, optional
            Radius for the Sakoe-Chiba global constraint in DTW. Default is 1.

        Returns
        -------
        float
            Calinski-Harabasz index score (higher is better).

        Notes
        -----
        Requires a `dtw` function that supports the `global_constraint` and
        `sakoe_chiba_radius` arguments. Assumes integer labels from 0 to k-1.
        """

        n_samples = len(X)
        k = len(centers)

        # Compute intra-cluster dispersion (within-cluster sum of squares)
        SSW = 0.0
        for i in range(k):
            cluster_points = X[labels == i]
            center = centers[i]
            SSW += np.sum([
                dtw(
                    x.ravel(), center.ravel(),
                    global_constraint="sakoe_chiba",
                    sakoe_chiba_radius=sakoe_chiba_radius
                )
                for x in cluster_points
            ])

        # Compute global "center" as the average of all cluster centers
        global_center = np.mean(centers, axis=0)

        # Compute inter-cluster dispersion (between-cluster sum of squares)
        SSB = 0.0
        for i in range(k):
            n_i = np.sum(labels == i)
            SSB += n_i * dtw(
                centers[i].ravel(), global_center.ravel(),
                global_constraint="sakoe_chiba",
                sakoe_chiba_radius=sakoe_chiba_radius
            )

        # Calinski-Harabasz index formula
        ch_score = (SSB / (k - 1)) / (SSW / (n_samples - k))
        return ch_score

    def _cluster_shape(self, k, in_array, kmax=None, filename=None):
        """
        Perform time series clustering using DTW-based K-means.

        Clusters time series using Dynamic Time Warping (DTW) with Sakoe-Chiba
        constraint and computes various evaluation metrics.

        Parameters
        ----------
        k : int or None
            Number of clusters. If None, will search from 2 to kmax.
        in_array : np.ndarray
            Input time series data array.
        kmax : int, optional
            Maximum number of clusters to try. Default is None.
        filename : str, optional
            Filename for saving the model. Default is None.

        Returns
        -------
        clusters : np.ndarray
            Cluster assignments for each time series.
        db_scores : list
            Davies-Bouldin scores using DTW distance.
        ch_scores : list
            Calinski-Harabasz scores using DTW distance.
        silhouette_scores : list
            Silhouette scores using DTW distance.
        variances : list
            Within-cluster variances.
        distances : list
            Average DTW distances within each cluster.
        """
        self.set_seed(42)
        db_scores = []
        ch_scores = []
        silhouette_scores = []
        variances = []
        distances = []
        in_array_plot = in_array.copy()
        if kmax!=None:
            kmax+=1
        if k!=None:
            kmin = k
            kmax = k+1
        else:
            kmin = 2

        for k in range(kmin, kmax):
            model = TimeSeriesKMeans(n_clusters=k, metric="dtw", metric_params={"global_constraint":"sakoe_chiba", "sakoe_chiba_radius":1}, max_iter=50, n_init=1)
            clusters = model.fit_predict(in_array)
            self.dataloader.save_submodel(model, 'cluster_models', '{}_model.pkl'.format(filename))

            idx = np.argsort(clusters)

            in_array = np.array(in_array_plot)
            sorted_in_array = in_array[idx,:]
            sorted_clusters = clusters[idx]

            db_scores.append(self._davies_bouldin_dtw(X=sorted_in_array, labels=sorted_clusters, centers=model.cluster_centers_, sakoe_chiba_radius=1))
            ch_scores.append(self._calinski_harabasz_dtw(X=sorted_in_array, labels=sorted_clusters, centers=model.cluster_centers_, sakoe_chiba_radius=1))
            silhouette_scores.append(ts_silhouette_score(sorted_in_array, sorted_clusters, metric="dtw", metric_params={"global_constraint":"sakoe_chiba", "sakoe_chiba_radius":1}))

            variance_tuple = []
            distances_tuple = []
            for cluster_id in np.unique(clusters):
                cluster_data = sorted_in_array[sorted_clusters == cluster_id]
                variance_tuple.append(np.mean(np.var(cluster_data, axis=0)))
                distances_tuple.append(np.mean(cdist_dtw(cluster_data, global_constraint="sakoe_chiba", sakoe_chiba_radius=1)))
                # distances_tuple.append(np.mean([dtw(x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=1) for i, x in enumerate(cluster_data) for y in cluster_data[i+1:]]))
            variances.append(variance_tuple)
            distances.append(distances_tuple)
        
        return clusters, db_scores, ch_scores, silhouette_scores, variances, distances
    
    def _cluster_tree(self, names, in_array, timeframe, csv_file_names, stop=10, id=None):
        """
        Recursively build a hierarchical cluster tree.

        Creates a tree structure by recursively subdividing clusters until
        each cluster has fewer than 'stop' members.

        Parameters
        ----------
        names : np.ndarray
            Names/identifiers for the data points.
        in_array : np.ndarray
            Input time series data array.
        timeframe : str
            Timeframe descriptor for the data.
        csv_file_names : list
            Names of CSV files for the data points.
        stop : int, optional
            Minimum cluster size threshold for stopping subdivision. Default is 10.
        id : str, optional
            Cluster identifier string for tracking position in tree. Default is None.

        Returns
        -------
        None
            Results are saved to disk during tree construction.
        """
        self.set_seed(42)
        
        _, db_scores, ch_scores, silhouette_scores, _, _ = self._cluster_shape(k=None, in_array=in_array, kmax=5, filename=id)
        mean_index = self._decide_per_voting(db_scores=db_scores, ch_scores=ch_scores, silhouette_scores=silhouette_scores)

        clusters, _, _, _, variances, distances = self._cluster_shape(mean_index, in_array=in_array, kmax=None, filename=id)
        
        ids = [f'{id}{str(i)}' for i in np.unique(clusters)]
        self.dataloader.save_csv_files(clusters, names, ids, variances, distances)
        
        if all(len(clusters[clusters == c]) <= stop for c in np.unique(clusters)):
            return

        for i, c in enumerate(np.unique(clusters)):
            if len(clusters[clusters == c]) > stop:
                csv_file_names = names[clusters == c]
                self._cluster_tree(csv_file_names, in_array[clusters == c], timeframe, csv_file_names, stop, ids[i])

    def _build_tree(self, variance_improvement_root_0=0.85, variance_improvement_root_1=0.85):
        """
        Build and finalize the cluster tree from both root branches.

        Collects valid leaf nodes from both branches (root '0' and root '1')
        based on variance/distance improvement criteria.

        Parameters
        ----------
        variance_improvement_root_0 : float, optional
            Minimum improvement threshold (α) for branch '0'. Must be in (0, 1).
            Default is 0.85.
        variance_improvement_root_1 : float, optional
            Minimum improvement threshold (α) for branch '1'. Must be in (0, 1).
            Default is 0.85.

        Returns
        -------
        tuple
            Contains 8 elements (4 per root):
            - filtered_last_valid_nodes_0 : list of valid clusters from root '0'
            - positions_0 : dict of cluster positions for root '0'
            - df_sorted_0 : pd.DataFrame of sorted clusters for root '0'
            - root_0 : str, root identifier '0'
            - filtered_last_valid_nodes_1 : list of valid clusters from root '1'
            - positions_1 : dict of cluster positions for root '1'
            - df_sorted_1 : pd.DataFrame of sorted clusters for root '1'
            - root_1 : str, root identifier '1'

        Raises
        ------
        AssertionError
            If variance improvement parameters are not between 0 and 1.
        """
        assert 0 < variance_improvement_root_0 < 1, "Variance improvement must be between 0 and 1."
        assert 0 < variance_improvement_root_1 < 1, "Variance improvement must be between 0 and 1."

        filtered_last_valid_nodes_1, positions_1, df_sorted_1, root_1 = self._collect_tree(root='1', threshold_size=self.threshold_size, alpha=variance_improvement_root_1)
        filtered_last_valid_nodes_0, positions_0, df_sorted_0, root_0 = self._collect_tree(root='0', threshold_size=self.threshold_size, alpha=variance_improvement_root_0)

        return filtered_last_valid_nodes_0, positions_0, df_sorted_0, root_0, filtered_last_valid_nodes_1, positions_1, df_sorted_1, root_1

    def _calculate_positions(self, df, parent=None, depth=0, pos=0, positions=None, subtree_sizes=None):
        """
        Calculate positions for tree visualization using recursive layout.

        Computes (depth, horizontal_position) coordinates for each cluster
        in the tree, ensuring proper spacing based on subtree sizes.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing cluster hierarchy (Cluster, Parent columns).
        parent : str, optional
            Parent cluster identifier. Default is None.
        depth : int, optional
            Current depth in the tree. Default is 0.
        pos : float, optional
            Current horizontal position. Default is 0.
        positions : dict, optional
            Dictionary to store position mappings. Default is None.
        subtree_sizes : dict, optional
            Pre-computed subtree sizes for each cluster. Default is None.

        Returns
        -------
        dict
            Dictionary mapping cluster IDs to (depth, position) tuples.
        """
        if positions is None:
            positions = {}
        if subtree_sizes is None:
            subtree_sizes = {}
            for cluster in df['Cluster']:
                subtree_sizes[cluster] = self._calculate_subtree_size(df, cluster)
        children = df[df['Parent'] == parent]
        num_children = len(children)
        if num_children == 0:
            positions[parent] = (depth, pos)
            return positions
        total_size = sum(subtree_sizes[child['Cluster']] for _, child in children.iterrows())
        child_pos = pos - (total_size - 1) / 2
        for _, child in children.iterrows():
            child_size = subtree_sizes[child['Cluster']]
            positions[child['Cluster']] = (depth, child_pos + (child_size - 1) / 2)
            self._calculate_positions(df, child['Cluster'], depth + 1, child_pos + (child_size - 1) / 2, positions, subtree_sizes)
            child_pos += child_size
        return positions
    
    def _calculate_subtree_size(self, df, parent=None):
        """
        Calculate the total size of a subtree rooted at a parent node.

        Recursively counts all descendant nodes in the tree.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing cluster hierarchy (Cluster, Parent columns).
        parent : str, optional
            Parent cluster identifier. Default is None.

        Returns
        -------
        int
            Total number of nodes in the subtree (including the parent).
        """
        size = 1
        children = df[df['Parent'] == parent]
        for _, child in children.iterrows():
            size += self._calculate_subtree_size(df, child['Cluster'])
        return size
    
    def _get_paths(self, df, parent=None, path=None, paths=None):
        """
        Get all paths from root to leaf nodes in the cluster tree.

        Recursively traverses the tree to collect all root-to-leaf paths.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing cluster hierarchy (Cluster, Parent columns).
        parent : str, optional
            Current parent cluster identifier. Default is None.
        path : list, optional
            Current path being constructed. Default is None.
        paths : list, optional
            List to accumulate all complete paths. Default is None.

        Returns
        -------
        list of lists
            List where each element is a path (list of cluster IDs) from root to leaf.
        """
        if path is None:
            path = []
        if paths is None:
            paths = []
        children = df[df['Parent'] == parent]
        if children.empty:
            paths.append(path)
        else:
            for _, child in children.iterrows():
                self._get_paths(df, child['Cluster'], path + [child['Cluster']], paths)
        return paths
    
    def _sort_family_tree(self, df, parent=None):
        """
        Sort DataFrame in family tree order (depth-first traversal).

        Recursively orders clusters by traversing the tree in depth-first order.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing cluster hierarchy.
        parent : str, optional
            Parent cluster to start sorting from. Default is None.

        Returns
        -------
        pd.DataFrame
            Sorted DataFrame in depth-first tree traversal order.
        """
        sorted_df = pd.DataFrame()
        children = df[df['Parent'] == parent]
        for _, child in children.iterrows():
            sorted_df = pd.concat([sorted_df, pd.DataFrame([child])])
            sorted_df = pd.concat([sorted_df, self._sort_family_tree(df, child['Cluster'])])
        return sorted_df
    
    def _calc_var_root_cluster(self, root):
        """
        Calculate variance for a root cluster.

        Computes the mean variance of time series in a root cluster
        based on weekly average load profiles.

        Parameters
        ----------
        root : str
            Root cluster identifier ('0' or '1').

        Returns
        -------
        float
            Mean variance across time points for the root cluster.
        """
        in_array, csv_file_names = self.dataloader.get_in_array('weekly_average',self.AC_model, self.DC_model)
        ids = [name.split('_')[0] for name in csv_file_names]
        df = pd.read_csv(f'{self.dataloader.base_path}/results/{self.experiment_name}/csv_files/kmeans_2_profile_Cluster_{root}.csv', dtype={'Cluster': str})
        df = df.drop(columns=['mean', 'barycenter'])
        cluster_names = df.columns.to_list()
        selected_indices = [i for i, id in enumerate(ids) if id in cluster_names]
        selected_in_array = in_array[selected_indices]
        return np.mean(np.var(selected_in_array,  axis=0))
    
    def _calc_dist_root_cluster(self, root):
        """
        Calculate average DTW distance for a root cluster.

        Computes the mean pairwise DTW distance between time series
        in a root cluster.

        Parameters
        ----------
        root : str
            Root cluster identifier ('0' or '1').

        Returns
        -------
        float
            Mean DTW distance within the root cluster.
        """
        in_array, csv_file_names = self.dataloader.get_in_array('weekly_average',self.AC_model, self.DC_model)
        ids = [name.split('_')[0] for name in csv_file_names]
        df = pd.read_csv(f'{self.dataloader.base_path}/results/{self.experiment_name}/csv_files/kmeans_2_profile_Cluster_{root}.csv', dtype={'Cluster': str})
        df = df.drop(columns=['mean', 'barycenter'])
        cluster_names = df.columns.to_list()
        selected_indices = [i for i, id in enumerate(ids) if id in cluster_names]
        selected_in_array = in_array[selected_indices]
        return np.mean(cdist_dtw(selected_in_array, global_constraint="sakoe_chiba", sakoe_chiba_radius=1))

    def _collect_tree(self, root, threshold_size=21, alpha=0.8):
        """
        Collect valid leaf clusters from a tree branch based on quality criteria.

        Traverses paths in the tree and selects the deepest valid node in each path
        based on distance/variance improvement and size thresholds.

        Parameters
        ----------
        root : str
            Root branch identifier ('0' or '1').
        threshold_size : int, optional
            Minimum cluster size for validity. Default is 21.
        alpha : float, optional
            Maximum ratio of child distance to parent distance for improvement.
            Default is 0.8.

        Returns
        -------
        filtered_last_valid_nodes : list
            List of valid leaf cluster identifiers.
        positions : dict
            Dictionary mapping cluster IDs to visualization positions.
        df_sorted : pd.DataFrame
            Sorted DataFrame of the tree branch.
        root : str
            The root identifier passed in.
        """
        # Load the DataFrame
        df = pd.read_csv(f'{self.dataloader.base_path}/results/{self.experiment_name}/csv_files/cluster_sizes.csv', dtype={'Cluster': str})
        df.Parent = df.Cluster.str[:-1]
        if root == '0':
            variance = self._calc_var_root_cluster('0')
            distance = self._calc_dist_root_cluster('0')
            df = df[~df['Cluster'].str.startswith('1')]
            df = pd.concat([df, pd.DataFrame([{'Cluster': '0', 'Parent': '-1', 'Variance': variance, 'Distance': distance, 'Size': df[df.Parent=='0'].Size.sum()}])], ignore_index=True)
        elif root == '1':
            variance = self._calc_var_root_cluster('1')
            distance = self._calc_dist_root_cluster('1')
            df = df[~df['Cluster'].str.startswith('0')]
            df = pd.concat([df, pd.DataFrame([{'Cluster': '1', 'Parent': '-1', 'Variance': variance, 'Distance': distance, 'Size': df[df.Parent=='1'].Size.sum()}])], ignore_index=True)
        # Calculate the positions of each cluster
        positions = self._calculate_positions(df, '-1')

        # Sort the DataFrame like a family tree
        df_sorted = self._sort_family_tree(df, '-1')

        # Get all paths from the root to the leaf nodes
        paths = self._get_paths(df, root, [root])

        # Select the last valid node for each path
        last_valid_nodes = []
        for path in paths:
            last_valid_node = None
            for i, cluster in enumerate(path):
                child_variance = df_sorted[df_sorted['Cluster'] == cluster]['Distance'].values[0] # or variance
                parent_cluster = df_sorted[df_sorted['Cluster'] == cluster]['Parent'].values[0]
                if parent_cluster in df_sorted['Cluster'].values:
                    parent_variance = df_sorted[df_sorted['Cluster'] == parent_cluster]['Distance'].values[0]
                    cluster_size = df_sorted[df_sorted['Cluster'] == cluster]['Size'].values[0]
                    min_parent_variance = min(df_sorted[df_sorted["Cluster"]==x]['Distance'].values[0] for x in path[0:i])
                    if child_variance <= alpha * parent_variance and cluster_size >= threshold_size and child_variance <= min_parent_variance:
                        last_valid_node = cluster
            if last_valid_node:
                last_valid_nodes.append(last_valid_node)

        # print(last_valid_nodes)
        filtered_last_valid_nodes = [node for node in np.unique(last_valid_nodes) if not any(child.startswith(node) and child != node for child in np.unique(last_valid_nodes))]
        self.final_clusters.extend(filtered_last_valid_nodes)

        return filtered_last_valid_nodes, positions, df_sorted, root

    def predict(self, fleet_id):
        """
        Predict the cluster assignment for a fleet.

        Hierarchically predicts cluster by first determining the root cluster,
        then traversing the tree to find the appropriate leaf cluster.

        Parameters
        ----------
        fleet_id : str
            Identifier for the fleet to predict.

        Returns
        -------
        str or None
            Cluster identifier for the fleet. If an exact match is not found in
            `self.final_clusters`, the method attempts to return the closest
            valid cluster identifier.
        """
        try:
            df_root = pd.read_csv(f'{self.dataloader.data_path}/power_demands_analysis/weekday_weekend_average/{fleet_id}_weekday_weekend_average.csv')
        except FileNotFoundError:
            print(f"File not found for fleet {fleet_id}")
            try:
                df_cdrs = pd.read_csv(f'{self.dataloader.data_path}/power_demands/cdrs/{fleet_id}_cdrs.csv')
                charging_data, charging_data_dict = find_p_demand_for_each_charging_event_from_charging_data(df_cdrs, self.dataloader.base_path)
                p_demand_timeseries, charging_data = generate_p_demand_occupancy_timeseries(charging_data, charging_data_dict, 15, localize_dt=False)
                _ = process_weekly_data(p_demand_timeseries, self.dataloader.data_path, fleet_id)
                _ = process_weekday_weekend_data(p_demand_timeseries, self.dataloader.data_path, fleet_id)
                df_root = pd.read_csv(f'{self.dataloader.data_path}/power_demands_analysis/weekday_weekend_average/{fleet_id}_weekday_weekend_average.csv')
            except FileNotFoundError:
                print(f"File not found for fleet {fleet_id}")
                return None

        root_cluster = self._predict_root_cluster(df_root)

        df = pd.read_csv(f'{self.dataloader.data_path}/power_demands_analysis/weekly_average/{fleet_id}_weekly_average.csv')
        _ = self._predict_cluster(df, str(root_cluster))
        if CLUSTER in self.final_clusters:
            return CLUSTER
        else:
            cluster = self._find_closest_cluster(CLUSTER)
            if pd.isnull(cluster):
                cluster = self._find_closest_cluster(CLUSTER)
            return cluster

    def _predict_root_cluster(self, df):
        """
        Predict the root cluster (0 or 1) for a fleet.

        Uses weekday/weekend maximum power demand features to classify
        into one of two root branches.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 72 rows containing 'Average_power_demand' column
            (24 hours weekday + 48 hours weekend).

        Returns
        -------
        int
            Root cluster identifier (0 or 1).

        Raises
        ------
        AssertionError
            If DataFrame does not have exactly 72 rows.
        """
        assert df.shape[0] == 72, "The input DataFrame must have exactly 72 rows."
        values = df['Average_power_demand']/df['Average_power_demand'].max()
        values = [np.max(values[0:24]), np.max(values[24::])]
        cluster_name = 'root'
        model = self.dataloader.load_submodel('cluster_models', cluster_name)

        return model.predict(np.array(values).reshape(1,2))[0]
    
    def _predict_cluster(self, df, cluster_name):
        """
        Recursively predict cluster by traversing the tree.

        Uses trained DTW-based models at each level to navigate down
        the hierarchical tree until reaching a final cluster.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'Average_power_demand' column with weekly average.
        cluster_name : str
            Current cluster node identifier in the tree.

        Returns
        -------
        str
            Final cluster identifier. Note: the implementation assigns the
            final value to the global variable `CLUSTER`.
        """
        global CLUSTER 
        values = df['Average_power_demand']/df['Average_power_demand'].max()
        model = self.dataloader.load_ts_submodel('cluster_models', cluster_name)
        
        cluster = model.predict(values.values.reshape(1, -1, 1))[0]
        cluster_name_new = (str(cluster_name)+str(cluster)).replace('None','')
        if cluster_name_new in self.final_clusters:
            CLUSTER = cluster_name_new
            return CLUSTER
        else:
            all_models = self.dataloader.get_all_models('cluster_models')
            
            if cluster_name_new in all_models:
                cluster = self._predict_cluster(df, cluster_name_new)
            else:
                CLUSTER = cluster_name_new
            return cluster
    
    def _find_closest_cluster(self, cluster_name, depth=5, cluster_width_search_space=10):
        """
        Find the closest valid cluster when exact match is not in final_clusters.

        Searches for a valid cluster by progressively expanding the search space
        in terms of tree depth and cluster digit variations.

        Parameters
        ----------
        cluster_name : str
            Target cluster identifier that's not in final_clusters.
        depth : int, optional
            Maximum depth to search for neighbors. Default is 5.
        cluster_width_search_space : int, optional
            Range of digits to try for each position (0 to this value-1). Default is 10.

        Returns
        -------
        str or None
            Closest valid cluster identifier from final_clusters, or None if not found.
        """
        for z in range(1, depth):  # Depth of neighbor
            for index in range(len(cluster_name), 0, -1):
                name_temp = cluster_name[:index]
                # Generate search space dynamically based on z
                search_space = [name_temp + ''.join(map(str, digits)) for digits in product(range(cluster_width_search_space), repeat=z)]
                for search_string in search_space:
                    if search_string in self.final_clusters:
                        return search_string