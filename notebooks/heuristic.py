import os
import sys
sys.path.append("/Users/antoine/Desktop/CS/CS-COURS/3A/IA/SDP/cs-sdp-2023-24/python")
sys.path.append("/Users/antoine/Desktop/CS/CS-COURS/3A/IA/SDP/cs-sdp-2023-24/data")

# import pickles 
import numpy as np
from gurobipy import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from metric import PairsExplained
from models import *
from data import Dataloader


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.model = self.instantiate()
        self.n_pieces = n_pieces
        self.n_clusters = n_clusters
        self.x_abs = None
        self.eps = 0.0001

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        
        model = Model("Heuristic")

        return model

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        #n_samples, n_features = X.shape

        all_features = np.concatenate([X, Y])

        PCA_data_preferences = PCA(n_components= self.n_clusters).fit_transform(all_features)
        PCA_cluster_clients = KMeans(n_clusters= self.n_clusters).fit(PCA_data_preferences)
        
        
        
        return (PCA_cluster_clients, PCA_data_preferences)

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return
    

    ### VISUALISATION



data_loader = Dataloader("/Users/antoine/Desktop/CS/CS-COURS/3A/IA/SDP/cs-sdp-2023-24/data/dataset_10") # Specify path to the dataset you want to load
X, Y = data_loader.load()

parameters = {"n_pieces": 5, "n_clusters" :3} # Can be completed
model = HeuristicModel(**parameters)
PCA_cluster_clients, PCA_data_preferences = model.fit(X,Y)

figure = plt.figure(figsize = (15,8))

# Visualisation des clusters en utilisant les deux premières composantes principales de la PCA
plt.subplot(2,2,1)
plt.scatter(PCA_data_preferences[:, 0], PCA_data_preferences[:, 1], c=PCA_cluster_clients.labels_, cmap='viridis')
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.title('Visualisation des clusters avec PCA 1/3')
plt.colorbar(label='Cluster')

# Visualisation des clusters en utilisant les première et troisième composantes principales de la PCA
plt.subplot(2,2,2)
plt.scatter(PCA_data_preferences[:, 0], PCA_data_preferences[:, 2], c=PCA_cluster_clients.labels_, cmap='viridis')
plt.xlabel('Première composante principale')
plt.ylabel('Troisième composante principale')
plt.title('Visualisation des clusters avec PCA 2/3')
plt.colorbar(label='Cluster')

# Visualisation des clusters en utilisant les deuxième et troisième composantes principales de la PCA
plt.subplot(2,2,3)
plt.scatter(PCA_data_preferences[:, 1], PCA_data_preferences[:, 2], c=PCA_cluster_clients.labels_, cmap='viridis')
plt.xlabel('Deuxième composante principale')
plt.ylabel('Troisième composante principale')
plt.title('Visualisation des clusters avec PCA 3/3')
plt.colorbar(label='Cluster')

plt.show()

print(figure.get_size_inches())
