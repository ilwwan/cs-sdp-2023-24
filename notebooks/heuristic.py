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

from metrics import PairsExplained
from models import *
from data import Dataloader


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters, n_iter):
        """Initialization of the Heuristic Model."""

        self.seed = 123
        self.n_pieces = n_pieces
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.x_abs = None
        self.eps = 0.0001
        self.model = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        
        models = []
        for i in range(self.n_clusters):
            models.append(Model(f"Heuristic_{i}"))

        return models

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        # data = np.concatenate([X, Y], axis = 1) #Met les x_j et y_j sur la même ligne j

        # PCA_all_features = PCA(n_components= self.n_clusters - 1).fit_transform(data)
        # PCA_cluster_clients = KMeans(n_clusters= self.n_clusters).fit(PCA_all_features)

        data = np.concatenate([X, Y], axis = 0) #Met les y en dessous des x, pour avoir la valeur min et max par feature 

        PCA_all_features = PCA(n_components= self.n_clusters - 1).fit_transform(data)

        n_samples, n_features = PCA_all_features.shape

        X_PCA = np.copy(PCA_all_features[: n_samples//2, :])
        Y_PCA = np.copy(PCA_all_features[n_samples//2 : , :])
        data_PCA = np.concatenate((X_PCA, Y_PCA), axis = 1)

        PCA_cluster_clients = KMeans(n_clusters= self.n_clusters).fit(data_PCA)

        
        PCA_all_elements = np.concatenate([X_PCA, Y_PCA], axis=1) #Met les y en dessous des x, pour avoir la valeur min et max par feature 

        #PCA pour réduire le nombre de critères de x et y 

        self.criteria_min = PCA_all_elements.min(axis=0)
        self.criteria_max = PCA_all_elements.max(axis=0)

        X_clusters = [select_lines(X, PCA_cluster_clients.labels_, k) # X_clusters[k] = liste des éléments (1 élément = ses 10 critères) qui sont dans le cluster k
                      for k in range(self.n_clusters)] 
        Y_clusters = [select_lines(Y, PCA_cluster_clients.labels_, k)
                      for k in range(self.n_clusters)]
        
        self.x_abs = []

        for i in range(n_features):
            current_feature_values = PCA_all_features[:, i]
            min_value = np.min(current_feature_values)
            max_value = np.max(current_feature_values)
            self.x_abs.append(np.linspace(min_value, max_value, self.n_pieces))

        for a in range(self.n_iter):

            print(f'Itération {a} sur {self.n_iter}')

            for m in range(self.n_clusters):

                # Variables

                # On construit les u[k], fonctions de décisions de chaque cluster k
                # On les construit vides, ils seront remplis après

                u = []
                for i in range(n_features):
                    u.append([])
                    for l in range(self.n_pieces):
                        u[i].append(self.models[m].addVar(
                            lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"u_{i}_{l}"))

                # Définition des erreurs d'estimation sigma_x plus et moins, sigma_y plus et moins

                sig_p = {}
                sig_m = {}

                for j in range(len(X_clusters[m])):
                    sig_p[j] = self.models[m].addVar(
                        lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"sig_y_p_{j}")
                    sig_m[j] = self.models[m].addVar(
                        lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"sig_y_m_{j}")

                # Contraintes
                # Croissance des fonctions de décision sur leur intervalle de définition
                for i in range(n_features):
                    for l in range(self.n_pieces-1):
                        self.models[m].addConstr(u[i][l] <= u[i][l+1])

                self.models[m].addConstr(quicksum(u[i][self.n_pieces-1]
                                                  for i in range(n_features)) == 1)

                # Les fonctions de décision u[k][i] commencent à 0
                for i in range(n_features):
                    self.models[m].addConstr(u[i][0] == 0)

                for j in range(len(X_clusters[m])):
                    uxj = quicksum(
                        fcpm(self.x_abs[i], u[i], X_clusters[m][j][i]) for i in range(n_features))
                    uyj = quicksum(
                        fcpm(self.x_abs[i], u[i], Y_clusters[m][j][i]) for i in range(n_features))
                    self.models[m].addConstr(
                        uxj-uyj-sig_m[j]+sig_p[j] >= self.eps)

                self.models[m].setObjective(
                    quicksum(sig_m[j] + sig_p[j] for j in range(len(X_clusters[m]))), GRB.MINIMIZE)

                self.models[m].optimize()

            new_X_clusters = [[] for k in range(len(X_clusters))]
            new_Y_clusters = [[] for k in range(len(X_clusters))]

            for k in range(len(X_clusters)):
                u = []
                for i in range(len(X_clusters[k])):
                    ux = self.predict_utility(
                        np.array(X_clusters[k][i]).reshape(1, len(X_clusters[k][i])))
                    uy = self.predict_utility(
                        np.array(X_clusters[k][i]).reshape(1, len(X_clusters[k][i])))
                    diff_utility = [ux[0][m] - uy[0][m]
                                    for m in range(len(ux[0]))]
                    kluster = diff_utility.index(max(diff_utility))
                    new_X_clusters[kluster].append(X_clusters[k][i])
                    new_Y_clusters[kluster].append(Y_clusters[k][i])

            X_clusters = new_X_clusters.copy()
            Y_clusters = new_Y_clusters.copy()

        return self
        # return (PCA_cluster_clients, data_PCA)

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
        n_samples, n_features = X.shape
        decision_values = np.zeros((n_samples, self.n_clusters))

        for j in range(n_samples):
            pred = []
            for m in range(self.K):
                s = 0
                for i in range(n_features):
                    ukil = [self.models[m].getVarByName(
                        f"u_{i}_{l}").x for l in range(self.n_pieces)]
                    s += fcpm(self.x_abs[i], ukil, X[j][i])
                pred.append(s)
            decision_values[j] = pred
        return decision_values
        

    
    
    
    
    
    
    
    
    
    
    ### VISUALISATION

def affiche(PCA_cluster_clients, PCA_data_preferences) :
    nb_composantes_PCA = np.shape(PCA_data_preferences)[1]

    if nb_composantes_PCA == 2 :
        plt.scatter(PCA_data_preferences[:, 0], PCA_data_preferences[:, 1], c=PCA_cluster_clients.labels_, cmap='viridis')
        plt.xlabel('Première composante principale')
        plt.ylabel('Deuxième composante principale')
        plt.title('Visualisation des clusters avec PCA')
        plt.colorbar(label='Cluster')
        plt.show()
        
    elif nb_composantes_PCA >= 3 : 
        PCA3_fig = plt.figure(figsize = (15,8))

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

        if nb_composantes_PCA == 4 :
            plt.close()

            #Test pour une PCA à 4 composantes 
            PCA4_fig = plt.figure(figsize = (15,8))

            # Visualisation des clusters en utilisant les première et quatrième composantes principales de la PCA
            plt.subplot(2,2,1)
            plt.scatter(PCA_data_preferences[:, 0], PCA_data_preferences[:, 3], c=PCA_cluster_clients.labels_, cmap='viridis')
            plt.xlabel('Première composante principale')
            plt.ylabel('Quatrième composante principale')
            plt.title('Visualisation des clusters avec 4 PCA 1/3')
            plt.colorbar(label='Cluster')

            # Visualisation des clusters en utilisant les deuxième et quatrième composantes principales de la PCA
            plt.subplot(2,2,2)
            plt.scatter(PCA_data_preferences[:, 1], PCA_data_preferences[:, 3], c=PCA_cluster_clients.labels_, cmap='viridis')
            plt.xlabel('Deuxième composante principale')
            plt.ylabel('Quatrième composante principale')
            plt.title('Visualisation des clusters avec 4 PCA 2/3')
            plt.colorbar(label='Cluster')

            # Visualisation des clusters en utilisant les troisième et quatrième composantes principales de la PCA
            plt.subplot(2,2,3)
            plt.scatter(PCA_data_preferences[:, 2], PCA_data_preferences[:, 3], c=PCA_cluster_clients.labels_, cmap='viridis')
            plt.xlabel('Troisième composante principale')
            plt.ylabel('Quatrième composante principale')
            plt.title('Visualisation des clusters avec 4 PCA 3/3')
            plt.colorbar(label='Cluster')

            plt.show()



data_loader = Dataloader("/Users/antoine/Desktop/CS/CS-COURS/3A/IA/SDP/cs-sdp-2023-24/data/dataset_10") # Specify path to the dataset you want to load
X, Y = data_loader.load()

parameters = {"n_pieces": 5, "n_clusters": 3, "n_iter": 1} # Can be completed
model = HeuristicModel(**parameters)
model.fit(X,Y)
#PCA_cluster_clients, PCA_data_preferences = model.fit(X,Y)
#affiche(PCA_cluster_clients, PCA_data_preferences)






