import pickle
from abc import abstractmethod

import numpy as np
from gurobipy import *


def fcpm(abscisses, ordonnees, x):
    # Trouver l'intervalle dans lequel x se trouve
    for i in range(len(abscisses) - 1):
        if abscisses[i] <= x <= abscisses[i + 1]:
            # Calculer la valeur de la fonction par morceaux dans cet intervalle
            pente = (ordonnees[i + 1] - ordonnees[i]) / \
                (abscisses[i + 1] - abscisses[i])
            y = ordonnees[i] + pente * (x - abscisses[i])
            return y


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features)  # Weights cluster 1
        weights_2 = np.random.rand(num_features)  # Weights cluster 2

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        u_1 = np.dot(X, self.weights[0])  # Utility for cluster 1 = X^T.w_1
        u_2 = np.dot(X, self.weights[1])  # Utility for cluster 2 = X^T.w_2
        # Stacking utilities over cluster on axis 1
        return np.stack([u_1, u_2], axis=1)


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n_clusters: int
            Number of clusters to implement in the MIP.
        """
        self.seed = 123
        self.model = self.instantiate()
        self.n_pieces = n_pieces
        self.n_clusters = n_clusters
        self.x_abs = None
        self.eps = 0.0001

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        # To be completed

        model = Model("TwoClustersMIP")
        model.setParam('TimeLimit', 5*60)

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

        n_samples, n_features = X.shape

        M = 2

        # Calcul des X_i_l

        # x_abs contient n_features élément. Chaque élément est une liste des abscisses des points de cassure pour la feature i

        all_features = np.concatenate([X, Y])

        self.x_abs = []

        for i in range(n_features):
            current_feature_values = all_features[:, i]
            min_value = np.min(current_feature_values)
            max_value = np.max(current_feature_values)
            self.x_abs.append(np.linspace(min_value, max_value, self.n_pieces))

        # Variables

# On construit les u[k], fonctions de décisions de chaque cluster k
# On les construit vides, ils seront remplis après
        u = []
        for k in range(self.n_clusters):
            u.append([])
            # pour chaque cluster k, on ajoute une fonction de décision associée à chaque critère i
            for i in range(n_features):
                u[k].append([])
                # enfin, pour chaque u[k][i], fonction de décision de d'un critère au sein d'un cluster, on crée la variable qui représente l'ordonnée à l'origine de la cassure
                for l in range(self.n_pieces):
                    u[k][i].append(self.model.addVar(
                        lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"u_{k}_{i}_{l}"))

# On construit les v[j][k], qui valent 1 si la paire j est dans le cluster k, 0 sinon
        v = []
        for j in range(n_samples):
            v.append([])
            for k in range(self.n_clusters):
                v[j].append(self.model.addVar(
                    vtype=GRB.BINARY, name=f"v_{j}_{k}"))


# Définition des erreurs d'estimation sigma_x plus et moins, sigma_y plus et moins
        sig_y_p = {}
        sig_y_m = {}
        sig_x_p = {}
        sig_x_m = {}

        for j in range(n_samples):
            sig_x_p[j] = self.model.addVar(
                lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"sig_x_p_{j}_{k}")
            sig_x_m[j] = self.model.addVar(
                lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"sig_x_m_{j}_{k}")
            sig_y_p[j] = self.model.addVar(
                lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"sig_y_p_{j}_{k}")
            sig_y_m[j] = self.model.addVar(
                lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"sig_y_m_{j}_{k}")
        # Contraintes
        # Croissance des fonctions de décision sur leur intervalle de définition
        for k in range(self.n_clusters):
            for i in range(n_features):
                for l in range(self.n_pieces-1):
                    self.model.addConstr(u[k][i][l] <= u[k][i][l+1])

        # Normalisation des critères : la somme des max des u[k] vaut 1.
        for k in range(self.n_clusters):
            self.model.addConstr(quicksum(u[k][i][self.n_pieces-1]
                                 for i in range(n_features)) == 1)

        # Chaque paire j est présente dans au moins 1 cluster
        for j in range(n_samples):
            self.model.addConstr(quicksum(v[j][k]
                                 for k in range(self.n_clusters)) >= 1)

        # Les fonctions de décision u[k][i] commencent à 0
        for k in range(self.n_clusters):
            for i in range(n_features):
                self.model.addConstr(u[k][i][0] == 0)

        for j in range(n_samples):
            for k in range(self.n_clusters):
                ukxj = quicksum(
                    fcpm(self.x_abs[i], u[k][i], X[j][i]) for i in range(n_features))
                ukyj = quicksum(
                    fcpm(self.x_abs[i], u[k][i], Y[j][i]) for i in range(n_features))
                self.model.addConstr(
                    ukxj - ukyj - sig_x_p[j] + sig_x_m[j] + sig_y_p[j] - sig_y_m[j] - self.eps >= M * (v[j][k] - 1))
                # self.model.addConstr(
                #     ukxj - ukyj - sig_x_p[j] + sig_x_m[j] + sig_y_p[j] - sig_y_m[j] - self.eps >= - M*v[j][k])

        self.model.setObjective(quicksum(sig_x_p[j] + sig_x_m[j] + sig_y_p[j] + sig_y_m[j]
                                for j in range(n_samples) for k in range(self.n_clusters)), GRB.MINIMIZE)

        self.model.optimize()

        return self

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
            for k in range(self.n_clusters):
                s = 0
                for i in range(n_features):
                    ukil = [self.model.getVarByName(
                        f"u_{k}_{i}_{l}").x for l in range(self.n_pieces)]
                    s += fcpm(self.x_abs[i], ukil, X[j][i])
                pred.append(s)
            decision_values[j] = pred
        return decision_values


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.model = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
        return

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # To be completed
        return

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
