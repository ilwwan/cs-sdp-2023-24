import pickle
from abc import abstractmethod

import numpy as np
from gurobipy import GRB, Model, quicksum
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d


def fcpm(abscisses, ordonnees, x):
    # Trouver l'intervalle dans lequel x se trouve
    for i in range(len(abscisses) - 1):
        if abscisses[i] <= x <= abscisses[i + 1]:
            # Calculer la valeur de la fonction par morceaux dans cet intervalle
            pente = (ordonnees[i + 1] - ordonnees[i]) / (
                abscisses[i + 1] - abscisses[i]
            )
            y = ordonnees[i] + pente * (x - abscisses[i])
            return y


def select_lines(X, clusters, k):
    mask = clusters == k
    return X[mask]


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
        self.eps = 0.0001

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        # To be completed

        model = Model("TwoClustersMIP")

        return model

    def u_k(self, x, k, eval=False):
        return sum(self.u_ki(x[i], k, i, eval) for i in range(len(x)))

    def u_ki(self, x, k, i, eval=False):
        get_val = (lambda v: v.X) if eval else (lambda v: v)

        w = 1 / self.n_pieces
        l = int(np.floor(x / w))
        x_l = l * w
        slope = (get_val(self.u[k][i][l + 1]) - get_val(self.u[k][i][l])) / w
        return slope * (x - x_l) + get_val(self.u[k][i][l])

    def u_k(self, x, k, eval=False):
        return sum(self.u_ki(x[i], k, i, eval) for i in range(len(x)))

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        P_samples, n_features = X.shape
        P_samples, n_features = X.shape

        M = 2  # Majorant pour les contraines

        ###
        M = 2  # Majorant pour les contraines

        ###
        # Variables
        ###
        ###

        # Points de cassure des fonctions de décision UTA
        self.u = []
        # Points de cassure des fonctions de décision UTA
        self.u = []
        for k in range(self.n_clusters):
            self.u.append([])
            self.u.append([])
            for i in range(n_features):
                self.u[k].append([])
                for l in range(self.n_pieces + 1):
                    self.u[k][i].append(
                        self.model.addVar(
                            lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"u_{k}_{i}_{l}"
                        )
                    )
                self.u[k].append([])
                for l in range(self.n_pieces + 1):
                    self.u[k][i].append(
                        self.model.addVar(
                            lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"u_{k}_{i}_{l}"
                        )
                    )

        # v[j][k] = 1 si la paire j est dans le cluster k, 0 sinon
        self.v = []
        for j in range(P_samples):
            self.v.append([])
        # v[j][k] = 1 si la paire j est dans le cluster k, 0 sinon
        self.v = []
        for j in range(P_samples):
            self.v.append([])
            for k in range(self.n_clusters):
                self.v[j].append(self.model.addVar(vtype=GRB.BINARY, name=f"v_{j}_{k}"))

        # Définition des erreurs de sur et sous-estimation sigma_x + et -, sigma_y + et -
        self.sig_y_p = []
        self.sig_y_m = []
        self.sig_x_p = []
        self.sig_x_m = []
        # Définition des erreurs de sur et sous-estimation sigma_x + et -, sigma_y + et -
        self.sig_y_p = []
        self.sig_y_m = []
        self.sig_x_p = []
        self.sig_x_m = []

        for j in range(P_samples):
            self.sig_x_p.append(
                self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"sig_x_p_{j}")
            )
            self.sig_x_m.append(
                self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"sig_x_m_{j}")
            )
            self.sig_y_p.append(
                self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"sig_y_p_{j}")
            )
            self.sig_y_m.append(
                self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"sig_y_m_{j}")
            )

        self.model.update()

        ###
        # Contraintes
        ###
        ###
        # Croissance des fonctions de décision sur leur intervalle de définition
        for k in range(self.n_clusters):
            for i in range(n_features):
                for l in range(self.n_pieces):
                    self.model.addConstr(self.u[k][i][l] <= self.u[k][i][l + 1])

        # Normalisation des critères : la somme des max des u[k] vaut 1.
        for k in range(self.n_clusters):
            self.model.addConstr(
                quicksum(self.u[k][i][self.n_pieces] for i in range(n_features)) == 1
            )

        # Chaque paire j est présente dans au moins 1 cluster
        for j in range(P_samples):
            self.model.addConstr(
                quicksum(self.v[j][k] for k in range(self.n_clusters)) >= 1
            )
        for j in range(P_samples):
            self.model.addConstr(
                quicksum(self.v[j][k] for k in range(self.n_clusters)) >= 1
            )

        # Les fonctions de décision u[k][i] commencent à 0
        for k in range(self.n_clusters):
            for i in range(n_features):
                self.model.addConstr(self.u[k][i][0] == 0)
                self.model.addConstr(self.u[k][i][0] == 0)

        # Définition des erreurs de sur et sous-estimation sigma_x + et -, sigma_y + et - par rapport aux utilités
        for j in range(P_samples):
            for k in range(self.n_clusters):
                ukxj = self.u_k(X[j], k)
                ukyj = self.u_k(Y[j], k)
                ukxj = self.u_k(X[j], k)
                ukyj = self.u_k(Y[j], k)
                self.model.addConstr(
                    (1 - self.v[j][k]) * M
                    + (ukxj - self.sig_x_p[j] + self.sig_x_m[j])
                    - (ukyj - self.sig_y_p[j] + self.sig_y_m[j])
                    >= self.eps
                )
        ###
        # Fonction objectif
        ###
        self.model.setObjective(
            quicksum(self.sig_x_m)
            + quicksum(self.sig_x_p)
            + quicksum(self.sig_y_m)
            + quicksum(self.sig_y_p),
            GRB.MINIMIZE,
        )

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
        ret = []
        P_samples, n_features = X.shape
        for j in range(P_samples):
            ret.append([])
        ret = []
        P_samples, n_features = X.shape
        for j in range(P_samples):
            ret.append([])
            for k in range(self.n_clusters):
                ret[j].append(self.u_k(X[j], k, eval=True))
        return np.array(ret)


class HeuristicModelOld(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_clusters):
        """Initialization of the Heuristic Model."""
        self.seed = 123
        self.model = self.instantiate()
        self.K = n_clusters

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        models = []
        for i in range(self.K):
            models.append(Model(f"Heuristic_{i}"))
            models[i].setParam("TimeLimit", 10)
            models[i].setParam("OutputFlag", 0)

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
        n_samples, n_features = X.shape

        all_features = np.concatenate([X, Y])

        self.x_abs = []

        for i in range(n_features):
            current_feature_values = all_features[:, i]
            min_value = np.min(current_feature_values)
            max_value = np.max(current_feature_values)
            self.x_abs.append(np.linspace(min_value, max_value, self.n_pieces))

        sample_cluster = np.random.randint(0, self.n_clusters, size=(n_samples))

        for a in range(self.n_iter):

            print(f"Itération {a} sur {self.n_iter}")

            X_clusters = [select_lines(X, sample_cluster, k) for k in range(self.K)]
            Y_clusters = [select_lines(Y, sample_cluster, k) for k in range(self.K)]

            for m in range(self.K):

                # Variables

                # On construit les u[k], fonctions de décisions de chaque cluster k
                # On les construit vides, ils seront remplis après

                u = []
                for i in range(n_features):
                    u.append([])
                    for l in range(self.n_pieces):
                        u[i].append(
                            self.models[m].addVar(
                                lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"u_{i}_{l}"
                            )
                        )

                # Définition des erreurs d'estimation sigma_x plus et moins, sigma_y plus et moins

                sig_p = {}
                sig_m = {}

                for j in range(len(X_clusters[m])):
                    sig_p[j] = self.models[m].addVar(
                        lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"sig_y_p_{j}"
                    )
                    sig_m[j] = self.models[m].addVar(
                        lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"sig_y_m_{j}"
                    )

                # Contraintes
                # Croissance des fonctions de décision sur leur intervalle de définition
                for i in range(n_features):
                    for l in range(self.n_pieces - 1):
                        self.models[m].addConstr(u[i][l] <= u[i][l + 1])

                self.models[m].addConstr(
                    quicksum(u[i][self.n_pieces - 1] for i in range(n_features)) == 1
                )

                # Les fonctions de décision u[k][i] commencent à 0
                for i in range(n_features):
                    self.models[m].addConstr(u[i][0] == 0)

                for j in range(len(X_clusters[m])):
                    uxj = quicksum(
                        fcpm(self.x_abs[i], u[i], X_clusters[m][j][i])
                        for i in range(n_features)
                    )
                    uyj = quicksum(
                        fcpm(self.x_abs[i], u[i], Y_clusters[m][j][i])
                        for i in range(n_features)
                    )
                    self.models[m].addConstr(
                        uxj - uyj - sig_m[j] + sig_p[j] >= self.eps
                    )

                self.models[m].setObjective(
                    quicksum(sig_m[j] + sig_p[j] for j in range(len(X_clusters[m]))),
                    GRB.MINIMIZE,
                )

                self.models[m].optimize()

            new_X_clusters = [[] for k in range(len(X_clusters))]
            new_Y_clusters = [[] for k in range(len(X_clusters))]

            for k in range(len(X_clusters)):
                u = []
                for i in range(len(X_clusters[k])):
                    ux = self.predict_utility([X_clusters[k][i]])
                    uy = self.predict_utility([X_clusters[k][i]])
                    diff_utility = [ux[0][m] - uy[0][m] for m in range(len(ux[0]))]
                    kluster = diff_utility.index(max(diff_utility))
                    new_X_clusters[kluster].append(X_clusters[k][i])
                    new_Y_clusters[kluster].append(Y_clusters[k][i])

            X_clusters = new_X_clusters.copy()
            Y_clusters = new_Y_clusters.copy()

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
        decision_values = np.zeros((n_samples, self.K))

        for j in range(n_samples):
            pred = []
            for m in range(self.K):
                s = 0
                for i in range(n_features):
                    ukil = [
                        self.models[m].getVarByName(f"u_{i}_{l}").x
                        for l in range(self.n_pieces)
                    ]
                    s += fcpm(self.x_abs[i], ukil, X[j][i])
                pred.append(s)
            decision_values[j] = pred
        return decision_values


class HeuristicModel(BaseModel):

    def __init__(self, n_clusters, n_pieces=5, n_iter=3):
        """Initialization of the Heuristic Model."""
        self.K = n_clusters
        self.n_pieces = n_pieces
        self.n_iter = n_iter
        self.eps = 0.0001
        self.models = self.instantiate()

    def instantiate(self):
        models = []
        for i in range(self.K):
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

        n_samples, n_features = X.shape

        data = np.concatenate([X, Y], axis=1)
        kmeans = KMeans(self.K)
        kmeans.fit(data)

        all_elements = np.concatenate([X, Y], axis=0)
        self.criteria_min = all_elements.min(axis=0)
        self.criteria_max = all_elements.max(axis=0)

        X_clusters = [select_lines(X, kmeans.labels_, k) for k in range(self.K)]
        Y_clusters = [select_lines(Y, kmeans.labels_, k) for k in range(self.K)]

        all_features = np.concatenate([X, Y])

        self.x_abs = []

        for i in range(n_features):
            current_feature_values = all_features[:, i]
            min_value = np.min(current_feature_values)
            max_value = np.max(current_feature_values)
            self.x_abs.append(np.linspace(min_value, max_value, self.n_pieces))

        for a in range(self.n_iter):

            print(f"Itération {a} sur {self.n_iter}")

            for m in range(self.K):

                # Variables

                # On construit les u[k], fonctions de décisions de chaque cluster k
                # On les construit vides, ils seront remplis après

                u = []
                for i in range(n_features):
                    u.append([])
                    for l in range(self.n_pieces):
                        u[i].append(
                            self.models[m].addVar(
                                lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"u_{i}_{l}"
                            )
                        )

                # Définition des erreurs d'estimation sigma_x plus et moins, sigma_y plus et moins

                sig_p = {}
                sig_m = {}

                for j in range(len(X_clusters[m])):
                    sig_p[j] = self.models[m].addVar(
                        lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"sig_y_p_{j}"
                    )
                    sig_m[j] = self.models[m].addVar(
                        lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"sig_y_m_{j}"
                    )

                # Contraintes
                # Croissance des fonctions de décision sur leur intervalle de définition
                for i in range(n_features):
                    for l in range(self.n_pieces - 1):
                        self.models[m].addConstr(u[i][l] <= u[i][l + 1])

                self.models[m].addConstr(
                    quicksum(u[i][self.n_pieces - 1] for i in range(n_features)) == 1
                )

                # Les fonctions de décision u[k][i] commencent à 0
                for i in range(n_features):
                    self.models[m].addConstr(u[i][0] == 0)

                for j in range(len(X_clusters[m])):
                    uxj = quicksum(
                        fcpm(self.x_abs[i], u[i], X_clusters[m][j][i])
                        for i in range(n_features)
                    )
                    uyj = quicksum(
                        fcpm(self.x_abs[i], u[i], Y_clusters[m][j][i])
                        for i in range(n_features)
                    )
                    self.models[m].addConstr(
                        uxj - uyj - sig_m[j] + sig_p[j] >= self.eps
                    )

                self.models[m].setObjective(
                    quicksum(sig_m[j] + sig_p[j] for j in range(len(X_clusters[m]))),
                    GRB.MINIMIZE,
                )

                self.models[m].optimize()

            new_X_clusters = [[] for k in range(len(X_clusters))]
            new_Y_clusters = [[] for k in range(len(X_clusters))]

            for k in range(len(X_clusters)):
                u = []
                for i in range(len(X_clusters[k])):
                    ux = self.predict_utility(
                        np.array(X_clusters[k][i]).reshape(1, len(X_clusters[k][i]))
                    )
                    uy = self.predict_utility(
                        np.array(X_clusters[k][i]).reshape(1, len(X_clusters[k][i]))
                    )
                    diff_utility = [ux[0][m] - uy[0][m] for m in range(len(ux[0]))]
                    kluster = diff_utility.index(max(diff_utility))
                    new_X_clusters[kluster].append(X_clusters[k][i])
                    new_Y_clusters[kluster].append(Y_clusters[k][i])

            X_clusters = new_X_clusters.copy()
            Y_clusters = new_Y_clusters.copy()

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
        decision_values = np.zeros((n_samples, self.K))

        for j in range(n_samples):
            pred = []
            for m in range(self.K):
                s = 0
                for i in range(n_features):
                    ukil = [
                        self.models[m].getVarByName(f"u_{i}_{l}").x
                        for l in range(self.n_pieces)
                    ]
                    s += fcpm(self.x_abs[i], ukil, X[j][i])
                pred.append(s)
            decision_values[j] = pred
        return decision_values
