import numpy as np
from ..utils.timing import timer
from ..dimred.pca import PCA
from ..dimred.autoencoder import Autoencoder
from ..clustering.kmeans import KMeans
from ..clustering.gmm import GMM
from ..metrics.internal import silhouette_score, davies_bouldin_index, calinski_harabasz_index, wcss
from ..metrics.external import adjusted_rand_index, normalized_mutual_info, purity_score, confusion_matrix

def compute_all_metrics(X, labels_pred, y_true=None, gmm_model=None):
    out = {}
    out["silhouette"] = silhouette_score(X, labels_pred)
    out["davies_bouldin"] = davies_bouldin_index(X, labels_pred)
    out["calinski_harabasz"] = calinski_harabasz_index(X, labels_pred)
    out["wcss"] = wcss(X, labels_pred)

    if gmm_model is not None:
        out["log_likelihood"] = gmm_model.log_likelihood(X)
        out["bic"] = gmm_model.bic(X)
        out["aic"] = gmm_model.aic(X)

    if y_true is not None:
        out["ari"] = adjusted_rand_index(y_true, labels_pred)
        out["nmi"] = normalized_mutual_info(y_true, labels_pred)
        out["purity"] = purity_score(y_true, labels_pred)
        cm, labs = confusion_matrix(y_true, labels_pred)
        out["confusion_matrix"] = cm
        out["confusion_labels"] = labs
    return out

def run_kmeans(X, k, init="kmeans++", seed=42, max_iter=300, tol=1e-4):
    with timer() as t:
        km = KMeans(n_clusters=k, init=init, seed=seed, max_iter=max_iter, tol=tol).fit(X)
    return km, t()

def run_gmm(X, k, covariance_type="full", seed=42, max_iter=200, tol=1e-4, reg_covar=1e-6):
    with timer() as t:
        gmm = GMM(n_components=k, covariance_type=covariance_type, seed=seed, max_iter=max_iter, tol=tol, reg_covar=reg_covar).fit(X)
    return gmm, t()

def pca_embed(X, n_components):
    pca = PCA(n_components=n_components).fit(X)
    Z = pca.transform(X)
    rec_err = pca.reconstruction_error(X)
    return pca, Z, rec_err

def ae_embed(X, bottleneck_dim, hidden_dims=(64, 32, 16), activation="relu",
             lr=1e-3, l2=1e-4, lr_decay=0.995, epochs=200, batch_size=64, seed=42, verbose=0):
    ae = Autoencoder(
        input_dim=X.shape[1],
        bottleneck_dim=bottleneck_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        lr=lr,
        l2=l2,
        lr_decay=lr_decay,
        seed=seed
    ).fit(X, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)
    Z = ae.transform(X)
    rec_err = ae.reconstruction_error(X)
    return ae, Z, rec_err
