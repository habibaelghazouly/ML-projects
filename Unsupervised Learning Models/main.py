import numpy as np

from utils.data import load_breast_cancer_kagglehub, standardize_fit_transform
from KMeans.kmeans_numpy import KMeans
from PCA.pca_numpy import PCA

def main():
    X, y, feature_names = load_breast_cancer_kagglehub()
    Xs, scaler = standardize_fit_transform(X)

    print("X shape:", Xs.shape, "y shape:", None if y is None else y.shape)

    # Quick PCA
    pca = PCA(n_components=5, random_state=42)
    Z = pca.fit_transform(Xs)
    Xrec = pca.inverse_transform(Z)
    print("PCA explained variance ratio (first 5):", pca.explained_variance_ratio_)
    print("PCA reconstruction MSE:", pca.reconstruction_error(Xs))

    # Quick KMeans
    km = KMeans(n_clusters=2, init="kmeans++", max_iter=100, tol=1e-4, random_state=42)
    labels = km.fit_predict(Xs)
    print("KMeans inertia:", km.inertia_)
    print("KMeans iterations:", km.n_iter_)

if __name__ == "__main__":
    main()
