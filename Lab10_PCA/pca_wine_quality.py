# Lab 10 : PCA on Wine Quality Dataset (13 features -> 2 features)
# Dataset : https://www.kaggle.com/datasets/yasserh/wine-quality-dataset  (WineQT.csv)
# Seed    : 77

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(77)


# ---------------- Load data ----------------
def load_data():
    # use WineQT.csv if it is in the same folder, else download from Kaggle
    if os.path.exists("WineQT.csv"):
        return pd.read_csv("WineQT.csv")
    import kagglehub
    path = kagglehub.dataset_download("yasserh/wine-quality-dataset")
    return pd.read_csv(glob.glob(os.path.join(path, "*.csv"))[0])


df = load_data()
X = df.values                        # all 13 columns as features
features = df.columns
classes = pd.cut(df["quality"], bins=[0, 5, 6, 10], labels=["Low", "Medium", "High"])

print("Dataset shape :", df.shape)
print("Features :", list(features))


# ---------------- PCA (13 -> 2) ----------------
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n========== PCA : 13 -> 2 ==========")
print("Shape before :", X_scaled.shape, " after :", X_pca.shape)
print("Explained variance ratio :", pca.explained_variance_ratio_.round(4))
print("Total variance retained  :", round(pca.explained_variance_ratio_.sum() * 100, 2), "%")


# ---------------- PCA mathematically (6 steps) ----------------
Z = (X - X.mean(axis=0)) / X.std(axis=0)          # 1. standardize
C = (Z.T @ Z) / (len(Z) - 1)                      # 2. covariance matrix (13 x 13)
eig_vals, eig_vecs = np.linalg.eigh(C)            # 3. eigenvalues, eigenvectors
idx = np.argsort(eig_vals)[::-1]                  # 4. rank (largest first)
eig_vals, eig_vecs = eig_vals[idx], eig_vecs[:, idx]
W = eig_vecs[:, :2]                               # 5. select top 2
Y = Z @ W                                         # 6. transform

print("\n========== PCA mathematically ==========")
print("Eigenvalues :", eig_vals.round(4))
print("Explained variance % :", (eig_vals / eig_vals.sum() * 100).round(2))
print("Matches sklearn (up to sign) :",
      all(np.allclose(abs(Y[:, i]), abs(X_pca[:, i])) for i in range(2)))


# ---------------- Question 1 : 2 -> 3 components ----------------
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)

print("\n========== Question 1 ==========")
print("Shape with 2 components :", X_pca.shape)
print("Shape with 3 components :", X_pca3.shape)
print("Features after reduction :", X_pca3.shape[1])
print("Samples changed? :", X_pca3.shape[0] != X.shape[0])
print("Variance retained (2) :", round(pca.explained_variance_ratio_.sum() * 100, 2), "%")
print("Variance retained (3) :", round(pca3.explained_variance_ratio_.sum() * 100, 2), "%")


# ---------------- Question 2 : without standardization ----------------
pca_raw = PCA(n_components=2).fit(X)

print("\n========== Question 2 ==========")
print("With standardization    :", pca.explained_variance_ratio_.round(4))
print("Without standardization :", pca_raw.explained_variance_ratio_.round(4))
print("\nVariance of each raw feature (largest first) :")
print(df.var().sort_values(ascending=False).round(3))


# ---------------- Question 3 : 1, 2, 3, 5, 10 components ----------------
print("\n========== Question 3 ==========")
for k in [1, 2, 3, 5, 10]:
    p = PCA(n_components=k).fit(X_scaled)
    print(f"{k:2d} components -> cumulative variance = {p.explained_variance_ratio_.sum() * 100:.2f} %")

cum = np.cumsum(PCA().fit(X_scaled).explained_variance_ratio_)
k90 = np.argmax(cum >= 0.90) + 1
print("Components needed for >= 90% variance :", k90)

plt.figure()
plt.plot(range(1, len(cum) + 1), cum * 100, marker="o")
plt.axhline(90, color="red", ls="--", label="90%")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance (%)")
plt.title("Q3 : Cumulative explained variance")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("q3_cumulative_variance.png", dpi=120)


# ---------------- Question 4 : visualization ----------------
print("\n========== Question 4 ==========")
plt.figure()
for c, col in zip(["Low", "Medium", "High"], ["red", "orange", "green"]):
    plt.scatter(X_pca[classes == c, 0], X_pca[classes == c, 1], s=12, c=col, label=c, alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Q4 : Wine Quality, 13 features -> 2 (PCA)")
plt.legend()
plt.savefig("q4_pca_scatter.png", dpi=120)
print("Information lost (13 -> 2) :", round((1 - pca.explained_variance_ratio_.sum()) * 100, 2), "%")
print("PCA was fitted only on X_scaled (no labels used).")
print("Class counts :")
print(classes.value_counts())


# ---------------- Question 5 : selection vs extraction ----------------
print("\n========== Question 5 ==========")
print("PC1 = weighted sum of ALL 13 original features :")
print(pd.Series(pca.components_[0], index=features).round(3))
print("\nPC2 weights :")
print(pd.Series(pca.components_[1], index=features).round(3))

print("\nPlots saved : q3_cumulative_variance.png, q4_pca_scatter.png")
plt.show()
