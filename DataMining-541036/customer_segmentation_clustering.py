import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# 1) Load dataset
df = pd.read_csv("Mall_Customers.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2) Select features
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# 3) Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Exploratory plots

# Histograms
for col in features:
    plt.figure(figsize=(10, 4))
    plt.hist(df[col], bins=15)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# Correlation heatmap
corr = df[features].corr()
plt.figure(figsize=(6, 5))
plt.imshow(corr, interpolation="nearest")
plt.title("Correlation Heatmap")
plt.xticks(range(len(features)), features, rotation=45, ha="right")
plt.yticks(range(len(features)), features)
plt.colorbar()
plt.tight_layout()
plt.show()



# Model selection plots

# Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(list(K), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# Silhouette Scores
sil_scores = []
K2 = range(2, 11)
for k in K2:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels_k))

plt.figure(figsize=(7, 4))
plt.plot(list(K2), sil_scores, marker="o")
plt.title("Silhouette Score vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.grid(True)
plt.show()


# Final K-Means + PCA visualization

k_final = 5
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df["Cluster"] = labels

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.title("Customer Segmentation (K-Means) - PCA View")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()
