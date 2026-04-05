# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
# Load dataset
data = load_iris()
X = data.data
# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# -----------------------------
# Elbow Method
# -----------------------------
wcss = []
for k in range(1, 11):
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
wcss.append(kmeans.inertia_)
# Plot Elbow Curve
plt.plot(range(1, 11), wcss, marker=&#39;o&#39;)
plt.xlabel(&quot;Number of Clusters (K)&quot;)
plt.ylabel(&quot;WCSS&quot;)
plt.title(&quot;Elbow Method&quot;)
plt.show()
# -----------------------------
# K-Means Clustering
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
# Plot Clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.xlabel(&quot;Feature 1&quot;)
plt.ylabel(&quot;Feature 2&quot;)
plt.title(&quot;K-Means Clustering&quot;)
plt.show()
