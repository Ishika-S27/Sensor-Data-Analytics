import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Load dataset
data = load_iris()
X = data.data
y = data.target
# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Visualization
plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.xlabel(&quot;Principal Component 1&quot;)
plt.ylabel(&quot;Principal Component 2&quot;)
plt.title(&quot;PCA Visualization&quot;)
plt.show()
