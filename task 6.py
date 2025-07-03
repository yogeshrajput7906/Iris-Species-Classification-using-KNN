# ------------------------------------------------------------
# Import necessary libraries
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------------------------------------
# Load the Iris dataset
# ------------------------------------------------------------
df = pd.read_csv("Iris.csv")
print(df.head())

# Drop ID column if present
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# ------------------------------------------------------------
# Separate features (X) and target (y)
# ------------------------------------------------------------
X = df.drop('Species', axis=1)
y = df['Species']

# ------------------------------------------------------------
# Encode the target labels to integers
# ------------------------------------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ------------------------------------------------------------
# Normalize the features
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# Split data into training and testing sets
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y_encoded)

# ------------------------------------------------------------
# Train KNN for different values of K
# ------------------------------------------------------------
for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    print(f"\n=== KNN with K={k} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------
# Plot decision boundaries (only on first two features)
# ------------------------------------------------------------
# Reduce data to first two features for 2D plotting
X_vis = X_scaled[:, :2]

# Split again for this reduced data
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train KNN on reduced features
knn_vis = KNeighborsClassifier(n_neighbors=5)
knn_vis.fit(X_train_vis, y_train_vis)

# Create a mesh grid to predict over
h = 0.02
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict over the grid
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# ------------------------------------------------------------
# Plot the decision boundary
# ------------------------------------------------------------
plt.figure(figsize=(10,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['red', 'green', 'blue']))
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_encoded, cmap=ListedColormap(['red', 'green', 'blue']))
plt.xlabel(df.columns[1])
plt.ylabel(df.columns[2])
plt.title("Decision Boundaries (K=5)")
plt.show()
