import numpy as np

# ------------------------------
# Load Saved Features
# ------------------------------
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

print("====================================")
print("Basic Info")
print("====================================")
print("Feature shape:", X_train.shape)
print("Label shape  :", y_train.shape)

# ------------------------------
# 1. Check for Zero Vectors
# ------------------------------
zero_rows = np.where(np.all(X_train == 0, axis=1))[0]
print("\nNumber of zero feature vectors:", len(zero_rows))

# ------------------------------
# 2. Global Mean and Std
# ------------------------------
print("\nGlobal Mean:", np.mean(X_train))
print("Global Std :", np.std(X_train))

# ------------------------------
# 3. Variance Per Dimension
# ------------------------------
feature_variances = np.var(X_train, axis=0)

print("\nMin variance across dimensions:", np.min(feature_variances))
print("Max variance across dimensions:", np.max(feature_variances))
print("Number of near-zero variance dims:",
      np.sum(feature_variances < 1e-8))

# ------------------------------
# 4. Norm Distribution
# ------------------------------
norms = np.linalg.norm(X_train, axis=1)

print("\nEmbedding Norm Statistics")
print("Min norm :", np.min(norms))
print("Max norm :", np.max(norms))
print("Mean norm:", np.mean(norms))
print("Std norm :", np.std(norms))

# ------------------------------
# 5. Distance Between Random Samples
# ------------------------------
if X_train.shape[0] >= 2:
    dist = np.linalg.norm(X_train[0] - X_train[1])
    print("\nDistance between sample 0 and 1:", dist)

# ------------------------------
# 6. Class-wise Mean Norms
# ------------------------------
print("\nClass-wise Mean Embedding Norms")
for c in np.unique(y_train):
    class_mean = np.mean(X_train[y_train == c], axis=0)
    print(f"Class {c} mean norm:", np.linalg.norm(class_mean))



print("\n====================================")
print("Sanity check complete.")
print("====================================")

import numpy as np

# Load features
X = np.load("X_train.npy")
y = np.load("y_train.npy")

print("Feature shape:", X.shape)

# -------------------------------------------------
# 1️⃣ Print first vector (first 20 dimensions)
# -------------------------------------------------
print("\nFirst sample (first 20 values):")
print(X[0][:20])

# -------------------------------------------------
# 2️⃣ Print second vector (first 20 dims)
# -------------------------------------------------
print("\nSecond sample (first 20 values):")
print(X[1][:20])

# -------------------------------------------------
# 3️⃣ Check if first two vectors are identical
# -------------------------------------------------
print("\nAre first two vectors identical?",
      np.allclose(X[0], X[1]))

# -------------------------------------------------
# 4️⃣ Print one sample from each class
# -------------------------------------------------
print("\nOne sample from each class (first 10 dims):")

for c in np.unique(y):
    idx = np.where(y == c)[0][0]
    print(f"\nClass {c} sample index {idx}:")
    print(X[idx][:10])

# -------------------------------------------------
# 5️⃣ Difference between two samples
# -------------------------------------------------
diff = X[0] - X[1]
print("\nFirst 10 values of (sample0 - sample1):")
print(diff[:10])

print("\nL2 distance between sample 0 and 1:",
      np.linalg.norm(diff))