#!/usr/bin/env python
# coding: utf-8

# # Setup

# First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20.

# In[1]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "unsupervised_learning"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# ## Creating a Face Detection Model using clustering techniques on the Olivetti Faces Dataset

# The classic Olivetti faces dataset contains 400 grayscale 64 × 64–pixel images of faces. Each image is flattened to a 1D vector of size 4,096. 40 different people were photographed (10 times each). We can load the dataset using the `sklearn.datasets.fetch_olivetti_faces()` function

# In[2]:


from sklearn.datasets import fetch_olivetti_faces

olivetti = fetch_olivetti_faces()


# In[3]:


print(olivetti.DESCR)


# In[4]:


olivetti.target


# Splitting the dataset into a training set, a validation set, and a test set. Please note that the dataset is already scaled between 0 and 1. Further, we will be using stratified sampling here since our dataset is quite small which will ensure that there are the same number of images per person in each set.

# In[5]:


from sklearn.model_selection import StratifiedShuffleSplit

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
train_valid_idx, test_idx = next(strat_split.split(olivetti.data, olivetti.target))
X_train_valid = olivetti.data[train_valid_idx]
y_train_valid = olivetti.target[train_valid_idx]
X_test = olivetti.data[test_idx]
y_test = olivetti.target[test_idx]

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid))
X_train = X_train_valid[train_idx]
y_train = y_train_valid[train_idx]
X_valid = X_train_valid[valid_idx]
y_valid = y_train_valid[valid_idx]


# In[6]:


print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)


# As we saw above the dimensionality of the data is quite high so, we'll reduce the data's dimensionality using PCA:

# In[7]:


from sklearn.decomposition import PCA

pca = PCA(0.99) #99% variance should be explained by the model
X_train_pca = pca.fit_transform(X_train)
X_valid_pca = pca.transform(X_valid)
X_test_pca = pca.transform(X_test)

pca.n_components_


# ##### Quick fact: We use the fit_transform method on training data and transform method on test data because  in fit_transform, we have the fit which calculates the mean and variance of the training data and transform uses this mean and variance to transform/scale the data. Using fit_transform on the test data will let the model know about the test data as well so this will be no surprise for the model. Thus, we use the transform method since this will use the training data mean and variance to transform the test data 

# Clustering images using K-Means

# In[8]:


from sklearn.cluster import KMeans

k_range = range(5, 150, 5)
kmeans_per_k = []
for k in k_range:
    print("k={}".format(k))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_train_pca)
    kmeans_per_k.append(kmeans)


# In[9]:


from sklearn.metrics import silhouette_score

silhouette_scores = [silhouette_score(X_train_pca, model.labels_)
                     for model in kmeans_per_k]
best_index = np.argmax(silhouette_scores)
best_k = k_range[best_index]
best_score = silhouette_scores[best_index]

plt.figure(figsize=(8, 3))
plt.plot(k_range, silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.plot(best_k, best_score, "rs")
plt.show()


# In[10]:


best_k


# It looks like the best number of clusters is quite high, at 120. An expectation around 40 would be more believable since there are 40 different people on the pictures. However, same person is clicked very differently using different angles, using specs which could have led to this high number of clusters.

# In[11]:


inertias = [model.inertia_ for model in kmeans_per_k]
best_inertia = inertias[best_index]

plt.figure(figsize=(8, 3.5))
plt.plot(k_range, inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.plot(best_k, best_inertia, "rs")
plt.show()


# Since we don't see an obvious elbow which would help us to give optimal number of clusters, so let's stick with k=120

# In[12]:


best_model = kmeans_per_k[best_index]


# *Visualizing the clusters to see if there are similar faces in the clusters*

# In[13]:


def plot_faces(faces, labels, n_cols=5):
    faces = faces.reshape(-1, 64, 64)
    n_rows = (len(faces) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols, n_rows * 1.1))
    for index, (face, label) in enumerate(zip(faces, labels)):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(face, cmap="gray")
        plt.axis("off")
        plt.title(label)
    plt.show()

for cluster_id in np.unique(best_model.labels_):
    print("Cluster", cluster_id)
    in_cluster = best_model.labels_==cluster_id
    faces = X_train[in_cluster]
    labels = y_train[in_cluster]
    plot_faces(faces, labels)


# We see that some of clusters are useful: that is, they contain at least 2 pictures, all of the same person. However, the rest of the clusters have either one or more intruders, or they have just a single picture.
# 
# Clustering images this way may be too imprecise to be directly useful when training a model (as we will see below), but it can be tremendously useful when labeling images in a new dataset: it will usually make labelling much faster.

# ## Using Clustering as Preprocessing for Classification

# *Training a classifier to predict which person is represented in each picture, and evaluate it on the validation set.*

# In[14]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train_pca, y_train)
clf.score(X_valid_pca, y_valid)


# *Using K-Means as a dimensionality reduction tool, and train a classifier on the reduced set.*

# In[17]:


X_train_reduced = best_model.transform(X_train_pca)
X_valid_reduced = best_model.transform(X_valid_pca)
X_test_reduced = best_model.transform(X_test_pca)

clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train_reduced, y_train)
    
clf.score(X_valid_reduced, y_valid)


# This did not help at all. Lets see if tuning the clusters will help or not.

# *Searching for the number of clusters that allows the classifier to get the best performance*

# Since we already have a validation set, we don't need K-fold cross-validation, and we're only exploring a single hyperparameter, so it's simpler to just run a loop manually:

# In[18]:


from sklearn.pipeline import Pipeline

for n_clusters in k_range:
    pipeline = Pipeline([
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42)),
        ("forest_clf", RandomForestClassifier(n_estimators=150, random_state=42))
    ])
    pipeline.fit(X_train_pca, y_train)
    print(n_clusters, pipeline.score(X_valid_pca, y_valid))


# Even after tuning, we never get beyond 80% accuracy. Looks like the distances to the cluster centroids are not as informative as the original images.

# *Now, we can try appending the features from the reduced set to the original features*

# In[20]:


X_train_extended = np.c_[X_train_pca, X_train_reduced]
X_valid_extended = np.c_[X_valid_pca, X_valid_reduced]
X_test_extended = np.c_[X_test_pca, X_test_reduced]


# In[21]:


clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train_extended, y_train)
clf.score(X_valid_extended, y_valid)


# That's a bit better, but still worse than without the cluster features. The clusters are not useful to directly train a classifier in this case (but they can still help when labelling new training instances).

# ## Lets try a Gaussian Mixture Model for the Olivetti Faces Dataset

# *Training a Gaussian mixture model on the Olivetti faces dataset and to speed up the algorithm, we are reducing the dataset's dimensionality (e.g., use PCA, preserving 99% of the variance).*

# In[23]:


from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=40, random_state=42)
y_pred = gm.fit_predict(X_train_pca)


# *Generating some new faces and visualizing them*

# In[24]:


n_gen_faces = 20
gen_faces_reduced, y_gen_faces = gm.sample(n_samples=n_gen_faces)
gen_faces = pca.inverse_transform(gen_faces_reduced)


# In[25]:


plot_faces(gen_faces, y_gen_faces)


# *Testing the model by modifying some images (e.g., rotate, flip, darken) and see if the model can detect the anomalies (i.e., compare the output of the `score_samples()` method for normal images and for anomalies).*

# In[26]:


n_rotated = 4
rotated = np.transpose(X_train[:n_rotated].reshape(-1, 64, 64), axes=[0, 2, 1])
rotated = rotated.reshape(-1, 64*64)
y_rotated = y_train[:n_rotated]

n_flipped = 3
flipped = X_train[:n_flipped].reshape(-1, 64, 64)[:, ::-1]
flipped = flipped.reshape(-1, 64*64)
y_flipped = y_train[:n_flipped]

n_darkened = 3
darkened = X_train[:n_darkened].copy()
darkened[:, 1:-1] *= 0.3
y_darkened = y_train[:n_darkened]

X_bad_faces = np.r_[rotated, flipped, darkened]
y_bad = np.concatenate([y_rotated, y_flipped, y_darkened])

plot_faces(X_bad_faces, y_bad)


# In[27]:


X_bad_faces_pca = pca.transform(X_bad_faces)


# In[28]:


gm.score_samples(X_bad_faces_pca)


# The bad faces are all considered highly unlikely by the Gaussian Mixture model. Comparing this to the scores of some training instances:

# In[29]:


gm.score_samples(X_train_pca[:10])


# ## Using Dimensionality Reduction Techniques for Anomaly Detection

# *Using dimensionality reduction techniques for anomaly detection by computing the reconstruction error for each image*

# Using the reduced dataset:

# In[30]:


X_train_pca


# In[31]:


def reconstruction_errors(pca, X):
    X_pca = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    mse = np.square(X_reconstructed - X).mean(axis=-1)
    return mse


# In[32]:


reconstruction_errors(pca, X_train).mean()


# In[34]:


reconstruction_errors(pca, X_bad_faces).mean()


# In[37]:


plot_faces(X_bad_faces, y_bad)


# In[38]:


X_bad_faces_reconstructed = pca.inverse_transform(X_bad_faces_pca)
plot_faces(X_bad_faces_reconstructed, y_bad)


# In[ ]:




