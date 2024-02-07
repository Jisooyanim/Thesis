#https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Read the image
image = cv2.imread('test_images/20211008_124250 (Custom).jpg') 

# Convert image to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

# Reshape image pixel values
pixel_vals = image.reshape((-1,3)) 

# Convert to float type for k-means
pixel_vals = np.float32(pixel_vals)

# Calculate silhouette scores for different values of k
sil = []
kmax = 10

# Minimum number of clusters should be 2
for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters=k).fit(pixel_vals)
    labels = kmeans.labels_
    sil.append(silhouette_score(pixel_vals, labels, metric='euclidean'))

# Plot the silhouette scores
plt.plot(range(2, kmax+1), sil)
plt.title('Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()
