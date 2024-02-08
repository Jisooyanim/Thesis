# https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
from sklearn.cluster import KMeans
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse

image = cv2.imread('dataset with preprocess/preprocessed/Bacterial Canker/IMG_20211106_121111 (Custom).jpg') 

# Convert image to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

# Reshape image pixel values
pixel_vals = image.reshape((-1,3)) 

# Convert to float type for k-means
pixel_vals = np.float32(pixel_vals)

# Calculate WSS for different values of k
kmax = 10
wss = calculate_WSS(pixel_vals, kmax)

# Plot the Elbow Method graph
plt.plot(range(1, kmax+1), wss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-Cluster Sum of Squares (WSS)')
plt.show()