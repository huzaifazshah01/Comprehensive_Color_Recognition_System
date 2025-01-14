import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import imutils

clusters = 8 

def classify_color_temperature(color):
    red, green, blue = color[::-1]  
    if (red + green) > blue:
        return 'Warm'
    else:
        return 'Cold'

img = cv2.imread('Default_Step_into_a_world_of_luxury_and_style_with_this_stunni_2.jpg')
org_img = img.copy()
img = imutils.resize(img, height=200)
flat_img = np.reshape(img, (-1, 3))

kmeans = KMeans(n_clusters=clusters, random_state=0)
kmeans.fit(flat_img)

dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')
percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
p_and_c = sorted(zip(percentages, dominant_colors), reverse=True, key=lambda x: x[0])

color_temperatures = [classify_color_temperature(color) for _, color in p_and_c]

plt.figure(figsize=(12, 8))
for i, ((percentage, color), temp) in enumerate(zip(p_and_c, color_temperatures)):
    plt.subplot(1, clusters, i + 1)
    block = np.ones((50, 50, 3), dtype='uint') * color[::-1]
    plt.imshow(block)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f"{round(percentage * 100, 2)}% - {temp}")
plt.show()

bar = np.ones((50, 500, 3), dtype='uint')
plt.figure(figsize=(12, 8))
plt.title('Proportions of colors in the image')
start = 0
for i, (p, c) in enumerate(p_and_c):
    end = start + int(p * bar.shape[1])
    bar[:, start:end] = c[::-1]
    start = end
plt.imshow(bar)
plt.xticks([])
plt.yticks([])
plt.show()

inertias = []
cluster_range = range(1, 15)
for k in cluster_range:
    kmeans_test = KMeans(n_clusters=k, random_state=0)
    kmeans_test.fit(flat_img)
    inertias.append(kmeans_test.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(cluster_range, inertias, marker='o')
plt.title("KMeans Inertia (Elbow Method)")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(cluster_range)
plt.grid(True)
plt.show()

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(flat_img)

optimal_clusters = 8
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
kmeans.fit(reduced_data)
labels = kmeans.labels_

plt.figure(figsize=(10, 8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
plt.title("Visualization of clustered data in 2D space")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

pca_full = PCA(n_components=min(flat_img.shape))
pca_full.fit(flat_img)
explained_variance = pca_full.explained_variance_ratio_.cumsum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
plt.title("Cumulative Explained Variance by PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.show()



rows, cols = 1000, int((org_img.shape[0] / org_img.shape[1]) * 1000)
img = cv2.resize(org_img, dsize=(rows, cols), interpolation=cv2.INTER_LINEAR)
copy = img.copy()
cv2.rectangle(copy, (rows//2-250, cols//2-90), (rows//2+250, cols//2+110), (255, 255, 255), -1)
final = cv2.addWeighted(img, 0.1, copy, 0.9, 0)
cv2.putText(final, 'Most Dominant Colors in the Image', (rows//2-230, cols//2-40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)

start = rows//2-220
for i in range(5):
    end = start + 70
    final[cols//2:cols//2+70, start:end] = p_and_c[i][1]
    cv2.putText(final, str(i+1), (start+25, cols//2+45), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    start = end + 20

plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

cv2.imwrite('output.png', final)
