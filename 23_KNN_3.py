import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.patches import Circle
import os

# Dataset
X = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y = np.array([0, 0, 0, 1, 1, 1])
new_point = np.array([[5, 4]])

# Class color map
class_colors = {0: 'blue', 1: 'green'}

# Store frames
images = []
k_values = [1, 2, 3, 4, 5]

for k in k_values:
    # 1. Train KNN (Indented 4 spaces)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    pred = knn.predict(new_point)[0]

    # 2. Find distances and neighbors
    distances = np.linalg.norm(X - new_point, axis=1)
    nearest_indices = np.argsort(distances)[:k]
    kth_distance = distances[nearest_indices[-1]]

    # 3. Plotting logic
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot existing points
    for class_value in np.unique(y):
        ax.scatter(X[y == class_value, 0], X[y == class_value, 1],
                   label=f'Class {class_value}', s=100, color=class_colors[class_value])

    # Plot new point
    ax.scatter(new_point[0, 0], new_point[0, 1],
               c=class_colors[pred], edgecolors='black',
               s=200, label='New Point', zorder=5, marker='*')

    # Dotted circle for k-th neighbor
    circle = Circle(new_point[0], kth_distance, color='gray',
                    linestyle='dotted', fill=False, linewidth=2, label=f'{k}-NN Radius')
    ax.add_patch(circle)

    # Highlight neighbors
    ax.scatter(X[nearest_indices, 0], X[nearest_indices, 1],
               s=200, facecolors='none', edgecolors='black', linewidths=2,
               label='Neighbors')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title(f'K = {k} → Predicted Class: {pred}', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True)

    # 4. Save frame
    os.makedirs("frames", exist_ok=True)
    filename = f"frames/frame_knn_{k}.png"
    plt.savefig(filename)
    images.append(imageio.imread(filename))
    plt.close()

# 5. Save as animated GIF (Outside the loop)
imageio.mimsave("knn_interactive.gif", images, fps=1)
print("GIF saved as 'knn_interactive.gif'")
