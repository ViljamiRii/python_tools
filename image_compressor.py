import os
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

'''
https://en.wikipedia.org/wiki/K-means_clustering
https://trepo.tuni.fi/bitstream/handle/10024/121055/PeltonenAnni.pdf?sequence=2

'''


def k_centroid_initiation(x, k):
    
    dimensions = len(x)
    return x[np.random.choice(dimensions, k, replace=False), :]

def find_centroids(x, centroids):
    
    dimensions = len(x)
    vector = np.zeros(dimensions)
    
    for i in range(dimensions):
        
        distances = np.linalg.norm(x[i] - centroids, axis=1)
        vector[i] = np.argmin(distances)
    
    return vector

def calc_means(x, idx, k):

    _, n = x.shape
    centroids = np.zeros((k, n))

    for k in range(k):

        examples = x[np.where(idx == k)]
        mean = [np.mean(column) for column in examples.T]
        centroids[k] = mean

    return centroids

def find_k_means(x, k, max_iterations=10):

    centroids = k_centroid_initiation(x, k)
    previous_centroids = centroids

    for _ in range(max_iterations):

        idx = find_centroids(x, centroids)
        centroids = calc_means(x, idx, k)

        if (centroids == previous_centroids).all():
            return centroids # centroids at halt
        
        else:
            previous_centroids = centroids
        
    return centroids, idx

def load_image(path):

    image = Image.open(path)
    return np.asarray(image) / 255


def main():

    
    try:
        image_path = input("Enter the name of the image:")
        assert os.path.isfile(image_path)
    except:
        print("Error getting the image")

    image = load_image(image_path)
    width, height, depth = image.shape
    print(f"Width: {width}, Height: {height}, Depth: {depth}")

    x = image.reshape((width * height, depth))
    k = 40
    iters = 20
    try:
        k = int(input("Enter the amount of clusters (colors) wanted:"))
    except:
        pass
    try:
        iters = int(input("Enter how many iterations wanted:"))
    except:
        pass

    print("Find K means")
    colors, _ = find_k_means(x, k, max_iterations=iters)

    idx = find_centroids(x, colors)
    idx = np.array(idx, dtype=np.uint8)

    x_reconstructed = np.array(colors[idx, :] * 255, dtype=np.uint8).reshape((width, height, depth))
    compressed_image = Image.fromarray(x_reconstructed)

    compressed_image.save("compressed.png")

main()