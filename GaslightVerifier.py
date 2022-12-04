import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

def graph_load(filename):
    graph_data = np.load(filename).tolist()
    best_scores = graph_data[0]
    best_misclass = graph_data[1]
    best_similarity = graph_data[2]

    timesteps = list(range(len(best_scores)))

    plt.figure()
    plt.plot(timesteps, best_misclass, label="Misclassification")
    plt.plot(timesteps, best_similarity, label="Similarity")
    plt.plot(timesteps, best_scores, label="Best Score")
    plt.xlabel("Timesteps")
    plt.ylabel("Score [0-1]")
    plt.legend()
    plt.savefig(f"{filename}.png")
    plt.close()

def mnist_load(filename):
    model = models.load_model("Classifiers/mnist")
    image = np.load(filename)
    image_input = image.reshape((1,) + image.shape) / 255.0
    print(np.argmax(model.predict(image_input)[0]))
    cv2.imwrite(f"{filename}.png", image)

mnist_load("Outputs/MNIST-Hard-Targeted.npy")
graph_load("Figures/MNIST-Hard-Targeted.npy")