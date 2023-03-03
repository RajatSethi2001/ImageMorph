import cv2
import numpy as np
import tensorflow as tf

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from tensorflow.keras import models

model_fn = models.load_model("Classifiers/cifar10")
x = cv2.imread("Inputs/CIFAR10.png")
x = x.reshape((1,) + x.shape) / 255.0
eps = 0.25
eps_iter = 0.1
nb_iter = 0.1
norm = 2

adv_x = fast_gradient_method(model_fn, x, eps, norm).numpy()[0]
adv_x_test = adv_x.reshape((1,) + adv_x.shape)
print(np.argmax(model_fn(adv_x_test)))
cv2.imwrite("Outputs/FastGM.png", (adv_x * 255).astype(np.int64))

adv_x = basic_iterative_method(model_fn, x, eps, eps_iter, nb_iter, norm).numpy()[0]
adv_x_test = adv_x.reshape((1,) + adv_x.shape)
print(np.argmax(model_fn(adv_x_test)))
cv2.imwrite("Outputs/BasicIter.png", (adv_x * 255).astype(np.int64))

adv_x = carlini_wagner_l2(model_fn, x.astype(np.float32))
print(np.argmax(model_fn(adv_x)))
cv2.imwrite("Outputs/Carlini.png", (adv_x[0] * 255).astype(np.int64))