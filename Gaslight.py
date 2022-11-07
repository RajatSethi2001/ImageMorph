from MorphEngine import run

#Filename of image to be morphed
image_file = "MNIST.png"

#If Gaslight has created a checkpoint image, put its filename here.
checkpoint_file = None

#Victim model to misclassify (works for TF models only)
victim = "mnist"

#List of classifications (ordered by how they come out of the model)
#For example, MNIST would be [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
classes = [x for x in range(10)]

#Class name that the image should be misclassified to.
new_class = 5

#Divide pixel values by 255 before plugging into model?
scale_image = True

#Which action space to use (0-3)
#Action 0 - Edit one pixel at a time by -1 or +1
#Action 1 - Edit one pixel at a time by changing it to a value between [0-255]
#Action 2 - Edit all pixels at a time by -1 or +1
#Action 3 - Edit all pixels at a time by changing it to a value between [0-255]
action = 0

#Minimum similarity needed for a successful morph [0-1]
similarity = 0.9

#Render images after how many steps. Set to 0 for no render.
render_interval = 100

#Save image after how many steps. Set to 0 for no save.
save_interval = 1000

#Which RL framework to use (PPO, TD3)
framework = "PPO"

run(image_file, victim, classes, new_class, action, similarity, scale_image, checkpoint_file, render_interval, save_interval, framework)