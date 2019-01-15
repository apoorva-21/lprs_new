#loads saved trained MNIST CNN model, then runs inference on it from the binarized digit images

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = './binary_digits/'
MODEL_NAME = './models/my_model.h5'
model = load_model(MODEL_NAME)

for image_name in os.listdir(DATA_PATH):
	image_name = os.path.join(DATA_PATH, image_name)
	print image_name
	image = plt.imread(fname = image_name, format = 'PNG')
	image = np.reshape(image/255, (1, 28, 28, 1))
	print(np.argmax(model.predict(image)))
