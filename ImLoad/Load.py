import tensorflow as tf
import numpy as np


def load_and_preprocess_image(filename, img_height, img_width):
	image = tf.io.read_file(filename)
	image = tf.image.decode_image(image, channels=3)
	image = tf.image.resize(image, [img_height, img_width]).numpy()
	image = image.reshape((1, img_height, img_width, 3))
	return image