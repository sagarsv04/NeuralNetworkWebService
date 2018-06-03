from flask import request
from flask import jsonify
from flask import Flask
import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
import tensorflow as tf  # To get default graph

# MobileNet
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

# from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools


app = Flask(__name__)  # Create instance of a flask class


def get_model():
	global model
	model = load_model('./VGG16_cats_and_dogs.h5')
	# model.get_weights()
	global graph
	graph = tf.get_default_graph() # without this i was having issues
	print(" * Model loaded!")


def get_mobilenet_model():
	global mobilenet_model
	mobilenet_model = keras.applications.mobilenet.MobileNet()
	# model.get_weights()
	global mobilenet_graph
	mobilenet_graph = tf.get_default_graph() # without this i was having issues
	print(" * MobileNet Model loaded!")


def prepare_image(image, target_size):
	if image.mode != "RGB":
		image = image.convert("RGB")
	image = image.resize(target_size)
	img_array = img_to_array(image)
	img_array_expanded_dims = np.expand_dims(img_array, axis=0)
	return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


def preprocess_image(image, target_size):
	if image.mode != "RGB":
		image = image.convert("RGB")
	image = image.resize(target_size)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	return image


@app.route('/')  # Decorator that tells flask at which url to call the function
def running():
	return 'Flask is running!'


@app.route('/sayhello')  # Decorator that tells flask at which url to call the function
def print_hello():
	return 'Hello!'


@app.route('/hello',methods=['POST'])
def hello():
	message = request.get_json(force=True) # force True means parse json even if its unsure of datatype
	name = message['name']
	response = {
		'greeting': 'Hello, ' + name + '!'
	}
	return jsonify(response)

@app.route("/predict", methods=["POST"])
def predict():
	message = request.get_json(force=True)
	encoded = message['image']
	# print(encoded)
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	# image.show()
	processed_image = preprocess_image(image, target_size=(224, 224))
	# Why we have to use the created graph as default for predicting
	with graph.as_default():
		prediction = model.predict(processed_image).tolist()
	response = {
		'prediction': {
			'dog': prediction[0][0],
			'cat': prediction[0][1]
		}
	}
	return jsonify(response)


@app.route("/mobilenet", methods=["POST"])
def imagenet_predict():
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	# image.show()
	preprocessed_image = prepare_image(image, target_size=(224, 224))
	# processed_image = preprocess_image(image, target_size=(224, 224))
	# Why we have to use the created graph as default for predicting
	with mobilenet_graph.as_default():
		predictions = mobilenet_model.predict(preprocessed_image)
	results = imagenet_utils.decode_predictions(predictions)

	response = {
		'prediction': {
			results[0][0][1]: np.float64(results[0][0][2]),
			results[0][1][1]: np.float64(results[0][1][2]),
			results[0][2][1]: np.float64(results[0][2][2]),
			results[0][3][1]: np.float64(results[0][3][2]),
			results[0][4][1]: np.float64(results[0][4][2])
		}
	}
	return jsonify(response)


if __name__ == '__main__':
	print(" * Loading Keras model...")
	get_model()
	get_mobilenet_model()
	app.run()
