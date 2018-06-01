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


app = Flask(__name__)  # Create instance of a flask class


def get_model():
	global model
	model = load_model('./VGG16_cats_and_dogs.h5')
	# model.get_weights()
	global graph
	graph = tf.get_default_graph() # without this i was having issues
	print(" * Model loaded!")


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
	print("Test5")
	return jsonify(response)


if __name__ == '__main__':
	print(" * Loading Keras model...")
	get_model()
	app.run()
