# NeuralNetworkWebService

This code is a demo app of creating a web application for predicting Dogs and Cats.

I have used FLASK for web service, and basic HTML + Javascript for web page.

The VGG16_cats_and_dogs.h5 file is too large to upload to Github.

The MobileNet model gets downloaded while loading the model.

Download it here: https://drive.google.com/open?id=19yICdtSbU_YkQBRxJ2if9KJwUL1oY5xs

MobileNet suffers in accuracy as it has very less parameters compare to VGG16.

Language
============

* Python 3


Packages
============
tensorflow (1.6.0)

Flask (1.0.2)

Keras (2.1.5)


Usage
============

python sample_app.py

VGG16 > Cat & Dog >>> visit http://ip:port/static/predict.html

MobileNet >>>> visit http://ip:port/static/mobilenet.html
