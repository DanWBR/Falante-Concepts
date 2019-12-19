import tensorflow as tf
import os

new_model= tf.keras.models.load_model(filepath="predictor_en.h5")

tflite_converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = tflite_converter.convert()

open("predictor_en.tflite", "wb").write(tflite_model)