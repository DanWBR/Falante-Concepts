import coremltools
import tensorflow as tf
import os
import tfcoreml

print(tf.__version__)

basedir =  os.path.dirname(__file__)
modelfile = os.path.join(basedir, 'predictor_en.h5')

keras_model = tf.keras.models.load_model(modelfile)

# print input name, output name, input shape
print(keras_model.input.name)
print(keras_model.input_shape)
print(keras_model.output.name)
print(keras_model.output_shape)

model = tfcoreml.convert(modelfile,
                         input_name_shape_dict={'embedding_input': (1, 1)},
                         output_feature_names=['Identity'],
	     minimum_ios_deployment_target='13')

outmodelfile = os.path.join(basedir, 'predictor_en.mlmodel')

model.save(outmodelfile)