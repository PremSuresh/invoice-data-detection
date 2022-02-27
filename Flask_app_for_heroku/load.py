import numpy as np
import keras.models
from keras.models import model_from_json
from keras.engine import Layer
import scipy
import tensorflow as tf
import tensorflow as tf
import tensorflow_hub as hub

from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model

import numpy as np
# from keras.utils import multi_gpu_model
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)
class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2',
                               trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights +=K.tf.trainable_variables(scope=
                                                           "^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


# def build_model1():
#     input_text = layers.Input(shape=(1,), dtype="string")
#     embedding = ElmoEmbeddingLayer()(input_text)
#     dense1 = layers.Dense(1024, activation='relu')(embedding)
#     dense2 = layers.Dense(256, activation="relu")(dense1)
#     pred = layers.Dense(19, activation='softmax')(dense2)

#     model = Model(inputs=[input_text], outputs=pred)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()  
#     return(model)

def init(): 
	json_file = open('halfnonehalfsac.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json, custom_objects={'ElmoEmbeddingLayer': ElmoEmbeddingLayer()})
	#load woeights into new model
	loaded_model.load_weights("halfnonehalfsac.h5")
	loaded_model._make_predict_function()

	print("Loaded Model from disk")

	#compile and evaluate loaded model
	# loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	#loss,accuracy = model.evaluate(X_test,y_test)
	#print('loss:', loss)
	#print('accuracy:', accuracy)
	graph = tf.get_default_graph()

	return loaded_model,graph
