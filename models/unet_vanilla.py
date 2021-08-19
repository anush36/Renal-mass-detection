# Base U-Net Model

import sys
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models

#UNCOMMENT THIS BEFORE RUNNING TRAINING
"""
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
"""


def unet_model(IMG_SIZE, DROPOUT):


    def conv_block(input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        #encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Dropout(DROPOUT)(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
        encoder = layers.Dropout(DROPOUT)(encoder)
        #encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder

    def encoder_block(input_tensor, num_filters):
        encoder = conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
        
        return encoder_pool, encoder

    def decoder_block(input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = layers.Dropout(DROPOUT)(decoder)
        #concat_tensor = attention_block(decoder, concat_tensor, num_filters)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        #decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.Dropout(DROPOUT)(decoder)
        #decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.Dropout(DROPOUT)(decoder)
        #decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    inputs = layers.Input(shape= (IMG_SIZE, IMG_SIZE, 2))
    # 256

    encoder0_pool, encoder0 = encoder_block(inputs, 16) #16
    # 128

    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 32)  #32
    # 64

    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 64) #128
    # 32

    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 128) #256
    # 16

    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 256) #256

    encoder5_pool, encoder5 = encoder_block(encoder4_pool, 512) #256

    center = conv_block(encoder5_pool, 1024) #512
    # 16

    decoder5 = decoder_block(center, encoder5, 512) #256

    decoder4 = decoder_block(decoder5, encoder4, 256) #256

    decoder3 = decoder_block(decoder4, encoder3, 128) #256
    # 64

    decoder2 = decoder_block(decoder3, encoder2, 64)   #128
    # 128

    decoder1 = decoder_block(decoder2, encoder1, 32)    #64
    # 256

    decoder0 = decoder_block(decoder1, encoder0, 16)    #32
    # 512

    # Looks like this layer just serves the purpose of an activation
    outputs = layers.Conv2D(1, (1,1), activation = 'sigmoid')(decoder0)
    # use_bias=True, bias_initializer=tf.keras.initializers.Constant(-1.9661128)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])

    return model

