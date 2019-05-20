
import tensorflow as tf
from keras.models import Model
from keras.layers import BatchNormalization, Dropout, concatenate, Input, Conv2D, Conv2DTranspose,Lambda

def residual_block(net, filter_size=3):
    tmp = tf.layers.conv2d( net, filters=128, kernel_size=filter_size, strides=(1, 1), padding='same', activation=tf.nn.relu)

    return net + tf.layers.conv2d( tmp, filters=128, kernel_size=filter_size, strides=(1, 1), padding='same', activation=tf.nn.relu)

def get_TransformNet(scope, image):

  image = image/127.5

  with tf.variable_scope(scope):
    conv1 = tf.layers.conv2d( image, filters=32, kernel_size=9, strides=(1, 1), padding='same', activation=tf.nn.relu)
    b_norm1 = tf.layers.batch_normalization(conv1)
    conv2 = tf.layers.conv2d( b_norm1, filters=64, kernel_size=3, strides=(2, 2), padding='same', activation=tf.nn.relu)
    b_norm2 = tf.layers.batch_normalization(conv2)
    conv3 = tf.layers.conv2d( b_norm2, filters=128, kernel_size=3, strides=(2, 2), padding='same', activation=tf.nn.relu)

    resid1 = residual_block(conv3, 3)
    resid2 = residual_block(resid1, 3)
    resid3 = residual_block(resid2, 3)
    resid4 = residual_block(resid3, 3)
    resid5 = residual_block(resid4, 3)

    conv_t1 = tf.layers.conv2d_transpose( resid5, filters=128, kernel_size=3, strides=(2, 2), padding='same', activation=tf.nn.relu)
    b_norm3 = tf.layers.batch_normalization(conv_t1)
    conv_t2 = tf.layers.conv2d_transpose( b_norm3, filters=32, kernel_size=3, strides=(2, 2), padding='same', activation=tf.nn.relu)
    conv_t3 = tf.layers.conv2d_transpose( conv_t2, filters=3, kernel_size=1, strides=(1, 1), padding='same', activation=tf.nn.tanh)


    output = conv_t3*127.5

  return output
