
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.models import Model

from functools import reduce


def get_VGG19(layers):

  # Load our model. We load pretrained VGG19, trained on imagenet data
  weights_path    = None # "vgg_weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
  vgg19           = VGG19(weights=weights_path, include_top=False)
  vgg19.trainable = False

  model_outputs = []

  for name in layers:
    model_outputs.append(vgg19.get_layer(name).output)


  m = Model(inputs = vgg19.inputs, outputs = model_outputs) 
  return m, vgg19



class Loss:

  def __init__(self, stylized_images, content_layers, style_layers):

    
    self.content_layers, self.style_layers = content_layers, style_layers

    self.len_content_layers = len(self.content_layers)
    
    self.vgg19, self.vg   = get_VGG19(list(content_layers+style_layers))


    self.stylized_activations   = self.vgg19(stylized_images)

  def load_weights_to_vgg19(self, path):

    self.vg.summary()
    self.vg.load_weights(path)

  def content_loss(self, content_input_batch):

    self.content_activations = self.vgg19(content_input_batch)

    for layer in self.content_activations:
      layer.trainable = False

    cont_loss = 0
    for l in range(len(self.content_layers)):
      _, height, width, channels = self.content_activations[l].get_shape().as_list()

      cont_loss +=  tf.reduce_sum(tf.square(self.content_activations[l] - self.stylized_activations[l]))/ (height * width * channels)

    return cont_loss

  def gram_matrix(self, input_tensor):

    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)

    return gram / tf.cast(n, tf.float32)

  def style_loss(self, style_image):

    self.style_activations = self.vgg19(style_image)

    style_score = 0

    for l in range(len(self.style_layers)):

      _, height, width, channels = self.style_activations[self.len_content_layers+l].get_shape().as_list()

      style_gram     = self.gram_matrix(self.style_activations[self.len_content_layers+l])
      transfrom_gram = self.gram_matrix(self.stylized_activations[self.len_content_layers+l])

      style_score += tf.reduce_sum(tf.square(style_gram - transfrom_gram)) / (height*width*(channels**2))

    return style_score


  def tv_loss(self, x):

    _, img_height, img_width, channels = x.get_shape().as_list()

    # total variation denoising
    a = tf.square(x[:, :img_height-1, :img_width-1, :] - x[:, 1:, :img_width-1, :])
    b = tf.square(x[:, :img_height-1, :img_width-1, :] - x[:, :img_height-1, 1:, :])

    return tf.reduce_sum(tf.pow(a + b, 1.25))



