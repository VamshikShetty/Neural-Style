
import tensorflow as tf
tf.reset_default_graph() 


from keras.applications.vgg19 import VGG19
import os 

from tensorflow.python.keras.preprocessing import image as kp_image
from keras.models import Model
from keras.layers import Dense, BatchNormalization,Dropout,concatenate 
from keras import backend as K
from keras.models import Model,load_model,model_from_json #,evaluate_generator
from keras import losses
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D,Flatten,GlobalAveragePooling2D

import numpy as np
from PIL import Image
import cv2
import scipy
import transform_network as TNET
from loss import Loss, get_VGG19

content_layers = ['block3_conv3']
style_layers   = ['block1_conv1','block2_conv2', 'block3_conv3', 'block4_conv3']

num_content_layers = len(content_layers)
num_style_layers   = len(style_layers)


seed = 791
tf.set_random_seed(seed)
np.random.seed(seed)

content_dir = 'content/'
style_image = 'udnie.jpg'


height = 352
width  = 352

def load_img(path_to_img, expand = True, img_shape=(height,width)):
  
  img = scipy.misc.imread(path_to_img)

  img = scipy.misc.imresize(img, img_shape)
  img = img.astype("float32")
  if expand:
    img = np.expand_dims(img, axis=0)
  
  img = tf.keras.applications.vgg19.preprocess_input(img)

  return img

def load_batch(image_paths):
  x = []
  for image_path in image_paths:
    img = load_img(image_path, False)

    x.append(img)

  x = np.asarray(x)
  return x


def deprocess_img(processed_img, shape):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessiing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68



  x = np.clip(x, 0, 255).astype('uint8')
  img = scipy.misc.imresize(x, shape)
  return img


def run_fast_style_transfer(content_training_images,  style_image_path, epochs, batch_size, content_weight=0.6,  style_weight=0.4, total_variation_weight = 1e-5): 

  with tf.Session() as sess:
    K.set_session(sess)

    
    input_batch = tf.placeholder(tf.float32, shape=(None, height, width, 3), name="input_batch")
    init_image  = TNET.get_TransformNet('transform_network', input_batch)

    loss        = Loss(init_image, content_layers, style_layers)


    content_loss = loss.content_loss(input_batch)

  
    style_var   = load_img(style_image_path)



    style_var   = tf.Variable(style_var)
    style_loss  = loss.style_loss(style_var)
    

    tv_loss = loss.tv_loss(init_image)

    total_loss    = style_weight*style_loss + content_weight*content_loss + total_variation_weight*tv_loss


    transform_net =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='transform_network')
    opt           = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, epsilon=1e-08).minimize(total_loss, var_list=[transform_net])


    #sess.run(tf.variables_initializer(var_list=[input_batch]))
    
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()

    Tnet_saver = tf.train.Saver(transform_net)

    # loading the weights again because tf.global_variables_initializer() resets the weights
    loss.load_weights_to_vgg19("vgg_weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")
    # init_image.load_weights('0-transform_network.h5')


    dir_model = "weights/"+style_image.split('.')[0]+"_weights/"
    if not os.path.exists(dir_model):
      os.makedirs(dir_model)
    
    # Tnet_saver.restore(sess, dir_model+"model.ckpt")


    for i in range(epochs):

      avg_loss = 0
      avg_cnt  = 1

      for j in range(0, int(len(content_training_images)/batch_size)):

        batch = load_batch(content_training_images[j: j+batch_size])

        temp = sess.run([total_loss, style_loss, content_loss, tv_loss, init_image, opt],feed_dict={input_batch:batch})

        print('epoch: ',i,'batch: ',j,'  loss: ', temp[:4], 'avg loss: ', avg_loss )

        avg_loss = (avg_loss*(avg_cnt-1) + temp[0] )/avg_cnt
        avg_cnt += 1


        if j%50==0: # and i%50==0:
          image =  deprocess_img(temp[4][2], batch[2].shape[:-1])
          cv2.imwrite(str(i)+'-'+str(j)+'-temp.jpg',image)
          if i==0:
            image_ori =  deprocess_img(batch[2], batch[2].shape[:-1])
            cv2.imwrite(str(i)+'-'+str(j)+'-temp-orgi.jpg',image_ori)


      # if (i+1)%100==0:
      print('\n Data Saved ... ')
      Tnet_saver.save(sess, dir_model+"model.ckpt")

    sess.close()



content_training_images = os.listdir(content_dir) # http://cocodataset.org/#download 2017 val images [5k/1GB]
for i in range(len(content_training_images)):
  content_training_images[i] = content_dir+content_training_images[i]

#print(content_training_images)
run_fast_style_transfer(content_training_images, style_image, epochs=5, batch_size=6)
#cv2.imwrite(str(num_iterations)+'-'+save_name,best)

