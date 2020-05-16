#neural style transfer
import os
import cv2

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

#tf.enable_eager_execution()
#print("Eager execution: {}".format(tf.executing_eagerly()))

# Set up some global values here
content_path = './content/BX7OQz.jpg'
style_path = './style/Vassily_Kandinsky,_1913_-_Composition_7.jpg'

def load_and_resize_image(img_path, size=512):
    img = cv2.imread(img_path)

    #resize
    scale = size/float(max(img.shape))
    img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

    ## bgr to rgb
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #broadcast to make it in batch dimension (1,h,w,c)
    img = np.expand_dims(img, axis=0)
    return img

def display_image(image, title=None):

    if len(image.shape)==4:
        #remove the batch dimension
        img = np.squeeze(image, axis=0)

    #convert from rgb to bgr
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow(title, img)
    cv2.waitKey()

# read the content image
content_img = load_and_resize_image(content_path, 512)
display_image(content_img)

# read the style image
style_img = load_and_resize_image(style_path, 512)
display_image(style_img)

# lets pre-process the images expected by the VGG19 net
def load_and_process_img(img_path):
    img = load_and_resize_image(img_path)

    #images are channel normalized by mean [B=103.939, G=116.779, R=123.68]
    img = keras.applications.vgg19.preprocess_input(img)
    return img

content_img = load_and_process_img(content_path)
display_image(content_img)

def inverse_process(img):
    img_cpy = img.copy()

    if len(img_cpy.shape) == 4:
        img_cpy = np.squeeze(img_cpy, axis=0)
    
    assert len(img_cpy.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")

    if len(img_cpy.shape) != 3:
        raise ValueError("Invalid input to inverse processing...")

    img_cpy[:, :, 0] += 103.939 #B
    img_cpy[:, :, 1] += 116.779 #G
    img_cpy[:, :, 2] += 123.68 #R

    #img_cpy = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2RGB)

    img_cpy = np.clip(img_cpy, 0, 255).astype('uint8') #clip the values between 0-255
    img_cpy = np.expand_dims(img_cpy, axis=0)
    return img_cpy

# Now lets get the model to represent the content and style images into higer order features
# for an input image we'll try to match the corresponding style and content target representation at the intermediate layers

# The intermediate layers work as a complex feature extractor which generalizes
#  well and agnostic to small variance in the image since its able to capture the intrinsic features of image

# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# since VGG19 is a relatively simple model compared to ResNet or InceptionNet
# its better at representing the features for style transfer

def get_model():
  """ Creates our model with access to intermediate layers.
  This function will load the VGG19 model and access the intermediate layers.
  These layers will then be used to create a new model that will take input image
  and return the outputs from these intermediate layers from the VGG model.
 
  Returns:
    returns a keras model that takes image inputs and outputs the style and
      content intermediate layers.
  """
  # Load our model. We load pretrained VGG, trained on imagenet data
  vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  # Get output layers corresponding to style and content layers
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs
  # Build model
  return models.Model(vgg.input, model_outputs)


## define loss functions

## 1. Content loss
# its nothing but euclidean distance between input image and output image (after passing through the network)

def get_content_loss(base_content, target):
  return keras.backend.mean(keras.backend.square(base_content - target), keepdims=False)

## 2. Style loss
# we calculate the loss between base input image and style image
# instead of comparing the raw loss of outputs of style and base input we see the gram metrics
def gram_matrix(input_tensor):
  # We make the image channels first
  channels = int(input_tensor.shape[-1])
  a = keras.backend.reshape(input_tensor, [-1, channels])
  n = keras.backend.shape(a)[0]
  gram = keras.backend.matmul(a, a, transpose_a=True)
  return gram / keras.backend.cast_to_floatx(n)

def get_style_loss(base_style, gram_target):
  """Expects two images of dimension h, w, c"""
  # height, width, num filters of each layer
  # We scale the loss at a given layer by the size of the feature map and the number of filters
  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
 
  return keras.backend.mean(keras.backend.square(gram_style - gram_target), keepdims=False)# / (4. * (channels ** 2) * (width * height) ** 2)

## Optimize using gradient discent
def get_feature_representations(model, content_path, style_path):
  """Helper function to compute our content and style feature representations.

  This function will simply load and preprocess both the content and style
  images from their path. Then it will feed them through the network to obtain
  the outputs of the intermediate layers.
 
  Arguments:
    model: The model that we are using.
    content_path: The path to the content image.
    style_path: The path to the style image
   
  Returns:
    returns the style features and the content features.
  """
  # Load our images in
  content_image = load_and_process_img(content_path)
  style_image = load_and_process_img(style_path)
 
  # batch compute content and style features
  style_outputs = model(style_image)
  content_outputs = model(content_image)
 
 
  # Get the style and content feature representations from our model  
  style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
  content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
  return style_features, content_features

## Compute loss
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
  """This function will compute the loss total loss.
 
  Arguments:
    model: The model that will give us access to the intermediate layers
    loss_weights: The weights of each contribution of each loss function.
      (style weight, content weight, and total variation weight)
    init_image: Our initial base image. This image is what we are updating with
      our optimization process. We apply the gradients wrt the loss we are
      calculating to this image.
    gram_style_features: Precomputed gram matrices corresponding to the
      defined style layers of interest.
    content_features: Precomputed outputs from defined content layers of
      interest.
     
  Returns:
    returns the total loss, style loss, content loss, and total variational loss
  """
  style_weight, content_weight = loss_weights
 
  # Feed our init image through our model. This will give us the content and
  # style representations at our desired layers. Since we're using eager
  # our model is callable just like any other function!
  model_outputs = model(init_image)
 
  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]
 
  style_score = 0
  content_score = 0

  # Accumulate style losses from all layers
  # Here, we equally weight each contribution of each loss layer
  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
   
  # Accumulate content losses from all layers
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features):
    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
 
  style_score *= style_weight
  content_score *= content_weight

  # Get total loss
  loss = style_score + content_score
  return loss, style_score, content_score

def compute_grads(cfg):
  with tf.GradientTape() as tape:
    all_loss = compute_loss(**cfg)
  # Compute gradients wrt input image
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss

import IPython.display
def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
  # We don't need to (or want to) train any layers of our model, so we set their
  # trainable to false.
  model = get_model()
  for layer in model.layers:
    layer.trainable = False
 
  # Get the style and content feature representations (from our specified intermediate layers)
  style_features, content_features = get_feature_representations(model, content_path, style_path)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
 
  # Set initial image
  init_image = load_and_process_img(content_path)
  init_image = tfe.Variable(init_image, dtype=tf.float32)
  # Create our optimizer
  opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

  # For displaying intermediate images
  iter_count = 1
 
  # Store our best result
  best_loss, best_img = float('inf'), None
 
  # Create a nice config
  loss_weights = (style_weight, content_weight)
  cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
  }
   
  # For displaying
  num_rows = 2
  num_cols = 5
  display_interval = num_iterations/(num_rows*num_cols)
  start_time = time.time()
  global_start = time.time()
 
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means  
 
  imgs = []
  for i in range(num_iterations):
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    end_time = time.time()
   
    if loss < best_loss:
      # Update best loss and best image from total loss.
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())

    if i % display_interval== 0:
      start_time = time.time()
     
      # Use the .numpy() method to get the concrete numpy array
      plot_img = init_image.numpy()
      plot_img = deprocess_img(plot_img)
      imgs.append(plot_img)
      IPython.display.clear_output(wait=True)
      IPython.display.display_png(Image.fromarray(plot_img))
      print('Iteration: {}'.format(i))        
      print('Total loss: {:.4e}, '
            'style loss: {:.4e}, '
            'content loss: {:.4e}, '
            'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
  print('Total time: {:.4f}s'.format(time.time() - global_start))
  IPython.display.clear_output(wait=True)
  plt.figure(figsize=(14,4))
  for i,img in enumerate(imgs):
      plt.subplot(num_rows,num_cols,i+1)
      plt.imshow(img)
      plt.xticks([])
      plt.yticks([])
     
  return best_img, best_loss
