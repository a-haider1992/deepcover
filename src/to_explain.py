import time
import numpy as np
from spectra_gen import *
from to_rank import *
from utils import *
from datetime import datetime
from mask import *
from multiprocessing import Pool
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, classification_report
from tensorflow import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization
#from keras.utils import np_utils
from keras.utils import to_categorical #from keras.utils.np_utils import to_categorical
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, model_from_json
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D, LeakyReLU
from keras.layers import MaxPooling2D,AveragePooling2D, GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,EarlyStopping,CSVLogger
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.mobilenet import MobileNet
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from skimage.segmentation import slic
from skimage.color import gray2rgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Normalizer
from lime import lime_image
from skimage.segmentation import mark_boundaries

import logging
logger = logging.getLogger(__name__)
# logging.basicConfig(filename='logging.log', level=logging.INFO)
# logger.info('Started')
# logger.info('Tensorflow version: %s', tf.__version__)
def get_lime_explanation(img_path, model, image_size):
  logger.info('get_lime_explanation')
  
  def generate_heatmap_image(img_boundry1, img_boundry2):
      # Create a heatmap image where positive features are labeled as 1 and negative features as 2
      heatmap_image = np.zeros_like(img_boundry1, dtype=np.uint8)  # Ensure the dtype is uint8
      heatmap_image[img_boundry1 > 0] = 255  # Set positive features to white
      heatmap_image[img_boundry2 > 0] = 150  # Set negative features to gray
      return heatmap_image
  
  # Read and preprocess the image
  img = cv2.imread(img_path)
  img = cv2.resize(img, (image_size, image_size))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
  img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
  img_array = np.expand_dims(img_tensor, axis=0)

  # Configure TensorFlow to use GPU if available
  if tf.test.is_gpu_available():
      logger.info('GPU is available. Using GPU for LimeImageExplainer.')
      device = "/gpu:0"
  else:
      logger.info('GPU is not available. Using CPU for LimeImageExplainer.')
      device = "/cpu:0"
  
  # Create the LimeImageExplainer
  with tf.device(device):
      explainer = lime_image.LimeImageExplainer()
  
  # Explain the instance
  explanation = explainer.explain_instance(img_array[0], model.predict, top_labels=3, hide_color=0, num_samples=1000)

  # Get the image and mask for the top label with positive-only and negative features
  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
  img_boundry1 = mark_boundaries(temp / 2 + 0.5, mask)

  temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
  img_boundry2 = mark_boundaries(temp / 2 + 0.5, mask)

  # Generate the heatmap image
  heatmap_image = generate_heatmap_image(img_boundry1, img_boundry2)

  return heatmap_image

def get_img_array(img_path, size):
  img = keras.utils.load_img(img_path, target_size=(size, size))
  array = keras.utils.img_to_array(img)
  array = np.expand_dims(array, axis=0)
  return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
  logger.info('make_gradcam_heatmap')
  # First, we create a model that maps the input image to the activations
  # of the last conv layer as well as the output predictions
  grad_model = keras.models.Model(
      model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
  )

  # Then, we compute the gradient of the top predicted class for our input image
  # with respect to the activations of the last conv layer
  with tf.GradientTape() as tape:
      last_conv_layer_output, preds = grad_model(img_array)
      if pred_index is None:
          pred_index = tf.argmax(preds[0])
      class_channel = preds[:, pred_index]

  # This is the gradient of the output neuron (top predicted or chosen)
  # with regard to the output feature map of the last conv layer
  # grads = tape.gradient(class_channel, last_conv_layer_output)

  grads = tape.gradient(class_channel, last_conv_layer_output)

  # This is a vector where each entry is the mean intensity of the gradient
  # over a specific feature map channel
  if grads is None:
      raise Exception("Tape gradients are None")
  else:
      pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

  # We multiply each channel in the feature map array
  # by "how important this channel is" with regard to the top predicted class
  # then sum all the channels to obtain the heatmap class activation
  last_conv_layer_output = last_conv_layer_output[0]
  heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
  heatmap = tf.squeeze(heatmap)

  # For visualization purpose, we will also normalize the heatmap between 0 & 1
  heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

  return heatmap.numpy()

def compute_gradcam(img_path, heatmap, activation_thresh, image_size, alpha=0.4):
  # Load the original image
  #img = keras.utils.load_img(img_path)
  img = tf.squeeze(get_img_array(img_path, size=image_size))

  # Rescale heatmap to a range 0-255
  heatmap = np.uint8(255 * heatmap)
  # Use jet colormap to colorize heatmap
  jet = plt.colormaps["jet"]

  # Use RGB values of the colormap
  jet_colors = jet(np.arange(256))[:, :3]
  jet_heatmap = jet_colors[heatmap]

  jet_heatmap = keras.utils.array_to_img(jet_heatmap)
  jet_heatmap = jet_heatmap.resize((image_size, image_size))
  #heatmap_im = cv2.resize(heatmap, (image_size, image_size));
  jet_heatmap = keras.utils.img_to_array(jet_heatmap)

  # Superimpose the heatmap on original image
  superimposed_img = jet_heatmap * alpha + img
  superimposed_img = keras.utils.array_to_img(superimposed_img)

  CV2_img = cv2.imread(img_path)
  CV2_img = cv2.resize(CV2_img,(image_size, image_size))
  activation_threhold = activation_thresh

  img2gray = cv2.cvtColor(CV2_img,cv2.COLOR_BGR2GRAY)
  ret,heatmask = cv2.threshold(img2gray,activation_threhold,255,cv2.THRESH_BINARY)
  img_fg = cv2.bitwise_and(CV2_img,CV2_img,mask = heatmask)

  # subtracted_img = CV2_img - jet_heatmap
  return heatmask, img_fg, superimposed_img

def RG_sal_maps(image_path, image_size, model, activation_thresh=75):
  # Prepare image
  img_array = preprocess_input(get_img_array(image_path, size=image_size))

  # Generate class activation heatmap
  heatmap = make_gradcam_heatmap(img_array, model, 'block5_conv3')
  lime_map = get_lime_explanation(image_path, model, image_size=image_size)

  # Display heatmap
  heatmask, bitwise_map, superimposed_map = compute_gradcam(image_path, heatmap, activation_thresh=activation_thresh, image_size=image_size)
  return bitwise_map, superimposed_map, lime_map

def compute_gradcam_maps(eobj, operation="BitwiseAND"):
  '''
  Computes Bitwise activation maps
  '''
  model=eobj.model
  all_dirs = eobj.fnames
  logger.info('Generating GradCAM maps for all images')
  output_dir = eobj.outputs
  THIS_DIR = os.getcwd()
  THIS_DIR = os.path.join(THIS_DIR, output_dir, eobj.explainable_method)
  if not os.path.exists(THIS_DIR):
    os.mkdir(THIS_DIR)

  for file in tqdm(all_dirs, desc="Computing maps"):
    bitwise_map, superimposed_map, lime_map = RG_sal_maps(file, eobj.image_size, model)
    # class label
    class_name = file.split("/")[2]
    file_path = os.path.join(THIS_DIR, class_name)
    if not os.path.exists(file_path):
      os.mkdir(file_path)
    file_name_bitwise = "Bitwise_" + class_name +"_"+ file.split("/")[-1]
    file_name = os.path.join(file_path, file_name_bitwise)

    file_name_super = "Superimposed_" + class_name +"_"+ file.split("/")[-1]
    file_name_super = os.path.join(file_path, file_name_super)

    file_name_lime = "Lime_" + class_name +"_"+ file.split("/")[-1]
    file_name_lime = os.path.join(file_path, file_name_lime)

    logger.info('Saving Bitwise activation map for %s', file_name)
    logger.info('Saving Superimposed activation map for %s', file_name_super)
    logger.info('Saving Lime explanation for %s', file_name_lime)

    try:
      cv2.imwrite(file_name, bitwise_map)
    except Exception as e:
      print(f'Exception {e} while writing file {file_name}!')
    
    try:
      superimposed_map.save(file_name_super)
    except Exception as e:
      print(f'Exception {e} while writing file {file_name_super}!')
    
    try:
      cv2.imwrite(file_name_lime, lime_map)
    except Exception as e:
      print(f'Exception {e} while writing file {file_name_lime}!')
    
    continue
    
    # try:
    #   cv2.imwrite(file_name, bitwise_map)
    #   cv2.imwrite(file_name_super, superimposed_map)
    #   cv2.imwrite(file_name_lime, lime_map)
    # except Exception as e:
    #   print(f'Exception {e} while writing file {file}!')
    #   continue

def to_explain(eobj):
  # Set GPU device
  pdb.set_trace()
  physical_devices = tf.config.list_physical_devices('GPU')
  if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
  else:
    print("No GPU devices found. Running on CPU.")

  print ('\n[To explain: SFL (Software Fault Localization) is used]')
  print ('  ### [Measures: {0}]'.format(eobj.measures))
  logger.info('To explain: SFL (Software Fault Localization) is used')
  model=eobj.model
  ## to create output DI
  #print ('\n[Create output folder: {0}]'.format(eobj.outputs))
  di=eobj.outputs

  try:
    os.system('mkdir -p {0}'.format(di))
  except: pass
  if not eobj.boxes is None:

      f = open(di+"/wsol-results.txt", "a")
      f.write('input_name   x_method    intersection_with_groundtruth\n')
      f.close()
  class_name = "class_name"

  for i in range(0, len(eobj.inputs)):
    x=eobj.inputs[i]
    res=model.predict(sbfl_preprocess(eobj, np.array([x])), verbose=10)
    y=np.argsort(res)[0][-eobj.top_classes:]

    print ('\n[Input {2}: {0} / Output Label (to Explain): {1}]'.format(eobj.fnames[i], y, i))

    # if eobj.fnames[i].split("/")[2] is not None:
    #     class_name = eobj.fnames[i].split("/")[2]
    #     print ('  #### [Target Class: {0}]'.format(class_name))
    # else:
    #     print ('  ### Target class not found...')

    ite=0
    reasonable_advs=False
    while ite<eobj.testgen_iter:
      print ('  #### [Start generating SFL spectra...]')
      start=time.time()
      ite+=1

      passing, failing=spectra_sym_gen(eobj, x, y[-1:], adv_value=eobj.adv_value, testgen_factor=eobj.testgen_factor, testgen_size=eobj.testgen_size)
      spectra=[]
      num_advs=len(failing)
      adv_xs=[]
      adv_ys=[]
      for e in passing:
        adv_xs.append(e)
        adv_ys.append(0)
      for e in failing:
        adv_xs.append(e)
        adv_ys.append(-1)
      tot=len(adv_xs)

      adv_part=num_advs*1./tot
      #print ('###### adv_percentage:', adv_part, num_advs, tot)
      end=time.time()
      print ('  #### [SFL spectra generation DONE: passing {0:.2f} / failing {1:.2f}, total {2}; time: {3:.0f} seconds]'.format(1-adv_part, adv_part, tot, end-start))

      if adv_part<=eobj.adv_lb:
        print ('  #### [too few failing tests: SFL explanation aborts]') 
        continue
      elif adv_part>=eobj.adv_ub:
        print ('  #### [too few many tests: SFL explanation aborts]') 
        continue
      else: 
        reasonable_advs=True
        break

    if not reasonable_advs:
      #print ('###### failed to explain')
      continue

    ## to obtain the ranking for Input i
    selement=sbfl_elementt(x, 0, adv_xs, adv_ys, model, eobj.fnames[i])
    # Creates nested directories
    # di = di + '/{0}'.format(class_name)
    dii=di+'/{0}'.format(str(datetime.now()).replace(' ', '-'))
    dii=dii.replace(':', '-')
    os.system('mkdir -p {0}'.format(dii+"_"+class_name))
    for measure in eobj.measures:
      print ('  #### [Measuring: {0} is used]'.format(measure))
      ranking_i, spectrum=to_rank(selement, measure)
      selement.y = y
      diii=dii+'/{0}'.format(measure)
      print ('  #### [Saving: {0}]'.format(diii))
      os.system('mkdir -p {0}'.format(diii))
      np.savetxt(diii+'/ranking_{0}.txt'.format(class_name), ranking_i, fmt='%s')

      # to plot the heatmap
      spectrum = np.array((spectrum/spectrum.max())*255)
      gray_img = np.array(spectrum[:,:,0],dtype='uint8')
      #print (gray_img)
      heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
      if x.shape[2]==1:
          x3d = np.repeat(x[:, :, 0][:, :, np.newaxis], 3, axis=2)
      else: x3d = x
      fin = cv2.addWeighted(heatmap_img, 0.7, x3d, 0.3, 0)
      plt.rcParams["axes.grid"] = False
      plt.imshow(cv2.cvtColor(fin, cv2.COLOR_BGR2RGB))
      plt.savefig(diii+'/heatmap_{0}_{1}.png'.format(measure, class_name if class_name else 'unknown'))
      logger.info('Saved heatmap_{0}_{1}.png - Deepcover'.format(measure, class_name if class_name else 'unknown'))

      # to plot the top ranked pixels
      if not eobj.text_only:
          ret=top_plot(selement, ranking_i, diii, measure, eobj)
          if not eobj.boxes is None:
              f = open(di+"/wsol-results.txt", "a")
              f.write('{0} {1} {2}\n'.format(eobj.fnames[i], measure, ret))
              f.close()

