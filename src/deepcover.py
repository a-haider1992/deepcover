
from keras.preprocessing import image
from keras.applications import vgg16
from keras.applications.vgg16 import VGG16
from keras.applications import inception_v3, mobilenet, xception
from keras.models import load_model
import matplotlib.pyplot as plt
import csv
import cv2

import argparse
import os
import numpy as np
import shutil

from utils import *
from to_explain import *
from comp_explain import *
from RPN import *

import logging

def main():
  logger = logging.getLogger(__name__)
  logging.basicConfig(filename='logging.log', level=logging.INFO)
  parser=argparse.ArgumentParser(description='To explain neural network decisions' )
  parser.add_argument(
    '--model', dest='model', default='-1', help='the input neural network model (.h5)')
  parser.add_argument("--inputs", dest="inputs", default="-1",
                    help="the input test data directory", metavar="DIR")
  parser.add_argument("--outputs", dest="outputs", default="outs",
                    help="the outputput test data directory", metavar="DIR")
  parser.add_argument("--measures", dest="measures", default=['tarantula', 'zoltar', 'ochiai', 'wong-ii'],
                    help="the SFL measures (tarantula, zoltar, ochiai, wong-ii)", metavar="" , nargs='+')
  parser.add_argument("--measure", dest="measure", default="None",
                    help="the SFL measure", metavar="MEASURE")
  parser.add_argument("--mnist-dataset", dest="mnist", help="MNIST dataset", action="store_true")
  parser.add_argument("--normalized-input", dest="normalized", help="To normalize the input", action="store_true")
  parser.add_argument("--cifar10-dataset", dest="cifar10", help="CIFAR-10 dataset", action="store_true")
  parser.add_argument("--grayscale", dest="grayscale", help="MNIST dataset", action="store_true")
  parser.add_argument("--vgg16-model", dest='vgg16', help="vgg16 model", action="store_true")
  parser.add_argument("--inception-v3-model", dest='inception_v3', help="inception v3 model", action="store_true")
  parser.add_argument("--xception-model", dest='xception', help="Xception model", action="store_true")
  parser.add_argument("--mobilenet-model", dest='mobilenet', help="mobilenet model", action="store_true")
  parser.add_argument("--attack", dest='attack', help="to atatck", action="store_true")
  parser.add_argument("--text-only", dest='text_only', help="for efficiency", action="store_true")
  parser.add_argument("--input-rows", dest="img_rows", default="224",
                    help="input rows", metavar="INT")
  parser.add_argument("--input-cols", dest="img_cols", default="224",
                    help="input cols", metavar="INT")
  parser.add_argument("--input-channels", dest="img_channels", default="3",
                    help="input channels", metavar="INT")
  parser.add_argument("--x-verbosity", dest="x_verbosity", default="0",
                    help="the verbosity level of explanation output", metavar="INT")
  parser.add_argument("--top-classes", dest="top_classes", default="1",
                    help="check the top-xx classifications", metavar="INT")
  parser.add_argument("--adversarial-ub", dest="adv_ub", default="1.",
                    help="upper bound on the adversarial percentage (0, 1]", metavar="FLOAT")
  parser.add_argument("--adversarial-lb", dest="adv_lb", default="0.",
                    help="lower bound on the adversarial percentage (0, 1]", metavar="FLOAT")
  parser.add_argument("--masking-value", dest="adv_value", default="234",
                    help="masking value for input mutation", metavar="INT")
  parser.add_argument("--testgen-factor", dest="testgen_factor", default="0.2",
                    help="test generation factor (0, 1]", metavar="FLOAT")
  parser.add_argument("--testgen-size", dest="testgen_size", default="2000",
                    help="testgen size ", metavar="INT")
  parser.add_argument("--testgen-iterations", dest="testgen_iter", default="1",
                    help="to control the testgen iteration", metavar="INT")
  parser.add_argument("--causal", dest='causal', help="causal explanation", action="store_true")
  parser.add_argument("--explainable-method", dest='explainable_method', help="Specify your preferred explainability method", type=str, default=None)
  parser.add_argument("--wsol", dest='wsol_file', help="weakly supervised object localization", metavar="FILE")
  parser.add_argument("--occlusion", dest='occlusion_file', help="to load the occluded images", metavar="FILE")

  args=parser.parse_args()

  # logger.info(args)
  logger.info(time.strftime("%Y-%m-%d %H:%M:%S"))

  img_rows, img_cols, img_channels = int(args.img_rows), int(args.img_cols), int(args.img_channels)

  ## some common used datasets
  if args.mnist:
    img_rows, img_cols, img_channels = 28, 28, 1
  elif args.cifar10:
    img_rows, img_cols, img_channels = 32, 32, 3
  elif args.inception_v3 or args.xception:
    img_rows, img_cols, img_channels = 299, 299, 3

  ## to load the input DNN model
  if args.model!='-1':
    dnn=load_model(args.model)
  elif args.vgg16:
    print ('to load VGG16')
    dnn=VGG16()
    print ('done')
  elif args.mobilenet:
    dnn=mobilenet.MobileNet()
  elif args.inception_v3:
    dnn=inception_v3.InceptionV3()
  elif args.xception:
    dnn=xception.Xception()
  else:
    raise Exception ('A DNN model needs to be provided...')

  ## to load the input data
  fnames=[]
  xs=[]
  if args.inputs!='-1':
    for path, subdirs, files in os.walk(args.inputs):
      for name in files:
        fname=(os.path.join(path, name))
        if fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.JPEG'):
            if args.grayscale is True or args.mnist:
              x=image.load_img(fname, target_size=(img_rows, img_cols), color_mode = "grayscale")
              x=np.expand_dims(x,axis=2)
            else: 
              x=image.load_img(fname, target_size=(img_rows, img_cols))
            x=np.expand_dims(x,axis=0)
            xs.append(x)
            fnames.append(fname)
  else:
    raise Exception ('What do you want me to do?')
  xs=np.vstack(xs)
  xs = xs.reshape(xs.shape[0], img_rows, img_cols, img_channels)
  print ('\n[Total data loaded: {0}]'.format(len(xs)))

  eobj=explain_objectt(dnn, xs)
  eobj.outputs=args.outputs
  eobj.top_classes=int(args.top_classes)
  eobj.adv_ub=float(args.adv_ub)
  eobj.adv_lb=float(args.adv_lb)
  eobj.adv_value=float(args.adv_value)
  eobj.testgen_factor=float(args.testgen_factor)
  eobj.testgen_size=int(args.testgen_size)
  eobj.testgen_iter=int(args.testgen_iter)
  eobj.vgg16=args.vgg16
  eobj.mnist=args.mnist
  eobj.cifar10=args.cifar10
  eobj.inception_v3=args.inception_v3
  eobj.xception=args.xception
  eobj.mobilenet=args.mobilenet
  eobj.attack=args.attack
  eobj.text_only=args.text_only
  eobj.normalized=args.normalized
  eobj.x_verbosity=int(args.x_verbosity)
  eobj.fnames=fnames
  eobj.occlusion_file=args.occlusion_file
  eobj.image_size=img_rows
  eobj.explainable_method = args.explainable_method
  measures = []
  if not args.measure=='None':
      measures.append(args.measure)
  else: measures = args.measures
  eobj.measures=measures

  if not args.wsol_file is None:
      print (args.wsol_file)
      boxes={}
      with open(args.wsol_file, 'r') as csvfile:
        res=csv.reader(csvfile, delimiter=' ')
        for row in res:
          boxes[row[0]]=[int(row[1]), int(row[2]), int(row[3]), int(row[4])]
      eobj.boxes=boxes

  def preprocess():
    fnames=[]
    root = os.path.dirname(os.path.dirname(__file__))
    if args.inputs!='-1':
      for path, subdirs, files in os.walk(args.inputs):
        for subdir in subdirs:
          # class_name = subdir.split("_")[-1] if "_" in subdir else "unknown"
          direc_name = os.path.join(path, subdir)
          direcs = os.listdir(direc_name)
          for direc in direcs:
            if direc == "tarantula":
              tarantula_path = os.path.join(direc_name, direc)
              tarantula_files = os.listdir(tarantula_path)
              for tarantula_file in tarantula_files:
                if "heatmap" in tarantula_file:
                  class_name = tarantula_file.split("_")[-1].split(".")[0]
                # if "explanation-found" in tarantula_file:
                if tarantula_file == "tarantula-5.jpg":
                  fname=(os.path.join(root, tarantula_path, tarantula_file))
                  obj = {"fname": fname, "class": class_name, "folder_name": subdir}
                  fnames.append(obj)

    patch_size = (224, 224)
    num_clusters = 2
    # Convert the image to grayscale

    directory_to_delete = eobj.outputs

    # Attempt to delete the directory
    try:
        # Use shutil.rmtree() to delete the directory and all its contents recursively
        shutil.rmtree(directory_to_delete)
        print(f"Directory '{directory_to_delete}' deleted successfully.")
    except FileNotFoundError:
        print(f"Directory '{directory_to_delete}' does not exist.")
    except Exception as e:
        print(f"An error occurred while deleting directory '{directory_to_delete}': {e}")

    with open("patch-data.txt", "w") as f:
      for files in fnames:
        image_path = files["fname"]
        class_name = files["class"]
        folder_name = files["folder_name"]
        # print("class_name: ", class_name)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Make directory for each Output folder
        os.makedirs(os.path.join(eobj.outputs, folder_name))
        
        previous_row = None
        start_row = None
        w, h = patch_size[0], patch_size[1]  
        roi_patches = []

        # Loop through each row of the image
        # for row_idx, row in enumerate(gray_image):
        #   # If it's not the first row, compare with the previous row

        #   if previous_row is not None:
        #       # Check if there's a change in pixel values
        #       if not (previous_row == row).all():
        #           # If change detected, extract rectangle along columns
        #           if start_row is not None:
        #               end_row = row_idx + h
        #               if end_row - start_row < h:
        #                   continue
        #               # Loop through each column
        #               cropped_image = gray_image[start_row:end_row, :]
        #               previous_col = None
        #               start_col = None
        #               for col_idx, col in enumerate(cropped_image.T):
        #                   # If it's not the first column, compare with the previous column
        #                   if previous_col is not None:
        #                     # Check if there's a change in pixel values
        #                     if not (previous_col == col).all():
        #                         # If change detected, extract rectangle along rows
        #                         if start_col is not None:
        #                             end_col = col_idx + w
        #                             if end_col - start_col < w:
        #                                 continue
        #                             extracted_patch = image[start_row:end_row, start_col:end_col]
        #                             resized_patch = cv2.resize(extracted_patch, patch_size)
        #                             roi_patches.append(resized_patch)
        #                             start_col = col_idx + (w // 4)
        #                   else:
        #                       start_col = col_idx
        #                   previous_col = col
        #           start_row = row_idx + (h // 4)
        #   # Update previous_row to current row
        #   previous_row = row

        cv2.imwrite("{0}/original_image.jpg".format(os.path.join(eobj.outputs, folder_name)), image)
        
        roi_patches, _, _ = RPN(image)

        for i, patch in enumerate(roi_patches):
          patch_name = "{0}/{1}.jpg".format(os.path.join(eobj.outputs, folder_name), i)
          # make patch compatible with cv2
          patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
          patch = cv2.resize(patch, patch_size)
          cv2.imwrite(patch_name, patch)
          f.write("{0},{1},{2}\n".format(folder_name, class_name, patch_name))

        
        ## Kmeans clustering
        # roi_patches = []
        # os.makedirs(os.path.join(eobj.outputs, "Kmeans", folder_name))
      
        # # Apply clustering algorithm to segment the image
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # _, labels, centers = cv2.kmeans(np.float32(gray_image.reshape(-1, 1)), num_clusters, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

        # # # pdb.set_trace()
        
        # # # Convert the labels to uint8 for visualization
        # segmented_image = np.uint8(labels.reshape(gray_image.shape))
        
        # # # Find contours in the segmented image
        # contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # # Extract patches for each contour (ROI)
        # for contour in contours:
        #     # Get the bounding box of the contour
        #     x, y, w, h = cv2.boundingRect(contour)
            
        #     # Extract the patch from the original image
        #     roi_patch = image[y:y+h, x:x+w]
            
        #     # Resize the patch to the specified patch size
        #     roi_patch = cv2.resize(roi_patch, patch_size)
            
        #     # Add the patch to the list of ROI patches
        #     roi_patches.append(roi_patch)

        # # if not os.path.exists(eobj.outputs):
        # # os.makedirs(os.path.join(eobj.outputs, folder_name))

        # # cv2.imwrite("{0}/segmented_image.jpg".format(os.path.join(eobj.outputs, class_name, folder_name)), segmented_image)
        # cv2.imwrite("{0}/original_image.jpg".format(os.path.join(eobj.outputs, "Kmeans", folder_name)), image)

        # for i, patch in enumerate(roi_patches):
        #   patch = cv2.resize(patch, patch_size)
        #   patch_name = "{0}/{1}.jpg".format(os.path.join(eobj.outputs, "Kmeans", folder_name), i)
        #   cv2.imwrite(patch_name, patch)
        #   f.write("{0},{1},{2}\n".format(folder_name, class_name, patch_name))
        
  # preprocess()

  # RPN("original_image.jpg")

  if args.causal:
    comp_explain(eobj)
  elif args.explainable_method == "GradCam" or args.explainable_method == "Lime":
    logger.info("Using GradCam or Lime")
    compute_gradcam_maps(eobj)
  elif args.explainable_method == "DeepCover":
    logger.info("Using DeepCover")
    to_explain(eobj)



if __name__=="__main__":
  main()

