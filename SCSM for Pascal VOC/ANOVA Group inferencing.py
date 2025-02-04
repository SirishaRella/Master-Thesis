# -*- coding: utf-8 -*-
"""test_2012_resnet50_v2_ANOVA

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11Y5aqzp61H8ePzLDqlqY22uIyejFbA-d
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/gdrive')
!pip install ipython-autotime
# %load_ext autotime
# %cd /content/gdrive/My Drive/Colab Notebooks/Separate Models 2012/
!pip install mxnet-cu100mkl gluoncv==0.5.0
!pip install -e git+https://github.com/tqdm/tqdm.git@master#egg=tqdm
!python -mpip install -U pip
!python -mpip install -U matplotlib
!python -mpip install -U opencv-python

import shutil
import os
import xml.etree.ElementTree as ET
from os import listdir, path
from os.path import isfile, join
import math
import shutil
from pathlib import Path
import ast
backbone = 'resnet50-20071000' #folder name
# backbone = 'resnet101'

CLASSES  = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
group_1  =['boat', 'diningtable', 'horse', 'person']
group_2 = ['bird', 'car', 'cat', 'dog', 'motorbike', 'pottedplant', 'train', 'tvmonitor']
group_3 = ['aeroplane', 'bicycle', 'bottle', 'bus', 'chair', 'cow', 'sheep', 'sofa']

ctx_list = [group_1, group_2, group_3]

artifact_path = 'Test-Artifacts/' + backbone + '/ANOVA/0.95/' 
output_path = artifact_path +'output/'
Path(output_path).mkdir(parents=True, exist_ok=True)

from tqdm import tqdm
import pandas as pd
import glob
import time
from matplotlib import patches
import gluoncv as gcv
import matplotlib.pyplot as plt
from bbox_context import plot_bbox
import matplotlib.pyplot as plot
import mxnet as mx

ctx = mx.gpu(0)
net_ensemble = []
if 'resnet50' in backbone:
    for group in ctx_list:
      net = gcv.model_zoo.get_model('faster_rcnn_resnet50_v1b_voc',  pretrained=True, ctx = ctx)
      net.reset_class(classes=group, reuse_weights=group)
      net_ensemble.append(net)
print(len(net_ensemble))
# else:
#     model_path = cls + '/faster_rcnn_resnet101_v1d_voc_best.params'
#     net = gcv.model_zoo.get_model('faster_rcnn_resnet101_v1d_custom', classes=CLASSES, pretrained_base=False)

shutil.copy('/content/gdrive/My Drive/Colab Notebooks/ContextModels/VOC2012/bbox_context.py', 'bbox_context.py')

"""# **1. Generate Test Datasets**"""

####THE 500 IMAGES###
img_directory = '2007_Test/original/' #change to 2007_Test/original/
img_path = img_directory + 'JPEGImages/*.jpg'
df = pd.read_csv('Test-Artifacts/' + backbone + '/pretrained_collab_fasterRCNN/inference_metrics_with_gt_500.csv')
img_list = []
for idx, row in df.iterrows():
  img_list.append(img_directory + 'JPEGImages/' + row['image_id'])
print(img_list)
display(df)

"""# **2. Inferencing and Creating inference metrics**

*2b. Inferencing based on calculated thresholds
"""

#edit: add threshold as a parameter in each class, create and save dataframes for each class
def inference(threshold):
  column_names = ['image_id','dimension', 'class_name', 'inference_time', 'bboxes', 'scores'] 
  start_time = time.time()

  
  df = pd.DataFrame(columns = column_names)
  artifact_path = 'Test-Artifacts/' + backbone + '/ANOVA/' + str(threshold) + '/' 
  output_path = artifact_path +'output/'
  csv_path = artifact_path + 'inference_metrics.csv'
  Path(output_path).mkdir(parents=True, exist_ok=True)
  print(output_path)
  if os.path.exists(csv_path):
    # print('EXISTS CSV!')
    # os.remove(csv_path)
    # df = pd.DataFrame(columns = column_names)
    # df_size = 0

    df= pd.read_csv(csv_path)
    df_size = len(df)
  else:
    print('NOT EXISTS!')
    df_size = 0
  print(df_size)
 
  for index, filename in enumerate(tqdm(img_list)):
    if index < df_size:
      continue
    # if index == 10:
    #   break
    img_path = filename
    img_score_list = []
    img_bbox_list = []
    img_class_list = []
    
    for idx, group in enumerate(ctx_list):

      print("CURRENT MODEL: ", group)
      net = net_ensemble[idx]
      image_id = filename.split('/')[-1]

      x, image = gcv.data.transforms.presets.rcnn.load_test(img_path, 640)
      cids, scores, bboxes = net(x.as_in_context(ctx))
      
      
      

      print()
      print(output_path)
      print(csv_path)
      # print(model_path)
      print()
  
      ax, co_dict, class_name, score, color, bbox = plot_bbox(image, bboxes[0], scores[0], cids[0],
                                                              class_names=group, thresh=threshold)
    
      coordinate_list = []
      score_list = []
      #Extract Prediction Coordinates
      for j in range(0, len(co_dict['coordinates']['xmin'])):
          xmin = co_dict['coordinates']['xmin'][j]
          ymin = co_dict['coordinates']['ymin'][j]
          xmax = co_dict['coordinates']['xmax'][j]
          ymax = co_dict['coordinates']['ymax'][j]
          score = co_dict['scores'][j]

          coordinate_list.append([xmin, ymin, xmax, ymax])
          score_list.append(score)

      width = plt.xlim()[1] - plt.xlim()[0]
      height = plt.ylim()[0] - plt.ylim()[1]
    
      print(image_id)


      plt.savefig(output_path + image_id)
      plt.close()
      img_score_list += score_list
      img_class_list += class_name
      img_bbox_list += coordinate_list
      print(img_class_list, img_score_list, img_bbox_list)
      # if (idx == 3):
      #   break
    row_data = [[image_id,[width, height], img_class_list, str(time.time() - start_time), img_bbox_list, img_score_list]]
    print(row_data)
    data_df = pd.DataFrame(row_data, columns = column_names)
    df = df.append(data_df, ignore_index = True)
    df.to_csv(csv_path, index=False)
    ctx.empty_cache()

    # break

  print("CURRENT THRESHOLD:", threshold)
  # artifact_path = 'Test-Artifacts/' + backbone + '/' + cls + '/' + str(threshold) + '/'
  # csv_path = artifact_path + 'inference_metrics.csv'
  # csv_path = artifact_path + 'inference_metrics_with_gt.csv'  
  # # display(df_list[index])
  # df = cleanup_df(df_list[index])
  

  display(df)

  # prepare_for_mAP(cls, threshold_list)

  # calculate_mAP(cls, threshold_list)

  # result = extract_mAP(cls, threshold_list)

  # print ("Best mAP of", cls, ":", result[0], 'with threshold:', result[1])
  # return result



def cleanup_df(df):
  gt_bbox_list = []
  gt_classname_list = []
  print(len(df))

  for index, row in df.iterrows():

    ##Extract IMG and XML name
    IMG_name = row['image_id']
    XML_name = row['image_id'].split('.jpg')[0] + '.xml'
    inference_width, inference_height = ast.literal_eval(row['dimension'])
    
    print(index, IMG_name, XML_name, [inference_width, inference_height])
    

    #Fix negative values
    bboxes = ast.literal_eval(row['bboxes'])
    print(row['bboxes'])
    for idx, bbox in enumerate(bboxes):
      print(bbox)
      for x in bbox:
        if x<=0:
          print("NEGATIVE!!!")
      bboxes[idx] = [x if x>=0 else 0 for x in bbox]
    df.loc[df['image_id'] == IMG_name, 'bboxes'] = str(bboxes)


    
    gt_XML_path = img_directory + 'Annotations/' + IMG_name.split('.jpg')[0] + '.xml'
    root = ET.parse(gt_XML_path).getroot()
    for object in root.findall('size'):
      original_width = float(object.find('width').text)
      original_height = float(object.find('height').text)
    
    gt_bboxes = []
    gt_classnames = []
    for object in root.findall('object'):
      className = object.find('name').text
      #original xmin,ymin,xmax,ymax
      boundingBox_xmin = float(object.find('bndbox').find('xmin').text)
      boundingBox_ymin = float(object.find('bndbox').find('ymin').text)
      boundingBox_xmax = float(object.find('bndbox').find('xmax').text)
      boundingBox_ymax = float(object.find('bndbox').find('ymax').text)

      #find factor
      boundingBox_xmin_factor = boundingBox_xmin / original_width 
      boundingBox_ymin_factor = boundingBox_ymin / original_height
      boundingBox_xmax_factor = boundingBox_xmax / original_width
      boundingBox_ymax_factor = boundingBox_ymax / original_height

      #rescale the original bounding boxes
      scaled_boundingBox_xmin = boundingBox_xmin_factor * inference_width
      scaled_boundingBox_ymin = boundingBox_ymin_factor * inference_height
      scaled_boundingBox_xmax = boundingBox_xmax_factor * inference_width
      scaled_boundingBox_ymax = boundingBox_ymax_factor * inference_height
      gt_bboxes.append([scaled_boundingBox_xmin, scaled_boundingBox_ymin ,scaled_boundingBox_xmax, scaled_boundingBox_ymax])
      gt_classnames.append(className)
    gt_bbox_list.append(gt_bboxes)
    gt_classname_list.append(gt_classnames)
  df['gt_bboxes'] = gt_bbox_list
  df['gt_class_name'] = gt_classname_list
  
  # display(df)
  # print("Saving to", output)
  # df.to_csv(output, index=False)
  return df



def prepare_for_mAP(threshold):
  ###PREPARING FOR MAP AND MAP CALCULATION###
  ###MAP SCORE###
  import shutil
  from pathlib import Path
  import ast
  import pandas as pd
  #Preparing and Structuring Data


  #Prepare folder
  test_root = 'Test-Artifacts/' + backbone + '/ANOVA/' + str(threshold) + '/mAP-report/input/' 
  gt_root = test_root + 'ground-truth/'
  prediction_root = test_root + 'detection-results/'
  test_img_write_root = test_root +'images-optional/'

  Path(gt_root).mkdir(parents=True, exist_ok=True)
  Path(prediction_root).mkdir(parents=True, exist_ok=True)
  Path(test_img_write_root).mkdir(parents=True, exist_ok=True)




  filename = 'Test-Artifacts/' + backbone + '/ANOVA/inference_metrics_with_gt.csv'
  df = pd.read_csv(filename)
  size = len(df)
  # display(df)

  for index, row in df.iterrows():
    print(index, '/', size)
    image_id = row['image_id'].split('.jpg')[0]
    image_src_path = '2012_TestImages/' + row['image_id']
    gt_path = gt_root + image_id + '.txt'
    prediction_path = prediction_root + image_id + '.txt'
    image_destination_path = test_img_write_root

    ##GROUND TRUTH VALUES###
    gt_class_names = ast.literal_eval(row['gt_class_name'])
    gt_bboxes = ast.literal_eval(row['gt_bboxes'])

    ##PREDICTION VALUES###
    p_class_names = ast.literal_eval(row['class_name'])
    p_bboxes = ast.literal_eval(row['bboxes'])
    scores = ast.literal_eval(row['scores'])
    # print(p_bboxes, scores)
    # print(gt_bbox)
    # print(p_bbox)
    #print(class_name, gt_path, prediction_path, image_src_path, image_destination_path)
    #print(image_id)
    #Write ground truth
    with open(gt_path, 'w') as g_write:
      for i, gt_bbox in enumerate(gt_bboxes):
        class_name = gt_class_names[i]
        for idx, val in enumerate(gt_bbox):
          gt_bbox[idx] = str(val)
        # print(gt_bbox)
        line = class_name + ' ' + gt_bbox[0] + ' ' + gt_bbox[1] + ' ' + gt_bbox[2] + ' ' + gt_bbox[3] + '\n'
        g_write.write(line)
        print("GROUND TRUTH:", class_name, gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3], '\n')

    #Write prediction results
    with open(prediction_path, 'w') as p_write:
      for i, p_bbox in enumerate(p_bboxes):
        score = str(scores[i])
        class_name = p_class_names[i]
        for idx, val in enumerate(p_bbox):
          p_bbox[idx] = str(val)
        # print(p_bbox, score)
        line = class_name + ' ' + score + ' ' + p_bbox[0] + ' ' + p_bbox[1] + ' ' + p_bbox[2] + ' ' + p_bbox[3] + '\n'
        p_write.write(line)
        print("PREDICTION:", class_name + ' ' + score + ' ' + p_bbox[0] + ' ' + p_bbox[1] + ' ' + p_bbox[2] + ' ' + p_bbox[3] + '\n')
    if os.path.exists(image_src_path):
      shutil.copy(image_src_path, image_destination_path)
    # if index == 3:
    #   break


    
  # break
def calculate_mAP(threshold):
  destination = 'Test-Artifacts/' + backbone + '/ANOVA/' + str(threshold) + '/mAP-report/' 
  shutil.copy('Test-Artifacts/resnet50/SCSM/mAP-report/main.py', destination)
  !python '$destination/main.py' --no-animation --iou 0.5
# def extract_mAP(threshold):
#   mAP_list = [] 
#   for threshold in threshold_list:
#     destination = 'Test-Artifacts/' + backbone + '/' + cls + '/' + str(threshold) + '/mAP-report/output'
#     path = destination + '/output.txt'
#     with open(path) as f:
#       lines = f.readlines()
#       mAP = float(lines[1].split('%')[0]) #get mAP of first class
#       print("Threshold: ", threshold, "mAP:", mAP)
#       mAP_list.append(mAP)
  
#   best_mAP = max(mAP_list)
#   best_mAP_index = mAP_list.index(best_mAP)
#   threshold = THRESHOLD_LIST[best_mAP_index]

#   return [best_mAP, threshold]
inference(0.95)
# extract_mAP()

import pandas as pd

# df = pd.read_csv('Test-Artifacts/' + backbone + '/ANOVA/0.95/inference_metrics.csv')
# # for i in range(2,4):
# #   new_df = pd.read_csv('Test-Artifacts/' + backbone + '/pretrained_collab_fasterRCNN/0.95/inference_metrics_remaining' + str(i) + '.csv')
# #   df = pd.concat([df,new_df], ignore_index = True)

# df = cleanup_df(df)
# df.to_csv('Test-Artifacts/' + backbone + '/ANOVA/inference_metrics_with_gt.csv', index=False)
#prepare_for_mAP(0.95)
calculate_mAP(0.95)

# %ls  'Test-Artifacts/$backbone/pretrained_collab_fasterRCNN/0.95/mAP-report/output' 
# df[3933:3935]
# import collections
# print([item for item, count in collections.Counter(df['image_id'].tolist()).items() if count > 1])

"""Display all test images in a grid format"""

#display all test images

from matplotlib.pyplot import figure, imshow, axis, savefig
from matplotlib.image import imread

import matplotlib.pyplot as plt

def grid_display(list_of_images, list_of_titles=[], no_of_columns=2, figsize=(10,10), theme=""):

    fig = figure(figsize=figsize)
    fig.suptitle(theme, fontsize=30, x = 0.52, y = 0.57)
    fig.tight_layout()
    column = 0
    for i in range(len(list_of_images)):
        column += 1
        #  check for end of column and create a new figure
        if column == no_of_columns+1:
            fig = figure(figsize=figsize)
            column = 1
        fig.add_subplot(1, no_of_columns, column)
        imshow(imread(list_of_images[i]))
        axis('off')
    savefig('aeroplane_test.png')
for cls in CLASSES:
  print (cls)
  file_list = []
  for img in os.listdir('Test-Artifacts/' + cls + '/output'):
    file_list.append('Test-Artifacts/' + cls + '/output/' + img)
  grid_display(file_list, no_of_columns=5, figsize = (30,35), theme = cls)

  print()
  print()

"""# **3. Fix Negative Values, Extract Ground Truth Bounding Boxes, Rescale, and save to CSV file**"""

from pathlib import Path
artifact_path = 'Test-Artifacts/' + backbone + '/pretrained_collab_fasterRCNN/' 
csv_path = artifact_path + 'inference_metrics.csv'
img_directory = '2007_Test/original/' #change to 2007_Test/original/
img_path = img_directory + 'JPEGImages/*.jpg'
output_path = artifact_path +'output/'
Path(output_path).mkdir(parents=True, exist_ok=True)

#Get Enhanced Predict BBoxes
import xml.etree.ElementTree as ET
import ast
import pandas as pd
filename = 'Test-Artifacts/' + backbone + '/pretrained_collab_fasterRCNN/inference_metrics.csv'
output = 'Test-Artifacts/' + backbone + '/pretrained_collab_fasterRCNN/inference_metrics_with_gt.csv'
df = pd.read_csv(filename) 

gt_bbox_list = []
gt_classname_list = []
for index, row in df.iterrows():
  #Fix negative values -- Cleaning up
  bboxes = ast.literal_eval(row['bboxes'])
  for idx, bbox in enumerate(bboxes):
    bboxes[idx] = [x if x>=0 else 0 for x in bbox]
  # print(df.at[index, 'bboxes'])
  df.at[index, 'bboxes'] = bboxes


  ##Extract IMG and XML name
  IMG_name = row['image_id']
  XML_name = row['image_id'].split('.jpg')[0] + '.xml'
  inference_width, inference_height = ast.literal_eval(row['dimension'])
  
  print(index, IMG_name, XML_name, [inference_width, inference_height])

  ###XML File Path###
  gt_XML_path = img_directory + 'Annotations/' + IMG_name.split('.jpg')[0] + '.xml'
  #################
  root = ET.parse(gt_XML_path).getroot()
  for object in root.findall('size'):
    original_width = float(object.find('width').text)
    original_height = float(object.find('height').text)
  
  gt_bboxes = []
  gt_classnames = []
  for object in root.findall('object'):
    className = object.find('name').text
    # print(className)
    #original xmin,ymin,xmax,ymax
    boundingBox_xmin = float(object.find('bndbox').find('xmin').text)
    boundingBox_ymin = float(object.find('bndbox').find('ymin').text)
    boundingBox_xmax = float(object.find('bndbox').find('xmax').text)
    boundingBox_ymax = float(object.find('bndbox').find('ymax').text)

    #find factor
    boundingBox_xmin_factor = boundingBox_xmin / original_width 
    boundingBox_ymin_factor = boundingBox_ymin / original_height
    boundingBox_xmax_factor = boundingBox_xmax / original_width
    boundingBox_ymax_factor = boundingBox_ymax / original_height

    #rescale the original bounding boxes
    scaled_boundingBox_xmin = boundingBox_xmin_factor * inference_width
    scaled_boundingBox_ymin = boundingBox_ymin_factor * inference_height
    scaled_boundingBox_xmax = boundingBox_xmax_factor * inference_width
    scaled_boundingBox_ymax = boundingBox_ymax_factor * inference_height
    gt_bboxes.append([scaled_boundingBox_xmin, scaled_boundingBox_ymin ,scaled_boundingBox_xmax, scaled_boundingBox_ymax])
    gt_classnames.append(className)
    # print(gt_classnames)
  gt_bbox_list.append(gt_bboxes)
  gt_classname_list.append(gt_classnames)
  # if index == 3:
  #   print(len(gt_bbox_list))
  #   print(len(gt_classname_list))
  #   break
df['gt_bboxes'] = gt_bbox_list
df['gt_class_name'] = gt_classname_list
display(df)
print("Saving to", output)
df.to_csv(output, index=False)

"""# **4. MAP CALCULATION**"""

###PREPARING FOR MAP AND MAP CALCULATION###
###MAP SCORE###
import shutil
from pathlib import Path
import ast
import pandas as pd
#Preparing and Structuring Data


#Prepare folder
test_root = 'Test-Artifacts/' + backbone + '/pretrained_collab_fasterRCNN/mAP-report/input/' 
gt_root = test_root + 'ground-truth/'
prediction_root = test_root + 'detection-results/'
test_img_write_root = test_root +'images-optional/'

Path(gt_root).mkdir(parents=True, exist_ok=True)
Path(prediction_root).mkdir(parents=True, exist_ok=True)
Path(test_img_write_root).mkdir(parents=True, exist_ok=True)




filename = 'Test-Artifacts/' + backbone + '/pretrained_collab_fasterRCNN/inference_metrics_with_gt.csv'
df = pd.read_csv(filename)

img_list = df['image_id'].tolist()
img_list
import collections
duplicate_list = [item for item, count in collections.Counter(img_list).items() if count > 1]
len(duplicate_list)
# print(duplicate_list)
from pprint import pprint
pprint(duplicate_list)
# display(df[479:480]['scores'])
# scores = ast.literal_eval(df[479:480]['scores'])
# for index, row in df.iterrows():
#   if index <= 576:
#     continue
#   print(index)
#   image_id = row['image_id'].split('.jpg')[0]
#   image_src_path = '2012_TestImages/' + row['image_id']
#   gt_path = gt_root + image_id + '.txt'
#   prediction_path = prediction_root + image_id + '.txt'
#   image_destination_path = test_img_write_root

#   print(image_id)
#   ##GROUND TRUTH VALUES###
#   gt_class_names = ast.literal_eval(row['gt_class_name'])
#   gt_bboxes = ast.literal_eval(row['gt_bboxes'])

#   ##PREDICTION VALUES###
#   p_class_names = ast.literal_eval(row['class_name'])
#   p_bboxes = ast.literal_eval(row['bboxes'])
#   print(row['scores'])
#   scores = ast.literal_eval(row['scores'])
#   # print(p_bboxes, scores)
#   # print(gt_bbox)
#   # print(p_bbox)
#   # print(class_name, gt_path, prediction_path, image_src_path, image_destination_path)
  
#   #Write ground truth
#   with open(gt_path, 'w') as g_write:
#     for i, gt_bbox in enumerate(gt_bboxes):
#       class_name = gt_class_names[i]
#       for idx, val in enumerate(gt_bbox):
#         gt_bbox[idx] = str(val)
#       # print(gt_bbox)
#       line = class_name + ' ' + gt_bbox[0] + ' ' + gt_bbox[1] + ' ' + gt_bbox[2] + ' ' + gt_bbox[3] + '\n'
#       g_write.write(line)
#       print("GROUND TRUTH:", class_name, gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3], '\n')

#   #Write prediction results
#   with open(prediction_path, 'w') as p_write:
#     for i, p_bbox in enumerate(p_bboxes):
#       score = str(scores[i])
#       class_name = p_class_names[i]
#       for idx, val in enumerate(p_bbox):
#         p_bbox[idx] = str(val)
#       # print(p_bbox, score)
#       line = class_name + ' ' + score + ' ' + p_bbox[0] + ' ' + p_bbox[1] + ' ' + p_bbox[2] + ' ' + p_bbox[3] + '\n'
#       p_write.write(line)
#       print("PREDICTION:", class_name + ' ' + score + ' ' + p_bbox[0] + ' ' + p_bbox[1] + ' ' + p_bbox[2] + ' ' + p_bbox[3] + '\n')
#   if os.path.exists(image_src_path):
#     shutil.copy(image_src_path, image_destination_path)

shutil.copy('Test-Artifacts/resnet50/regular_fasterRCNN/mAP-report/main.py', 'Test-Artifacts/' + backbone + '/pretrained_collab_fasterRCNN/mAP-report')
!python 'Test-Artifacts/$backbone/pretrained_collab_fasterRCNN/mAP-report/main.py' --no-animation --iou 0.5

#intersection over union: 
# 2 bounding boxes
# overlapping region

"""# **5. Create Inspection CSV and Generate Confusion Matrix**"""

import pandas as pd
import ast

backbone = 'resnet50-20071000' #folder name
# backbone = 'resnet101'

CLASSES  = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']



df = pd.read_csv('Test-Artifacts/' +  backbone + '/ANOVA/inference_metrics_with_gt.csv')

def create_inspect_csv(df):
  column_names =  ['image_id', 'prediction', 'ground_truth', 'false_positives']
  column_names2 =  ['image_id', 'prediction', 'ground_truth', 'false_positives', 'pt_bboxes', 'gt_bboxes', 'dimension']
  result_df = pd.DataFrame(columns = column_names)
  visualization_df = pd.DataFrame(columns = column_names2)
  size = len(df)
  for index, row in df.iterrows():
    print(index, '/', size)
    image_id = row['image_id']
    
    prediction= list(set(ast.literal_eval(row['class_name'])))
    gt = list(set(ast.literal_eval(row['gt_class_name'])))
    fp = [x for x in prediction if x not in gt]
    data = [[image_id, prediction, gt, fp]]
    print(data)
    data_row = pd.DataFrame(data = data, columns = column_names)
    result_df = result_df.append(data_row, ignore_index = True)
    
    dimension = ast.literal_eval(row['dimension'])
    prediction= ast.literal_eval(row['class_name'])
    gt = ast.literal_eval(row['gt_class_name'])
    p_bbox = ast.literal_eval(row['bboxes'])
    gt_bbox = ast.literal_eval(row['gt_bboxes'])
    data2 = [[image_id, prediction, gt, fp, p_bbox, gt_bbox, dimension]]

    data_row2 = pd.DataFrame(data = data2, columns = column_names2)
    visualization_df = visualization_df.append(data_row2, ignore_index = True)
  # print(new_df['class_name'][0])
  display(result_df)
  result_df.to_csv('Test-Artifacts/' +  backbone + '/ANOVA/predict_truth_report.csv')
  display(visualization_df)
  visualization_df.to_csv('Test-Artifacts/' +  backbone + '/ANOVA/predict_truth_report_with_bboxes.csv', index = False)
  return visualization_df
inspect_table = create_inspect_csv(df)

from sklearn.metrics import confusion_matrix
confusion_columns = ['ground_truth', 'prediction']
confusion_df = pd.DataFrame(columns = confusion_columns)
display(inspect_table.head())
IOU = 0.5
confusion_dict = {}
true_match = {}
fn_dict = {}
size = len(inspect_table)
for index, row in inspect_table.iterrows(): #for every image
  print(index, '/', size)
  display(row)
  print(row['image_id'])
  prediction_data = row['pt_bboxes']
  pd_classes = row['prediction']
  ground_truth_data = row['gt_bboxes']
  gt_classes = row['ground_truth']
  gt_match = {}
  
  for i, bb in enumerate(prediction_data): #for every pd bbox
      pd_class_name = pd_classes[i]
      print(pd_class_name, bb)
      ovmax = -1
      current_match_idx = -1
      for idx, bbgt in enumerate(ground_truth_data): #for every gt bbox
          gt_class_name = gt_classes[idx]
          # look for a class_name match
          print('\t--->', gt_class_name, bbgt)
          ov = -1
          #CALCULATE OVERLAP
          bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])] #intersection bbox
          iw = bi[2] - bi[0] + 1
          ih = bi[3] - bi[1] + 1
          if iw > 0 and ih > 0:
              # compute overlap (IoU) = area of intersection / area of union
              ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                              + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
              ov = iw * ih / ua
              if ov > ovmax:
                  ovmax = ov
                  current_match_idx = idx #gt index
      
          print('\toverlap:', ov)
          print('\tmax overlap:', ovmax, 'index:', current_match_idx)
          print()
      if (ovmax >= IOU):
        if pd_class_name == gt_classes[current_match_idx]:  #same box same class
          if current_match_idx in gt_match: #duplicate
            print("DUPLICATED PREDICTION! ALREADY MATCHED! NOT CONSIDERED!")
          else: #new match
            gt_match[current_match_idx] = i
            print("A NEW MATCH -> GT: {} and PD: {}".format(current_match_idx, i))
            if gt_classes[current_match_idx] in true_match:
               true_match[gt_classes[current_match_idx]].append(pd_class_name)
            else:
               true_match[gt_classes[current_match_idx]] = [pd_class_name]
        else: #same box but different classes
          print("CONFUSION BETWEEN {} and {}".format(pd_class_name, gt_classes[current_match_idx]))
          if gt_classes[current_match_idx] in confusion_dict:
            confusion_dict[gt_classes[current_match_idx]].append(pd_class_name)
          else:
            confusion_dict[gt_classes[current_match_idx]] = [pd_class_name]
      else: #totally different box => extra detection
        print("TOTALLY DIFFERENT BOX")
        print("EXTRA {} DETECTED".format(pd_class_name))
        if 'other' in confusion_dict:
          confusion_dict['other'].append(pd_class_name)
        else:
          confusion_dict['other'] = [pd_class_name]
        print()
  print(gt_match.keys())
  matched_gt = gt_match.keys()
  false_negatives = [gt_classes[x] for x in range(len(gt_classes)) if x not in matched_gt]
  print('FALSE NEGATIVES:', false_negatives)
  print('CONFUSION DICT:', confusion_dict)
  print()
  print('########')
  for fn in false_negatives:
    if fn in fn_dict:
      fn_dict[fn] += 1
    else:
      fn_dict[fn] = 1
  # if (index == 100):
  #   break


y_pred = []
y_true = []
print("#######CONFUSION DICT EXPANSION########")
for gt, preds in confusion_dict.items():
  for pred in preds:
    print("GT: {} <- PD: {}".format(gt, pred))
    y_pred.append(pred)
    y_true.append(gt)
print("#######REAL MATCH EXPANSION########")
for gt, preds in true_match.items():
  for pred in preds:
    print("GT: {} <- PD: {}".format(gt, pred))
    y_pred.append(pred)
    y_true.append(gt)
print("#######FALSE NEGATIVES########")
print(fn_dict) 
confusion_matrix = confusion_matrix(y_true, y_pred, labels=CLASSES + ['other'])
confusion_matrix

fn_df = pd.DataFrame(columns = ['false_negatives'])
for key, value in sorted(fn_dict.items()):
  row_data = pd.DataFrame(data = [[value]], columns = ['false_negatives'])
  fn_df = fn_df.append(row_data, ignore_index = True )
fn_df = fn_df.set_index([pd.Index(sorted(fn_dict.keys()))])
fn_df.to_csv('Test-Artifacts/' +  backbone + '/ANOVA/fn_report.csv', index = False)
fn_dict, fn_df

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

output_path = 'Test-Artifacts/' +  backbone + '/ANOVA/confusion_matrix.png'
#'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 
# 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 
# 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2',
#  'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 
#  'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 
#  'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 
#  'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 
#  'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 
#  'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 
#  'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 
#  'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 
#  'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_...
confusion_list = confusion_matrix.tolist()
gt_classes = []
pt_classes = []
val_list = []
new_classes = CLASSES + ['other']
for i, row in enumerate(confusion_list):
  gt_class = new_classes[i]
  for idx, col in enumerate(row):
    pt_class = new_classes[idx]
    # print(gt_class, pt_class, col)
    gt_classes.append(gt_class)
    pt_classes.append(pt_class)
    val_list.append(col)
  # break
gt_classes, pt_classes, val_list
df = pd.DataFrame({'Ground_Truth':gt_classes,'Prediction':pt_classes,'Occurences':val_list})
df = df.pivot_table(index='Ground_Truth', columns='Prediction', values='Occurences')
##MOVE COL###
cols = list(df)
print(cols)
cols.insert(len(cols), cols.pop(14))
print(cols)
df = df.reindex(columns= cols)

##MOVE ROW###
rows = df.index.tolist()
print(rows)
rows.insert(len(rows), rows.pop(14))
df = df.reindex(rows)
print(backbone)
df.to_csv('Test-Artifacts/' +  backbone + '/ANOVA/confusion_matrix.csv', index = False)
plt.figure(figsize = (30,15))
sn.heatmap(df, annot=True, annot_kws={"size": 20}, cmap='CMRmap', cbar=False)
plt.savefig(output_path)

