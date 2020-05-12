import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import ast
import matplotlib.patches as patches
from PIL import Image
import cv2


df = pd.read_csv('C:/Users/Siri/Desktop/Final Thesis Results/SCSM FasterRCNN-2 -ANOVA/ANOVA/ANOVA/inference_metrics_with_gt.csv')

image_id = df.iloc[:,0]
dimensions = df.iloc[:,1]
class_names = df.iloc[:,2]
bboxes = df.iloc[:,4]
scores = df.iloc[:,5]

# print(image_id)
# print(dimensions[0])
# print(class_names)
# print(len(bboxes[0]))
# print(scores)
# print(dimensions[0])
print(len(image_id))
for i in range(0, len(image_id)):

    dim = ast.literal_eval(dimensions[i])
    print(image_id)
    img = cv2.imread('C:/Users/Siri/Downloads/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/'+image_id[i])
    size = (int(dim[0]), int(dim[1]))
    print(type(size))
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(resized)
    bboxes[i] = ast.literal_eval(bboxes[i])
    class_names1 = ast.literal_eval(class_names[i])
    print(class_names1)
    scores1 = ast.literal_eval(scores[i])
    print(len(bboxes[i]))
    # Create a Rectangle patch
    for j in range(0, len(bboxes[i])):
        print(bboxes[i])

        width = int(bboxes[i][j][2])-int(bboxes[i][j][0])
        height = int(bboxes[i][j][3])-int(bboxes[i][j][1])
        rect = patches.Rectangle((int(bboxes[i][j][0]), int(bboxes[i][j][1])),width , height, linewidth=3, edgecolor='g', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        ax.text(int(bboxes[i][j][0]), int(bboxes[i][j][1])- 2,
                 '{:s} {:s}'.format(class_names1[j], scores1[j]),
                 bbox=dict(facecolor='g', alpha=0.5),
                 fontsize=12, color='white')

    plt.savefig('C:/Users/Siri/Desktop/Final Thesis Results/SCSM FasterRCNN-2 -ANOVA/ANOVA/ANOVA/0.95/output_my_color/'+image_id[i])


# with open('checking_best.txt','w') as f_write:
#     f_write.write('scsm Class names|Regular class names')
#     f_write.write("\n")
#     import pandas as pd
#
#     df = pd.read_csv('C:/Users/Siri/Desktop/Final Thesis Results/SCSM Faster RCNN/inference_metrics_with_gt_500.csv')
#     df1 = pd.read_csv('C:/Users/Siri/Desktop/Final Thesis Results/regular_fasterRCNN/inference_metrics_with_gt_500_regular.csv')
#
#
#     c =df.iloc[:,2]
#     c_gt = df.iloc[:,-1]
#     c1= df1.iloc[:,2]
#     for i in range(0, 500):
#         print(df.iloc[:,0][i])
#         image_id = df.iloc[:,0][i]
#         for j in range(0, 500):
#             #print(image_id)
#             if image_id == df1.iloc[:,0][j]:
#
#                 f_write.write(image_id + "|")
#                 f_write.write(str(c[i]) + "|")
#                 f_write.write(str(c1[j]) + "|")
#                 f_write.write(c_gt[i])
#                 f_write.write("\n")

# import ast
# with open('checking_best.txt', 'r') as f_read:
#     lines = f_read.readlines()
#     for line in lines:
#         split_line = line.split('|')
#         scsm = ast.literal_eval(split_line[1])
#         collaborative = ast.literal_eval(split_line[2])
#         gt = ast.literal_eval(split_line[3])
#         if len(scsm)== len(collaborative) and len(scsm) == len(gt) and len(scsm)>1:
#             print(split_line[0])

#Checking for Anova Samples:
# import ast
# with open('anova_validation.txt', 'w') as f_write:
#     f_write.write("image_id|wrong predictions|ground_truth")
#     f_write.write("\n")
#     with open('checking_best.txt', 'r') as f_read:
#         lines = f_read.readlines()
#         for line in lines:
#             split_line = line.split('|')
#             scsm = ast.literal_eval(split_line[1])
#             collaborative = ast.literal_eval(split_line[2])
#             gt = ast.literal_eval(split_line[3])
#
#             false = []
#             for i in range(0, len(scsm)):
#
#                 if scsm[i] not in gt:
#                     false.append(scsm[i])
#
#             if len(false) >0:
#                 f_write.write(split_line[0] + '|')
#                 f_write.write(str(false) + '|')
#                 f_write.write(str(gt))
#
#                 f_write.write("\n")


# import pandas as pd
# df = pd.read_csv('VOC 2007 Anova matrix.csv')
#
# l = df.values.tolist()
# print(len(l))
#
# with open('counting_threhsolds_0.5.txt','w') as f_write:
#     for i in range(0, len(l)):
#         count =0
#         for j in range(1, len(l[i])):
#             print(l[i][j])
#             if l[i][j] > 0.5:
#                 count = count + 1
#         f_write.write(str(l[i][0])+ "|")
#         f_write.write(str(count))
#         f_write.write("\n")

# import pandas as pd
# df = pd.read_csv('VOC 2007 Anova matrix.csv')
#
# l = df.values.tolist()
# print(len(l))
#
# with open('counting_threhsolds_0.5.txt','w') as f_write:
#     for i in range(0, len(l)):
#         count =0
#         for j in range(1, len(l[i])):
#             print(l[i][j])
#             if l[i][j] > 0.5:
#                 count = count + 1
#         f_write.write(str(l[i][0])+ "|")
#         f_write.write(str(count))
#         f_write.write("\n")


#Sorted order ANOVA
import pandas as pd
df = pd.read_csv('VOC 2007 Anova matrix.csv')

l = df.values.tolist()
print(len(l))





