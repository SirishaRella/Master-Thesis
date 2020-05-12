import glob

import Image

CLASSES  = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

source_path = "C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/" + CLASSES[19]+"/images"

image_files = [f for f in glob.glob(source_path+'/*.jpg')]

print(image_files)
with open('C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/' + CLASSES[19] +'/metadata.tsv','w') as file:
    id = 1
    file.write('id' +' '+'image')
    file.write('\n')
    for filepath in image_files:
        image_name = filepath.split('\\')[-1]
        print(image_name)
        file.write(str(id) +' ' +image_name)
        file.write('\n')
        id = id+1
