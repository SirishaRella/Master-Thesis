import xml.etree.ElementTree as ET
import glob as glob
import copy
import tempfile, shutil, os

#Classes to keep track of
CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

#Looping through all xml files
for filename in glob.glob('C:/Users/Siri/Downloads/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/*.xml'):
    root = ET.parse(filename).getroot()
    temp = copy.copy(root)

    track_objects = []

    #Creating a temp file by removing all the objects in an XML file.
    for object in root.findall('object'):
        temp.remove(object)

    #Finding all objects in root element
    for object in root.findall('object'):
        className =object.find('name').text
        temp_set = set()
        temp_set.add(className)

        #Created a set of unique classes.
        for e in temp_set:
            dynamicName = copy.copy(temp)
            for object in root.findall('object'):
                className = object.find('name').text

                if className == e:
                    dynamicName.append(object)
                    tree = ET.ElementTree(dynamicName)
                    tree.write('C:/Users/Siri/PycharmProjects/Thesis_benchmarkDataset/classes/'+className+'/'+ filename.split('\\')[-1])



