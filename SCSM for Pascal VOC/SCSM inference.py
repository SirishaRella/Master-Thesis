import glob


from gluoncv.utils import viz

import time

from matplotlib import patches

import gluoncv as gcv
import matplotlib.pyplot as plt
import bbox
import matplotlib.pyplot as plot


CLASSES  = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

with open('inference_metrics.txt', mode='w') as file_write:
    inference_time_track = []
    output_bbox_track = []
    ouput_class_name_track = []


    for filename in glob.glob('/content/drive/My Drive/Research/BenchMark_Models_Inferece/2007/TestImages/*.jpg'):
        print(filename.split('\\')[-1].split('.jpg')[0])

        x, image = gcv.data.transforms.presets.rcnn.load_test(filename,640)
        # Create figure and axes
        fig, ax1 = plot.subplots(3)

        # Display the image
        # ax1.imshow(image)
        indoor = ['bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
        co_dict_temp = {}
        class_name_temp = ""
        for i in range(0, len(CLASSES)):
            startTime = time.time()
            if CLASSES[i] in indoor:
                classes = [CLASSES[i]]
                net = gcv.model_zoo.get_model('faster_rcnn_resnet50_v1b_custom', classes=classes, pretrained_base=False)
                net.load_parameters('/content/drive/My Drive/Research/BenchMark_Models_Inferece/2007/Benchmark_models_2007/'+ CLASSES[i] +'.params')

                cid, score, bbox = net(x)
                print(CLASSES[i])
                print(time.time() - startTime)
                inference_time_track.append(time.time() - startTime)
                ax2, co_dict, class_name, score, color, bboxes = bbox.plot_bbox(image, bbox[0], score[0], cid[0],class_names=[CLASSES[i]], thresh=0.75)
                co_dict_temp = co_dict
                class_name_temp = class_name
                ax2.axis("off")
            if CLASSES[i] not in indoor:
                net = gcv.model_zoo.get_model('faster_rcnn_resnet50_v1b_custom', classes=CLASSES, pretrained_base=False)
                net.load_parameters('/content/drive/My Drive/Research/BenchMark_Models_Inferece/2007/Benchmark_models_2007/'+ CLASSES[i] +'.params')
                print("which class")
                print(CLASSES[i])

                cid, score, bbox = net(x)
                print(CLASSES[i])
                print(time.time() - startTime)
                inference_time_track.append(time.time() - startTime)
                ax2,co_dict, class_name, score, color, bboxes = bbox.plot_bbox(image, bbox[0], score[0], cid[0], class_names=[CLASSES[i]], thresh=0.75)
                co_dict_temp = co_dict
                class_name_temp = class_name
                ax2.axis('off')

            print("Checking for boundary positions")
            print(co_dict_temp)

            for j in range(0, len(co_dict_temp['coordinates']['xmin'])):
                output_bbox_track.append(co_dict_temp)
                ouput_class_name_track.append(class_name_temp)
                rect = patches.Rectangle((co_dict_temp['coordinates']['xmin'][j], co_dict_temp['coordinates']['ymin'][j]),
                                         co_dict_temp['coordinates']['xmax'][j] - co_dict_temp['coordinates']['xmin'][j],
                                         co_dict_temp['coordinates']['ymax'][j] - co_dict_temp['coordinates']['ymin'][j], linewidth=3,
                                         edgecolor='g', facecolor='none')
                ax1.add_patch(rect)
                ax1.text(co_dict_temp['coordinates']['xmin'][j], co_dict_temp['coordinates']['ymin'][j] - 2,
                         '{:s} {:s}'.format(class_name_temp, co_dict_temp['scores'][j]),
                         bbox=dict(facecolor='g', alpha=0.5),
                         fontsize=12, color='white')
            #
            # plt.show()


        file_write.write(str([filename, max(inference_time_track), ouput_class_name_track, output_bbox_track]))
        file_write.write("\n")
        print("Final Output to store as a row in CSV file")
        print(filename)
        print(inference_time_track)
        print(ouput_class_name_track)
        print(output_bbox_track)

        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plot.savefig('/content/drive/My Drive/Research/BenchMark_Models_Inferece/2007/Inference_output/' +
                     filename.split('\\')[-1].split('.jpg')[0], dpi=300)
        #plt.show()
        plt.close()
file_write.close()