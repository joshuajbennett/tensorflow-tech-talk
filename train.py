import sys
import argparse
import pdb
from PIL import Image
import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

NUM_CLASSES = 20

def load_pascal(data_directory, split='train'): # train/test/val
    image_file = open(data_directory + "/VOCdevkit/VOC2007/ImageSets/Main/" + split + ".txt","r").read()
    split_image_file = image_file.split('\n')[:-1]
    num_images = len(split_image_file)

    images = np.empty([num_images,224,224,3],dtype=np.float32)
    labels = np.empty([num_images,NUM_CLASSES],dtype=np.int32)
    weights = np.empty([num_images,NUM_CLASSES],dtype=np.int32)

    for i in range(num_images):
        image_name  = split_image_file[i]
        input_img   = Image.open(data_directory + "/VOCdevkit/VOC2007/JPEGImages/" + image_name + ".jpg")
        scaled_img  = np.asarray(input_img.resize([224,224]))
        images[i,:,:,:] = scaled_img[np.newaxis,:,:,:]
        image_label     = np.empty([0])
        image_weight    = np.empty([0])
        for class_name in CLASS_NAMES:
            split_file = open(data_directory + "/VOCdevkit/VOC2007/ImageSets/Main/" + class_name + "_" + split + ".txt","r").read()
            img_pos = split_file.find(image_name)+len(image_name)+1
            img = split_file[img_pos:img_pos+2]
            if img == " 1":
                image_label  = np.append(image_label,[1],axis=0)
                image_weight = np.append(image_weight,[1],axis=0)
            else:
                image_label  = np.append(image_label,[0],axis=0)
                if img == " 0":
                    image_weight = np.append(image_weight,[0],axis=0)
                else:
                    image_weight = np.append(image_weight,[1],axis=0)
        labels[i,:] = image_label
        weights[i,:] = image_weight
    return (images, labels, weights)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Alexnet in Pytorch.')
    parser.add_argument(
        'data_directory', type=str, default='data/VOC2007',
        help='Path to PASCAL data set')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    (train_images, train_labels, train_weights) = load_pascal(args.data_directory, split='train')
    (test_images, test_labels, test_weights) = load_pascal(args.data_directory, split='test')
    (val_images, val_labels, val_weights) = load_pascal(args.data_directory, split='val')



if __name__ == "__main__":
    main()
