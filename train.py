import alexnet
import argparse
from math import floor
import numpy as np
import os.path
import pdb
from PIL import Image
from random import shuffle
import sys
import tensorflow as tf

# Classes for the Pascal data set
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

# Loads the pascal data set
def load_pascal(data_directory, split='train'): # split = train/test/val
    assert split == 'train' or split == 'test' or split == 'val'
    # Get the cached npy file names    
    cached_images = data_directory + '/images_' + split
    cached_labels = data_directory + '/labels_' + split
    cached_weights = data_directory + '/weights_' + split
    # Load the cached npy files if they exist
    if os.path.isfile(cached_images + '.npy') and \
            os.path.isfile(cached_labels + '.npy') and \
            os.path.isfile(cached_weights + '.npy'):
        images = np.load(cached_images + '.npy')
        labels = np.load(cached_labels + '.npy')
        weights = np.load(cached_weights + '.npy')
    # If there are no cached files, load the data from file
    else:
        # Open the train/test/val split definition
        image_file = open(data_directory + '/VOCdevkit/VOC2007/ImageSets/Main/' + split + '.txt','r').read()
        split_image_file = image_file.split('\n')[:-1]
        num_images = len(split_image_file)
	
        # Pre-allocate the data for loading
        images = np.empty([num_images,224,224,3],dtype=np.float32)
        labels = np.empty([num_images,NUM_CLASSES],dtype=np.int32)
        weights = np.empty([num_images,NUM_CLASSES],dtype=np.int32)

	# Loop through and load each image
        for i in range(num_images):
            image_name  = split_image_file[i]
            input_img   = Image.open(data_directory + '/VOCdevkit/VOC2007/JPEGImages/' + image_name + '.jpg')
            # Scale the image to the expected AlexNet size 
            scaled_img  = np.asarray(input_img.resize([224,224]))
            images[i,:,:,:] = scaled_img[np.newaxis,:,:,:]
            image_label     = np.empty([0])
            image_weight    = np.empty([0])
            
            # Populate the labels and weights 
            for class_name in CLASS_NAMES:
                split_file = open(data_directory + '/VOCdevkit/VOC2007/ImageSets/Main/' + class_name + '_' + split + '.txt','r').read()
                img_pos = split_file.find(image_name)+len(image_name)+1
                img = split_file[img_pos:img_pos+2]
                if img == ' 1':
                    image_label  = np.append(image_label,[1],axis=0)
                    image_weight = np.append(image_weight,[1],axis=0)
                else:
                    image_label  = np.append(image_label,[0],axis=0)
                    if img == ' 0':
                        image_weight = np.append(image_weight,[0],axis=0)
                    else:
                        image_weight = np.append(image_weight,[1],axis=0)
            labels[i,:] = image_label
            weights[i,:] = image_weight

        # Cache the loaded data
        np.save(cached_images, images)
        np.save(cached_labels, labels)
        np.save(cached_weights, weights)
    return (images, labels, weights)

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description='Alexnet in TensorFlow.')
    parser.add_argument(
        'data_directory', type=str, default='data/VOC2007',
        help='Path to PASCAL data set')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = parse_args()
    # Load the train data
    (train_images, train_labels, train_weights) = load_pascal(args.data_directory, split='train')
	
    # Avoid loading other data for now
    # (test_images, test_labels, test_weights) = load_pascal(args.data_directory, split='test')
    # (val_images, val_labels, val_weights) = load_pascal(args.data_directory, split='val')

    # Convert the training data to tensors 
    with tf.device("/cpu:0"):
        train_images = tf.convert_to_tensor(train_images)
        train_labels = tf.convert_to_tensor(train_labels)
        train_weights = tf.convert_to_tensor(train_weights)

    # Setup a TensorFlow session 
    with tf.Session() as sess:
        # Initialize global variables
        sess.run(tf.global_variables_initializer())
        
        # Setup a way to write summary data
        summary_writer = tf.summary.FileWriter('/tmp/tensorboard_example', graph=tf.get_default_graph())
      
        num_train_images = train_images.shape[0].value
        batch_size = 10
        num_epochs = 5
        # Epoch = a full cycle through all of the training images 
        for epoch in range(num_epochs):
            # Randomly order the training images
            training_order = range(num_train_images) 
            shuffle(training_order) 
            
            # Split the training images into batches
            for batch_num in range(int(floor(num_train_images/batch_size))):
                # Extract the data for this batch
                batch_indices = training_order[batch_num*batch_size:(batch_num + 1)*batch_size]
                batch_images = tf.gather(train_images, batch_indices)
                batch_labels = tf.gather(train_labels, batch_indices)
                batch_weights = tf.gather(train_weights, batch_indices)

                # Run the batch through the alexnet model
                with tf.contrib.slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
                    outputs, end_points = alexnet.alexnet_v2(batch_images, NUM_CLASSES)
                pdb.set_trace() 
                print(outputs)

if __name__ == '__main__':
    main()
