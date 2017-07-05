import numpy as np
import pickle
import cv2
import csv

#Constants
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 80
TOP_PADDING = 10
BOTTOM_PADDING = -30
LEFT_PADDING = 4
RIGHT_PADDING = -6
FX = 1
FY = 1
NUM_IMAGE_CLASSES = 1
NUM_CHANNELS = 20

VALIDATION_RATIO = 0.2

CNN_FILTER1 = 32
CNN_FILTER2 = 64
CNN_FILTER3 = 128
CNN_FILTER4 = 256
KERNEL_SIZE = 5
STRIDES = 2
POOL_SIZE = 2
DENSE_UNITS = 1024

OUTPUT_PATH = "./output/results7.csv"


#######################################################################
## Helper functions.
#######################################################################
def recursiveSplit(sourceDict, sourceIndices, splitIndices):
    splitDict = {}
    for key in sourceDict.keys():
        splitDict[key] = sourceDict[key][splitIndices]
        sourceDict[key] = sourceDict[key][sourceIndices]
    return (sourceDict, splitDict)  

def split_data(sourceDict, data_size, validation_ratio=VALIDATION_RATIO):
    randomIndices = np.arange(data_size)
    np.random.shuffle(randomIndices)
    numValidationSamples = round(validation_ratio*data_size)
    return recursiveSplit(sourceDict, randomIndices[numValidationSamples:], randomIndices[0:numValidationSamples])

def crop_mask_images(data, image_type, mask, top=TOP_PADDING, bottom=BOTTOM_PADDING, left=LEFT_PADDING, right=RIGHT_PADDING, apply_mask=True):
    images = []
    length = len(data[image_type])
    print(length)

    for i in range(0, length):
        image = data[image_type][i]
        image = image[top:bottom, left:right]
        if len(image.shape) > 2:
            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

        if apply_mask is True:
            segmentation = data[mask][i]
            segmentation = segmentation[top:bottom, left:right]
            mask2 = np.mean(segmentation, axis=2) > 150
            image = image*mask2
        image = cv2.resize(image, (0,0), fx=FX, fy=FY, interpolation=cv2.INTER_AREA) 
        images.append(image)
        
    return np.array(images)

def export_csv(Results):
    index = 1;
    with open(OUTPUT_PATH, 'w') as csvfile:   
    #configure writer to write standard csv file
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Id', 'Prediction'])
        for row in Results:
            writer.writerow([index, row])
            index+=1

def data_iterator(data, labels, batch_size, num_epochs=1, shuffle=True):
    """
    A simple data iterator for samples and labels.
    @param data: Numpy tensor where the samples are in the first dimension.
    @param labels: Numpy array.
    @param batch_size:
    @param num_epochs:
    @param shuffle: Boolean to shuffle data before partitioning the data into batches.
    """
    data_size = data.shape[0]
    for epoch in range(num_epochs):
        # shuffle labels and features
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_samples = data[shuffle_indices]
            shuffled_labels = labels[shuffle_indices]
        else:
            shuffled_samples = data
            shuffled_labels = labels
        for batch_idx in range(0, data_size-batch_size, batch_size):
            batch_samples = shuffled_samples[batch_idx:batch_idx + batch_size]
            batch_labels = shuffled_labels[batch_idx:batch_idx + batch_size]
            yield batch_samples, batch_labels

def data_iterator_samples(data, batch_size):
    """
    A simple data iterator for samples.
    @param data: Numpy tensor where the samples are in the first dimension.
    @param batch_size:
    @param num_epochs:
    """
    data_size = data.shape[0]
    for batch_idx in range(0, data_size, batch_size):
        batch_samples = data[batch_idx:batch_idx + batch_size]
        yield batch_samples
