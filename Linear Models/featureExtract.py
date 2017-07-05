import pickle
import numpy as np
import random
import cv2
from skimage import feature
from config import *
np.random.seed(1)

# Feature Extractors
def featureExtractHOG(imageData, orientations=9, pixels_per_cell=(5,5), cells_per_block=(2,2), widthPadding=10, heightPadding=50):
    '''
    Calculates HOG feature vector for every image.

    imageData is a numpy tensor consisting of 2- or 3-dimensional images (i.e.,
    grayscale or rgb). Color-images are first transformed to grayscale since
    HOG requires grayscale images.
    '''
    numImages = imageData.shape[0]
    featureList = []

    if len(imageData.shape) > 3: # Color image
        for i in range(0, numImages):
            img = cv2.cvtColor(imageData[i][widthPadding:-heightPadding, widthPadding+widthPadding:-widthPadding],cv2.COLOR_RGB2GRAY)
            hogFeatures = feature.hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualise=False)
            featureList.append(hogFeatures)
    else:
        for i in range(0, numImages):
            hogFeatures = feature.hog(imageData[i][widthPadding:-heightPadding, widthPadding+widthPadding:-widthPadding], orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualise=False)
            featureList.append(hogFeatures)

    return np.asarray(featureList)


def featureExtractSIFT(imageData, nfeatures, contrastThreshold=0.04, edgeThreshold=10, widthPadding=10, heightPadding=50):
    '''
    Calculates SIFT feature vector for every image. Note that this is not the
    best practice since new keypoints are estimated for every image.

    imageData is a numpy tensor consisting of 2- or 3-dimensional images (i.e.,
    grayscale or rgb).

    Since SIFT can extract different # of key points per image, nfeatures parameter
    of the SIFT create() function is required. For details:
        [nfeatures]:The number of best features to retain. The features are
        ranked by their scores (measured in SIFT algorithm as the local contrast)
        [contrastThreshold]:The contrast threshold used to filter out weak features
        in semi-uniform (low-contrast) regions. The larger the threshold, the less
        features are produced by the detector.
        [edgeThreshold]:The threshold used to filter out edge-like features. Note
        that the its meaning is different from the contrastThreshold, i.e. the
        larger the edgeThreshold, the less features are filtered out (more
        features are retained).
    '''
    numImages = imageData.shape[0]
    featureList = []

    # number of key points can be less than nfeatures. For sanity check, run sift
    # on a reference image.
    referenceSift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold)
    keypoints = referenceSift.detect(imageData[random.randint(1,numImages-1)][widthPadding:-heightPadding, widthPadding+widthPadding:-widthPadding], None)
    # Update nfeatures
    if len(keypoints) < nfeatures:
        print("[SIFT] nfeatures (" + str(nfeatures) + ") is updated (" + str(len(keypoints)) + ")")
        nfeatures = len(keypoints)

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold)

    for i in range(0, numImages):
        keypoints = sift.detect(imageData[i], None)
        (keypoints, descriptions) = sift.compute(imageData[i][widthPadding:-heightPadding, widthPadding+widthPadding:-widthPadding], keypoints[0:nfeatures]) # Sometimes sift may return more keypoints than nfeatures.
        featureList.append(descriptions.flatten())

    return np.asarray(featureList)

def featureExtractSIFTDense(imageData, pixelStepSize=10, widthPadding=10, heightPadding=50):
    '''
    Calculates SIFT feature vector for every image. First creates a list of
    keypoints by scanning through the pixel locations of the image so that
    features will have the same locality for every image.

    imageData is a numpy tensor consisting of 2- or 3-dimensional images (i.e.,
    grayscale or rgb).
    '''
    numImages = imageData.shape[0]
    featureList = []

    # Create key points.
    keypointGrid = [cv2.KeyPoint(x, y, pixelStepSize)
                    for y in range(widthPadding, imageData.shape[1]-heightPadding, pixelStepSize)
                        for x in range(widthPadding+widthPadding, imageData.shape[2]-widthPadding, pixelStepSize)]

    sift = cv2.xfeatures2d.SIFT_create()

    for i in range(0, numImages):
        (keypoints, descriptions) = sift.compute(imageData[i], keypointGrid)
        featureList.append(descriptions.flatten())

    return np.asarray(featureList)

def applyMask(imageData, maskData):
    """
    Applies mask on the given data.
    """
    numImages = imageData.shape[0]
    if len(imageData.shape) > 3: # Color image
        for i in range(0, numImages):
            segmentedUser = maskData[i]
            mask2 = np.mean(segmentedUser, axis=2) > 150
            mask3 = np.tile(mask2, (3,1,1)) # 3-channel
            mask3 = mask3.transpose((1,2,0))
            imageData[i] = imageData[i] * mask3
    else:
        for i in range(0, numImages):
            segmentedUser = maskData[i]
            mask2 = np.mean(segmentedUser, axis=2) > 150
            imageData[i] = imageData[i] * mask2

def recursiveSplit(sourceDict, sourceIndices, splitIndices):
    splitDict = {}
    for key in sourceDict.keys():
        splitDict[key] = sourceDict[key][splitIndices]
        sourceDict[key] = sourceDict[key][sourceIndices]
    return (sourceDict, splitDict)

def main():
    print('Loading Data.')
    dataTrain = pickle.load(open(TRAINING_DATA, 'rb'))
    dataTest = pickle.load(open(TEST_DATA, 'rb'))
    if VALIDATION_DATA is None:
        randomIndices = np.arange(dataTrain['gestureLabels'].shape[0])
        np.random.shuffle(randomIndices)
        numValidationSamples = round(VALIDATION_DATA_RATIO*dataTrain['gestureLabels'].shape[0])
        (dataTrain, dataValidation) = recursiveSplit(dataTrain, randomIndices[numValidationSamples:], randomIndices[0:numValidationSamples])
    else:
        dataValidation = pickle.load(open(VALIDATION_DATA, 'rb'))

    if APPLY_MASK is True and FEATURE_POOL['skeleton'] is False:
        applyMask(dataTrain[DATA_MODALITY], dataTrain['segmentation'])
        applyMask(dataValidation[DATA_MODALITY], dataValidation['segmentation'])
        applyMask(dataTest[DATA_MODALITY], dataTest['segmentation'])

    if FEATURE_POOL['hog'] is True:
        print("Extracting '" + DATA_MODALITY + " hog' Features.")
        featureMatrixTraining = featureExtractHOG(dataTrain[DATA_MODALITY], orientations=9, pixels_per_cell=(5,5), cells_per_block=(2,2))
        featureMatrixValidation = featureExtractHOG(dataValidation[DATA_MODALITY], orientations=9, pixels_per_cell=(5,5), cells_per_block=(2,2))
        featureMatrixTesting = featureExtractHOG(dataTest[DATA_MODALITY], orientations=9, pixels_per_cell=(5,5), cells_per_block=(2,2))
    elif FEATURE_POOL['sift'] is True:
        print("Extracting '" + DATA_MODALITY + " sift' Features.")
        featureMatrixTraining = featureExtractSIFT(dataTrain[DATA_MODALITY], nfeatures=30, contrastThreshold=0.04, edgeThreshold=10)
        featureMatrixValidation = featureExtractSIFT(dataValidation[DATA_MODALITY], nfeatures=30, contrastThreshold=0.04, edgeThreshold=10)
        featureMatrixTesting = featureExtractSIFT(dataTest[DATA_MODALITY], nfeatures=30, contrastThreshold=0.04, edgeThreshold=10)
    elif FEATURE_POOL['sift_dense'] is True:
        print("Extracting '" + DATA_MODALITY + " sift_dense' Features.")
        featureMatrixTraining = featureExtractSIFTDense(dataTrain[DATA_MODALITY], pixelStepSize=9)
        featureMatrixValidation = featureExtractSIFTDense(dataValidation[DATA_MODALITY], pixelStepSize=9)
        featureMatrixTesting = featureExtractSIFTDense(dataTest[DATA_MODALITY], pixelStepSize=9)
    elif FEATURE_POOL['skeleton'] is True:
        print("Extracting 'skeleton' Features.")
        featureMatrixTraining = dataTrain['skeleton']
        featureMatrixValidation = dataValidation['skeleton']
        featureMatrixTesting = dataTest['skeleton']

    data = {}
    data['featureMatrixTraining'] = featureMatrixTraining
    data['featureMatrixValidation'] = featureMatrixValidation
    data['targetsTraining'] = dataTrain['gestureLabels']
    data['targetsValidation'] = dataValidation['gestureLabels']
    pickle.dump(data, open(FEATURE_SAVE_PATH, 'wb'))

    TestData = {}
    TestData['featureMatrixTesting'] = featureMatrixTesting
    pickle.dump(TestData, open(TEST_FEATURE_SAVE_PATH, 'wb'))


if __name__ == '__main__':
    main()
