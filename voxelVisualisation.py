import numpy
import math
from functools import reduce

def factors(n):    
    return list(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if not n % i)))

def createImageGrid(images, numberOfColumns = None):
    numberOfImages, imageHeight, imageWidth, colourChannels = images.shape
    if numberOfColumns is None:
        numberOfImagesFactors = factors(numberOfImages)
        numberOfColumns = numberOfImagesFactors[numpy.argsort([abs(x - math.sqrt(numberOfImages)) for x in numberOfImagesFactors])[0]]
    numberOfRows = int(numberOfImages/numberOfColumns)
    # print('{} rows, {} columns.'.format(numberOfRows, numberOfColumns))
    assert numberOfImages == numberOfRows*numberOfColumns
    imageGrid = (images.reshape(numberOfRows, numberOfColumns, imageHeight, imageWidth, colourChannels)
                .swapaxes(1,2)
                .reshape(imageHeight*numberOfRows, imageWidth*numberOfColumns, colourChannels))
    return imageGrid