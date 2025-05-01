import torch
from torch import nn
from torch.nn.functional import affine_grid, grid_sample
import pytorch3d.transforms as transforms
import numpy
import numbers
import math
import voxelVisualisationPytorch3D
import os
import hdf5storage
import matplotlib.pyplot as plt
import pytorch3DRenderer

class GaussianSmoothing(nn.Module):
    """
    Author: Adrian Sahlman (https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7)
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim  = 2, paddingMode = 'default'):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim   
        kernel_size = [k+1 if k%2==0 else k for k in kernel_size]
        self.padding = []
        for k in kernel_size:
            self.padding.append(int(round((k-1)/2)))
            self.padding.append(int(round((k-1)/2)))

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if paddingMode == 'default':
            if dim == 1:
                self.conv = torch.nn.functional.conv1d
                self.paddingMode = 'reflect'
            elif dim == 2:
                self.conv = torch.nn.functional.conv2d
                self.paddingMode = 'reflect'
            elif dim == 3:
                self.conv = torch.nn.functional.conv3d
                self.paddingMode = 'circular'
            else:
                raise RuntimeError(
                    'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
                )
        elif paddingMode == 'zeros':
            if dim == 1:
                self.conv = torch.nn.functional.conv1d
                self.paddingMode = 'constant'
            elif dim == 2:
                self.conv = torch.nn.functional.conv2d
                self.paddingMode = 'constant'
            elif dim == 3:
                self.conv = torch.nn.functional.conv3d
                self.paddingMode = 'constant'
            else:
                raise RuntimeError(
                    'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
                )
        else:
            raise ValueError('Unknown padding mode: {}'.format(paddingMode))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        input = torch.nn.functional.pad(input,self.padding,mode=self.paddingMode)
        return self.conv(input, weight=self.weight, groups=self.groups)

def importMATLABVariable(fileName,variableName):
    if not os.path.isfile(fileName):
        raise FileNotFoundError(fileName)
    return hdf5storage.loadmat(fileName,variable_names=[variableName])[variableName]

def loadObjectData(dataDirectory, objectInformationFile, binariseVoxels = False):
    objectInformation = importMATLABVariable(objectInformationFile, 'objectInformation')[0]
    objects = []
    for ii in range(len(objectInformation)):
        currentObject = {}
        currentObjectName = objectInformation[ii]['name'][0]
        currentObjectDataFile = '{}/{}Voxels.mat'.format(dataDirectory,currentObjectName)
        currentObject['voxelGridMaximum'] = torch.as_tensor(importMATLABVariable(currentObjectDataFile,'voxelGridMaximum')[0], dtype = torch.float)
        currentObject['voxelGridMinimum'] = torch.as_tensor(importMATLABVariable(currentObjectDataFile,'voxelGridMinimum')[0], dtype = torch.float)
        if binariseVoxels:            
            currentObject['voxelGrid'] = torch.as_tensor(importMATLABVariable(currentObjectDataFile,'voxelGrid')>1e-4, dtype = torch.float)
        else:
            currentObject['voxelGrid'] = torch.as_tensor(importMATLABVariable(currentObjectDataFile,'voxelGrid'), dtype = torch.float)
        objects.append(currentObject)
    return objects

def spatialTransformer(inputVoxels, affineTransformations, voxelGridSize):
    """
    Spatial transformer implementation. Does not currently perform blurring/high frequency removal when downsampling.
    """
    grid = affine_grid(affineTransformations, voxelGridSize, align_corners = False)
    outputVoxels = grid_sample(inputVoxels, grid, align_corners = False)
    return outputVoxels

def createTransformationsfromMatrices(rotationMatrices, translations):
    output = torch.zeros(rotationMatrices.size(0), 4, 4)
    output[:,:3,:3] = rotationMatrices
    output[:,:3,3] = translations
    return output

def createTransformationsfromQuaternions(rotationQuaternions, translations):
    rotationMatrices = transforms.quaternion_to_matrix(rotationQuaternions)
    return createTransformationsfromMatrices(rotationMatrices, translations)

@ torch.no_grad()
def findOptimalTranslation(referenceVoxelGrid, testVoxelGrid):
    while referenceVoxelGrid.dim() < 5:
        referenceVoxelGrid = referenceVoxelGrid.unsqueeze(dim = 0)
    while testVoxelGrid.dim() < 5:
        testVoxelGrid = testVoxelGrid.unsqueeze(dim = 0)
    referenceVoxelGrid = referenceVoxelGrid/(referenceVoxelGrid.pow(2).sum(dim=[2,3,4]).sqrt()[:,:,None,None,None] + 1e-7)
    testVoxelGrid = testVoxelGrid/(testVoxelGrid.pow(2).sum(dim=[2,3,4]).sqrt()[:,:,None,None,None] + 1e-7)
    if testVoxelGrid.size(0) == 1 and testVoxelGrid.size(1) > 1:
        testVoxelGrid = testVoxelGrid.transpose(0,1)
    elif testVoxelGrid.size(0) > 1 and testVoxelGrid.size(1) > 1:
        raise ValueError('Unknown testVoxelGrid shape: {}'.format(testVoxelGrid.shape))
    leftPadding = [ int(math.ceil(x/2)) for x in testVoxelGrid.shape[2:]]
    leftPadding = [ x + (x%2) for x in leftPadding]
    paddingOffsetTensor = torch.as_tensor(leftPadding, dtype = torch.float, device = referenceVoxelGrid.device).unsqueeze(dim = 0)
    padding = [ (x,x) for x in leftPadding]
    padding = [x for y in padding for x in y]
    padding.reverse()
    # referenceVoxelGrid = torch.nn.functional.pad(referenceVoxelGrid, pad = padding, mode = 'constant', value = 0)
    if referenceVoxelGrid.size(1) == 1 and testVoxelGrid.size(0) > 1:
        referenceVoxelGrid = referenceVoxelGrid.expand(referenceVoxelGrid.size(0), testVoxelGrid.size(0), *referenceVoxelGrid.shape[2:])
    convolution = torch.nn.functional.conv3d(referenceVoxelGrid, weight=testVoxelGrid, padding = leftPadding, groups=testVoxelGrid.size(0))
    maximumValues = convolution.max(dim = 2)[0].max(dim = 2)[0].max(dim = 2)[0][:,:,None,None,None]
    indices = (convolution == maximumValues).nonzero().to(dtype = torch.float)
    indices[:,2:] = indices[:,2:] + (torch.as_tensor(testVoxelGrid.shape[2:], dtype = torch.float, device = indices.device)-1)/2 - paddingOffsetTensor
    return indices, maximumValues

def calculateCompositeTransformation(transformationsPytorchToTest, transformationsTestToReference, transformationsReferenceToPytorch):
    forwardTransformations = torch.matmul(transformationsTestToReference, transformationsPytorchToTest)
    forwardTransformations = torch.matmul(transformationsReferenceToPytorch, forwardTransformations)
    backwardsTransformations = forwardTransformations.inverse()
    return forwardTransformations, backwardsTransformations

def extractTestToReferenceFromForwardComposite(forwardComposite, transformationsPytorchToTest, transformationsReferenceToPytorch):
    forwardComposite = torch.matmul(transformationsReferenceToPytorch.inverse(), torch.matmul(forwardComposite, transformationsPytorchToTest.inverse()))
    return forwardComposite
    
def extractTestToReferenceFromBackwardComposite(backwardComposite, transformationsPytorchToTest, transformationsReferenceToPytorch):
    forwardComposite = backwardComposite.inverse()
    return extractTestToReferenceFromForwardComposite(forwardComposite, transformationsPytorchToTest, transformationsReferenceToPytorch)

def convertFromPytorchCoordinatesToOriginalCoordinates(transformation):
    coordinatePermutationMatrix = torch.as_tensor([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]], dtype = torch.float, device = transformation.device).unsqueeze(dim = 0)
    convertedTransformation = torch.matmul(transformation, coordinatePermutationMatrix.inverse())
    convertedTransformation = torch.matmul(coordinatePermutationMatrix, convertedTransformation)
    return convertedTransformation

def convertFromOriginalCoordinatesToPytorchCoordinates(transformation):
    coordinatePermutationMatrix = torch.as_tensor([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]], dtype = torch.float, device = transformation.device).unsqueeze(dim = 0)
    convertedTransformation = torch.matmul(transformation, coordinatePermutationMatrix)
    convertedTransformation = torch.matmul(coordinatePermutationMatrix.inverse(), convertedTransformation)
    return convertedTransformation

def calculateTransformations(referenceBBMin, referenceBBMax, testBBMin, testBBMax, rotations, translations, device):
    batchSize = rotations.size(0)
    testDimensions = (testBBMax - testBBMin).to(device)
    testCentre = ((testBBMax + testBBMin)/2).to(device)
    transformationsPytorchToTest = torch.eye(4, 4, device = device)
    transformationsPytorchToTest[:3,:3] = torch.diag_embed(testDimensions/2)
    transformationsPytorchToTest[:3,3] = testCentre

    transformationsTestToReference = torch.eye(4, 4, dtype = torch.float, device = device).repeat(batchSize,1,1)
    transformationsTestToReference[:,:3,:3] = rotations
    transformationsTestToReference[:,:3,3] = translations

    referenceDimensions = (referenceBBMax - referenceBBMin).to(device)
    referenceCentre = ((referenceBBMax + referenceBBMin)/2).to(device)
    scaleMatrices = torch.diag_embed(referenceDimensions/2).repeat(batchSize,1,1)
    transformationsPytorchToReference = torch.eye(4,4).repeat(batchSize,1,1)
    transformationsPytorchToReference[:,:3,:3] = scaleMatrices
    transformationsPytorchToReference[:,:3,3] = referenceCentre

    transformationsPytorchToReference = transformationsPytorchToReference.to(device = device)
    transformationsReferenceToPytorch = transformationsPytorchToReference.inverse()
    forwardTransformations, backwardTransformations = calculateCompositeTransformation(transformationsPytorchToTest = transformationsPytorchToTest, 
        transformationsTestToReference = transformationsTestToReference, transformationsReferenceToPytorch = transformationsReferenceToPytorch)
    transformation = backwardTransformations
    finalTransformation = convertFromOriginalCoordinatesToPytorchCoordinates(transformation)
    return finalTransformation, forwardTransformations, backwardTransformations

def binVoxels(voxels):
    
    if len(voxels.shape) == 3:
        voxels = (voxels[0::2,0::2,0::2]+voxels[1::2,0::2,0::2]+voxels[0::2,1::2,0::2]+voxels[0::2,0::2,1::2]+
                    voxels[1::2,1::2,0::2]+voxels[1::2,0::2,1::2]+voxels[0::2,1::2,1::2]+voxels[1::2,1::2,1::2])/8
    elif len(voxels.shape) == 4:
        voxels = (voxels[:,0::2,0::2,0::2]+voxels[:,1::2,0::2,0::2]+voxels[:,0::2,1::2,0::2]+voxels[:,0::2,0::2,1::2]+
                    voxels[:,1::2,1::2,0::2]+voxels[:,1::2,0::2,1::2]+voxels[:,0::2,1::2,1::2]+voxels[:,1::2,1::2,1::2])/8
    elif len(voxels.shape) == 5:
        voxels = (voxels[:,:,0::2,0::2,0::2]+voxels[:,:,1::2,0::2,0::2]+voxels[:,:,0::2,1::2,0::2]+voxels[:,:,0::2,0::2,1::2]+
                    voxels[:,:,1::2,1::2,0::2]+voxels[:,:,1::2,0::2,1::2]+voxels[:,:,0::2,1::2,1::2]+voxels[:,:,1::2,1::2,1::2])/8
    else:
        raise ValueError('Unknown voxel data shape: {}.'.format(voxels.shape))
    return voxels

def estimatePose(   referenceVoxelGrid, referenceBBMin, referenceBBMax, testVoxelGrid, testBBMin, testBBMax, initialRotations, numberOfIterations, 
                    lossBlurKernelSize, lossBlurKernelSigma, lossType, device):
    batchSize = initialRotations.size(0)
    referenceShape = torch.as_tensor(referenceVoxelGrid.shape, device = device)
    referenceVoxelSize = (referenceBBMax - referenceBBMin)/referenceShape
    testVoxelSize = (testBBMax - testBBMin)/torch.as_tensor(testVoxelGrid.shape, device = device)
    scaling = referenceVoxelSize/testVoxelSize
    testVoxelCentre = (testBBMax + testBBMin)/2
    testVoxelDiameter = (testBBMax - testBBMin).norm()
    initialTestVoxelGridSize = (testVoxelDiameter/referenceVoxelSize).ceil().to(dtype = torch.int32)
    if initialRotations.dim() == 3:
        rotationMatrices = initialRotations
        rotationQuaternions = transforms.matrix_to_quaternion(initialRotations)
    elif initialRotations.dim() == 2:
        rotationMatrices = transforms.quaternion_to_matrix(initialRotations)
        rotationQuaternions = initialRotations
    else:
        raise ValueError('Unknown rotation format')
    initialTransforms = torch.cat([torch.matmul(torch.diag_embed(scaling),rotationMatrices),torch.zeros(batchSize,3,1, device = device)], dim = 2)
    initialOrientedVoxels = spatialTransformer(testVoxelGrid[None,None,:,:,:].expand(batchSize,1,*testVoxelGrid.shape), initialTransforms, [batchSize,1]+initialTestVoxelGridSize.tolist())
    initialOrientedVoxels = initialOrientedVoxels.squeeze(dim = 1)

    #Bin initial translation estimation to reduce convolution kernel sizes
    translationEstimationBinningPower = 1
    binnedReferenceVoxelGrid = referenceVoxelGrid
    binnedInitialOrientedVoxels = initialOrientedVoxels
    for ii in range(translationEstimationBinningPower):
        referencePadding = [(x%2,0) for x in binnedReferenceVoxelGrid.shape]
        referencePadding = [x for y in referencePadding for x in y]
        referencePadding.reverse()
        binnedReferenceVoxelGrid = torch.nn.functional.pad(input = binnedReferenceVoxelGrid, pad = referencePadding, mode = 'constant', value = 0)
        binnedReferenceVoxelGrid = binVoxels(binnedReferenceVoxelGrid)        
        initialPadding = [(x%2,0) for x in binnedInitialOrientedVoxels.shape]
        initialPadding = [x for y in initialPadding for x in y]
        initialPadding.reverse()
        binnedInitialOrientedVoxels = torch.nn.functional.pad(input = binnedInitialOrientedVoxels, pad = initialPadding, mode = 'constant', value = 0)
        binnedInitialOrientedVoxels = binVoxels(binnedInitialOrientedVoxels)
    estimatedTranslationVoxels, maximumValues = findOptimalTranslation(binnedReferenceVoxelGrid, binnedInitialOrientedVoxels)
    
    #Remove non-useful translations
    keepMask = maximumValues.squeeze() > 0.1
    if keepMask.sum() == 0:
        return None, None, float('inf'), 0
    estimatedTranslationVoxels = estimatedTranslationVoxels[keepMask[estimatedTranslationVoxels[:,1].to(dtype = torch.long)],:]

    values, indices = estimatedTranslationVoxels[:,1].unique(return_inverse = True)
    indices = indices[:values.numel()]

    estimatedTranslations = estimatedTranslationVoxels[indices,2:]*math.pow(2, translationEstimationBinningPower)

    estimatedTranslations = referenceBBMin.unsqueeze(dim = 0) + (referenceBBMax - referenceBBMin).unsqueeze(dim = 0)*(estimatedTranslations/referenceShape.unsqueeze(dim = 0))

    optimisationTranslations = estimatedTranslations.detach().clone().requires_grad_(True)
    optimisationQuaternions = rotationQuaternions[values.to(dtype = torch.long),:].detach().clone().requires_grad_(True)
    initialQuaternions = optimisationQuaternions.detach().clone()

    optimiser = torch.optim.Adam(params = [optimisationTranslations, optimisationQuaternions], lr = 1e-2)

    testVoxelGrid = testVoxelGrid[None,None,:,:,:].expand(optimisationTranslations.size(0),1,*testVoxelGrid.shape)
    referenceVoxelGrid = referenceVoxelGrid[None,None,:,:,:].expand(optimisationTranslations.size(0),1,*referenceVoxelGrid.shape)
    if lossBlurKernelSize > 0 and lossBlurKernelSigma > 0:
        blur = GaussianSmoothing(channels = 1, kernel_size = lossBlurKernelSize, sigma = lossBlurKernelSigma, dim  = 3, paddingMode = 'zeros').to(device = device)
    else:
        blur = None
    bestLoss = float('inf')
    bestRotation = None
    bestTranslation = None
    for ii in range(numberOfIterations):
        optimisationTransforms, forward, backward = calculateTransformations(referenceBBMin, referenceBBMax, testBBMin, testBBMax, 
                                                                        transforms.quaternion_to_matrix(optimisationQuaternions), optimisationTranslations, device)
        outputs = spatialTransformer(testVoxelGrid, optimisationTransforms[:,:3,:], referenceVoxelGrid.shape)
        if blur is None:
            blurredReference = referenceVoxelGrid
            blurredTest = outputs
        else:
            # blurredReference = blur(referenceVoxelGrid)
            blurredReference = referenceVoxelGrid
            blurredTest = blur(outputs)
            
        if lossType == 'meanSquaredError':
            loss = (blurredReference - blurredTest).pow(2).mean(dim = [1,2,3,4])
        elif lossType == 'intersectionOverUnion':
            loss = 1/((blurredReference*blurredTest).mean(dim = [1,2,3,4]) + 1e-7)
        else:
            raise ValueError('Unknown loss type: {}'.format(lossType))

        for rotationQuaternion, translation, lossValue in zip(optimisationQuaternions, optimisationTranslations, loss):
            if lossValue < bestLoss:
                bestRotation = rotationQuaternion.detach().clone()
                bestTranslation = translation.detach().clone()
                bestLoss = lossValue.item()
        optimiser.zero_grad()
        loss.mean().backward()
        # print(optimisationQuaternions.grad.pow(2).mean().item())
        optimiser.step()
        with torch.no_grad():
            optimisationQuaternions /= optimisationQuaternions.pow(2).sum(dim = 1, keepdim = True).sqrt()
        optimisationQuaternions.requires_grad_(True)
        # print('Iteration {}, loss {:.1e}'.format(ii, loss.min().item()))
    # print('Average quaternion dot product is {}'.format((optimisationQuaternions*initialQuaternions).sum(dim = 1).mean().item()))
    # bestRotation /= bestRotation.pow(2).sum().sqrt()
    finalRotationMatrix = transforms.quaternion_to_matrix(bestRotation)
    finalTranslation = bestTranslation

    optimisationTransforms, forward, backward = calculateTransformations(referenceBBMin, referenceBBMax, testBBMin, testBBMax, 
                                                                        finalRotationMatrix.unsqueeze(dim = 0), finalTranslation.unsqueeze(dim = 0), device)
    outputs = spatialTransformer(testVoxelGrid[0:1,:,:,:,:], optimisationTransforms[:,:3,:], [1]+list(referenceVoxelGrid.shape[1:]))
    intersection = ((referenceVoxelGrid[0:1,:,:,:,:] > 0.5) & (outputs > 0.5)).to(dtype = torch.float).sum().item()
    union = ((referenceVoxelGrid[0:1,:,:,:,:] > 0.5) | (outputs > 0.5)).to(dtype = torch.float).sum().item()
    intersectionOverUnion = intersection/union
    meanSquaredError = (referenceVoxelGrid[0:1,:,:,:,:] - outputs).pow(2).mean().item()
    return finalRotationMatrix, finalTranslation, meanSquaredError, intersectionOverUnion

class VoxelParser(nn.Module):
    def __init__(self, objectVoxels, objectBoundsMax, objectBoundsMin, seedRotations, iouThreshold):
        super().__init__()
        self.objectVoxels = objectVoxels
        self.objectBoundsMax = objectBoundsMax
        self.objectBoundsMin = objectBoundsMin
        self.seedRotations = seedRotations
        self.iouThreshold = iouThreshold

    def forward(self, voxelScenes, voxelBoundsMax, voxelBoundsMin, numberOfIterations = 100, blurKernelSize = 5, blurKernelSigma = 1, lossType = 'intersectionOverUnion'):
        if voxelScenes.dim() == 5:
            scenes = [x for x in voxelScenes]
        elif voxelScenes.dim() == 4:
            scenes = [voxelScenes]
        else:
            raise ValueError('Unknown voxel scene shape')

        outputScenes = list()
        for scene in scenes:
            outputScene = list()
            for index, (reference, test, testMax, testMin) in enumerate(zip(scene, self.objectVoxels, self.objectBoundsMax, self.objectBoundsMin)):
                finalRotationMatrix, finalTranslation, meanSquaredError, finalIntersectionOverUnion = estimatePose(   
                                            referenceVoxelGrid = reference, 
                                            referenceBBMin = voxelBoundsMin, referenceBBMax = voxelBoundsMax, 
                                            testVoxelGrid = test, 
                                            testBBMin = testMin, 
                                            testBBMax = testMax, 
                                            initialRotations = self.seedRotations, numberOfIterations = numberOfIterations, 
                                            lossBlurKernelSize = blurKernelSize, lossBlurKernelSigma = blurKernelSigma, lossType = lossType, device = reference.device)
                if finalIntersectionOverUnion >= self.iouThreshold:
                    output = dict(  objectClass = index,
                                    rotationMatrix = finalRotationMatrix,
                                    translationVector = finalTranslation,
                                    iou = finalIntersectionOverUnion,
                                    meanSquaredError = meanSquaredError)
                    outputScene.append(output)
            outputScenes.append(outputScene)
        return outputScenes
            

if __name__ == '__main__':
    device = torch.device('cuda')
    dataDirectory = 'canonicalVoxelRepresentations'
    objectInformationFile = 'objectInformation.mat'
    voxelSize = 0.005
    voxelBinningPower = 0
    voxelGridBounds = torch.as_tensor([[-0.24,0.24],[-0.24,0.24],[-0.02,0.30]], dtype = torch.float, device = device)
    voxelSpatialDimensions = (voxelGridBounds[:,1] - voxelGridBounds[:,0]/voxelSize).round()
    voxelBinnedSpatialDimensions = (voxelSpatialDimensions/math.pow(2, voxelBinningPower)).round()

    objectDirectory = 'ycbobj'
    sceneData = pytorch3DRenderer.loadObjects(objectDirectory, device = device)
    renderer = pytorch3DRenderer.Pytorch3DRenderer(   cameraIntrinsics = sceneData['K'], cameraExtrinsics = sceneData['T'], lightPositions = sceneData['lightPosition'], 
                                    objectMeshes = sceneData['objectMeshes'], tableMesh = sceneData['tableMesh'], device = device)

    rotationData = hdf5storage.loadmat('rotations/quaternions32.mat')
    quaternions = torch.as_tensor(rotationData['quaternions'], dtype = torch.float, device = device)
    rotationMatrices = transforms.quaternion_to_matrix(quaternions)
    classColours = hdf5storage.loadmat('officialColours.mat')['colours'][1:17,:]
    classColours = torch.as_tensor(classColours, dtype = torch.float)
    objectData = loadObjectData(dataDirectory, objectInformationFile, binariseVoxels = True)
    objectVoxels = []
    objectBoundsMax = []
    objectBoundsMin = []
    for currentObject in objectData:
        objectVoxels.append(currentObject['voxelGrid'].to(dtype = torch.float, device = device))
        objectBoundsMax.append(currentObject['voxelGridMaximum'].to(dtype = torch.float, device = device))
        objectBoundsMin.append(currentObject['voxelGridMinimum'].to(dtype = torch.float, device = device))
    parser = VoxelParser(objectVoxels = objectVoxels, objectBoundsMax = objectBoundsMax, objectBoundsMin = objectBoundsMin, seedRotations = quaternions, iouThreshold = 0.5)
    
    scene = torch.as_tensor(numpy.load('Data14Heap854_o.npz')['initialVoxelScene'], dtype = torch.float, device = device)

    lossBlurKernelSigma = 1
    lossBlurKernelSize = 2*int(math.ceil(lossBlurKernelSigma*3)) - 1
    # lossBlurKernelSigma = 0
    # lossBlurKernelSize = 0
    # lossType = 'meanSquaredError'
    lossType = 'intersectionOverUnion'
    numberOfIterations = 150
    parsing = parser(   voxelScenes = scene, voxelBoundsMax = voxelGridBounds[:,1], voxelBoundsMin= voxelGridBounds[:,0], 
                        numberOfIterations = numberOfIterations, blurKernelSize = lossBlurKernelSize, blurKernelSigma = lossBlurKernelSigma, lossType = lossType)
    objectPoses = [dict(objectClass = obj['objectClass'], transform = torch.cat([obj['rotationMatrix'], obj['translationVector'][:,None]], dim = 1)) for obj in parsing[0]]
    with torch.no_grad():
        images = renderer(objectPoses)
    rgbImage = images['occludedRGBImage'].squeeze()[:,:,:3].cpu().numpy()
    plt.imshow(rgbImage)
    plt.show()
    print('Testing finished')
