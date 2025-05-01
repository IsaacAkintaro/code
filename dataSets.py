import torch
from torch import nn
from torch.nn.functional import grid_sample, pad
from torch.utils.data import Dataset
import numpy
import hdf5storage
import math
import os

def importMATLABVariable(fileName,variableName):
    if not os.path.isfile(fileName):
        raise FileNotFoundError(fileName)
    return hdf5storage.loadmat(fileName,variable_names=[variableName])[variableName]

def importMATLABStructArrayField(fileName,variableName,variableIndex,fieldName):
    return importMATLABVariable(fileName,variableName)[0][variableIndex][fieldName]

def calculateTrigonometricEncoding(data, minimumPower, maximumPower):
    encodings = list()
    for power in range(minimumPower, maximumPower+1):
        encodings.append((math.pow(2, power)*math.pi*data).sin())
        encodings.append((math.pow(2, power)*math.pi*data).cos())
    return torch.cat(encodings, dim = 0)

def binVoxels(voxels):
    if len(voxels.shape) == 4:
        voxels = (voxels[:,0::2,0::2,0::2]+voxels[:,1::2,0::2,0::2]+voxels[:,0::2,1::2,0::2]+voxels[:,0::2,0::2,1::2]+
                    voxels[:,1::2,1::2,0::2]+voxels[:,1::2,0::2,1::2]+voxels[:,0::2,1::2,1::2]+voxels[:,1::2,1::2,1::2])/8
    elif len(voxels.shape) == 5:
        voxels = (voxels[:,:,0::2,0::2,0::2]+voxels[:,:,1::2,0::2,0::2]+voxels[:,:,0::2,1::2,0::2]+voxels[:,:,0::2,0::2,1::2]+
                    voxels[:,:,1::2,1::2,0::2]+voxels[:,:,1::2,0::2,1::2]+voxels[:,:,0::2,1::2,1::2]+voxels[:,:,1::2,1::2,1::2])/8
    else:
        raise ValueError('Unknown voxel data shape: {}.'.format(voxels.shape))
    return voxels

def parallel_variance(avg_a, count_a, var_a, avg_b, count_b, var_b):
    # From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    delta = avg_b - avg_a
    m_a = var_a * (count_a - 1)
    m_b = var_b * (count_b - 1)
    M2 = m_a + m_b + delta ** 2 * count_a * count_b / (count_a + count_b)
    return M2 / (count_a + count_b - 1)

def calculateDatasetStatistics(dataLoader):
    means = dict()
    variances = dict()
    counts = dict()
    for data in dataLoader:
        for key in data.keys():
            if key in means:
                oldMean = means[key]
            else:
                oldMean = numpy.float64(0)
            if key in variances:
                oldVariance = variances[key]
            else:
                oldVariance = numpy.float64(0)
            if key in counts:
                oldCount = counts[key]
            else:
                oldCount = numpy.uint64(0)
            currentData = data[key]
            if key == 'depthImage':
                currentData = currentData[(currentData > 0) & ~torch.isinf(currentData)]
            if currentData.dim() > 2: 
                #statistics over data dimensions
                #batch size, channels, data dimensions
                currentCount = numpy.prod(currentData.shape[2:], dtype = numpy.uint64)
                currentMean = currentData.to(dtype = torch.float64).mean(dim = list(range(2, len(currentData.shape))))
                currentVariance = currentData.to(dtype = torch.float64).var(dim = list(range(2, len(currentData.shape))))
            elif currentData.dim() == 2:
                #statistics over channels
                #batchSize, channels
                currentCount = numpy.uint64(currentData.shape[1])
                currentMean = currentData.to(dtype = torch.float64).mean(dim = 1)
                currentVariance = currentData.to(dtype = torch.float64).var(dim = 1)
            elif currentData.dim() == 1:
                #statistics over batch
                #batchSize
                currentCount = numpy.uint64(currentData.shape[0])
                currentMean = currentData.to(dtype = torch.float64).mean()
                currentVariance = currentData.to(dtype = torch.float64).var()
            else:
                raise ValueError('Unknown data shape: {}'.format(currentData.shape))
            if currentMean.dim() == 0:
                oldVariance = parallel_variance(oldMean, oldCount, oldVariance, currentMean, currentCount, currentVariance)
                oldMean = (oldCount*oldMean + currentCount*currentMean)/(oldCount+currentCount)
                oldCount = oldCount + currentCount
            else:
                for pointMean, pointVariance in zip(currentMean, currentVariance):
                    oldVariance = parallel_variance(oldMean, oldCount, oldVariance, pointMean, currentCount, pointVariance)
                    oldMean = (oldCount*oldMean + currentCount*pointMean)/(oldCount+currentCount)
                    oldCount = oldCount + currentCount
            means[key] = oldMean
            variances[key] = oldVariance
            counts[key] = oldCount
    return means, variances

def calculatePixelCoordinates(voxelGridBounds, voxelSize, imageResolution, cameraIntrinsics, cameraRotationMatrix, cameraTranslationVector):
    # gridVectors = [torch.arange(minValue, maxValue, voxelSize) for values in voxelGridBounds for minValue, maxValue in values]
    gridVectors = [torch.arange(min(values)+voxelSize/2,max(values)+voxelSize/2,voxelSize) for values in voxelGridBounds]
    xGrid, yGrid, zGrid = torch.meshgrid(*gridVectors)
    gridVector = torch.stack([xGrid.contiguous().view(-1),yGrid.contiguous().view(-1),zGrid.contiguous().view(-1)])
    rotatedGridVector = torch.matmul(cameraRotationMatrix,gridVector)
    transformedGridVector = rotatedGridVector + cameraTranslationVector
    # transformedGridVector = transformedGridVector[[0,2,1],:]#Convert to camera-based coordinate system (i.e. y is depth)  ##  
    pixelDepths = transformedGridVector[2,:]
    projectedGridVector = torch.matmul(cameraIntrinsics,transformedGridVector/transformedGridVector[2,:])
    pixelCoordinates = projectedGridVector.permute(1,0)
    pixelCoordinates = pixelCoordinates[:,[1,0]]#Conversion to y,x pixel coordinates for grid sampling
    scalingTensor = (torch.as_tensor(imageResolution)/2).unsqueeze(0).type(torch.float32)
    normalisedPixelCoordinates = (pixelCoordinates - scalingTensor)/scalingTensor
    return pixelCoordinates, pixelDepths, normalisedPixelCoordinates

def embedRepresentationMapInVoxelGrid(voxelGridBounds, voxelSize, depthMap, representationMap, cameraIntrinsics, cameraRotationMatrix, 
        cameraTranslationVector, voxelDepths = None, voxelNormalisedPixelCoordinates = None):
    gridVectors = [torch.arange(min(values)+voxelSize/2,max(values)+voxelSize/2,voxelSize) for values in voxelGridBounds]
    gridSize = [len(x) for x in gridVectors]
    voxelMaps = []
    if voxelDepths is None or voxelNormalisedPixelCoordinates is None:
        pixelCoordinates, voxelDepths, voxelNormalisedPixelCoordinates = calculatePixelCoordinates(voxelGridBounds, voxelSize, depthMap.shape[1:], 
                cameraIntrinsics, cameraRotationMatrix, cameraTranslationVector)
    # normalisedPixelCoordinates[:,:,:,0] = -normalisedPixelCoordinates[:,:,:,0]#Reversing height coordinate system
    sampledDepths = grid_sample(depthMap.unsqueeze(0),voxelNormalisedPixelCoordinates[None,None,:,[1,0]],mode='nearest',padding_mode='zeros').squeeze()
    dataVoxels = sampledDepths.gt(0)*sampledDepths.lt(float('Inf'))
    relativeDepth = voxelDepths-sampledDepths
    emptySpaceVoxels = (relativeDepth < - voxelSize/1)*dataVoxels
    surfaceVoxels = (relativeDepth.abs() <= voxelSize/1)*dataVoxels
    unseenVoxels = (relativeDepth > voxelSize/1)*dataVoxels 
    representationSamples = grid_sample(representationMap.unsqueeze(0),voxelNormalisedPixelCoordinates[None,None,:,[1,0]],mode='bilinear',padding_mode='zeros')
    representationVoxelGrid = representationSamples.reshape(representationMap.shape[0],*gridSize)
    emptySpaceVoxelGrid = emptySpaceVoxels.reshape(1,*gridSize)
    surfaceVoxelGrid = surfaceVoxels.reshape(1,*gridSize)
    unseenVoxelGrid = unseenVoxels.reshape(1,*gridSize)
    depthVoxelGrid = sampledDepths.reshape(1,*gridSize)
    relativeDepthVoxelGrid = relativeDepth.reshape(1,*gridSize)
    dataVoxelGrid = dataVoxels.reshape(1,*gridSize)
    output = dict(
        representationVoxelGrid = representationVoxelGrid,
        emptySpaceVoxelGrid = emptySpaceVoxelGrid,
        surfaceVoxelGrid = surfaceVoxelGrid,
        unseenVoxelGrid = unseenVoxelGrid,
        depthVoxelGrid = depthVoxelGrid,
        relativeDepthVoxelGrid = relativeDepthVoxelGrid,
        dataVoxelGrid = dataVoxelGrid
    )
    return output

class CompleteVoxelDataset(Dataset):

    def __init__(self, sceneSizes, firstSceneNumbers, lastSceneNumbers, dataDirectory, fileTemplate, mean = None, standardDeviation = None):
        super().__init__()
        self.dataDirectory = dataDirectory
        self.fileTemplate = fileTemplate
        self.mean = mean
        self.standardDeviation = standardDeviation
        self.sceneSize = list()
        self.sceneNumber = list()
        for sceneSize, firstSceneNumber, lastSceneNumber in zip(sceneSizes, firstSceneNumbers, lastSceneNumbers):
            sceneNumbers = list(range(firstSceneNumber, lastSceneNumber+1))
            self.sceneSize.extend([sceneSize]*len(sceneNumbers))
            self.sceneNumber.extend(sceneNumbers)

    def __getitem__(self, index):
        currentSceneSize = self.sceneSize[index]
        currentSceneNumber = self.sceneNumber[index]
        fileName = self.fileTemplate.format(currentSceneSize, currentSceneNumber)
        filePath = '{}/{}'.format(self.dataDirectory, fileName)
        data = hdf5storage.loadmat(filePath, variable_names=['originalClassLabelVoxelGridTS'])['originalClassLabelVoxelGridTS']
        data = data[:,:,:,1:16].transpose(3,0,1,2).astype(numpy.float32)
        if self.mean is not None:
            data = data - self.mean
        if self.standardDeviation is not None:
            data = data/self.standardDeviation
        data = binVoxels(data)
        return data        

    def __len__(self):
        return len(self.sceneSize)

class PreprocessedStabilityDataset(Dataset):

    def __init__(   self, sceneSizes, firstSceneNumbers, lastSceneNumbers, dataDirectory, fileTemplate, 
                    includeOriginalScenes, includePerturbedScenes, includeScenesWithMissingObjects,
                    voxelBinningPower = 0, dataMeans = dict(), dataStandardDeviations = dict(), enforceBinaryScenes = True):
        super().__init__()
        self.dataDirectory = dataDirectory
        self.fileTemplate = fileTemplate
        self.dataMeans = dataMeans
        self.dataStandardDeviations = dataStandardDeviations
        self.voxelBinningPower = voxelBinningPower
        self.enforceBinaryScenes = enforceBinaryScenes
        self.sceneSize = list()
        self.sceneNumber = list()
        self.sceneType = list()#0=original, 1=perturbed, 2=missingObject
        self.focusObject = list()
        for sceneSize, firstSceneNumber, lastSceneNumber in zip(sceneSizes, firstSceneNumbers, lastSceneNumbers):
            for sceneNumber in range(firstSceneNumber, lastSceneNumber+1):
                if includeOriginalScenes:
                    self.sceneSize.append(sceneSize)
                    self.sceneNumber.append(sceneNumber)
                    self.sceneType.append(0)
                    self.focusObject.append(0)
                if includePerturbedScenes:
                    for objectNumber in range(sceneSize):
                        self.sceneSize.append(sceneSize)
                        self.sceneNumber.append(sceneNumber)
                        self.sceneType.append(1)
                        self.focusObject.append(objectNumber)
                if includeScenesWithMissingObjects and (sceneSize > 1):
                    for objectNumber in range(sceneSize):
                        self.sceneSize.append(sceneSize)
                        self.sceneNumber.append(sceneNumber)
                        self.sceneType.append(2)
                        self.focusObject.append(objectNumber)

    def __getitem__(self, index):
        currentSceneSize = self.sceneSize[index]
        currentSceneNumber = self.sceneNumber[index]
        currentSceneType = self.sceneType[index]
        currentFocusObject = self.focusObject[index]
        if currentSceneType == 0:
            fileName = self.fileTemplate.format(currentSceneSize, currentSceneNumber, '_o')
        elif currentSceneType == 1:
            fileName = self.fileTemplate.format(currentSceneSize, currentSceneNumber, '_p{}'.format(currentFocusObject))
        elif currentSceneType == 2:
            fileName = self.fileTemplate.format(currentSceneSize, currentSceneNumber, '_m{}'.format(currentFocusObject))
        filePath = '{}/{}'.format(self.dataDirectory, fileName)
        data = numpy.load(filePath)
        data = {key:torch.as_tensor(value) for (key,value) in data.items()}
        
        #Binarize complete scene voxels if appropriate  (this is particularly relevant if the voxels have been binned)
        if self.enforceBinaryScenes and 'initialVoxelScene' in data:
            data['initialVoxelScene'] = (data['initialVoxelScene'] > 1e-5).to(dtype = torch.float)
        if self.enforceBinaryScenes and 'finalVoxelScene' in data:
            data['finalVoxelScene'] = (data['finalVoxelScene'] > 1e-5).to(dtype = torch.float)

        #Bin voxels if appropriate
        for ii in range(self.voxelBinningPower):
            if 'initialVoxelScene' in data:
                data['initialVoxelScene'] = binVoxels(data['initialVoxelScene'])
                if self.enforceBinaryScenes:
                    data['initialVoxelScene'] = (data['initialVoxelScene'] > 1e-5).to(dtype = torch.float)
            if 'finalVoxelScene' in data:
                data['finalVoxelScene'] = binVoxels(data['finalVoxelScene'])
                if self.enforceBinaryScenes:
                    data['finalVoxelScene'] = (data['finalVoxelScene'] > 1e-5).to(dtype = torch.float)

        # #Finally normalise if appropriate
        # if 'completeVoxelScene' in self.dataMeans and 'initialVoxelScene' in data:
        #     data['initialVoxelScene'] = data['initialVoxelScene'] - self.dataMeans['completeVoxelScene'][:,None,None,None].to(dtype = torch.float)
        # if 'completeVoxelScene' in self.dataStandardDeviations and 'initialVoxelScene' in data:
        #     data['initialVoxelScene'] = data['initialVoxelScene']/self.dataStandardDeviations['completeVoxelScene'][:,None,None,None].to(dtype = torch.float)
        # if 'completeVoxelScene' in self.dataMeans and 'finalVoxelScene' in data:
        #     data['finalVoxelScene'] = data['finalVoxelScene'] - self.dataMeans['completeVoxelScene'][:,None,None,None].to(dtype = torch.float)
        # if 'completeVoxelScene' in self.dataStandardDeviations and 'finalVoxelScene' in data:
        #     data['finalVoxelScene'] = data['finalVoxelScene']/self.dataStandardDeviations['completeVoxelScene'][:,None,None,None].to(dtype = torch.float)

        for key in data:
            if key in self.dataMeans:
                data[key] = data[key] - self.dataMeans[key]
            if key in self.dataStandardDeviations:
                data[key] = data[key]/self.dataStandardDeviations[key]

        return data

    def __len__(self):
        return len(self.sceneSize)

class StabilityDataset(Dataset):

    def __init__(   self, sceneSizes, firstSceneNumbers, lastSceneNumbers, dataDirectory, fileTemplate, 
                    includeOriginalScenes, includePerturbedScenes, includeScenesWithMissingObjects, numberOfObjectClasses,
                    voxelBinningPower = 0, dataMeans = dict(), dataStandardDeviations = dict(), enforceBinaryScenes = True):
        super().__init__()
        self.dataDirectory = dataDirectory
        self.fileTemplate = fileTemplate
        self.numberOfObjectClasses = numberOfObjectClasses
        self.dataMeans = dataMeans
        self.dataStandardDeviations = dataStandardDeviations
        self.voxelBinningPower = voxelBinningPower
        self.enforceBinaryScenes = enforceBinaryScenes
        self.sceneSize = list()
        self.sceneNumber = list()
        self.sceneType = list()#0=original, 1=perturbed, 2=missingObject
        self.focusObject = list()
        for sceneSize, firstSceneNumber, lastSceneNumber in zip(sceneSizes, firstSceneNumbers, lastSceneNumbers):
            for sceneNumber in range(firstSceneNumber, lastSceneNumber+1):
                if includeOriginalScenes:
                    self.sceneSize.append(sceneSize)
                    self.sceneNumber.append(sceneNumber)
                    self.sceneType.append(0)
                    self.focusObject.append(0)
                if includePerturbedScenes:
                    for objectNumber in range(sceneSize):
                        self.sceneSize.append(sceneSize)
                        self.sceneNumber.append(sceneNumber)
                        self.sceneType.append(1)
                        self.focusObject.append(objectNumber)
                if includeScenesWithMissingObjects and (sceneSize > 1):
                    for objectNumber in range(sceneSize):
                        self.sceneSize.append(sceneSize)
                        self.sceneNumber.append(sceneNumber)
                        self.sceneType.append(2)
                        self.focusObject.append(objectNumber)

    def __getitem__(self, index):
        currentSceneSize = self.sceneSize[index]
        currentSceneNumber = self.sceneNumber[index]
        currentSceneType = self.sceneType[index]
        currentFocusObject = self.focusObject[index]
        fileName = self.fileTemplate.format(currentSceneSize, currentSceneNumber)
        voxelFilePath = '{}/{}'.format(self.dataDirectory, fileName)
        if currentSceneType == 0:#Original
            data = hdf5storage.loadmat(voxelFilePath, variable_names=['originalClassLabelVoxelGridTS', 'originalClassLabelVoxelGridTE', 'originalTrajectories', 'originalCollisions', 'heap'])
            initialVoxelScene = torch.as_tensor(data['originalClassLabelVoxelGridTS'], dtype = torch.float)
            finalVoxelScene = torch.as_tensor(data['originalClassLabelVoxelGridTE'], dtype = torch.float)
            trajectoryData = data['originalTrajectories']
            rotationData = trajectoryData['rotationMatrix']
            translationData = trajectoryData['translationVector']
            heapData = data['heap']
            rotations = torch.zeros(self.numberOfObjectClasses,3,3,rotationData[0,0].shape[2])
            translations = torch.zeros(self.numberOfObjectClasses,3,rotationData[0,0].shape[2])
            classes = torch.zeros(self.numberOfObjectClasses)
            for index in range(heapData.shape[1]):
                currentObjectClass = heapData[0, index]['objectLibraryIndex'][0,0].item() - 1
                classes[currentObjectClass] = 1
                rotations[currentObjectClass,:,:,:] = torch.as_tensor(rotationData[0,index], dtype = torch.float)
                translations[currentObjectClass,:,:] = torch.as_tensor(translationData[0,index], dtype = torch.float)
            collisionData = data['originalCollisions'].squeeze()
            contacts = torch.zeros(self.numberOfObjectClasses+1,self.numberOfObjectClasses+1,rotationData[0,0].shape[2])
            for timeStep in range(collisionData.size):
                currentCollisions = collisionData[timeStep]
                currentDynamicCollisions = currentCollisions['dynamicObjectCollisions'][0,0]
                if currentDynamicCollisions.size > 0:
                    for collisionIndex in range(currentDynamicCollisions.size):
                        currentCollision = currentDynamicCollisions[0,collisionIndex]
                        object1Index = heapData[0, currentCollision['objectIndex'][0,0].item()-1]['objectLibraryIndex'][0,0].item()
                        object1IsDynamic = currentCollision['isDynamic'][0,0].item()
                        object2Index = heapData[0, currentCollision['objectIndex'][1,0].item()-1]['objectLibraryIndex'][0,0].item()
                        object2IsDynamic = currentCollision['isDynamic'][1,0].item()
                        contacts[object1Index,object2Index,timeStep] = 1
                        contacts[object2Index,object1Index,timeStep] = 1
                currentMixedCollisions = currentCollisions['mixedObjectCollisions'][0,0]
                if currentMixedCollisions.size > 0:
                    for collisionIndex in range(currentMixedCollisions.size):
                        currentCollision = currentMixedCollisions[0,collisionIndex]
                        object1IsDynamic = currentCollision['isDynamic'][0,0].item() == 1
                        object2IsDynamic = currentCollision['isDynamic'][1,0].item() == 1
                        if object1IsDynamic:
                            object1Index = heapData[0, currentCollision['objectIndex'][0,0].item()-1]['objectLibraryIndex'][0,0].item()
                            contacts[object1Index,-1,timeStep] = 1
                            contacts[-1,object1Index,timeStep] = 1
                        elif object2IsDynamic:
                            object2Index = heapData[0, currentCollision['objectIndex'][1,0].item()-1]['objectLibraryIndex'][0,0].item()
                            contacts[object2Index,-1,timeStep] = 1
                            contacts[-1,object2Index,timeStep] = 1
                        else:
                            raise ValueError('Collision has no dynamic objects')

        elif currentSceneType == 1:#Perturbed
            data = hdf5storage.loadmat(voxelFilePath, variable_names=['perturbations'])
            initialVoxelScene = torch.as_tensor(data['perturbations'][0,currentFocusObject]['perturbedClassLabelVoxelGridTS'], dtype = torch.float)
            finalVoxelScene = torch.as_tensor(data['perturbations'][0,currentFocusObject]['perturbedClassLabelVoxelGridTE'], dtype = torch.float)
            trajectoryData = data['perturbations'][0,currentFocusObject]['perturbedTrajectories']
            rotationData = trajectoryData['rotationMatrix']
            translationData = trajectoryData['translationVector']
            heapData = data['perturbations'][0,currentFocusObject]['perturbedHeap']
            rotations = torch.zeros(self.numberOfObjectClasses,3,3,rotationData[0,0].shape[2])
            translations = torch.zeros(self.numberOfObjectClasses,3,rotationData[0,0].shape[2])
            classes = torch.zeros(self.numberOfObjectClasses)
            for index in range(heapData.shape[1]):
                currentObjectClass = heapData[0, index]['objectLibraryIndex'][0,0].item() - 1
                classes[currentObjectClass] = 1
                rotations[currentObjectClass,:,:,:] = torch.as_tensor(rotationData[0,index], dtype = torch.float)
                translations[currentObjectClass,:,:] = torch.as_tensor(translationData[0,index], dtype = torch.float)
            collisionData = data['perturbations'][0,currentFocusObject]['perturbedCollisions'].squeeze()
            contacts = torch.zeros(self.numberOfObjectClasses+1,self.numberOfObjectClasses+1,rotationData[0,0].shape[2])
            for timeStep in range(collisionData.size):
                currentCollisions = collisionData[timeStep]
                currentDynamicCollisions = currentCollisions['dynamicObjectCollisions'][0,0]
                if currentDynamicCollisions.size > 0:
                    for collisionIndex in range(currentDynamicCollisions.size):
                        currentCollision = currentDynamicCollisions[0,collisionIndex]
                        object1Index = heapData[0, currentCollision['objectIndex'][0,0].item()-1]['objectLibraryIndex'][0,0].item()
                        object1IsDynamic = currentCollision['isDynamic'][0,0].item()
                        object2Index = heapData[0, currentCollision['objectIndex'][1,0].item()-1]['objectLibraryIndex'][0,0].item()
                        object2IsDynamic = currentCollision['isDynamic'][1,0].item()
                        contacts[object1Index,object2Index,timeStep] = 1
                        contacts[object2Index,object1Index,timeStep] = 1
                currentMixedCollisions = currentCollisions['mixedObjectCollisions'][0,0]
                if currentMixedCollisions.size > 0:
                    for collisionIndex in range(currentMixedCollisions.size):
                        currentCollision = currentMixedCollisions[0,collisionIndex]
                        object1IsDynamic = currentCollision['isDynamic'][0,0].item() == 1
                        object2IsDynamic = currentCollision['isDynamic'][1,0].item() == 1
                        if object1IsDynamic:
                            object1Index = heapData[0, currentCollision['objectIndex'][0,0].item()-1]['objectLibraryIndex'][0,0].item()
                            contacts[object1Index,-1,timeStep] = 1
                            contacts[-1,object1Index,timeStep] = 1
                        elif object2IsDynamic:
                            object2Index = heapData[0, currentCollision['objectIndex'][1,0].item()-1]['objectLibraryIndex'][0,0].item()
                            contacts[object2Index,-1,timeStep] = 1
                            contacts[-1,object2Index,timeStep] = 1
                        else:
                            raise ValueError('Collision has no dynamic objects')
        elif currentSceneType == 2:#Missing object
            data = hdf5storage.loadmat(voxelFilePath, variable_names=['missingObjects'])
            initialVoxelScene = torch.as_tensor(data['missingObjects'][0,currentFocusObject]['missingObjectClassLabelVoxelGridTS'], dtype = torch.float)
            finalVoxelScene = torch.as_tensor(data['missingObjects'][0,currentFocusObject]['missingObjectClassLabelVoxelGridTE'], dtype = torch.float)
            trajectoryData = data['missingObjects'][0,currentFocusObject]['missingObjectTrajectories']
            rotationData = trajectoryData['rotationMatrix']
            translationData = trajectoryData['translationVector']
            heapData = data['missingObjects'][0,currentFocusObject]['missingObjectHeap']
            rotations = torch.zeros(self.numberOfObjectClasses,3,3,rotationData[0,0].shape[2])
            translations = torch.zeros(self.numberOfObjectClasses,3,rotationData[0,0].shape[2])
            classes = torch.zeros(self.numberOfObjectClasses)
            for index in range(heapData.shape[1]):
                currentObjectClass = heapData[0, index]['objectLibraryIndex'][0,0].item() - 1
                classes[currentObjectClass] = 1
                rotations[currentObjectClass,:,:,:] = torch.as_tensor(rotationData[0,index], dtype = torch.float)
                translations[currentObjectClass,:,:] = torch.as_tensor(translationData[0,index], dtype = torch.float)
            collisionData = data['missingObjects'][0,currentFocusObject]['missingObjectCollisions'].squeeze()
            contacts = torch.zeros(self.numberOfObjectClasses+1,self.numberOfObjectClasses+1,rotationData[0,0].shape[2])
            for timeStep in range(collisionData.size):
                currentCollisions = collisionData[timeStep]
                currentDynamicCollisions = currentCollisions['dynamicObjectCollisions'][0,0]
                if currentDynamicCollisions.size > 0:
                    for collisionIndex in range(currentDynamicCollisions.size):
                        currentCollision = currentDynamicCollisions[0,collisionIndex]
                        object1Index = heapData[0, currentCollision['objectIndex'][0,0].item()-1]['objectLibraryIndex'][0,0].item()
                        object1IsDynamic = currentCollision['isDynamic'][0,0].item()
                        object2Index = heapData[0, currentCollision['objectIndex'][1,0].item()-1]['objectLibraryIndex'][0,0].item()
                        object2IsDynamic = currentCollision['isDynamic'][1,0].item()
                        contacts[object1Index,object2Index,timeStep] = 1
                        contacts[object2Index,object1Index,timeStep] = 1
                currentMixedCollisions = currentCollisions['mixedObjectCollisions'][0,0]
                if currentMixedCollisions.size > 0:
                    for collisionIndex in range(currentMixedCollisions.size):
                        currentCollision = currentMixedCollisions[0,collisionIndex]
                        object1IsDynamic = currentCollision['isDynamic'][0,0].item() == 1
                        object2IsDynamic = currentCollision['isDynamic'][1,0].item() == 1
                        if object1IsDynamic:
                            object1Index = heapData[0, currentCollision['objectIndex'][0,0].item()-1]['objectLibraryIndex'][0,0].item()
                            contacts[object1Index,-1,timeStep] = 1
                            contacts[-1,object1Index,timeStep] = 1
                        elif object2IsDynamic:
                            object2Index = heapData[0, currentCollision['objectIndex'][1,0].item()-1]['objectLibraryIndex'][0,0].item()
                            contacts[object2Index,-1,timeStep] = 1
                            contacts[-1,object2Index,timeStep] = 1
                        else:
                            raise ValueError('Collision has no dynamic objects')
        initialVoxelScene = initialVoxelScene[:,:,:,1:16].permute(3,0,1,2)
        finalVoxelScene = finalVoxelScene[:,:,:,1:16].permute(3,0,1,2)
        if self.enforceBinaryScenes:
            initialVoxelScene = (initialVoxelScene > 1e-5).to(dtype = torch.float)
            finalVoxelScene = (finalVoxelScene > 1e-5).to(dtype = torch.float)
        for ii in range(self.voxelBinningPower):
            initialVoxelScene = binVoxels(initialVoxelScene)
            finalVoxelScene = binVoxels(finalVoxelScene)
            if self.enforceBinaryScenes:
                initialVoxelScene = (initialVoxelScene > 1e-5).to(dtype = torch.float)
                finalVoxelScene = (finalVoxelScene > 1e-5).to(dtype = torch.float)

        # if 'completeVoxelScene' in self.dataMeans:
        #     initialVoxelScene = initialVoxelScene - self.dataMeans['completeVoxelScene'][:,None,None,None].to(dtype = torch.float)
        #     finalVoxelScene = finalVoxelScene - self.dataMeans['completeVoxelScene'][:,None,None,None].to(dtype = torch.float)
        # if 'completeVoxelScene' in self.dataStandardDeviations:
        #     initialVoxelScene = initialVoxelScene/self.dataStandardDeviations['completeVoxelScene'][:,None,None,None].to(dtype = torch.float)
        #     finalVoxelScene = finalVoxelScene/self.dataStandardDeviations['completeVoxelScene'][:,None,None,None].to(dtype = torch.float)

        output = dict(  initialVoxelScene = initialVoxelScene,
                        finalVoxelScene = finalVoxelScene,
                        classes = classes,
                        translations = translations,
                        rotations = rotations,
                        contacts = contacts,
                        sceneSize = currentSceneSize,
                        sceneNumber = currentSceneNumber,
                        sceneType = currentSceneType,
                        focusObject = currentFocusObject)

        for key in output:
            if key in self.dataMeans:
                output[key] = output[key] - self.dataMeans[key]
            if key in self.dataStandardDeviations:
                output[key] = output[key]/self.dataStandardDeviations[key]

        return output

    def __len__(self):
        return len(self.sceneSize)

class PreprocessedMixedDataset(Dataset):
    def __init__(   self, sceneSizes, firstSceneNumbers, lastSceneNumbers, dataDirectory, fileTemplate, voxelBinningPower = 0, 
                    dataMeans = dict(), dataStandardDeviations = dict(), enforceBinaryCompleteScenes = True):
        sceneSizeList = list()
        sceneNumberList = list()
        for sceneSize, firstSceneNumber, lastSceneNumber in zip(sceneSizes, firstSceneNumbers, lastSceneNumbers):
            sceneNumbers = list(range(firstSceneNumber, lastSceneNumber+1))
            sceneSizeList.extend([sceneSize]*len(sceneNumbers))
            sceneNumberList.extend(sceneNumbers)
        self.dataSet = PreprocessedMixedListDataset(sceneSizeList, sceneNumberList, dataDirectory, fileTemplate, voxelBinningPower, 
                    dataMeans, dataStandardDeviations, enforceBinaryCompleteScenes)

    def __getitem__(self, index):
        return self.dataSet[index]

    def __len__(self):
        return len(self.dataSet)

class PreprocessedMixedListDataset(Dataset):
    def __init__(   self, sceneSize, sceneNumber, dataDirectory, fileTemplate, voxelBinningPower = 0, 
                    dataMeans = dict(), dataStandardDeviations = dict(), enforceBinaryCompleteScenes = True):
        self.dataDirectory = dataDirectory
        self.fileTemplate = fileTemplate
        self.voxelBinningPower = voxelBinningPower
        self.dataMeans = dataMeans
        self.dataStandardDeviations = dataStandardDeviations
        self.enforceBinaryCompleteScenes = enforceBinaryCompleteScenes
        self.sceneSize = sceneSize
        self.sceneNumber = sceneNumber

    def __getitem__(self, index):
        currentSceneSize = self.sceneSize[index]
        currentSceneNumber = self.sceneNumber[index]
        fileName = self.fileTemplate.format(currentSceneSize, currentSceneNumber)
        filePath = '{}/{}'.format(self.dataDirectory, fileName)
        data = numpy.load(filePath)
        data = {key:torch.as_tensor(value) for (key,value) in data.items()}
        
        #Binarize complete scene voxels if appropriate  (this is particularly relevant if the voxels have been binned)
        if self.enforceBinaryCompleteScenes and 'completeVoxelScene' in data:
            data['completeVoxelScene'] = (data['completeVoxelScene'] > 1e-5).to(dtype = torch.float)

        #Bin voxels if appropriate
        for ii in range(self.voxelBinningPower):
            if 'completeVoxelScene' in data:
                data['completeVoxelScene'] = binVoxels(data['completeVoxelScene'])
                if self.enforceBinaryCompleteScenes:
                    data['completeVoxelScene'] = (data['completeVoxelScene'] > 1e-5).to(dtype = torch.float)
            if 'surfaceVoxelEmbedding' in data:
                data['surfaceVoxelEmbedding'] = binVoxels(data['surfaceVoxelEmbedding'])
            if 'projectionVoxelEmbedding' in data:
                data['projectionVoxelEmbedding'] = binVoxels(data['projectionVoxelEmbedding'])

        #Finally normalise if appropriate
        # if 'completeVoxelScene' in self.dataMeans and 'completeVoxelScene' in data:
        #     data['completeVoxelScene'] = data['completeVoxelScene'] - self.dataMeans['completeVoxelScene'][:,None,None,None].to(dtype = torch.float)
        # if 'completeVoxelScene' in self.dataStandardDeviations and 'completeVoxelScene' in data:
        #     data['completeVoxelScene'] = data['completeVoxelScene']/self.dataStandardDeviations['completeVoxelScene'][:,None,None,None].to(dtype = torch.float)
        # if 'surfaceVoxelEmbedding' in self.dataMeans and 'surfaceVoxelEmbedding' in data:
        #     data['surfaceVoxelEmbedding'] = data['surfaceVoxelEmbedding'] - self.dataMeans['surfaceVoxelEmbedding'][:,None,None,None].to(dtype = torch.float)
        # if 'surfaceVoxelEmbedding' in self.dataStandardDeviations and 'surfaceVoxelEmbedding' in data:
        #     data['surfaceVoxelEmbedding'] = data['surfaceVoxelEmbedding']/self.dataStandardDeviations['surfaceVoxelEmbedding'][:,None,None,None].to(dtype = torch.float)
        # if 'projectionVoxelEmbedding' in self.dataMeans and 'projectionVoxelEmbedding' in data:
        #     data['projectionVoxelEmbedding'] = data['projectionVoxelEmbedding'] - self.dataMeans['projectionVoxelEmbedding'][:,None,None,None].to(dtype = torch.float)
        # if 'projectionVoxelEmbedding' in self.dataStandardDeviations and 'projectionVoxelEmbedding' in data:
        #     data['projectionVoxelEmbedding'] = data['projectionVoxelEmbedding']/self.dataStandardDeviations['projectionVoxelEmbedding'][:,None,None,None].to(dtype = torch.float)

        for key in data:
            if key in self.dataMeans:
                data[key] = data[key] - self.dataMeans[key]
            if key in self.dataStandardDeviations:
                data[key] = data[key]/self.dataStandardDeviations[key]
                
        return data

    def __len__(self):
        return len(self.sceneSize)

class MixedDataset(Dataset):

    def __init__(self, sceneSizes, firstSceneNumbers, lastSceneNumbers, imageDataDirectory, imageFileTemplate, voxelDataDirectory, voxelFileTemplate,
                voxelGridBounds, voxelSize, cameraIntrinsics, cameraRotationMatrix, cameraTranslationVector, 
                cameraResolution, voxelBinningPower = 0, classLabels = None, numberOfObjectClasses = 14,
                dataMeans = dict(), dataStandardDeviations = dict(), enforceBinaryCompleteScenes = True):
        super().__init__()
        sceneSizeList = list()
        sceneNumberList = list()
        for sceneSize, firstSceneNumber, lastSceneNumber in zip(sceneSizes, firstSceneNumbers, lastSceneNumbers):
            sceneNumbers = list(range(firstSceneNumber, lastSceneNumber+1))
            sceneSizeList.extend([sceneSize]*len(sceneNumbers))
            sceneNumberList.extend(sceneNumbers)
        self.dataSet = MixedListDataset(sceneSizeList, sceneNumberList, imageDataDirectory, imageFileTemplate, voxelDataDirectory, voxelFileTemplate,
                voxelGridBounds, voxelSize, cameraIntrinsics, cameraRotationMatrix, cameraTranslationVector, 
                cameraResolution, voxelBinningPower, classLabels, numberOfObjectClasses,
                dataMeans, dataStandardDeviations, enforceBinaryCompleteScenes)

    def __getitem__(self, index):
        return self.dataSet[index]
        
    def __len__(self):
        return len(self.dataSet)


class MixedListDataset(Dataset):

    def __init__(self, sceneSize, sceneNumber, imageDataDirectory, imageFileTemplate, voxelDataDirectory, voxelFileTemplate,
                voxelGridBounds, voxelSize, cameraIntrinsics, cameraRotationMatrix, cameraTranslationVector, 
                cameraResolution, voxelBinningPower = 0, classLabels = None, numberOfObjectClasses = 14,
                dataMeans = dict(), dataStandardDeviations = dict(), enforceBinaryCompleteScenes = True):
        super().__init__()
        self.imageDataDirectory = imageDataDirectory
        self.imageFileTemplate = imageFileTemplate
        self.voxelDataDirectory = voxelDataDirectory
        self.voxelFileTemplate = voxelFileTemplate
        self.voxelGridBounds = voxelGridBounds
        self.voxelSize = voxelSize
        self.cameraIntrinsics = cameraIntrinsics
        self.cameraRotationMatrix = cameraRotationMatrix
        self.cameraTranslationVector = cameraTranslationVector
        self.cameraResolution = cameraResolution
        self.dataMeans = dataMeans
        self.dataStandardDeviations = dataStandardDeviations
        self.enforceBinaryCompleteScenes = enforceBinaryCompleteScenes
        self.voxelBinningPower = voxelBinningPower
        if classLabels is not None and not isinstance(classLabels, torch.Tensor):
            classLabels = torch.as_tensor(classLabels)
        self.classLabels = classLabels
        self.numberOfObjectClasses = numberOfObjectClasses
        self.sceneSize = sceneSize
        self.sceneNumber = sceneNumber
        
        pixelCoordinates, voxelDepths, voxelNormalisedPixelCoordinates = calculatePixelCoordinates(voxelGridBounds, voxelSize, cameraResolution, 
                cameraIntrinsics, cameraRotationMatrix, cameraTranslationVector)
        self.voxelDepths = voxelDepths
        self.voxelNormalisedPixelCoordinates = voxelNormalisedPixelCoordinates

    def __getitem__(self, index):
        currentSceneSize = self.sceneSize[index]
        currentSceneNumber = self.sceneNumber[index]

        imageFileName = self.imageFileTemplate.format(currentSceneSize, currentSceneNumber)
        imageFilePath = '{}/{}'.format(self.imageDataDirectory, imageFileName)
        imageData = hdf5storage.loadmat(imageFilePath, variable_names=['heapDepthImage', 'heapRGBCompositeImage', 'heapClassLabelImage','heapInstanceLabelImage','heap','heapUnoccludedObjectInstanceMaps'])
        depthImage = torch.as_tensor(imageData['heapDepthImage'], dtype = torch.float).unsqueeze(dim = 0)
        rgbImage = torch.as_tensor(imageData['heapRGBCompositeImage'], dtype = torch.float).permute(2,0,1)
        classLabelImage = torch.as_tensor(imageData['heapClassLabelImage'], dtype = torch.int16)
        instanceLabelImage = torch.as_tensor(imageData['heapInstanceLabelImage'], dtype = torch.int16)
        unoccludedObjectInstanceMaps = torch.as_tensor(imageData['heapUnoccludedObjectInstanceMaps'], dtype = torch.bool)
        if unoccludedObjectInstanceMaps.dim() == 2:
            unoccludedObjectInstanceMaps = unoccludedObjectInstanceMaps.unsqueeze(dim = 2)
        if self.classLabels is None:
            uniqueLabels = classLabelImage.unique()
        else:
            uniqueLabels = self.classLabels
        classSegmentationImage = torch.zeros(uniqueLabels.numel(),*classLabelImage.shape)
        for sliceIndex, currentSlice in enumerate(classSegmentationImage):
            currentSlice[classLabelImage == uniqueLabels[sliceIndex]] = 1
        classLabelImage = classLabelImage.unsqueeze(dim = 0)

        classes = torch.zeros(self.numberOfObjectClasses)
        visibility = torch.zeros(self.numberOfObjectClasses)
        heapData = imageData['heap']
        for instanceIndex in range(heapData.shape[1]):
            currentObjectClass = int(round(heapData[0, instanceIndex]['objectLibraryIndex'][0,0].item()))
            classes[currentObjectClass - 1] = 1
            unoccludedInstanceMap = (unoccludedObjectInstanceMaps[:,:,instanceIndex]).to(torch.float)
            occludedInstanceMap = (instanceLabelImage == instanceIndex + 1).to(torch.float)
            visibility[currentObjectClass - 1] = occludedInstanceMap.sum()/unoccludedInstanceMap.sum()

        embeddingData = embedRepresentationMapInVoxelGrid(voxelGridBounds = self.voxelGridBounds, voxelSize = self.voxelSize, depthMap = depthImage, 
            representationMap = classSegmentationImage, cameraIntrinsics = self.cameraIntrinsics, cameraRotationMatrix = self.cameraRotationMatrix, 
            cameraTranslationVector = self.cameraTranslationVector, voxelDepths = self.voxelDepths, voxelNormalisedPixelCoordinates = self.voxelNormalisedPixelCoordinates)

        surfaceVoxelEmbedding = torch.cat([embeddingData['representationVoxelGrid'].to(dtype = torch.float)*embeddingData['surfaceVoxelGrid'].to(dtype=torch.float),
            embeddingData['unseenVoxelGrid'].to(dtype=torch.float)], dim= 0)
        relativeDepth = embeddingData['relativeDepthVoxelGrid'].to(dtype = torch.float)
        relativeDepthSign = relativeDepth.sign()
        relativeDepthSign[relativeDepthSign == 0] = 1
        relativeDepth = 1/(relativeDepth + relativeDepthSign*1e-3)
        # relativeDepth[~embeddingData['dataVoxelGrid']] = relativeDepth[embeddingData['dataVoxelGrid']].max() + 1
        projectionVoxelEmbedding = torch.cat([embeddingData['representationVoxelGrid'].to(dtype = torch.float)*embeddingData['dataVoxelGrid'].to(dtype=torch.float),relativeDepth], dim = 0)

        voxelFileName = self.voxelFileTemplate.format(currentSceneSize, currentSceneNumber)
        voxelFilePath = '{}/{}'.format(self.voxelDataDirectory, voxelFileName)
        completeVoxelScene = torch.as_tensor(hdf5storage.loadmat(voxelFilePath, variable_names=['originalClassLabelVoxelGridTS'])['originalClassLabelVoxelGridTS'], dtype = torch.float)
        completeVoxelScene = completeVoxelScene[:,:,:,1:16].permute(3,0,1,2)
        if self.enforceBinaryCompleteScenes:
            completeVoxelScene = (completeVoxelScene > 1e-5).to(dtype = torch.float)
        if self.voxelBinningPower > 0:
            for ii in range(self.voxelBinningPower):
                completeVoxelScene = binVoxels(completeVoxelScene)
                surfaceVoxelEmbedding = binVoxels(surfaceVoxelEmbedding)
                projectionVoxelEmbedding = binVoxels(projectionVoxelEmbedding)
                if self.enforceBinaryCompleteScenes:
                    completeVoxelScene = (completeVoxelScene > 1e-5).to(dtype = torch.float)

        # if 'completeVoxelScene' in self.dataMeans:
        #     completeVoxelScene = completeVoxelScene - self.dataMeans['completeVoxelScene'][:,None,None,None].to(dtype = torch.float)
        # if 'completeVoxelScene' in self.dataStandardDeviations:
        #     completeVoxelScene = completeVoxelScene/self.dataStandardDeviations['completeVoxelScene'][:,None,None,None].to(dtype = torch.float)
        # if 'surfaceVoxelEmbedding' in self.dataMeans:
        #     surfaceVoxelEmbedding = surfaceVoxelEmbedding - self.dataMeans['surfaceVoxelEmbedding'][:,None,None,None].to(dtype = torch.float)
        # if 'surfaceVoxelEmbedding' in self.dataStandardDeviations:
        #     surfaceVoxelEmbedding = surfaceVoxelEmbedding/self.dataStandardDeviations['surfaceVoxelEmbedding'][:,None,None,None].to(dtype = torch.float)
        # if 'projectionVoxelEmbedding' in self.dataMeans:
        #     projectionVoxelEmbedding = projectionVoxelEmbedding - self.dataMeans['projectionVoxelEmbedding'][:,None,None,None].to(dtype = torch.float)
        # if 'projectionVoxelEmbedding' in self.dataStandardDeviations:
        #     projectionVoxelEmbedding = projectionVoxelEmbedding/self.dataStandardDeviations['projectionVoxelEmbedding'][:,None,None,None].to(dtype = torch.float)

        output = dict(
                        depthImage = depthImage,
                        rgbImage = rgbImage,
                        classLabelImage = classLabelImage,
                        classSegmentationImage = classSegmentationImage,
                        classes = classes,
                        visibility = visibility,
                        completeVoxelScene = completeVoxelScene,
                        surfaceVoxelEmbedding = surfaceVoxelEmbedding,
                        projectionVoxelEmbedding = projectionVoxelEmbedding,
                        sceneSize = torch.as_tensor([currentSceneSize]),
                        sceneNumber = torch.as_tensor([currentSceneNumber])
                    )

        for key in output:
            if key in self.dataMeans:
                output[key] = output[key] - self.dataMeans[key]
            if key in self.dataStandardDeviations:
                output[key] = output[key]/self.dataStandardDeviations[key]
        return output      

    def __len__(self):
        return len(self.sceneSize)
