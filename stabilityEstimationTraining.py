import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy
import math
import dataSets
import hdf5storage
import os
import sys
import shutil
import time
import stabilityEstimationTrainingConfiguration as configuration
import networkArchitectures
import matplotlib.pyplot as plt

def getDataBatch(dataLoader, dataIterator):
    try:
        dataBatch = next(dataIterator)
    except:
        dataIterator = iter(dataLoader)
        dataBatch = next(dataIterator)
    return dataBatch, dataIterator

def calculateInterpolationGradientPenalty(inputs1, inputs2, network, penaltyType = 'equality'):
    batchSize = inputs1.shape[0]
    device = inputs1.device
    weighting = torch.randn(batchSize, *[1 for x in inputs1.shape[1:]], device = device)
    interpolatedInputs = weighting*inputs1.detach() + (1-weighting)*inputs2.detach()
    interpolatedInputs = interpolatedInputs.requires_grad_(True)
    outputs = network(interpolatedInputs)
    if isinstance(outputs, tuple) or isinstance(outputs, list):
        outputs = outputs[0]
    gradients = torch.autograd.grad(outputs=outputs.sum(),
                                    inputs=interpolatedInputs,
                                    create_graph=True)[0]
    gradientNorm = gradients.view(batchSize, -1).norm(dim = 1)
    if penaltyType == 'equality':
        output = (gradientNorm - 1).pow(2)
    elif penaltyType == 'inequality':
        output = (gradientNorm - 1).clamp_min(0).pow(2)
    else:
        raise ValueError('Unknown penalty type: {}'.format(penaltyType))
    return output

def processQuantity(quantity, quantisationBinCentres, quantisationBinSpacing, trainingConfig):
    if trainingConfig.predictionScale == 'scene':
        quantity, _ = quantity.max(dim = 1, keepdim = True)
        if trainingConfig.predictionType == 'quantised':
            if trainingConfig.quantisationBinScaling == 'exponential':
                quantity = quantity.log10() - quantisationBinCentres[None, :]
            elif trainingConfig.quantisationBinScaling == 'linear':
                quantity = quantity - quantisationBinCentres[None, :]
            else:
                raise ValueError('Unknown quantisation bin scaling: {}'.format(trainingConfig.quantisationBinScaling))
            if trainingConfig.lossType == 'crossEntropy':
                values, quantity = quantity.abs().min(dim = 1)
            else:
                quantity = torch.exp(-(quantity/quantisationBinSpacing/2).pow(2)/2)
                quantity = quantity/(quantity.sum(dim = 1, keepdim = True) + 1e-7)
    elif trainingConfig.predictionScale == 'object':
        if trainingConfig.predictionType == 'quantised':
            if trainingConfig.quantisationBinScaling == 'exponential':
                quantity = (quantity + 1e-7).log10().unsqueeze(dim = 2) - quantisationBinCentres[None, None, :]
            elif trainingConfig.quantisationBinScaling == 'linear':
                quantity = quantity.unsqueeze(dim = 2) - quantisationBinCentres[None, None, :]
            else:
                raise ValueError('Unknown quantisation bin scaling: {}'.format(trainingConfig.quantisationBinScaling))
            if trainingConfig.lossType == 'crossEntropy':
                values, quantity = quantity.abs().min(dim = 2)
            else:
                quantity = torch.exp(-(quantity/quantisationBinSpacing/2).pow(2)/2)
                quantity = quantity/(quantity.sum(dim = 2, keepdim = True) + 1e-7)
    else:
        raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
    return quantity       

def klDivergence( logProbabilities1, logProbabilities2):
    divergence = logProbabilities1.exp()*(logProbabilities1 - logProbabilities2)
    return divergence

def jsDivergence( logProbabilities1, logProbabilities2):
    probabilityDifference = logProbabilities1.exp() - logProbabilities2.exp()
    divergence = (probabilityDifference*logProbabilities1 - probabilityDifference*logProbabilities2)/2
    return divergence.sum(dim = 1)

def trainNetwork(   network, trainingDataloader, trainingDataIterator, gradientPenaltyDataIterator, 
                    numberOfObjectClasses, quantisationBinCentres, quantisationBinSpacing, classWeights, device, networkOptimiser, networkConfig, dataConfig, trainingConfig):
    network.train() 
    if trainingConfig.predictionScale == 'scene':
        networkOutputDimension = trainingConfig.numberOfQuantisationBins
    elif trainingConfig.predictionScale == 'object':
        networkOutputDimension = numberOfObjectClasses*trainingConfig.numberOfQuantisationBins
    else:
        raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))

    trainingData, trainingDataIterator = getDataBatch(trainingDataloader, trainingDataIterator )
    trainingScenes = trainingData['initialVoxelScene'].to(dtype = torch.float, device = device).requires_grad_(True)
    currentBatchSize = trainingScenes.size(0)
    trainingClasses = trainingData['classes'].to(dtype = torch.float, device = device)
    if trainingConfig.estimateMissingObjects:
        presenceMask = torch.ones(trainingClasses.numel(), dtype = torch.bool, device = device)
    else:
        presenceMask = trainingClasses.round().reshape(-1) == 1
    if trainingConfig.inputNoise > 0:
        trainingScenes = trainingScenes + trainingConfig.inputNoise*torch.randn_like(trainingScenes)
    if trainingConfig.predictionQuantity == 'finalDisplacement':
        quantity = trainingData['finalMeanDisplacement']
    elif trainingConfig.predictionQuantity == 'maximumDisplacement':
        quantity, _ = trainingData['meanDisplacementFromStart'].max(dim = 2)
    else:
        raise ValueError('Unknown prediction quantity: {}'.format(trainingConfig.predictionQuantity))
    quantity = processQuantity(quantity, quantisationBinCentres, quantisationBinSpacing, trainingConfig).to(dtype = torch.float, device = device)
    if classWeights is None:
        if trainingConfig.predictionScale == 'object':
            predictionWeights = torch.ones(currentBatchSize, numberOfObjectClasses, dtype = torch.float, device = device)
        elif trainingConfig.predictionScale == 'scene':
            predictionWeights = torch.ones(currentBatchSize, dtype = torch.float, device = device)
        else:
            raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
    else:
        if trainingConfig.lossType == 'crossEntropy':
            predictionClasses = quantity.to(dtype = torch.long)
        else:
            _, predictionClasses = quantity.max(dim = -1)
        if trainingConfig.predictionScale == 'object':
            predictionWeights = torch.zeros(currentBatchSize, numberOfObjectClasses, dtype = torch.float, device = device)
            for dataIndex in range(currentBatchSize):
                for classIndex in range(numberOfObjectClasses):
                    if trainingConfig.estimateMissingObjects or trainingClasses[dataIndex, classIndex] == 1:
                        predictionWeights[dataIndex, classIndex] = classWeights[classIndex, predictionClasses[dataIndex, classIndex]]
        elif trainingConfig.predictionScale == 'scene':
            predictionWeights = classWeights[predictionClasses]
        else:
            raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
        # predictionWeights.unsqueeze(dim = -1)
    
    stabilityEstimate, _ = network(trainingScenes, returnIntermediateRepresentations = False)
    if trainingConfig.predictionScale == 'object' and trainingConfig.predictionType == 'quantised':
        stabilityEstimate = stabilityEstimate.reshape(currentBatchSize*numberOfObjectClasses, trainingConfig.numberOfQuantisationBins)
    if trainingConfig.lossType == 'crossEntropy':
        quantity = quantity.reshape(-1).round().to(torch.long)
        predictionWeights = predictionWeights.reshape(-1)
        if trainingConfig.predictionScale == 'object':
            distributionLoss = predictionWeights[presenceMask]*torch.nn.functional.cross_entropy(stabilityEstimate[presenceMask,:], quantity[presenceMask], weight=None, reduction='none')
        elif trainingConfig.predictionScale == 'scene':
            distributionLoss = predictionWeights*torch.nn.functional.cross_entropy(stabilityEstimate, quantity, weight=None, reduction='none')
        else:
            raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
    elif trainingConfig.lossType == 'klDivergence':
        if trainingConfig.predictionScale == 'object' and trainingConfig.predictionType == 'quantised':
            quantity = quantity.reshape(currentBatchSize*numberOfObjectClasses, trainingConfig.numberOfQuantisationBins)
        logQuantity = (quantity + 1e-7).log()
        predictionWeights = predictionWeights.reshape(-1)
        if trainingConfig.predictionScale == 'object':
            distributionLoss = predictionWeights[presenceMask]*klDivergence(logQuantity[presenceMask,:], (stabilityEstimate[presenceMask,:].softmax(dim = 1) + 1e-7).log())
        elif trainingConfig.predictionScale == 'scene':
            distributionLoss = predictionWeights*klDivergence(logQuantity, (stabilityEstimate.softmax(dim = 1) + 1e-7).log())
        else:
            raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
    elif trainingConfig.lossType == 'jsDivergence':
        if trainingConfig.predictionScale == 'object' and trainingConfig.predictionType == 'quantised':
            quantity = quantity.reshape(currentBatchSize*numberOfObjectClasses, trainingConfig.numberOfQuantisationBins)
        logQuantity = (quantity + 1e-7).log()
        predictionWeights = predictionWeights.reshape(-1)
        if trainingConfig.predictionScale == 'object':
            distributionLoss = predictionWeights[presenceMask]*jsDivergence(logQuantity[presenceMask,:], (stabilityEstimate[presenceMask,:].softmax(dim = 1) + 1e-7).log())
        elif trainingConfig.predictionScale == 'scene':
            distributionLoss = predictionWeights*jsDivergence(logQuantity, (stabilityEstimate.softmax(dim = 1) + 1e-7).log())
        else:
            raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
    elif trainingConfig.lossType == 'leastSquares':
        distributionLoss = (predictionWeights.unsqueeze(dim = 0)*(quantity - stabilityEstimate.exp())).pow(2).mean()
    else:
        raise ValueError('Unknown loss type: {}'.format(trainingConfig.lossType))
    distributionLoss = distributionLoss.mean()
    totalLoss = distributionLoss

    if trainingConfig.useGradientPenalty:
        gradientPenaltyData, gradientPenaltyDataIterator = getDataBatch(trainingDataloader, gradientPenaltyDataIterator )
        gradientPenaltyScenes = gradientPenaltyData['initialVoxelScene'].to(dtype = torch.float, device = device).requires_grad_(True)
        if trainingConfig.inputNoise > 0:
            gradientPenaltyScenes = gradientPenaltyScenes + trainingConfig.inputNoise*torch.randn_like(gradientPenaltyScenes)
        gradientPenaltyLoss = calculateInterpolationGradientPenalty(trainingScenes, gradientPenaltyScenes, network, penaltyType = 'equality').mean()
        totalLoss += 10*gradientPenaltyLoss
    else:
        gradientPenaltyLoss = torch.as_tensor(0)

    networkOptimiser.zero_grad()
    totalLoss.backward()
    networkOptimiser.step()

    output = dict(  distributionLoss = distributionLoss.item(),
                    gradientPenaltyLoss = gradientPenaltyLoss.item(),
                    totalLoss = totalLoss.item(),
                    stabilityEstimate = stabilityEstimate.detach().cpu())

    return output, trainingDataIterator, gradientPenaltyDataIterator

def evaluateOnDataset(  network, dataloader, numberOfObjectClasses, quantisationBinCentres, quantisationBinSpacing, 
                        device, useEvalMode, networkConfig, dataConfig, trainingConfig):
    if useEvalMode:
        network.eval()              
    if trainingConfig.predictionScale == 'scene':
        networkOutputDimension = trainingConfig.numberOfQuantisationBins
    elif trainingConfig.predictionScale == 'object':
        networkOutputDimension = numberOfObjectClasses*trainingConfig.numberOfQuantisationBins
    else:
        raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))

    distributionLosses = list()
    if trainingConfig.predictionType == 'quantised':
        truePositives = torch.zeros(trainingConfig.numberOfQuantisationBins)
        falsePositives = torch.zeros(trainingConfig.numberOfQuantisationBins)
        trueNegatives = torch.zeros(trainingConfig.numberOfQuantisationBins)
        falseNegatives = torch.zeros(trainingConfig.numberOfQuantisationBins)
        confusionMatrix = torch.zeros(trainingConfig.numberOfQuantisationBins, trainingConfig.numberOfQuantisationBins)
    else:
        truePositives = None
        falsePositives = None
        trueNegatives = None
        falseNegatives = None
        confusionMatrix = None
    with torch.no_grad():
        for data in dataloader:
            scenes = data['initialVoxelScene'].to(dtype = torch.float, device = device).requires_grad_(False)
            currentBatchSize = scenes.size(0)
            classes = data['classes'].to(dtype = torch.float, device = device).requires_grad_(False)
            if trainingConfig.estimateMissingObjects:
                presenceMask = torch.ones(classes.numel(), dtype = torch.bool, device = device)
            else:
                presenceMask = classes.round().reshape(-1) == 1
            if trainingConfig.predictionQuantity == 'finalDisplacement':
                quantity = data['finalMeanDisplacement']
            elif trainingConfig.predictionQuantity == 'maximumDisplacement':
                quantity, _ = data['meanDisplacementFromStart'].max(dim = 2)
            else:
                raise ValueError('Unknown prediction quantity: {}'.format(trainingConfig.predictionQuantity))
            quantity = processQuantity(quantity, quantisationBinCentres, quantisationBinSpacing, trainingConfig).to(dtype = torch.float, device = device)
            stabilityEstimate, _ = network(scenes, returnIntermediateRepresentations = False)
            if trainingConfig.predictionScale == 'object' and trainingConfig.predictionType == 'quantised':
                stabilityEstimate = stabilityEstimate.reshape(currentBatchSize*numberOfObjectClasses, trainingConfig.numberOfQuantisationBins)
            if trainingConfig.lossType == 'crossEntropy':
                quantity = quantity.reshape(-1).round().to(torch.long)
                _, predictedClasses = stabilityEstimate.max(dim = 1)
                if trainingConfig.predictionScale == 'object':
                    distributionLoss = torch.nn.functional.cross_entropy(stabilityEstimate[presenceMask,:], quantity[presenceMask], weight=None, reduction='none')
                    for predictedClass, targetClass in zip(predictedClasses[presenceMask],quantity[presenceMask]):
                        confusionMatrix[targetClass, predictedClass] += 1
                        for binIndex in range(trainingConfig.numberOfQuantisationBins):
                            if binIndex == targetClass and binIndex == predictedClass:
                                truePositives[binIndex] += 1
                            elif binIndex == targetClass and binIndex != predictedClass:
                                falseNegatives[binIndex] += 1
                            elif binIndex != targetClass and binIndex == predictedClass:
                                falsePositives[binIndex] += 1
                            elif binIndex != targetClass and binIndex != predictedClass:
                                trueNegatives[binIndex] += 1

                elif trainingConfig.predictionScale == 'scene':
                    distributionLoss = torch.nn.functional.cross_entropy(stabilityEstimate, quantity, weight=None, reduction='none')
                    for predictedClass, targetClass in zip(predictedClasses,quantity):
                        confusionMatrix[targetClass, predictedClass] += 1
                        for binIndex in range(trainingConfig.numberOfQuantisationBins):
                            if binIndex == targetClass and binIndex == predictedClass:
                                truePositives[binIndex] += 1
                            elif binIndex == targetClass and binIndex != predictedClass:
                                falseNegatives[binIndex] += 1
                            elif binIndex != targetClass and binIndex == predictedClass:
                                falsePositives[binIndex] += 1
                            elif binIndex != targetClass and binIndex != predictedClass:
                                trueNegatives[binIndex] += 1
                else:
                    raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
            elif trainingConfig.lossType == 'klDivergence':
                if trainingConfig.predictionScale == 'object' and trainingConfig.predictionType == 'quantised':
                    quantity = quantity.reshape(currentBatchSize*numberOfObjectClasses, trainingConfig.numberOfQuantisationBins)
                _, targetClasses = quantity.max(dim = 1)
                _, predictedClasses = stabilityEstimate.max(dim = 1)
                logQuantity = (quantity + 1e-7).log()
                if trainingConfig.predictionScale == 'object':
                    distributionLoss = klDivergence(logQuantity[presenceMask,:], (stabilityEstimate[presenceMask,:].softmax(dim = 1) + 1e-7).log())
                    for predictedClass, targetClass in zip(predictedClasses[presenceMask],targetClasses[presenceMask]):
                        confusionMatrix[targetClass, predictedClass] += 1
                        for binIndex in range(trainingConfig.numberOfQuantisationBins):
                            if binIndex == targetClass and binIndex == predictedClass:
                                truePositives[binIndex] += 1
                            elif binIndex == targetClass and binIndex != predictedClass:
                                falseNegatives[binIndex] += 1
                            elif binIndex != targetClass and binIndex == predictedClass:
                                falsePositives[binIndex] += 1
                            elif binIndex != targetClass and binIndex != predictedClass:
                                trueNegatives[binIndex] += 1
                elif trainingConfig.predictionScale == 'scene':
                    distributionLoss = klDivergence(logQuantity, (stabilityEstimate.softmax(dim = 1) + 1e-7).log())
                    for predictedClass, targetClass in zip(predictedClasses,targetClasses):
                        confusionMatrix[targetClass, predictedClass] += 1
                        for binIndex in range(trainingConfig.numberOfQuantisationBins):
                            if binIndex == targetClass and binIndex == predictedClass:
                                truePositives[binIndex] += 1
                            elif binIndex == targetClass and binIndex != predictedClass:
                                falseNegatives[binIndex] += 1
                            elif binIndex != targetClass and binIndex == predictedClass:
                                falsePositives[binIndex] += 1
                            elif binIndex != targetClass and binIndex != predictedClass:
                                trueNegatives[binIndex] += 1
                else:
                    raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
            elif trainingConfig.lossType == 'jsDivergence':
                if trainingConfig.predictionScale == 'object' and trainingConfig.predictionType == 'quantised':
                    quantity = quantity.reshape(currentBatchSize*numberOfObjectClasses, trainingConfig.numberOfQuantisationBins)
                _, targetClasses = quantity.max(dim = 1)
                _, predictedClasses = stabilityEstimate.max(dim = 1)
                logQuantity = (quantity + 1e-7).log()
                if trainingConfig.predictionScale == 'object':
                    distributionLoss = jsDivergence(logQuantity[presenceMask,:], (stabilityEstimate[presenceMask,:].softmax(dim = 1) + 1e-7).log())
                    for predictedClass, targetClass in zip(predictedClasses[presenceMask],targetClasses[presenceMask]):
                        confusionMatrix[targetClass, predictedClass] += 1
                        for binIndex in range(trainingConfig.numberOfQuantisationBins):
                            if binIndex == targetClass and binIndex == predictedClass:
                                truePositives[binIndex] += 1
                            elif binIndex == targetClass and binIndex != predictedClass:
                                falseNegatives[binIndex] += 1
                            elif binIndex != targetClass and binIndex == predictedClass:
                                falsePositives[binIndex] += 1
                            elif binIndex != targetClass and binIndex != predictedClass:
                                trueNegatives[binIndex] += 1
                elif trainingConfig.predictionScale == 'scene':
                    distributionLoss = jsDivergence(logQuantity, (stabilityEstimate.softmax(dim = 1) + 1e-7).log())
                    for predictedClass, targetClass in zip(predictedClasses,targetClasses):
                        confusionMatrix[targetClass, predictedClass] += 1
                        for binIndex in range(trainingConfig.numberOfQuantisationBins):
                            if binIndex == targetClass and binIndex == predictedClass:
                                truePositives[binIndex] += 1
                            elif binIndex == targetClass and binIndex != predictedClass:
                                falseNegatives[binIndex] += 1
                            elif binIndex != targetClass and binIndex == predictedClass:
                                falsePositives[binIndex] += 1
                            elif binIndex != targetClass and binIndex != predictedClass:
                                trueNegatives[binIndex] += 1
                else:
                    raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
            elif trainingConfig.lossType == 'leastSquares':
                distributionLoss = (quantity - stabilityEstimate.exp()).pow(2).mean()
            else:
                raise ValueError('Unknown loss type: {}'.format(trainingConfig.lossType))
            for lossValue in distributionLoss:
                distributionLosses.append(lossValue.item())
            # for lossValue, stabilityValue, quantityValue in zip(distributionLoss, stabilityEstimate, quantity):
            #     distributionLosses.append(lossValue.item())
            #     if trainingConfig.predictionScale == 'object':
            #         if trainingConfig.predictionType == 'quantised':
            #             _, predictedClasses = stabilityValue.max(dim = 0)
            #             if trainingConfig.lossType == 'crossEntropy':
                            
            #             else:
            #                 _, targetClasses = quantity.max(dim = 0)
            #             accuracy.append(((predictedClasses == targetClasses).sum()/predictedClasses.numel()).item())
            #         elif trainingConfig.predictionType == 'scalar':
            #             pass
            #     elif trainingConfig.predictionScale == 'scene':
            #         if trainingConfig.predictionType == 'quantised':
            #             _, predictedClass = stabilityValue.max(dim = 0)
            #             _, targetClass = quantity.max(dim = 0)
            #             accuracy.append((predictedClass == targetClass).item())
            #         elif trainingConfig.predictionType == 'scalar':
            #             pass

    output = dict(  distributionLoss = torch.as_tensor(distributionLosses),
                    truePositives = truePositives,
                    falsePositives = falsePositives,
                    trueNegatives = trueNegatives,
                    falseNegatives = falseNegatives,
                    confusionMatrix = confusionMatrix)

    return output

if __name__ == '__main__':  
    config = configuration.getDefaultConfiguration()
    config.merge_from_file('stabilityEstimationDefaults.yaml')
    if len(sys.argv) > 1:
        configurationFileName = sys.argv[1]
    else:
        configurationFileName = 'localStabilityEstimationConfig.yaml'
    config.merge_from_file(configurationFileName)
    config.freeze()
    systemConfig = config.SYSTEM
    dataConfig = config.DATA
    networkConfig = config.NETWORK
    trainingConfig = config.TRAINING

    device = torch.device('cuda')
    if systemConfig.disableCuDNN:
        torch.backends.cudnn.enabled = False

    print('Setting up configuration')
    sceneSizes = list(range(1,dataConfig.maximumSceneSize+1))
    trainingFirstSceneNumbers = [dataConfig.trainingFirstSceneNumber for x in sceneSizes]
    trainingLastSceneNumbers = [dataConfig.trainingLastSceneNumber for x in sceneSizes]

    validationFirstSceneNumbers = [dataConfig.validationFirstSceneNumber for x in sceneSizes]
    validationLastSceneNumbers = [dataConfig.validationFirstSceneNumber for x in sceneSizes]

    classLabels = list(range(15))
    numberOfObjectClasses = 14
    voxelSize = 0.005
    voxelGridBounds = [[-0.24,0.24],[-0.24,0.24],[-0.02,0.30]]
    voxelSpatialDimensions = [int(round((y-x)/voxelSize)) for (x,y) in voxelGridBounds]
    voxelBinnedSpatialDimensions = [int(round(x/math.pow(2, dataConfig.voxelBinningPower))) for x in voxelSpatialDimensions]
    # voxelSpatialDimensions = [96,96,64]
    # voxelSpatialDimensions = [int(round(x/math.pow(2,dataConfig.voxelBinningPower))) for x in voxelSpatialDimensions]
    xGridCoordinates, yGridCoordinates, zGridCoordinates = numpy.indices([x + 1 for x in voxelSpatialDimensions])
    xGridCoordinates = xGridCoordinates.astype(numpy.float32)*voxelSize + voxelGridBounds[0][0]
    yGridCoordinates = yGridCoordinates.astype(numpy.float32)*voxelSize + voxelGridBounds[1][0]
    zGridCoordinates = zGridCoordinates.astype(numpy.float32)*voxelSize + voxelGridBounds[2][0]
    xBinnedGridCoordinates, yBinnedGridCoordinates, zBinnedGridCoordinates = numpy.indices([x + 1 for x in voxelBinnedSpatialDimensions])
    xBinnedGridCoordinates = xBinnedGridCoordinates.astype(numpy.float32)*voxelSize*math.pow(2, dataConfig.voxelBinningPower) + voxelGridBounds[0][0]
    yBinnedGridCoordinates = yBinnedGridCoordinates.astype(numpy.float32)*voxelSize*math.pow(2, dataConfig.voxelBinningPower) + voxelGridBounds[1][0]
    zBinnedGridCoordinates = zBinnedGridCoordinates.astype(numpy.float32)*voxelSize*math.pow(2, dataConfig.voxelBinningPower) + voxelGridBounds[2][0]

    preprocessedDataType = 'numpy'
    # preprocessedDataType = 'pytorch_gzip'

    preprocessedFileTemplate = 'Data{}Heap{}{}'
    if preprocessedDataType == 'numpy':
        preprocessedFileTemplate += '.npz'
    elif preprocessedDataType == 'pytorch_gzip':
        preprocessedFileTemplate += '.pytg'
    if systemConfig.type == 'bear':
        imageDataDirectory = '/rds/projects/2018/leonarda-muri/hector/data/dataFiles_d435_single_instance'
        voxelDataDirectory = '/rds/projects/2018/leonarda-muri/hector/data/dataFiles_d435_single_instance_stability'
        preprocessedDataDirectory = '/rds/projects/2018/leonarda-muri/hector/data/dataFiles_d435_single_instance_preprocessed_{}'.format(preprocessedDataType)
        preprocessedStabilityDataDirectory = '/rds/projects/2018/leonarda-muri/hector/data/dataFiles_d435_single_instance_stability_preprocessed_{}'.format(preprocessedDataType)
    elif systemConfig.type == 'desktop':
        imageDataDirectory = 'F:/data/dataFiles_d435_single_instance'
        voxelDataDirectory = 'F:/data/dataFiles_d435_single_instance_stability'
        preprocessedDataDirectory = 'F:/data/dataFiles_d435_single_instance_preprocessed_{}'.format(preprocessedDataType)
        preprocessedStabilityDataDirectory = 'F:/data/dataFiles_d435_single_instance_stability_preprocessed_{}'.format(preprocessedDataType)
    elif systemConfig.type == 'hpc':
        imageDataDirectory = '/home/research/users/basevhra/data/dataFiles_d435_single_instance'
        voxelDataDirectory = '/home/research/users/basevhra/data/dataFiles_d435_single_instance_stability'
        preprocessedDataDirectory = '/home/research/users/basevhra/data/dataFiles_d435_single_instance_preprocessed_{}'.format(preprocessedDataType)
        preprocessedStabilityDataDirectory = '/home/research/users/basevhra/data/dataFiles_d435_single_instance_stability_preprocessed_{}'.format(preprocessedDataType)
    else:
        raise ValueError('Unknown system type: {}'.format(config.type))

    enforceBinaryCompleteScenes = True

    sceneStatisticsMeansFileName = 'trainingSceneStatisticsMeans_1-{}_b{}.mat'.format(dataConfig.maximumSceneSize, dataConfig.voxelBinningPower)
    sceneStatisticsStandardDeviationsFileName = 'trainingSceneStatisticsStandardDeviations_1-{}_b{}.mat'.format(dataConfig.maximumSceneSize, dataConfig.voxelBinningPower)
    if os.path.exists(sceneStatisticsMeansFileName) and os.path.exists(sceneStatisticsStandardDeviationsFileName):
        print('Loading data statistics')
        statisticsMeansData = hdf5storage.loadmat(sceneStatisticsMeansFileName)
        statisticsMeansData = {key:torch.as_tensor(value) for (key, value) in statisticsMeansData.items()}
        statisticsStandardDeviationsData = hdf5storage.loadmat(sceneStatisticsStandardDeviationsFileName)
        statisticsStandardDeviationsData = {key:torch.as_tensor(value) for (key, value) in statisticsStandardDeviationsData.items()}
    else:
        raise ValueError('Unable to find precomputed scene statistics.')

    currentStatisticsMeans = dict(
                                    initialVoxelScene = statisticsMeansData['completeVoxelScene'][:,None,None,None].to(dtype = torch.float),
                                    finalVoxelScene = statisticsMeansData['completeVoxelScene'][:,None,None,None].to(dtype = torch.float),
                                )
    currentStatisticsStandardDeviations = dict(
                                    initialVoxelScene = statisticsStandardDeviationsData['completeVoxelScene'][:,None,None,None].to(dtype = torch.float),
                                    finalVoxelScene = statisticsStandardDeviationsData['completeVoxelScene'][:,None,None,None].to(dtype = torch.float),
                                )

    logRootDirectory = 'logs'
    if not os.path.exists(logRootDirectory):
        try:
            os.mkdir(logRootDirectory)
        except:
            if not os.path.exists(logRootDirectory):
                raise IOError('Unable to create logging root directory: {}.'.format(logRootDirectory))

    logDirectory = '{}/{}'.format(logRootDirectory, config.name)
    if not os.path.exists(logDirectory):
        try:
            os.mkdir(logDirectory)
        except:
            if not os.path.exists(logDirectory):
                raise IOError('Unable to create logging directory: {}.'.format(logDirectory))
    logger = SummaryWriter(logDirectory, flush_secs = 10)
    shutil.copy2('stabilityEstimationTraining.py','{}/stabilityEstimationTraining.py'.format(logDirectory))
    shutil.copy2('networkArchitectures.py','{}/networkArchitectures.py'.format(logDirectory))
    shutil.copy2('dataSets.py','{}/dataSets.py'.format(logDirectory))
    configuration.saveConfiguration(config, '{}/config.yaml'.format(logDirectory))

    trainingSet = dataSets.PreprocessedStabilityDataset(sceneSizes, trainingFirstSceneNumbers, trainingLastSceneNumbers, preprocessedStabilityDataDirectory, preprocessedFileTemplate, 
                    includeOriginalScenes = True, includePerturbedScenes = True, includeScenesWithMissingObjects = True,
                    voxelBinningPower = dataConfig.voxelBinningPower, dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations,
                    enforceBinaryScenes = enforceBinaryCompleteScenes)
    trainingDataloader = DataLoader(trainingSet, batch_size = trainingConfig.batchSize, shuffle = True, num_workers = systemConfig.numberOfWorkers, drop_last = True, pin_memory = True)
    trainingDataIterator = iter(trainingDataloader)
    print('There are {} training scenes'.format(len(trainingSet)))

    gradientPenaltyDataloader = DataLoader(trainingSet, batch_size = trainingConfig.batchSize, shuffle = True, num_workers = systemConfig.numberOfWorkers, drop_last = True, pin_memory = True)
    gradientPenaltyDataIterator = iter(gradientPenaltyDataloader)

    validationSet = dataSets.PreprocessedStabilityDataset(sceneSizes, validationFirstSceneNumbers, validationLastSceneNumbers, preprocessedStabilityDataDirectory, preprocessedFileTemplate, 
                    includeOriginalScenes = True, includePerturbedScenes = True, includeScenesWithMissingObjects = True,
                    voxelBinningPower = dataConfig.voxelBinningPower, dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations,
                    enforceBinaryScenes = enforceBinaryCompleteScenes)
    validationDataloader = DataLoader(validationSet, batch_size = trainingConfig.batchSize, shuffle = False, num_workers = systemConfig.numberOfWorkers, drop_last = False, pin_memory = True)
    validationDataIterator = iter(validationDataloader)
    print('There are {} validation scenes'.format(len(validationSet)))

    if trainingConfig.predictionType == 'quantised':
        if trainingConfig.quantisationBinScaling == 'exponential':
            quantisationBinSpacing = (math.log10(trainingConfig.highestQuantisationBinCentre/trainingConfig.lowestQuantisationBinCentre))/(trainingConfig.numberOfQuantisationBins-1)
            quantisationBinCentres = torch.arange(  math.log10(trainingConfig.lowestQuantisationBinCentre),math.log10(trainingConfig.highestQuantisationBinCentre)+1e-7,
                                                quantisationBinSpacing)                                         
        elif trainingConfig.quantisationBinScaling == 'linear':
            quantisationBinSpacing = (trainingConfig.highestQuantisationBinCentre-trainingConfig.lowestQuantisationBinCentre)/(trainingConfig.numberOfQuantisationBins-1)
            quantisationBinCentres = torch.arange(  trainingConfig.lowestQuantisationBinCentre,trainingConfig.highestQuantisationBinCentre+1e-7,
                                                quantisationBinSpacing)  
        if trainingConfig.predictionScale == 'scene':
            networkOutputDimension = trainingConfig.numberOfQuantisationBins
        elif trainingConfig.predictionScale == 'object':
            networkOutputDimension = numberOfObjectClasses*trainingConfig.numberOfQuantisationBins
        else:
            raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
    elif trainingConfig.predictionType == 'scalar':
        if trainingConfig.predictionScale == 'scene':
            networkOutputDimension = 1
        elif trainingConfig.predictionScale == 'object':
            networkOutputDimension = numberOfObjectClasses
        else:
            raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
    else:
        raise ValueError('Unknown prediction type: {}'.format(trainingConfig.predictionType))

    # Generate class weightings where appropriate
    if trainingConfig.predictionType == 'quantised':
        if trainingConfig.predictionScale == 'scene':
            classCounts = torch.zeros(trainingConfig.numberOfQuantisationBins, dtype = torch.int64)
        elif trainingConfig.predictionScale == 'object':
            classCounts = torch.zeros(numberOfObjectClasses, trainingConfig.numberOfQuantisationBins, dtype = torch.int64)
        else:
            raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
        for trainingData in trainingDataloader:
            trainingClasses = trainingData['classes'].to(dtype = torch.float, device = device)
            if trainingConfig.predictionQuantity == 'finalDisplacement':
                quantity = trainingData['finalMeanDisplacement']
            elif trainingConfig.predictionQuantity == 'maximumDisplacement':
                quantity, _ = trainingData['meanDisplacementFromStart'].max(dim = 2)
            else:
                raise ValueError('Unknown prediction quantity: {}'.format(trainingConfig.predictionQuantity))
            quantity = processQuantity(quantity, quantisationBinCentres, quantisationBinSpacing, trainingConfig).to(dtype = torch.float, device = device)
            if trainingConfig.lossType == 'crossEntropy':
                predictionClasses = quantity.round().to(dtype = torch.long)
            else:
                _, predictionClasses = quantity.max(dim = -1)
            for sceneIndex in range(predictionClasses.size(0)):
                if trainingConfig.predictionScale == 'scene':
                    classCounts[predictionClasses[sceneIndex]] += 1
                elif trainingConfig.predictionScale == 'object':
                    for objectIndex in range(predictionClasses.size(1)):
                        if trainingConfig.estimateMissingObjects:
                            classCounts[objectIndex, predictionClasses[sceneIndex, objectIndex]] += 1
                        else:
                            if trainingClasses[sceneIndex, objectIndex]:
                                classCounts[objectIndex, predictionClasses[sceneIndex, objectIndex]] += 1
                else:
                    raise ValueError('Unknown prediction scale: {}'.format(trainingConfig.predictionScale))
        classProbabilities = classCounts.to(dtype = torch.float64)/classCounts.to(dtype = torch.float64).sum(dim = -1, keepdim = True)   
        classProbabilities = classProbabilities.to(dtype = torch.float, device = device)
        classWeights = 1 - classProbabilities + 1e-7
        weightingFilePath = '{}/{}'.format(logDirectory, 'class_weighting.npz')
        numpy.savez_compressed(weightingFilePath, classProbabilities = classProbabilities.detach().cpu().numpy(), classWeights = classWeights.detach().cpu().numpy())
    elif trainingConfig.predictionType == 'scalar':
        classCounts = None
        classProbabilities = None
        classWeights = None

    print('Constructing network')
    numberOfSpatialScales = int(math.floor(min([math.log2(x) for x in voxelBinnedSpatialDimensions]))-1)
    completeSceneVoxelChannels = 15
    networkDownscaling = numberOfSpatialScales
    networkUseTrigonometricCoordinateEmbedding = True
    if networkConfig.useSpectralNorm:
        networkSpectralNormalisationSettings = dict(useSpectralNormalisation = True, numberOfPowerIterations = 5)
    else:
        networkSpectralNormalisationSettings = dict(useSpectralNormalisation = False)
    network = networkArchitectures.Encoder(inputDimensions = [completeSceneVoxelChannels]+voxelBinnedSpatialDimensions, 
        initialProjectionDimension = networkConfig.initialProjectionDimension, 
        numberOfDownscalingOperations = networkDownscaling, representationDimension = networkOutputDimension, 
        numberOfIntermediateComputationLayers = networkConfig.numberOfIntermediateComputationLayers, kernelSize = networkConfig.kernelSize,
        addCoordinatesToRepresentation = networkConfig.injectCoordinateInput,
        useTrigonometricCoordinateEmbedding = networkUseTrigonometricCoordinateEmbedding,
        residualMode = networkConfig.residualMode, useDepthWiseSeparableConvolutions = networkConfig.useDSConvolutions,
        spectralNormalisationSettings = networkSpectralNormalisationSettings, useBatchNorm = False, useBiasThroughout = networkConfig.useBiasThroughout,
        downscalingChannelScaling = networkConfig.downscalingChannelScaling)
    network = network.to(device = device)
    
    networkOptimiser = torch.optim.Adam(network.parameters(), lr = trainingConfig.learningRate)
    
    # torch.backends.cudnn.deterministic = True
    if systemConfig.cudnnBenchmarking:
        torch.backends.cudnn.benchmark = True
    
    validationInterval = 5000

    iterationToProfile = 2
    print('Training')
    iterationStartTime = time.time()
    for iterationNumber in range(trainingConfig.numberOfIterations):
        torch.cuda.reset_max_memory_allocated()
        if iterationNumber == iterationToProfile:
            print('Profiling iteration {}'.format(iterationNumber))
            profile = torch.autograd.profiler.profile(enabled = True, use_cuda = True, record_shapes = False)
        else:
            profile = torch.autograd.profiler.profile(enabled = False, use_cuda = False, record_shapes = False)

        with profile:
            output, trainingDataIterator, gradientPenaltyDataIterator = \
                trainNetwork(   network, trainingDataloader, trainingDataIterator, gradientPenaltyDataIterator, 
                        numberOfObjectClasses, quantisationBinCentres, quantisationBinSpacing, classWeights, device, networkOptimiser, networkConfig, dataConfig, trainingConfig)
            
        if iterationNumber == iterationToProfile:
            tracePath = '{}/iteration_{}.tracing'.format(logDirectory, iterationToProfile)
            profile.export_chrome_trace(tracePath)

        logger.add_scalar(tag = 'training/distributionLoss', scalar_value = output['distributionLoss'], global_step = iterationNumber)
        logger.add_scalar(tag = 'training/gradientPenaltyLoss', scalar_value = output['gradientPenaltyLoss'], global_step = iterationNumber)
        logger.add_scalar(tag = 'training/totalLoss', scalar_value = output['totalLoss'], global_step = iterationNumber)
        logger.add_scalar('training/memoryUsage', scalar_value = torch.cuda.max_memory_allocated(), global_step = iterationNumber)

        if (validationInterval > 0) and (iterationNumber % validationInterval == 0):
            networkStateFileName = 'stabilityEstimationNetwork_state_iteration_{}.pts'.format(iterationNumber)
            networkStateFilePath = '{}/{}'.format(logDirectory, networkStateFileName)
            print('Saving network state')
            torch.save(network.state_dict(), networkStateFilePath)
            print('Evaluating validation dataset')
            validationResults = evaluateOnDataset(  network, dataloader = validationDataloader, numberOfObjectClasses = numberOfObjectClasses, quantisationBinCentres = quantisationBinCentres, 
                                                    quantisationBinSpacing = quantisationBinSpacing, device = device, useEvalMode = True, 
                                                    networkConfig = networkConfig, dataConfig = dataConfig, trainingConfig = trainingConfig)
            logger.add_scalar(tag = 'validation/distributionLoss', scalar_value = validationResults['distributionLoss'].mean().item(), global_step = iterationNumber)
            truePositives = validationResults['truePositives']
            falsePositives = validationResults['falsePositives']
            trueNegatives = validationResults['trueNegatives']
            falseNegatives = validationResults['falseNegatives']
            confusionMatrix = validationResults['confusionMatrix']
            validationResultsFileName = 'stabilityEstimationNetwork_validation_iteration_{}.npz'.format(iterationNumber)
            validationResultsFilePath = '{}/{}'.format(logDirectory, validationResultsFileName)
            numpy.savez_compressed(validationResultsFilePath, **{key:value.numpy() if isinstance(value, torch.Tensor) else value for (key, value) in validationResults.items()})
            normalisedConfusionMatrix = confusionMatrix/(confusionMatrix.sum(dim = 1, keepdim = True)+1e-7)
            if trainingConfig.quantisationBinScaling == 'exponential':
                binLabels = ['{:.1e}'.format(10**x) for x in quantisationBinCentres]
            elif trainingConfig.quantisationBinScaling == 'linear':
                binLabels = ['{:.1e}'.format(x) for x in quantisationBinCentres]
            logger.add_scalar(tag = 'validation/accuracy', scalar_value = (truePositives.sum()/(truePositives+falseNegatives).sum()).item(), global_step = iterationNumber)

            figure = plt.figure()
            plt.bar(binLabels,(truePositives/(truePositives+falseNegatives+1e-7)).numpy())
            plt.xlabel("Class centre")
            plt.ylabel("Recall")
            logger.add_figure(tag = 'graphs/recall', figure = figure, global_step = iterationNumber, close = True)
            plt.close(figure)

            figure = plt.figure()
            plt.bar(binLabels,(truePositives/(truePositives+falsePositives+1e-7)).numpy())
            plt.xlabel("Class centre")
            plt.ylabel("Precision")
            logger.add_figure(tag = 'graphs/precision', figure = figure, global_step = iterationNumber, close = True)
            plt.close(figure)

            figure = plt.figure()
            plt.bar(binLabels,confusionMatrix.sum(dim = 1).numpy())
            plt.xlabel("Class centre")
            plt.ylabel("Counts")
            logger.add_figure(tag = 'graphs/ground truth class distribution', figure = figure, global_step = iterationNumber, close = True)
            plt.close(figure)

            figure = plt.figure()
            plt.bar(binLabels,confusionMatrix.sum(dim = 0).numpy())
            plt.xlabel("Class centre")
            plt.ylabel("Counts")
            logger.add_figure(tag = 'graphs/predicted class distribution', figure = figure, global_step = iterationNumber, close = True)
            plt.close(figure)

            figure = plt.figure()
            axis = figure.add_subplot(111)
            colourAxis = axis.matshow(confusionMatrix.numpy(), interpolation='nearest')
            figure.colorbar(colourAxis)
            axis.set_xticklabels(['']+binLabels)
            axis.set_yticklabels(['']+binLabels)
            logger.add_figure(tag = 'graphs/confusionMatrix', figure = figure, global_step = iterationNumber, close = True)
            plt.close(figure)

            figure = plt.figure()
            axis = figure.add_subplot(111)
            colourAxis = axis.matshow(normalisedConfusionMatrix.numpy(), interpolation='nearest')
            figure.colorbar(colourAxis)
            axis.set_xticklabels(['']+binLabels)
            axis.set_yticklabels(['']+binLabels)
            logger.add_figure(tag = 'graphs/normalisedConfusionMatrix', figure = figure, global_step = iterationNumber, close = True)
            plt.close(figure)

            logger.add_histogram(tag = 'validation/distributionLossValues', values = validationResults['distributionLoss'], global_step = iterationNumber)
            # logger.add_histogram(tag = 'validation/accuracyValues', values = validationResults['accuracy'], global_step = iterationNumber)

        iterationEndTime = time.time()
        logger.add_scalar(tag = 'training/iterationTime', scalar_value = iterationEndTime - iterationStartTime, global_step = iterationNumber)
        iterationStartTime = iterationEndTime

