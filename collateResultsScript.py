import torch
import numpy
import hdf5storage
from scipy.linalg import logm as logm
import os
import sys
import math

def rotationDistance(r1, r2):
    #Metric 6 from Du Q. Huynh, Metrics for 3D Rotations: Comparison and Analysis (no gradients)
    return numpy.linalg.norm(logm(torch.matmul(r1, r2.transpose(0,1)).cpu().numpy(), disp = False)[0]).item()

def calculateSceneStabilityStatistics(classes, translations, rotations, pointClouds, timeStepSize = 1):
    instantaneousLinearVelocity = (translations[:,:,1:] - translations[:,:,:-1]).norm(dim = 1)
    instantaneousAngularVelocity = torch.zeros(rotations.size(0),rotations.size(3)-1)
    instantaneousMeanDisplacement = torch.zeros(translations.size(0),translations.size(2)-1)
    meanDisplacementFromStart = torch.zeros(translations.size(0),translations.size(2)-1)
    finalMeanDisplacement = torch.zeros(translations.size(0))
    for ii in classes.nonzero().squeeze(dim = 1):
        firstPoints = torch.matmul(pointClouds[ii],rotations[ii,:,:,0].transpose(0,1)) + translations[ii,:,0].unsqueeze(dim = 0)
        currentPoints = firstPoints
        for jj in range(rotations.size(3)-1):
            instantaneousAngularVelocity[ii,jj] = rotationDistance(rotations[ii,:,:,jj],rotations[ii,:,:,jj+1])
            newPoints = torch.matmul(pointClouds[ii],rotations[ii,:,:,jj+1].transpose(0,1)) + translations[ii,:,jj+1].unsqueeze(dim = 0)
            pointDisplacement = (newPoints - currentPoints).norm(dim = 1)
            instantaneousMeanDisplacement[ii,jj] = pointDisplacement.mean()
            pointDisplacement = (newPoints - firstPoints).norm(dim = 1)
            meanDisplacementFromStart[ii,jj] = pointDisplacement.mean()
            currentPoints = newPoints
        finalMeanDisplacement[ii] = (currentPoints - firstPoints).norm(dim = 1).mean()
    output = dict(  instantaneousLinearVelocity = instantaneousLinearVelocity/timeStepSize, 
                    instantaneousAngularVelocity = instantaneousAngularVelocity/timeStepSize, 
                    instantaneousMeanDisplacement = instantaneousMeanDisplacement/timeStepSize,
                    meanDisplacementFromStart = meanDisplacementFromStart,
                    finalMeanDisplacement = finalMeanDisplacement )
    return output

def generateSceneResults(groundTruthData, resultData, simulationData, pointCloudData):
    groundTruthVisibility = torch.as_tensor(groundTruthData['visibility'])
    groundTruthClasses = torch.as_tensor(groundTruthData['classes'], dtype = torch.bool)
    predictedClasses = torch.zeros(14, dtype = torch.bool)
    translations = torch.zeros(14,3,simulationData['translations'].shape[2])
    rotations = torch.zeros(14,3,3,simulationData['rotations'].shape[3])
    oneClasses = torch.as_tensor(simulationData['classes'].squeeze(), dtype = torch.long)
    zeroClasses = oneClasses - 1
    for index, zeroClass in enumerate(zeroClasses):
        predictedClasses[zeroClass] = True
        translations[zeroClass,:,:] = torch.as_tensor(simulationData['translations'][index,:,:], dtype = torch.float)
        rotations[zeroClass,:,:] = torch.as_tensor(simulationData['rotations'][index,:,:,:], dtype = torch.float)
    stabilityStatistics = calculateSceneStabilityStatistics(predictedClasses, translations, rotations, pointClouds = pointCloudData, timeStepSize = 1/240)
    finalDisplacement = stabilityStatistics['finalMeanDisplacement']
    groundTruthDepthImage = torch.as_tensor(groundTruthData['depthImage'], dtype = torch.float).squeeze()
    groundTruthClassImage = torch.as_tensor(groundTruthData['classLabelImage'], dtype = torch.float).squeeze()
    groundTruthRGBImage = torch.as_tensor(groundTruthData['rgbImage'], dtype = torch.float).squeeze()
    predictedDepthImage = torch.as_tensor(resultData['depthImage'], dtype = torch.float).squeeze()
    predictedClassImage = torch.as_tensor(resultData['classImage'], dtype = torch.float).squeeze()
    predictedClassImage[predictedClassImage < 0] = 0
    predictedRGBImage = torch.as_tensor(resultData['rgbImage'], dtype = torch.float).squeeze()
    depthMask = (groundTruthDepthImage < float('inf')) & (predictedDepthImage < float('inf'))
    depthDifference = predictedDepthImage - groundTruthDepthImage
    depthDifference[~depthMask] = 0
    meanDepthError = depthDifference[depthMask].abs().mean()
    maxDepthError = depthDifference[depthMask].abs().max()
    minDepthError = depthDifference[depthMask].abs().min()
    classTruePositives = torch.zeros(14)
    classFalsePositives = torch.zeros(14)
    classFalseNegatives = torch.zeros(14)
    for zeroClass, oneClass in zip(zeroClasses,oneClasses):
        classTruePositives[zeroClass] = ((groundTruthClassImage == oneClass) & (predictedClassImage == oneClass)).sum()
        classFalsePositives[zeroClass] = ((groundTruthClassImage != oneClass) & (predictedClassImage == oneClass)).sum()
        classFalseNegatives[zeroClass] = ((groundTruthClassImage == oneClass) & (predictedClassImage != oneClass)).sum()
    classPrecision = classTruePositives[zeroClasses]/(classTruePositives[zeroClasses]+classFalsePositives[zeroClasses] + 1e-7)
    classRecall = classTruePositives[zeroClasses]/(classTruePositives[zeroClasses]+classFalseNegatives[zeroClasses] + 1e-7)
    misclassificationRatio = (groundTruthClassImage != predictedClassImage).to(dtype = torch.float).sum()/groundTruthClassImage.numel()
    fractionalSceneSize = predictedClasses.to(dtype = torch.float).sum()/groundTruthClasses.to(dtype = torch.float).sum()
    fractionOfCorrectObjectsRecovered = (predictedClasses & groundTruthClasses).to(dtype = torch.float).sum()/groundTruthClasses.to(dtype = torch.float).sum()
    outputs = dict(
                    predictedClasses = zeroClasses,
                    groundTruthClassVisibility = groundTruthVisibility,
                    finalDisplacement = finalDisplacement[zeroClasses],
                    classPrecision = classPrecision,
                    classRecall = classRecall,
                    meanDepthError = meanDepthError,
                    maxDepthError = maxDepthError,
                    minDepthError = minDepthError,
                    misclassificationRatio = misclassificationRatio,
                    fractionalSceneSize = fractionalSceneSize,
                    fractionOfCorrectObjectsRecovered = fractionOfCorrectObjectsRecovered
                )
    return outputs

if __name__ == '__main__':
    networkTypes = ['Adversarial_no_stability', 'Adversarial_object_stability', 'Adversarial_scene_stability', 'Regression_no_stability', 'Regression_object_stability', 'Regression_scene_stability']
    sceneSetTypes = ['difficult', 'hidden']
    if len(sys.argv) > 1:
        machineType = sys.argv[1]
    else:
        machineType = 'desktop'
    if len(sys.argv) > 2:
        dataIndex = int(sys.argv[2])
        networkIndex = int(math.floor(dataIndex/len(sceneSetTypes)))
        setIndex = dataIndex % len(sceneSetTypes)
        if networkIndex >= len(networkTypes):
            raise ValueError('Data index out of range')
    else:
        networkIndex = 0
        setIndex = 0
    networkType = networkTypes[networkIndex]
    sceneSetType = sceneSetTypes[setIndex]

    pointCloudFileName = 'objectSurfacePointSamples.mat'
    pointCloudData = hdf5storage.loadmat(pointCloudFileName, variable_names = ['objectSurfacePointSamples'])['objectSurfacePointSamples']
    pointCloudData = [torch.as_tensor(x, dtype = torch.float) for x in pointCloudData.squeeze()]

    if machineType == 'desktop':
        resultDirectory = 'U:/hector/bear/code/sceneImagination_binned_results/results'
        dataDirectory = 'U:/hector/data/dataFiles_d435_single_instance_preprocessed_v3_numpy'
    elif machineType == 'bear':
        resultDirectory = '/rds/projects/2018/leonarda-muri/hector/bear/code/sceneImagination_binned_results/results'
        dataDirectory = '/rds/projects/2018/leonarda-muri/hector/data/dataFiles_d435_single_instance_preprocessed_v3_numpy'
    else:
        raise ValueError('Unknown machine type')

    dataFileFormat = 'Data{}Heap{}.npz'
    resultFileFormat = 'sceneSize_{}_sceneNumber_{}_results.npz'
    simulationFileFormat = 'sceneSize_{}_sceneNumber_{}_simulation.mat'

    currentResultDirectory = '{}/{}'.format(resultDirectory, networkType)
    print('Processing scene type {} for network {}'.format(sceneSetType, networkType))
    sceneSetFileName = '{}TestingScenes.npz'.format(sceneSetType)
    sceneListData = numpy.load(sceneSetFileName)
    typeResults = dict()
    for index, (sceneSize, sceneNumber) in enumerate(zip(sceneListData['sceneSize'],sceneListData['sceneNumber'])):
        print('Progress {:.0f}%'.format(100.0*index/len(sceneListData['sceneSize'])))
        currentDataFilePath = '{}/{}'.format(dataDirectory, dataFileFormat.format(sceneSize, sceneNumber))
        currentData = numpy.load(currentDataFilePath)
        currentResultsFilePath = '{}/{}'.format(currentResultDirectory, resultFileFormat.format(sceneSize, sceneNumber))
        currentResults = numpy.load(currentResultsFilePath)
        currentSimulationFilePath = '{}/{}'.format(currentResultDirectory, simulationFileFormat.format(sceneSize, sceneNumber))
        currentSimulation = hdf5storage.loadmat(currentSimulationFilePath)
        processedResults = generateSceneResults(groundTruthData = currentData, resultData = currentResults, simulationData = currentSimulation, pointCloudData = pointCloudData)
        maxFinalDisplacement = processedResults['finalDisplacement'].max()
        averagePrecision = processedResults['classPrecision'].mean()
        averageRecall = processedResults['classRecall'].mean()
        meanDepthError = processedResults['meanDepthError']
        misclassificationRatio = processedResults['misclassificationRatio']
        fractionalSceneSize = processedResults['fractionalSceneSize']
        fractionOfCorrectObjectsRecovered = processedResults['fractionOfCorrectObjectsRecovered']

        if 'maxFinalDisplacement' in typeResults:
            typeResults['maxFinalDisplacement'].append(maxFinalDisplacement)
        else:
            typeResults['maxFinalDisplacement'] = [maxFinalDisplacement]

        if 'averagePrecision' in typeResults:
            typeResults['averagePrecision'].append(averagePrecision)
        else:
            typeResults['averagePrecision'] = [averagePrecision]

        if 'averageRecall' in typeResults:
            typeResults['averageRecall'].append(averageRecall)
        else:
            typeResults['averageRecall'] = [averageRecall]

        if 'meanDepthError' in typeResults:
            typeResults['meanDepthError'].append(meanDepthError)
        else:
            typeResults['meanDepthError'] = [meanDepthError]

        if 'misclassificationRatio' in typeResults:
            typeResults['misclassificationRatio'].append(misclassificationRatio)
        else:
            typeResults['misclassificationRatio'] = [misclassificationRatio]

        if 'fractionalSceneSize' in typeResults:
            typeResults['fractionalSceneSize'].append(fractionalSceneSize)
        else:
            typeResults['fractionalSceneSize'] = [fractionalSceneSize]

        if 'fractionOfCorrectObjectsRecovered' in typeResults:
            typeResults['fractionOfCorrectObjectsRecovered'].append(fractionOfCorrectObjectsRecovered)
        else:
            typeResults['fractionOfCorrectObjectsRecovered'] = [fractionOfCorrectObjectsRecovered]

    typeResults = {a:torch.as_tensor(b).numpy() for a,b in typeResults.items()}
    outputFileName = 'testingResults_{}_{}.npz'.format(networkType, sceneSetType)
    numpy.savez_compressed(outputFileName, **typeResults)

