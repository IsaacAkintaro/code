import torch
from torch.utils.data import DataLoader
from yacs.config import CfgNode
import pytorch3DRenderer
import voxelVisualisationPytorch3D
import voxelParser
import networkArchitectures
import dataSets
import conditionalTrainingConfiguration
import stabilityEstimationTrainingConfiguration
import math
import numpy
import os
import sys
import hdf5storage

def getDefaultConfig():
    config = CfgNode()
    config.type = 'desktop'
    config.batchSize = 1
    config.numberOfWorkers = 4
    config.rootDirectory = 'none'
    config.networkType = 'none'
    config.iterationNumberToLoad = 30000
    config.numberOfScenesToGenerate = 1
    config.estimateSceneStability = True
    config.sceneStabilityDirectory = 'none'
    config.sceneStabilityWeightFile = 'none'
    config.sceneStabilityConfigFile = 'none'
    config.estimateObjectStability = True
    config.objectStabilityDirectory = 'none'
    config.objectStabilityWeightFile = 'none'
    config.objectStabilityConfigFile = 'none'
    config.parseScenes = True
    config.parsingIoUThreshold = 0.25
    config.renderVoxelViews = True
    config.renderImages = True
    config.testingFirstSceneSize = 1
    config.testingLastSceneSize = 13
    config.testingFirstSceneNumber = 901
    config.testingLastSceneNumber = 1000
    config.extrapolationFirstSceneSize = 14
    config.extrapolationLastSceneSize = 14
    config.extrapolationFirstSceneNumber = 1
    config.extrapolationLastSceneNumber = 1000

    return config

def generateSceneResults(  generatorEncoder, generatorDecoder, sceneStabilityEstimator, objectStabilityEstimator, partialScene, encoderLatent, decoderLatent,
                        completeSceneMean, completeSceneStandardDeviation, numberOfStabilityBins,
                        idealBackground, voxelGridBounds, parser, renderer,
                        visualisationThreshold, maximumFillThreshold, visualisationAngles,classColours,
                        encoderConfig, decoderConfig, resultsConfig, device):
    outputs = dict()
    generatorEncoder.eval()
    generatorDecoder.eval()
    with torch.no_grad():
        if encoderConfig.stochasticRepresentationDimension > 0:
            encoderInput = torch.cat([partialScene, encoderLatent.expand(-1,-1,*partialScene.shape[2:])], dim = 1).to(device)
        else:
            encoderInput = partialScene.to(device)
        if generatorConfig.useSkipConnections:
            partialSceneEncoding, skipConnections = generatorEncoder(encoderInput, returnIntermediateRepresentations = True)
        else:
            partialSceneEncoding, _ = generatorEncoder(encoderInput, returnIntermediateRepresentations = False)
        del encoderInput
        if decoderConfig.stochasticRepresentationDimension > 0:
            decoderInput = torch.cat([partialSceneEncoding, decoderLatent.to(device)], dim = 1)
        else:
            decoderInput = partialSceneEncoding
        del partialSceneEncoding
        if generatorConfig.useSkipConnections:
            generatedScene = generatorDecoder(decoderInput, skipConnections)
            del skipConnections
        else:
            generatedScene = generatorDecoder(decoderInput)
        del decoderInput
        if not generatorConfig.reconstructBackground:
            generatedScene = torch.cat([generatedScene, idealBackground[None,None,:,:,:].expand(generatedScene.size(0),-1,-1,-1,-1)], dim = 1)
        denormalisedGeneratedScene = (generatedScene*completeSceneStandardDeviation + completeSceneMean)
        outputs['generatedScene'] = denormalisedGeneratedScene.cpu().numpy()
        if resultsConfig.estimateSceneStability:
            sceneStability, _ = sceneStabilityEstimator(generatedScene)
            outputs['sceneStability'] = sceneStability.squeeze().softmax(dim = 0).cpu().numpy()
        if resultsConfig.estimateObjectStability:
            objectStability, _ = objectStabilityEstimator(generatedScene)
            objectStability = objectStability.reshape(-1, numberOfStabilityBins).softmax(dim = 1)
            outputs['objectStability'] = objectStability.cpu().numpy()
    if resultsConfig.renderVoxelViews:
        flippedGeneratedScene = denormalisedGeneratedScene.squeeze(dim = 0).flip(dims=[1])
        images = voxelVisualisationPytorch3D.renderVoxelScene(  flippedGeneratedScene, visualisationThreshold, maximumFillThreshold, torch.as_tensor(classColours[:15,:]), 
                                    [2.7]*len(visualisationAngles), visualisationAngles, [45]*len(visualisationAngles), device)
        outputs['voxelImages'] = torch.stack(images, dim = 0).cpu().numpy()
        del images
    if resultsConfig.parseScenes:
        lossBlurKernelSigma = 1
        lossBlurKernelSize = 2*int(math.ceil(lossBlurKernelSigma*3)) - 1
        # lossBlurKernelSigma = 0
        # lossBlurKernelSize = 0
        # lossType = 'meanSquaredError'
        lossType = 'intersectionOverUnion'
        numberOfIterations = 150
        binarisedGeneratedScene = (denormalisedGeneratedScene.squeeze(dim = 0) >= 0.5).to(dtype = torch.float)
        parsing = parser(   voxelScenes = binarisedGeneratedScene, voxelBoundsMax = voxelGridBounds[:,1], voxelBoundsMin = voxelGridBounds[:,0], 
                            numberOfIterations = numberOfIterations, blurKernelSize = lossBlurKernelSize, blurKernelSigma = lossBlurKernelSigma, lossType = lossType)
        objectPoses = [dict(objectClass = obj['objectClass'], transform = torch.cat([obj['rotationMatrix'], obj['translationVector'][:,None]], dim = 1)) for obj in parsing[0]]
        outputs['objectPoseClasses'] = numpy.asarray([obj['objectClass'] for obj in parsing[0]])
        outputs['objectPoseTransforms'] = torch.stack([torch.cat([obj['rotationMatrix'], obj['translationVector'][:,None]], dim = 1) for obj in parsing[0]]).cpu().numpy()
        if resultsConfig.renderImages:
            with torch.no_grad():
                images = renderer(objectPoses)
                outputs['rgbImage'] = images['occludedRGBImage'].squeeze()[:,:,:3].cpu().numpy()
                outputs['unoccludedRGBImages'] = torch.stack([x.squeeze()[:,:,:3] for x in images['unoccludedRGBImages']]).cpu().numpy()
                outputs['unoccludedDepthImages'] = torch.stack([x.squeeze() for x in images['unoccludedDepthImages']]).cpu().numpy()
                outputs['depthImage'] = images['depthImage'].squeeze().cpu().numpy()
                outputs['classImage'] = images['classImage'].squeeze().cpu().numpy()
    return outputs


if __name__ == '__main__':
    resultsConfig = getDefaultConfig()
    defaultConfigFilePath = 'generateTrainingResultsDefaults.yaml'
    if os.path.exists(defaultConfigFilePath):
        resultsConfig.merge_from_file(defaultConfigFilePath)
    else:
        with os.fdopen(os.open(defaultConfigFilePath, os.O_WRONLY|os.O_CREAT),'w') as file:
            resultsConfig.dump(stream = file)
    if len(sys.argv) > 1:
        configurationFileName = sys.argv[1]
    else:
        configurationFileName = 'generateTrainingResultsLocal.yaml'
    resultsConfig.merge_from_file(configurationFileName)
    resultsConfig.freeze()

    networkDirectory = '{}/{}'.format(resultsConfig.rootDirectory, resultsConfig.networkType)
    networkConfigPath = '{}/config.yaml'.format(networkDirectory)
    networkConfig = conditionalTrainingConfiguration.getDefaultConfiguration()
    networkConfig.merge_from_file(networkConfigPath)
    networkConfig.freeze()

    networksConfig = networkConfig.NETWORKS
    dataConfig = networkConfig.DATA
    generatorConfig = networksConfig.GENERATOR
    encoderConfig = generatorConfig.ENCODER
    decoderConfig = generatorConfig.DECODER

    device = torch.device('cuda')

    print('Setting up configuration')
    testingSceneSizes = list(range(resultsConfig.testingFirstSceneSize,resultsConfig.testingLastSceneSize+1))
    testingFirstSceneNumbers = [resultsConfig.testingFirstSceneNumber for x in testingSceneSizes]
    testingLastSceneNumbers = [resultsConfig.testingLastSceneNumber for x in testingSceneSizes]
    extrapolationSceneSizes = list(range(resultsConfig.extrapolationFirstSceneSize,resultsConfig.extrapolationLastSceneSize+1))
    extrapolationFirstSceneNumbers = [resultsConfig.extrapolationFirstSceneNumber for x in extrapolationSceneSizes]
    extrapolationLastSceneNumbers = [resultsConfig.extrapolationLastSceneNumber for x in extrapolationSceneSizes]
    classLabels = list(range(15))
    numberOfObjectClasses = 14
    voxelSize = 0.005
    voxelGridBounds = [[-0.24,0.24],[-0.24,0.24],[-0.02,0.30]]
    tensorVoxelGridBounds = torch.as_tensor(voxelGridBounds, device = device)
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
    cameraParametersFileName = 'cameraParameters.mat'
    cameraData = hdf5storage.loadmat(cameraParametersFileName, variable_names = ['intrinsicsMatrix', 'extrinsicsRotationMatrix', 'extrinsicsTranslationVector'])
    cameraIntrinsics = torch.as_tensor(cameraData['intrinsicsMatrix'], dtype = torch.float)
    cameraRotationMatrix = torch.as_tensor(cameraData['extrinsicsRotationMatrix'], dtype = torch.float)
    cameraTranslationVector = torch.as_tensor(cameraData['extrinsicsTranslationVector'], dtype = torch.float)
    cameraResolution = [480,640]

    imageFileTemplate = 'Images{}ObjectHeap{}.mat'
    voxelFileTemplate = 'Data{}HeapStability{}Binary.mat'
    # dataSetType = 'runtime'
    dataSetType = 'preprocessed'
    preprocessedFileTemplate = 'Data{}Heap{}'
    preprocessedDataType = 'numpy'
    # preprocessedDataType = 'pytorch_gzip'
    if preprocessedDataType == 'numpy':
        preprocessedFileTemplate += '.npz'
    elif preprocessedDataType == 'pytorch_gzip':
        preprocessedFileTemplate += '.pytg'
    if resultsConfig.type == 'bear':
        imageDataDirectory = '/rds/projects/2018/leonarda-muri/hector/data/dataFiles_d435_single_instance'
        voxelDataDirectory = '/rds/projects/2018/leonarda-muri/hector/data/dataFiles_d435_single_instance_stability'
        preprocessedDataDirectory = '/rds/projects/2018/leonarda-muri/hector/data/dataFiles_d435_single_instance_preprocessed_v3_{}'.format(preprocessedDataType)
    elif resultsConfig.type == 'desktop':
        imageDataDirectory = 'F:/data/dataFiles_d435_single_instance'
        voxelDataDirectory = 'F:/data/dataFiles_d435_single_instance_stability'
        preprocessedDataDirectory = 'F:/data/dataFiles_d435_single_instance_preprocessed_v3_{}'.format(preprocessedDataType)
    elif resultsConfig.type == 'hpc':
        imageDataDirectory = '/home/research/users/basevhra/data/dataFiles_d435_single_instance'
        voxelDataDirectory = '/home/research/users/basevhra/data/dataFiles_d435_single_instance_stability'
        preprocessedDataDirectory = '/home/research/users/basevhra/data/dataFiles_d435_single_instance_preprocessed_v3_{}'.format(preprocessedDataType)
    else:
        raise ValueError('Unknown system type: {}'.format(resultsConfig.type))

    enforceBinaryCompleteScenes = True

    difficultTestingScenes = numpy.load('difficultTestingScenes.npz')
    difficultExtrapolationScenes = numpy.load('difficultExtrapolationScenes.npz')
    hiddenTestingScenes = numpy.load('hiddenTestingScenes.npz')
    hiddenExtrapolationScenes = numpy.load('hiddenExtrapolationScenes.npz')

    sceneStatisticsMeansFileName = 'trainingSceneStatisticsMeans_1-{}_b{}.mat'.format(dataConfig.maximumSceneSize, dataConfig.voxelBinningPower)
    sceneStatisticsStandardDeviationsFileName = 'trainingSceneStatisticsStandardDeviations_1-{}_b{}.mat'.format(dataConfig.maximumSceneSize, dataConfig.voxelBinningPower)
    if os.path.exists(sceneStatisticsMeansFileName) and os.path.exists(sceneStatisticsStandardDeviationsFileName):
        print('Loading data statistics')
        statisticsMeansData = hdf5storage.loadmat(sceneStatisticsMeansFileName)
        statisticsMeansData = {key:torch.as_tensor(value) for (key, value) in statisticsMeansData.items()}
        statisticsStandardDeviationsData = hdf5storage.loadmat(sceneStatisticsStandardDeviationsFileName)
        statisticsStandardDeviationsData = {key:torch.as_tensor(value) for (key, value) in statisticsStandardDeviationsData.items()}
    else:
        raise RuntimeError('Data set requires precomputed scene statistics')
    
    currentStatisticsMeans = dict(
                                    completeVoxelScene = statisticsMeansData['completeVoxelScene'][:,None,None,None].to(dtype = torch.float),
                                    surfaceVoxelEmbedding = statisticsMeansData['surfaceVoxelEmbedding'][:,None,None,None].to(dtype = torch.float),
                                    projectionVoxelEmbedding = statisticsMeansData['projectionVoxelEmbedding'][:,None,None,None].to(dtype = torch.float),
                                )
    currentStatisticsStandardDeviations = dict(
                                    completeVoxelScene = statisticsStandardDeviationsData['completeVoxelScene'][:,None,None,None].to(dtype = torch.float),
                                    surfaceVoxelEmbedding = statisticsStandardDeviationsData['surfaceVoxelEmbedding'][:,None,None,None].to(dtype = torch.float),
                                    projectionVoxelEmbedding = statisticsStandardDeviationsData['projectionVoxelEmbedding'][:,None,None,None].to(dtype = torch.float),
                                )

    completeSceneMean = currentStatisticsMeans['completeVoxelScene'].unsqueeze(dim = 0).to(dtype = torch.float32, device = device)
    completeSceneStandardDeviation = currentStatisticsStandardDeviations['completeVoxelScene'].unsqueeze(dim = 0).to(dtype = torch.float32,device = device)

    difficultTestingDataSet = dataSets.PreprocessedMixedListDataset(difficultTestingScenes['sceneSize'], difficultTestingScenes['sceneNumber'], preprocessedDataDirectory, preprocessedFileTemplate, 
                voxelBinningPower = dataConfig.voxelBinningPower, dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations, 
                enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
    difficultTestingDataloader = DataLoader(difficultTestingDataSet, batch_size = resultsConfig.batchSize, shuffle = False, num_workers = resultsConfig.numberOfWorkers, drop_last = False, pin_memory = True)

    hiddenTestingDataSet = dataSets.PreprocessedMixedListDataset(hiddenTestingScenes['sceneSize'], hiddenTestingScenes['sceneNumber'], preprocessedDataDirectory, preprocessedFileTemplate, 
                voxelBinningPower = dataConfig.voxelBinningPower, dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations, 
                enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
    hiddenTestingDataloader = DataLoader(hiddenTestingDataSet, batch_size = resultsConfig.batchSize, shuffle = False, num_workers = resultsConfig.numberOfWorkers, drop_last = False, pin_memory = True)
   
    difficultExtrapolationDataSet = dataSets.PreprocessedMixedListDataset(difficultExtrapolationScenes['sceneSize'], difficultExtrapolationScenes['sceneNumber'], preprocessedDataDirectory, preprocessedFileTemplate, 
                voxelBinningPower = dataConfig.voxelBinningPower, dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations, 
                enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
    difficultExtrapolationDataloader = DataLoader(difficultExtrapolationDataSet, batch_size = resultsConfig.batchSize, shuffle = False, num_workers = resultsConfig.numberOfWorkers, drop_last = False, pin_memory = True)

    hiddenExtrapolationDataSet = dataSets.PreprocessedMixedListDataset(hiddenExtrapolationScenes['sceneSize'], hiddenExtrapolationScenes['sceneNumber'], preprocessedDataDirectory, preprocessedFileTemplate, 
                voxelBinningPower = dataConfig.voxelBinningPower, dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations, 
                enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
    hiddenExtrapolationDataloader = DataLoader(hiddenExtrapolationDataSet, batch_size = resultsConfig.batchSize, shuffle = False, num_workers = resultsConfig.numberOfWorkers, drop_last = False, pin_memory = True)

    numberOfSpatialScales = int(math.floor(min([math.log2(x) for x in voxelBinnedSpatialDimensions]))-1)

    partialSceneVoxelChannels = 16
    completeSceneVoxelChannels = 15

    if resultsConfig.estimateSceneStability:
        print('Constructing scene stability estimator')
        stabilityConditioningConfig = stabilityEstimationTrainingConfiguration.getDefaultConfiguration()
        stabilityConfigFile = '{}/{}'.format(resultsConfig.sceneStabilityDirectory, resultsConfig.sceneStabilityConfigFile)
        stabilityConditioningConfig.merge_from_file(stabilityConfigFile)
        stabilityConditioningConfig.freeze()
        stabilityNetworkConfig = stabilityConditioningConfig.NETWORK 
        stabilityTrainingConfig = stabilityConditioningConfig.TRAINING 
        numberOfStabilityBins = stabilityTrainingConfig.numberOfQuantisationBins
        if stabilityTrainingConfig.predictionType == 'quantised':
            if stabilityTrainingConfig.quantisationBinScaling == 'exponential':
                quantisationBinSpacing = (math.log10(stabilityTrainingConfig.highestQuantisationBinCentre/stabilityTrainingConfig.lowestQuantisationBinCentre))/(stabilityTrainingConfig.numberOfQuantisationBins-1)
                quantisationBinCentres = torch.arange(  math.log10(stabilityTrainingConfig.lowestQuantisationBinCentre),math.log10(stabilityTrainingConfig.highestQuantisationBinCentre)+1e-7,
                                                    quantisationBinSpacing)                                         
            elif stabilityTrainingConfig.quantisationBinScaling == 'linear':
                quantisationBinSpacing = (stabilityTrainingConfig.highestQuantisationBinCentre-stabilityTrainingConfig.lowestQuantisationBinCentre)/(stabilityTrainingConfig.numberOfQuantisationBins-1)
                quantisationBinCentres = torch.arange(  stabilityTrainingConfig.lowestQuantisationBinCentre,stabilityTrainingConfig.highestQuantisationBinCentre+1e-7,
                                                    quantisationBinSpacing)  
            if stabilityTrainingConfig.predictionScale == 'scene':
                stabilityEstimatorOutputDimension = stabilityTrainingConfig.numberOfQuantisationBins
            elif stabilityTrainingConfig.predictionScale == 'object':
                stabilityEstimatorOutputDimension = numberOfObjectClasses*stabilityTrainingConfig.numberOfQuantisationBins
            else:
                raise ValueError('Unknown prediction scale: {}'.format(stabilityTrainingConfig.predictionScale))
        elif stabilityTrainingConfig.predictionType == 'scalar':
            if stabilityTrainingConfig.predictionScale == 'scene':
                stabilityEstimatorOutputDimension = 1
            elif stabilityTrainingConfig.predictionScale == 'object':
                stabilityEstimatorOutputDimension = numberOfObjectClasses
            else:
                raise ValueError('Unknown prediction scale: {}'.format(stabilityTrainingConfig.predictionScale))
        else:
            raise ValueError('Unknown prediction type: {}'.format(stabilityTrainingConfig.predictionType))

        stabilityEstimatorDownscaling = numberOfSpatialScales
        stabilityEstimatorUseTrigonometricCoordinateEmbedding = True
        if stabilityNetworkConfig.useSpectralNorm:
            stabilityEstimatorSpectralNormalisationSettings = dict(useSpectralNormalisation = True, numberOfPowerIterations = 5)
        else:
            stabilityEstimatorSpectralNormalisationSettings = dict(useSpectralNormalisation = False)
        sceneStabilityEstimator = networkArchitectures.Encoder(inputDimensions = [completeSceneVoxelChannels]+voxelBinnedSpatialDimensions, 
            initialProjectionDimension = stabilityNetworkConfig.initialProjectionDimension, 
            numberOfDownscalingOperations = stabilityEstimatorDownscaling, representationDimension = stabilityEstimatorOutputDimension, 
            numberOfIntermediateComputationLayers = stabilityNetworkConfig.numberOfIntermediateComputationLayers, kernelSize = stabilityNetworkConfig.kernelSize,
            addCoordinatesToRepresentation = stabilityNetworkConfig.injectCoordinateInput,
            useTrigonometricCoordinateEmbedding = stabilityEstimatorUseTrigonometricCoordinateEmbedding,
            residualMode = stabilityNetworkConfig.residualMode, useDepthWiseSeparableConvolutions = stabilityNetworkConfig.useDSConvolutions,
            spectralNormalisationSettings = stabilityEstimatorSpectralNormalisationSettings, useBatchNorm = False, useBiasThroughout = stabilityNetworkConfig.useBiasThroughout,
            downscalingChannelScaling = stabilityNetworkConfig.downscalingChannelScaling)
        sceneStabilityEstimator = sceneStabilityEstimator.to(device = device)
        stabilityWeightFile = '{}/{}'.format(resultsConfig.sceneStabilityDirectory,resultsConfig.sceneStabilityWeightFile)
        sceneStabilityEstimator.load_state_dict(torch.load(stabilityWeightFile, map_location=device))
        sceneStabilityEstimator.eval()

        
    if resultsConfig.estimateObjectStability:
        print('Constructing object stability estimator')
        stabilityConditioningConfig = stabilityEstimationTrainingConfiguration.getDefaultConfiguration()
        stabilityConfigFile = '{}/{}'.format(resultsConfig.objectStabilityDirectory, resultsConfig.objectStabilityConfigFile)
        stabilityConditioningConfig.merge_from_file(stabilityConfigFile)
        stabilityConditioningConfig.freeze()
        stabilityNetworkConfig = stabilityConditioningConfig.NETWORK 
        stabilityTrainingConfig = stabilityConditioningConfig.TRAINING 
        numberOfStabilityBins = stabilityTrainingConfig.numberOfQuantisationBins

        if stabilityTrainingConfig.predictionType == 'quantised':
            if stabilityTrainingConfig.quantisationBinScaling == 'exponential':
                quantisationBinSpacing = (math.log10(stabilityTrainingConfig.highestQuantisationBinCentre/stabilityTrainingConfig.lowestQuantisationBinCentre))/(stabilityTrainingConfig.numberOfQuantisationBins-1)
                quantisationBinCentres = torch.arange(  math.log10(stabilityTrainingConfig.lowestQuantisationBinCentre),math.log10(stabilityTrainingConfig.highestQuantisationBinCentre)+1e-7,
                                                    quantisationBinSpacing)                                         
            elif stabilityTrainingConfig.quantisationBinScaling == 'linear':
                quantisationBinSpacing = (stabilityTrainingConfig.highestQuantisationBinCentre-stabilityTrainingConfig.lowestQuantisationBinCentre)/(stabilityTrainingConfig.numberOfQuantisationBins-1)
                quantisationBinCentres = torch.arange(  stabilityTrainingConfig.lowestQuantisationBinCentre,stabilityTrainingConfig.highestQuantisationBinCentre+1e-7,
                                                    quantisationBinSpacing)  
            if stabilityTrainingConfig.predictionScale == 'scene':
                stabilityEstimatorOutputDimension = stabilityTrainingConfig.numberOfQuantisationBins
            elif stabilityTrainingConfig.predictionScale == 'object':
                stabilityEstimatorOutputDimension = numberOfObjectClasses*stabilityTrainingConfig.numberOfQuantisationBins
            else:
                raise ValueError('Unknown prediction scale: {}'.format(stabilityTrainingConfig.predictionScale))
        elif stabilityTrainingConfig.predictionType == 'scalar':
            if stabilityTrainingConfig.predictionScale == 'scene':
                stabilityEstimatorOutputDimension = 1
            elif stabilityTrainingConfig.predictionScale == 'object':
                stabilityEstimatorOutputDimension = numberOfObjectClasses
            else:
                raise ValueError('Unknown prediction scale: {}'.format(stabilityTrainingConfig.predictionScale))
        else:
            raise ValueError('Unknown prediction type: {}'.format(stabilityTrainingConfig.predictionType))

        stabilityEstimatorDownscaling = numberOfSpatialScales
        stabilityEstimatorUseTrigonometricCoordinateEmbedding = True
        if stabilityNetworkConfig.useSpectralNorm:
            stabilityEstimatorSpectralNormalisationSettings = dict(useSpectralNormalisation = True, numberOfPowerIterations = 5)
        else:
            stabilityEstimatorSpectralNormalisationSettings = dict(useSpectralNormalisation = False)
        objectStabilityEstimator = networkArchitectures.Encoder(inputDimensions = [completeSceneVoxelChannels]+voxelBinnedSpatialDimensions, 
            initialProjectionDimension = stabilityNetworkConfig.initialProjectionDimension, 
            numberOfDownscalingOperations = stabilityEstimatorDownscaling, representationDimension = stabilityEstimatorOutputDimension, 
            numberOfIntermediateComputationLayers = stabilityNetworkConfig.numberOfIntermediateComputationLayers, kernelSize = stabilityNetworkConfig.kernelSize,
            addCoordinatesToRepresentation = stabilityNetworkConfig.injectCoordinateInput,
            useTrigonometricCoordinateEmbedding = stabilityEstimatorUseTrigonometricCoordinateEmbedding,
            residualMode = stabilityNetworkConfig.residualMode, useDepthWiseSeparableConvolutions = stabilityNetworkConfig.useDSConvolutions,
            spectralNormalisationSettings = stabilityEstimatorSpectralNormalisationSettings, useBatchNorm = False, useBiasThroughout = stabilityNetworkConfig.useBiasThroughout,
            downscalingChannelScaling = stabilityNetworkConfig.downscalingChannelScaling)
        objectStabilityEstimator = objectStabilityEstimator.to(device = device)
        stabilityWeightFile = '{}/{}'.format(resultsConfig.objectStabilityDirectory,resultsConfig.objectStabilityWeightFile)
        objectStabilityEstimator.load_state_dict(torch.load(stabilityWeightFile, map_location=device))
        objectStabilityEstimator.eval()

    print('Constructing generator encoder')
    voxelGeneratorEncoderInputDimension = partialSceneVoxelChannels + encoderConfig.stochasticRepresentationDimension
    voxelGeneratorEncoderDownscaling = numberOfSpatialScales
    voxelGeneratorEncoderUseTrigonometricCoordinateEmbedding = True
    if encoderConfig.useSpectralNorm:
        voxelGeneratorEncoderSpectralNormalisationSettings = dict(useSpectralNormalisation = True, numberOfPowerIterations = 5)
    else:
        voxelGeneratorEncoderSpectralNormalisationSettings = dict(useSpectralNormalisation = False)
    generatorEncoder = networkArchitectures.Encoder(inputDimensions = [voxelGeneratorEncoderInputDimension]+voxelBinnedSpatialDimensions, initialProjectionDimension = encoderConfig.initialProjectionDimension, 
        numberOfDownscalingOperations = voxelGeneratorEncoderDownscaling, representationDimension = generatorConfig.representationVectorDimension, 
        numberOfIntermediateComputationLayers = encoderConfig.numberOfIntermediateComputationLayers, kernelSize = encoderConfig.kernelSize,
        addCoordinatesToRepresentation = encoderConfig.injectCoordinateInput,
        useTrigonometricCoordinateEmbedding = voxelGeneratorEncoderUseTrigonometricCoordinateEmbedding,
        residualMode = encoderConfig.residualMode, useDepthWiseSeparableConvolutions = encoderConfig.useDSConvolutions,
        spectralNormalisationSettings = voxelGeneratorEncoderSpectralNormalisationSettings, useBatchNorm = encoderConfig.useBatchNorm, useBiasThroughout = encoderConfig.useBiasThroughout,
        downscalingChannelScaling = encoderConfig.downscalingChannelScaling)
    generatorEncoder = generatorEncoder.to(device = device)
    generatorEncoderWeightFile = '{}/generatorEncoder_state_iteration_{}.pts'.format(networkDirectory,resultsConfig.iterationNumberToLoad)
    generatorEncoder.load_state_dict(torch.load(generatorEncoderWeightFile, map_location=device))
    generatorEncoder.eval()

    print('Constructing generator decoder')
    voxelRepresentationDimension = generatorConfig.representationVectorDimension + decoderConfig.stochasticRepresentationDimension
    voxelGeneratorDecoderInitialSpatialDimensions = [3,3,2]
    voxelGeneratorDecoderOutputDimension = completeSceneVoxelChannels
    if not generatorConfig.reconstructBackground:
        voxelGeneratorDecoderOutputDimension -= 1 #Not reconstructing background/table
    voxelGeneratorDecoderNumberOfUpscalingOperations = numberOfSpatialScales
    voxelGeneratorDecoderUseTrigonometricCoordinateEmbedding = True
    if decoderConfig.useSpectralNorm:
        voxelGeneratorDecoderSpectralNormalisationSettings = dict(useSpectralNormalisation = True, numberOfPowerIterations = 5)
    else:
        voxelGeneratorDecoderSpectralNormalisationSettings = dict(useSpectralNormalisation = False)
    if generatorConfig.useSkipConnections:
        skipConnectionDimensions = generatorEncoder.intermediateRepresentationShapes
    else:
        skipConnectionDimensions = None
    generatorDecoder = networkArchitectures.Decoder(representationDimension = voxelRepresentationDimension, initialSpatialDimensions = voxelGeneratorDecoderInitialSpatialDimensions, 
        initialProjectionDimension = decoderConfig.initialProjectionDimension, outputDimension = voxelGeneratorDecoderOutputDimension, 
        numberOfUpscalingOperations = voxelGeneratorDecoderNumberOfUpscalingOperations, 
        numberOfIntermediateComputationLayers = decoderConfig.numberOfIntermediateComputationLayers, kernelSize = decoderConfig.kernelSize,
        skipConnectionDimensions = skipConnectionDimensions, addCoordinatesToRepresentation = decoderConfig.injectCoordinateInput, 
        useTrigonometricCoordinateEmbedding = voxelGeneratorDecoderUseTrigonometricCoordinateEmbedding,
        residualMode = decoderConfig.residualMode, useDepthWiseSeparableConvolutions = decoderConfig.useDSConvolutions,
        spectralNormalisationSettings = voxelGeneratorDecoderSpectralNormalisationSettings, useBatchNorm = decoderConfig.useBatchNorm, useBiasThroughout = decoderConfig.useBiasThroughout,
        upscalingChannelScaling = decoderConfig.upscalingChannelScaling)
    generatorDecoder = generatorDecoder.to(device = device)
    generatorDecoderWeightFile = '{}/generatorDecoder_state_iteration_{}.pts'.format(networkDirectory,resultsConfig.iterationNumberToLoad)
    generatorDecoder.load_state_dict(torch.load(generatorDecoderWeightFile, map_location=device))
    generatorDecoder.eval()

    classColours = hdf5storage.loadmat('officialColours.mat')['colours'][1:17,:]
    classColours = classColours.astype(numpy.float32)

    visualisationThreshold = 0.5
    maximumFillThreshold = 1.0
    visualisationAngles = list(range(-60,300,20))

    if resultsConfig.parseScenes:
        dataDirectory = 'canonicalVoxelRepresentations'
        objectInformationFile = 'objectInformation.mat'
        rotationData = hdf5storage.loadmat('rotations/quaternions64.mat')
        quaternions = torch.as_tensor(rotationData['quaternions'], dtype = torch.float, device = device)
        objectData = voxelParser.loadObjectData(dataDirectory, objectInformationFile, binariseVoxels = True)
        objectVoxels = []
        objectBoundsMax = []
        objectBoundsMin = []
        for currentObject in objectData:
            objectVoxels.append(currentObject['voxelGrid'].to(dtype = torch.float, device = device))
            objectBoundsMax.append(currentObject['voxelGridMaximum'].to(dtype = torch.float, device = device))
            objectBoundsMin.append(currentObject['voxelGridMinimum'].to(dtype = torch.float, device = device))
        parser = voxelParser.VoxelParser(objectVoxels = objectVoxels, objectBoundsMax = objectBoundsMax, objectBoundsMin = objectBoundsMin, seedRotations = quaternions, iouThreshold = resultsConfig.parsingIoUThreshold)
    else:
        parser = None

    if resultsConfig.renderImages:
        objectDirectory = 'ycbobj'
        sceneData = pytorch3DRenderer.loadObjects(objectDirectory, device = device)
        renderer = pytorch3DRenderer.Pytorch3DRenderer(   cameraIntrinsics = sceneData['K'], cameraExtrinsics = sceneData['T'], lightPositions = sceneData['lightPosition'], 
                                        objectMeshes = sceneData['objectMeshes'], tableMesh = sceneData['tableMesh'], device = device)

    resultsRootDirectory = 'results'
    if not os.path.exists(resultsRootDirectory):
        try:
            os.mkdir(resultsRootDirectory)
        except:
            if not os.path.exists(resultsRootDirectory):
                raise IOError('Unable to create results directory: {}.'.format(resultsRootDirectory))

    resultsDirectory = '{}/{}'.format(resultsRootDirectory, resultsConfig.networkType)
    if not os.path.exists(resultsDirectory):
        try:
            os.mkdir(resultsDirectory)
        except:
            if not os.path.exists(resultsDirectory):
                raise IOError('Unable to create results directory: {}.'.format(resultsDirectory))

    for dataPoint in difficultTestingDataloader:
        if encoderConfig.stochasticRepresentationDimension > 0:
            encoderLatent = torch.randn(1, encoderConfig.stochasticRepresentationDimension, 1, 1, 1, requires_grad = False, device = device)
        else:
            encoderLatent = None
        if decoderConfig.stochasticRepresentationDimension > 0:
            decoderLatent = torch.randn(1, decoderConfig.stochasticRepresentationDimension, requires_grad = False, device = device)
        else:
            decoderLatent = None
        sceneSize = dataPoint['sceneSize'].item()
        sceneNumber = dataPoint['sceneNumber'].item()
        partialVoxelData = dataPoint[dataConfig.voxelEmbeddingType].to(dtype = torch.float, device = device)
        completeVoxelData = dataPoint['completeVoxelScene']
        idealBackground = completeVoxelData[0,14,:,:,:].to(device)
        dataPointResults = generateSceneResults(  generatorEncoder, generatorDecoder, sceneStabilityEstimator, objectStabilityEstimator, partialVoxelData, encoderLatent, decoderLatent,
                        completeSceneMean, completeSceneStandardDeviation, numberOfStabilityBins,
                        idealBackground, tensorVoxelGridBounds, parser, renderer,
                        visualisationThreshold, maximumFillThreshold, visualisationAngles, classColours,
                        encoderConfig, decoderConfig, resultsConfig, device)

        outputFileName = 'sceneSize_{}_sceneNumber_{}_results.npz'.format(sceneSize, sceneNumber)
        outputFilePath = '{}/{}'.format(resultsDirectory,outputFileName)
        numpy.savez_compressed(outputFilePath, **dataPointResults)
        outputMatlabFileName = 'sceneSize_{}_sceneNumber_{}_results.mat'.format(sceneSize, sceneNumber)
        outputMatlabFilePath = '{}/{}'.format(resultsDirectory,outputMatlabFileName)
        hdf5storage.savemat(outputMatlabFilePath, dataPointResults)
        
    for dataPoint in hiddenTestingDataloader:
        if encoderConfig.stochasticRepresentationDimension > 0:
            encoderLatent = torch.randn(1, encoderConfig.stochasticRepresentationDimension, 1, 1, 1, requires_grad = False, device = device)
        else:
            encoderLatent = None
        if decoderConfig.stochasticRepresentationDimension > 0:
            decoderLatent = torch.randn(1, decoderConfig.stochasticRepresentationDimension, requires_grad = False, device = device)
        else:
            decoderLatent = None
        sceneSize = dataPoint['sceneSize'].item()
        sceneNumber = dataPoint['sceneNumber'].item()
        partialVoxelData = dataPoint[dataConfig.voxelEmbeddingType].to(dtype = torch.float, device = device)
        completeVoxelData = dataPoint['completeVoxelScene']
        idealBackground = completeVoxelData[0,14,:,:,:].to(device)
        dataPointResults = generateSceneResults(  generatorEncoder, generatorDecoder, sceneStabilityEstimator, objectStabilityEstimator, partialVoxelData, encoderLatent, decoderLatent,
                        completeSceneMean, completeSceneStandardDeviation, numberOfStabilityBins,
                        idealBackground, tensorVoxelGridBounds, parser, renderer,
                        visualisationThreshold, maximumFillThreshold, visualisationAngles, classColours,
                        encoderConfig, decoderConfig, resultsConfig, device)

        outputFileName = 'sceneSize_{}_sceneNumber_{}_results.npz'.format(sceneSize, sceneNumber)
        outputFilePath = '{}/{}'.format(resultsDirectory,outputFileName)
        numpy.savez_compressed(outputFilePath, **dataPointResults)
        outputMatlabFileName = 'sceneSize_{}_sceneNumber_{}_results.mat'.format(sceneSize, sceneNumber)
        outputMatlabFilePath = '{}/{}'.format(resultsDirectory,outputMatlabFileName)
        hdf5storage.savemat(outputMatlabFilePath, dataPointResults)

    for dataPoint in difficultExtrapolationDataloader:
        if encoderConfig.stochasticRepresentationDimension > 0:
            encoderLatent = torch.randn(1, encoderConfig.stochasticRepresentationDimension, 1, 1, 1, requires_grad = False, device = device)
        else:
            encoderLatent = None
        if decoderConfig.stochasticRepresentationDimension > 0:
            decoderLatent = torch.randn(1, decoderConfig.stochasticRepresentationDimension, requires_grad = False, device = device)
        else:
            decoderLatent = None
        sceneSize = dataPoint['sceneSize'].item()
        sceneNumber = dataPoint['sceneNumber'].item()
        partialVoxelData = dataPoint[dataConfig.voxelEmbeddingType].to(dtype = torch.float, device = device)
        completeVoxelData = dataPoint['completeVoxelScene']
        idealBackground = completeVoxelData[0,14,:,:,:].to(device)
        dataPointResults = generateSceneResults(  generatorEncoder, generatorDecoder, sceneStabilityEstimator, objectStabilityEstimator, partialVoxelData, encoderLatent, decoderLatent,
                        completeSceneMean, completeSceneStandardDeviation, numberOfStabilityBins,
                        idealBackground, tensorVoxelGridBounds, parser, renderer,
                        visualisationThreshold, maximumFillThreshold, visualisationAngles, classColours,
                        encoderConfig, decoderConfig, resultsConfig, device)

        outputFileName = 'sceneSize_{}_sceneNumber_{}_results.npz'.format(sceneSize, sceneNumber)
        outputFilePath = '{}/{}'.format(resultsDirectory,outputFileName)
        numpy.savez_compressed(outputFilePath, **dataPointResults)
        outputMatlabFileName = 'sceneSize_{}_sceneNumber_{}_results.mat'.format(sceneSize, sceneNumber)
        outputMatlabFilePath = '{}/{}'.format(resultsDirectory,outputMatlabFileName)
        hdf5storage.savemat(outputMatlabFilePath, dataPointResults)

    for dataPoint in hiddenExtrapolationDataloader:
        if encoderConfig.stochasticRepresentationDimension > 0:
            encoderLatent = torch.randn(1, encoderConfig.stochasticRepresentationDimension, 1, 1, 1, requires_grad = False, device = device)
        else:
            encoderLatent = None
        if decoderConfig.stochasticRepresentationDimension > 0:
            decoderLatent = torch.randn(1, decoderConfig.stochasticRepresentationDimension, requires_grad = False, device = device)
        else:
            decoderLatent = None
        sceneSize = dataPoint['sceneSize'].item()
        sceneNumber = dataPoint['sceneNumber'].item()
        partialVoxelData = dataPoint[dataConfig.voxelEmbeddingType].to(dtype = torch.float, device = device)
        completeVoxelData = dataPoint['completeVoxelScene']
        idealBackground = completeVoxelData[0,14,:,:,:].to(device)
        dataPointResults = generateSceneResults(  generatorEncoder, generatorDecoder, sceneStabilityEstimator, objectStabilityEstimator, partialVoxelData, encoderLatent, decoderLatent,
                        completeSceneMean, completeSceneStandardDeviation, numberOfStabilityBins,
                        idealBackground, tensorVoxelGridBounds, parser, renderer,
                        visualisationThreshold, maximumFillThreshold, visualisationAngles, classColours,
                        encoderConfig, decoderConfig, resultsConfig, device)

        outputFileName = 'sceneSize_{}_sceneNumber_{}_results.npz'.format(sceneSize, sceneNumber)
        outputFilePath = '{}/{}'.format(resultsDirectory,outputFileName)
        numpy.savez_compressed(outputFilePath, **dataPointResults)
        outputMatlabFileName = 'sceneSize_{}_sceneNumber_{}_results.mat'.format(sceneSize, sceneNumber)
        outputMatlabFilePath = '{}/{}'.format(resultsDirectory,outputMatlabFileName)
        hdf5storage.savemat(outputMatlabFilePath, dataPointResults)

