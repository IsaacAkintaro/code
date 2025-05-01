from yacs.config import CfgNode as CN
import os
import sys

_CN = CN()
_CN.name = 'Default_configuration'
_CN.DATA = CN()
_CN.DATA.maximumSceneSize = 12
# surfaceVoxelEmbedding, projectionVoxelEmbedding
_CN.DATA.voxelEmbeddingType = 'surfaceVoxelEmbedding'
_CN.DATA.voxelBinningPower = 0
_CN.DATA.trainingFirstSceneNumber = 1
_CN.DATA.trainingLastSceneNumber = 800
_CN.DATA.validationFirstSceneNumber = 801
_CN.DATA.validationLastSceneNumber = 900
_CN.DATA.testingFirstSceneNumber = 901
_CN.DATA.testingLastSceneNumber = 1000

_CN.SYSTEM = CN()
#desktop, hpc, bear
_CN.SYSTEM.type = 'desktop'
_CN.SYSTEM.numberOfWorkers = 0
_CN.SYSTEM.cudnnBenchmarking = True
_CN.SYSTEM.disableCuDNN = False

_CN.NETWORK = CN()
_CN.NETWORK.useDSConvolutions = False                 
#depthwiseSeparable, None
_CN.NETWORK.residualMode = 'none'
_CN.NETWORK.useSpectralNorm = False
_CN.NETWORK.kernelSize = 3
_CN.NETWORK.numberOfIntermediateComputationLayers = 0
_CN.NETWORK.useBatchNorm = False
_CN.NETWORK.useBiasThroughout = False
_CN.NETWORK.injectCoordinateInput = False
_CN.NETWORK.initialProjectionDimension = 64
_CN.NETWORK.downscalingChannelScaling = 1.5

_CN.TRAINING = CN()
_CN.TRAINING.batchSize = 1
#finalDisplacement, maximumDisplacement
_CN.TRAINING.predictionQuantity = 'finalDisplacement'
#scene, object
_CN.TRAINING.predictionScale = 'scene'
#quantised, scalar
_CN.TRAINING.predictionType = 'quantised'
_CN.TRAINING.numberOfQuantisationBins = 8
_CN.TRAINING.lowestQuantisationBinCentre = 1e-3
_CN.TRAINING.highestQuantisationBinCentre = 1e-0
#exponential, linear
_CN.TRAINING.quantisationBinScaling = 'exponential'
#crossEntropy, leastSquares
_CN.TRAINING.lossType = 'crossEntropy'
_CN.TRAINING.estimateMissingObjects = True
_CN.TRAINING.useGradientPenalty = True
_CN.TRAINING.numberOfIterations = 50001
_CN.TRAINING.inputNoise = 0.1
_CN.TRAINING.learningRate = 1e-4

def getDefaultConfiguration():
    return _CN.clone()

def saveConfiguration(configuration, filePath):
    with os.fdopen(os.open(filePath, os.O_WRONLY|os.O_CREAT),'w') as file:
        configuration.dump(stream = file)

if __name__ == '__main__':
    defaults = getDefaultConfiguration()
    outputFileName = 'stabilityEstimationDefaults.yaml'
    if os.path.exists(outputFileName):
        os.remove(outputFileName)
    saveConfiguration(defaults, outputFileName)
    defaults2 = getDefaultConfiguration()
    defaults2.merge_from_file(outputFileName)
    print('Complete')
    
