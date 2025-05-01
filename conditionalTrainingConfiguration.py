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

_CN.SYSTEM = CN()
#desktop, hpc, bear
_CN.SYSTEM.type = 'desktop'
_CN.SYSTEM.numberOfWorkers = 0
_CN.SYSTEM.cudnnBenchmarking = False
_CN.SYSTEM.disableCuDNN = False

_CN.NETWORKS = CN()
_CN.NETWORKS.GENERATOR = CN()
_CN.NETWORKS.GENERATOR.useSkipConnections = True
_CN.NETWORKS.GENERATOR.reconstructBackground = False
_CN.NETWORKS.GENERATOR.representationVectorDimension = 128
_CN.NETWORKS.GENERATOR.useBootstrap = False
_CN.NETWORKS.GENERATOR.bootstrapResidual = True

_CN.NETWORKS.GENERATOR.ENCODER = CN()
_CN.NETWORKS.GENERATOR.ENCODER.useSpectralNorm = False
_CN.NETWORKS.GENERATOR.ENCODER.kernelSize = 3
_CN.NETWORKS.GENERATOR.ENCODER.numberOfIntermediateComputationLayers = 0
_CN.NETWORKS.GENERATOR.ENCODER.useBatchNorm = True
_CN.NETWORKS.GENERATOR.ENCODER.useBiasThroughout = False
_CN.NETWORKS.GENERATOR.ENCODER.injectCoordinateInput = False
_CN.NETWORKS.GENERATOR.ENCODER.initialProjectionDimension = 128
_CN.NETWORKS.GENERATOR.ENCODER.stochasticRepresentationDimension = 64
_CN.NETWORKS.GENERATOR.ENCODER.downscalingChannelScaling = 2.0
_CN.NETWORKS.GENERATOR.ENCODER.useDSConvolutions = False                 
#depthwiseSeparable, residual, residualX
_CN.NETWORKS.GENERATOR.ENCODER.residualMode = 'none'

_CN.NETWORKS.GENERATOR.DECODER = CN()
_CN.NETWORKS.GENERATOR.DECODER.useSpectralNorm = False
_CN.NETWORKS.GENERATOR.DECODER.kernelSize = 3
_CN.NETWORKS.GENERATOR.DECODER.numberOfIntermediateComputationLayers = 0
_CN.NETWORKS.GENERATOR.DECODER.useBatchNorm = True
_CN.NETWORKS.GENERATOR.DECODER.useBiasThroughout = False
_CN.NETWORKS.GENERATOR.DECODER.injectCoordinateInput = True
_CN.NETWORKS.GENERATOR.DECODER.initialProjectionDimension = 512
_CN.NETWORKS.GENERATOR.DECODER.stochasticRepresentationDimension = 64
_CN.NETWORKS.GENERATOR.DECODER.upscalingChannelScaling = 0.5
_CN.NETWORKS.GENERATOR.DECODER.useDSConvolutions = False                  
#depthwiseSeparable, residual, residualX
_CN.NETWORKS.GENERATOR.DECODER.residualMode = 'none'

_CN.NETWORKS.DISCRIMINATOR = CN()
_CN.NETWORKS.DISCRIMINATOR.useSpectralNorm = False
_CN.NETWORKS.DISCRIMINATOR.kernelSize = 3
_CN.NETWORKS.DISCRIMINATOR.numberOfIntermediateComputationLayers = 0
_CN.NETWORKS.DISCRIMINATOR.useBiasThroughout = False
_CN.NETWORKS.DISCRIMINATOR.injectCoordinateInput = False
_CN.NETWORKS.DISCRIMINATOR.initialProjectionDimension = 64
_CN.NETWORKS.DISCRIMINATOR.downscalingChannelScaling = 2.0
_CN.NETWORKS.DISCRIMINATOR.useDSConvolutions = False                 
#depthwiseSeparable, residual, residualX
_CN.NETWORKS.DISCRIMINATOR.residualMode = 'none'

_CN.NETWORKS.STABILITY = CN()
_CN.NETWORKS.STABILITY.configFileName = 'none'
_CN.NETWORKS.STABILITY.weightFileName = 'none'

_CN.NETWORKS.BOOTSTRAP = CN()
_CN.NETWORKS.BOOTSTRAP.configFileName = 'none'
_CN.NETWORKS.BOOTSTRAP.encoderWeightFileName = 'none'
_CN.NETWORKS.BOOTSTRAP.decoderWeightFileName = 'none'

_CN.TRAINING = CN()
_CN.TRAINING.batchSize = 1
#adversarial, regression
_CN.TRAINING.trainingType = 'adversarial'
_CN.TRAINING.useStability = False
#hinge, wasserstein
_CN.TRAINING.lossType = 'wasserstein'
_CN.TRAINING.useGradientPenalty = True
_CN.TRAINING.l1ResidualPenaltyWeight = 1.0
_CN.TRAINING.numberOfDiscriminatorIterationsPerGeneratorIteration = 5
_CN.TRAINING.discriminatorInputNoise = 0.1
_CN.TRAINING.baseLearningRate = 1e-4
#joint, disjoint
_CN.TRAINING.generatorDiscriminatorDataSplit = 'joint'
_CN.TRAINING.voxelDataLoggingInterval = 5000
_CN.TRAINING.visualisationLoggingInterval = 1000
_CN.TRAINING.networkStateLoggingInterval = 1000


def getDefaultConfiguration():
    return _CN.clone()

def getBlankConfiguration():
    return CN()

def saveConfiguration(configuration, filePath):
    with os.fdopen(os.open(filePath, os.O_WRONLY|os.O_CREAT),'w') as file:
        configuration.dump(stream = file)

if __name__ == '__main__':
    defaults = getDefaultConfiguration()
    outputFileName = 'defaults.yaml'
    if os.path.exists(outputFileName):
        os.remove(outputFileName)
    saveConfiguration(defaults, outputFileName)
    defaults2 = getDefaultConfiguration()
    defaults2.merge_from_file(outputFileName)
    print('Complete')
    
