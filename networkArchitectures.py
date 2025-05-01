import torch
from torch import nn
import math
import numpy

def isConvolution(layer):
    return isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv3d)

def generateNormalisedCoordinateGrid(gridSize):    
    gridVectors = [torch.arange(0, size, dtype = torch.float) for size in gridSize]
    gridVectors = [2*x/x.max() - 1 for x in gridVectors]
    xGrid, yGrid, zGrid = torch.meshgrid(*gridVectors)
    return torch.stack([xGrid, yGrid, zGrid])

def calculateTrigonometricEncoding(data, minimumPower, maximumPower):
    encodings = list()
    for power in range(minimumPower, maximumPower+1):
        encodings.append((math.pow(2, power)*math.pi*data).sin())
        encodings.append((math.pow(2, power)*math.pi*data).cos())
    return torch.cat(encodings, dim = 0)

class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        return x.flatten(start_dim = 1)

class DSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, groups = in_channels, bias = bias)
        self.pointwise = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = bias)

    def forward(self, x):
        x = self.pointwise(self.depthwise(x))
        return x

    def applySpectralNorm(self, numberOfPowerIterations = 5):
        self.depthwise = nn.utils.spectral_norm(self.depthwise, n_power_iterations = numberOfPowerIterations)
        self.pointwise = nn.utils.spectral_norm(self.pointwise, n_power_iterations = numberOfPowerIterations)

class DSConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, groups = in_channels, bias = bias)
        self.pointwise = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = bias)

    def forward(self, x):
        x = self.pointwise(self.depthwise(x))
        return x

    def applySpectralNorm(self, numberOfPowerIterations = 5):
        self.depthwise = nn.utils.spectral_norm(self.depthwise, n_power_iterations = numberOfPowerIterations)
        self.pointwise = nn.utils.spectral_norm(self.pointwise, n_power_iterations = numberOfPowerIterations)

class DSConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.depthwise = nn.ConvTranspose2d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, 
            padding = padding, output_padding = output_padding, groups = in_channels, bias = bias)
        self.pointwise = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

    def applySpectralNorm(self, numberOfPowerIterations = 5):
        self.depthwise = nn.utils.spectral_norm(self.depthwise, n_power_iterations = numberOfPowerIterations)
        self.pointwise = nn.utils.spectral_norm(self.pointwise, n_power_iterations = numberOfPowerIterations)

class DSConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.depthwise = nn.ConvTranspose3d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, 
            padding = padding, output_padding = output_padding, groups = in_channels, bias = bias)
        self.pointwise = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

    def applySpectralNorm(self, numberOfPowerIterations = 5):
        self.depthwise = nn.utils.spectral_norm(self.depthwise, n_power_iterations = numberOfPowerIterations)
        self.pointwise = nn.utils.spectral_norm(self.pointwise, n_power_iterations = numberOfPowerIterations)

class Residual2dBlock(nn.Module):
    def __init__(self, in_channels, internalDimension, kernel_size, bias, useBatchNorm = False):
        super().__init__()
        self.activation = nn.LeakyReLU()
        padding = int(round((kernel_size-1)/2))
        self.inwardProjection = nn.Conv2d(in_channels = in_channels, out_channels = internalDimension, kernel_size = kernel_size, stride = 1, padding = padding, groups = 1, bias = bias)
        self.outwardProjection = nn.Conv2d(in_channels = internalDimension, out_channels = in_channels, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = bias)
        if useBatchNorm:
            self.inwardBatchNorm = nn.BatchNorm2d(num_features = internalDimension)
            self.outwardBatchNorm = nn.BatchNorm2d(num_features = in_channels)
        else:
            self.inwardBatchNorm = None
            self.outwardBatchNorm = None

    def forward(self, x):
        r = self.activation(x)
        r = self.inwardProjection(r)
        if self.inwardBatchNorm is not None:
            r = self.inwardBatchNorm(r)
        r = self.activation(r)
        r = self.outwardProjection(r)
        if self.outwardBatchNorm is not None:
            r = self.outwardBatchNorm(r)
        out = r + x
        return out

    def applySpectralNorm(self, numberOfPowerIterations = 5):
        self.inwardProjection = nn.utils.spectral_norm(self.inwardProjection, n_power_iterations = numberOfPowerIterations)
        self.outwardProjection = nn.utils.spectral_norm(self.outwardProjection, n_power_iterations = numberOfPowerIterations)

class ResidualX2dBlock(nn.Module):
    def __init__(self, in_channels, internalDimension, kernel_size, bias, useBatchNorm = False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batchNorms = nn.ModuleList()
        self.activation = nn.LeakyReLU()
        padding = int(round((kernel_size-1)/2))
        self.layers.append(nn.Conv2d(in_channels = in_channels, out_channels = internalDimension, kernel_size = 1, stride = 1, padding = 0, bias = bias))    
        if useBatchNorm:
            self.batchNorms.append(nn.BatchNorm2d(num_features = internalDimension))
        else:
            self.batchNorms.append(None)
        self.layers.append(nn.Conv2d(in_channels = internalDimension, out_channels = internalDimension, kernel_size = kernel_size, stride = 1, padding = padding, bias = bias))
        if useBatchNorm:
            self.batchNorms.append(nn.BatchNorm2d(num_features = internalDimension))
        else:
            self.batchNorms.append(None)
        self.layers.append(nn.Conv2d(in_channels = internalDimension, out_channels = in_channels, kernel_size = 1, stride = 1, padding = 0, bias = bias))  
        if useBatchNorm:
            self.batchNorms.append(nn.BatchNorm2d(num_features = in_channels))
        else:
            self.batchNorms.append(None)

    def forward(self, x):
        r = x
        for layer, batchNorm in zip(self.layers, self.batchNorms):
            r = layer(self.activation(r))
            if batchNorm is not None:
                r = batchNorm(r)
        return r + x

    def applySpectralNorm(self, numberOfPowerIterations = 5):
        for index in range(len(self.layers)):
            self.layers[index] = nn.utils.spectral_norm(self.layers[index], n_power_iterations = numberOfPowerIterations)

class DSResidual2dBlock(nn.Module):
    def __init__(self, in_channels, internalDimension, kernel_size, bias, useBatchNorm = False):
        super().__init__()
        self.activation = nn.LeakyReLU()
        padding = int(round((kernel_size-1)/2))
        self.inwardProjection = nn.Conv2d(in_channels = in_channels, out_channels = internalDimension, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = bias)
        self.depthwise = nn.Conv2d(in_channels = internalDimension, out_channels = internalDimension, kernel_size = kernel_size, stride = 1, padding = padding, groups = internalDimension, bias = bias)
        self.outwardProjection = nn.Conv2d(in_channels = internalDimension, out_channels = in_channels, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = bias)
        if useBatchNorm:
            self.inwardBatchNorm = nn.BatchNorm2d(num_features = internalDimension)
            self.depthwiseBatchNorm = nn.BatchNorm2d(num_features = internalDimension)
            self.outwardBatchNorm = nn.BatchNorm2d(num_features = in_channels)
        else:
            self.inwardBatchNorm = None
            self.depthwiseBatchNorm = None
            self.outwardBatchNorm = None

    def forward(self, x):
        r = x
        r = self.inwardProjection(r)
        if self.inwardBatchNorm is not None:
            r = self.inwardBatchNorm(r)
        r = self.depthwise(self.activation(r))
        if self.inwardBatchNorm is not None:
            r = self.depthwiseBatchNorm(r)
        r = self.outwardProjection(self.activation(r))
        if self.inwardBatchNorm is not None:
            r = self.outwardBatchNorm(r)
        return r + x

    def applySpectralNorm(self, numberOfPowerIterations = 5):
        self.inwardProjection = nn.utils.spectral_norm(self.inwardProjection, n_power_iterations = numberOfPowerIterations)
        self.depthwise = nn.utils.spectral_norm(self.depthwise, n_power_iterations = numberOfPowerIterations)
        self.outwardProjection = nn.utils.spectral_norm(self.outwardProjection, n_power_iterations = numberOfPowerIterations)

class Residual3dBlock(nn.Module):
    def __init__(self, in_channels, internalDimension, kernel_size, bias, useBatchNorm = False):
        super().__init__()
        self.activation = nn.LeakyReLU()
        padding = int(round((kernel_size-1)/2))
        self.inwardProjection = nn.Conv3d(in_channels = in_channels, out_channels = internalDimension, kernel_size = kernel_size, stride = 1, padding = padding, groups = 1, bias = bias)
        self.outwardProjection = nn.Conv3d(in_channels = internalDimension, out_channels = in_channels, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = bias)
        if useBatchNorm:
            self.inwardBatchNorm = nn.BatchNorm3d(num_features = internalDimension)
            self.outwardBatchNorm = nn.BatchNorm3d(num_features = in_channels)
        else:
            self.inwardBatchNorm = None
            self.outwardBatchNorm = None

    def forward(self, x):
        r = self.activation(x)
        r = self.inwardProjection(r)
        if self.inwardBatchNorm is not None:
            r = self.inwardBatchNorm(r)
        r = self.activation(r)
        r = self.outwardProjection(r)
        if self.outwardBatchNorm is not None:
            r = self.outwardBatchNorm(r)
        return r + x
        
    def applySpectralNorm(self, numberOfPowerIterations = 5):
        self.inwardProjection = nn.utils.spectral_norm(self.inwardProjection, n_power_iterations = numberOfPowerIterations)
        self.outwardProjection = nn.utils.spectral_norm(self.outwardProjection, n_power_iterations = numberOfPowerIterations)


class ResidualX3dBlock(nn.Module):
    def __init__(self, in_channels, internalDimension, kernel_size, bias, useBatchNorm = False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batchNorms = nn.ModuleList()
        self.activation = nn.LeakyReLU()
        padding = int(round((kernel_size-1)/2))
        self.layers.append(nn.Conv3d(in_channels = in_channels, out_channels = internalDimension, kernel_size = 1, stride = 1, padding = 0, bias = bias))
        if useBatchNorm:
            self.batchNorms.append(nn.BatchNorm3d(num_features = internalDimension))
        else:
            self.batchNorms.append(None)
        self.layers.append(nn.Conv3d(in_channels = internalDimension, out_channels = internalDimension, kernel_size = kernel_size, stride = 1, padding = padding, bias = bias))
        if useBatchNorm:
            self.batchNorms.append(nn.BatchNorm3d(num_features = internalDimension))
        else:
            self.batchNorms.append(None)
        self.layers.append(nn.Conv3d(in_channels = internalDimension, out_channels = in_channels, kernel_size = 1, stride = 1, padding = 0, bias = bias))
        if useBatchNorm:
            self.batchNorms.append(nn.BatchNorm3d(num_features = in_channels))
        else:
            self.batchNorms.append(None)

    def forward(self, x):
        r = x
        for layer, batchNorm in zip(self.layers, self.batchNorms):
            r = layer(self.activation(r))
            if batchNorm is not None:
                r = batchNorm(r)
        return r + x

    def applySpectralNorm(self, numberOfPowerIterations = 5):
        for index in range(len(self.layers)):
            self.layers[index] = nn.utils.spectral_norm(self.layers[index], n_power_iterations = numberOfPowerIterations)

class DSResidual3dBlock(nn.Module):
    def __init__(self, in_channels, internalDimension, kernel_size, bias, useBatchNorm = False):
        super().__init__()
        self.activation = nn.LeakyReLU()
        padding = int(round((kernel_size-1)/2))
        self.inwardProjection = nn.Conv3d(in_channels = in_channels, out_channels = internalDimension, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = bias)
        self.depthwise = nn.Conv3d(in_channels = internalDimension, out_channels = internalDimension, kernel_size = kernel_size, stride = 1, padding = padding, groups = internalDimension, bias = bias)
        self.outwardProjection = nn.Conv3d(in_channels = internalDimension, out_channels = in_channels, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = bias)
        if useBatchNorm:
            self.inwardBatchNorm = nn.BatchNorm3d(num_features = internalDimension)
            self.depthwiseBatchNorm = nn.BatchNorm3d(num_features = internalDimension)
            self.outwardBatchNorm = nn.BatchNorm3d(num_features = in_channels)
        else:
            self.inwardBatchNorm = None
            self.depthwiseBatchNorm = None
            self.outwardBatchNorm = None

    def forward(self, x):
        r = x
        r = self.inwardProjection(r)
        if self.inwardBatchNorm is not None:
            r = self.inwardBatchNorm(r)
        r = self.depthwise(self.activation(r))
        if self.depthwiseBatchNorm is not None:
            r = self.depthwiseBatchNorm(r)
        r = self.outwardProjection(self.activation(r))
        if self.outwardBatchNorm is not None:
            r = self.outwardBatchNorm(r)
        return r + x

    def applySpectralNorm(self, numberOfPowerIterations = 5):
        self.inwardProjection = nn.utils.spectral_norm(self.inwardProjection, n_power_iterations = numberOfPowerIterations)
        self.depthwise = nn.utils.spectral_norm(self.depthwise, n_power_iterations = numberOfPowerIterations)
        self.outwardProjection = nn.utils.spectral_norm(self.outwardProjection, n_power_iterations = numberOfPowerIterations)

class Encoder(nn.Module):
    def __init__(self, inputDimensions, initialProjectionDimension, numberOfDownscalingOperations, representationDimension, numberOfIntermediateComputationLayers = 0, 
                kernelSize = 3, addCoordinatesToRepresentation = True, useTrigonometricCoordinateEmbedding = True,
                residualMode = 'depthwiseSeparable', useDepthWiseSeparableConvolutions = True, 
                spectralNormalisationSettings = dict(useSpectralNormalisation = True, numberOfPowerIterations = 5), useBatchNorm = False, useBiasThroughout = False,
                downscalingChannelScaling = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.inputDimensions = list()
        self.outputDimensions = list()
        self.intermediateRepresentationFlag = list()
        self.intermediateRepresentationShapes = list()

        padding = int(round((kernelSize-1)/2))
        currentSpatialDimensions = inputDimensions[1:]
        if len(currentSpatialDimensions) == 2:
            convolution = nn.Conv2d
            dsConvolution = DSConv2d
            residualBlock = Residual2dBlock
            residualXBlock = ResidualX2dBlock
            dsResidualBlock = DSResidual2dBlock
            batchNorm = nn.BatchNorm2d
        elif len(currentSpatialDimensions) == 3:
            convolution = nn.Conv3d
            dsConvolution = DSConv3d
            residualBlock = Residual3dBlock
            residualXBlock = ResidualX3dBlock
            dsResidualBlock = DSResidual3dBlock
            batchNorm = nn.BatchNorm3d
        if addCoordinatesToRepresentation:
            coordinates = generateNormalisedCoordinateGrid(currentSpatialDimensions)
            if useTrigonometricCoordinateEmbedding:
                coordinates = calculateTrigonometricEncoding(coordinates, 0, 5)
            self.register_buffer('coordinates', coordinates)
            coordinateDimension = coordinates.shape[0]
        else:
            self.coordinates = None
            coordinateDimension = 0
        currentNumberOfChannels = inputDimensions[0] + coordinateDimension
        currentConvolution = convolution(in_channels = currentNumberOfChannels, out_channels = initialProjectionDimension, kernel_size = 1, stride = 1, padding = 0, bias = True)
        # Do not regularise the initial projection layer
        # if spectralNormalisationSettings['useSpectralNormalisation']:
        #     currentConvolution = nn.utils.spectral_norm(currentConvolution, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
        self.layers.append(currentConvolution)
        self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
        currentNumberOfChannels = initialProjectionDimension
        self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
        self.intermediateRepresentationFlag.append(False)
        self.intermediateRepresentationShapes.append(None)
        if useBatchNorm:
            self.layers.append(batchNorm(num_features = currentNumberOfChannels))
            self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
            self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
            self.intermediateRepresentationFlag.append(False)
            self.intermediateRepresentationShapes.append(None)
        for downscaleIndex in range(numberOfDownscalingOperations):
            if residualMode == 'none':
                self.layers.append(nn.LeakyReLU())
                self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.intermediateRepresentationFlag.append(False)
                self.intermediateRepresentationShapes.append(None)
                if useDepthWiseSeparableConvolutions:
                    currentConvolution = dsConvolution(in_channels = currentNumberOfChannels, out_channels = currentNumberOfChannels, kernel_size = kernelSize, stride = 1, padding = padding, bias = useBiasThroughout)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                else:
                    currentConvolution = convolution(in_channels = currentNumberOfChannels, out_channels = currentNumberOfChannels, kernel_size = kernelSize, stride = 1, padding = padding, bias = useBiasThroughout)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution = nn.utils.spectral_norm(currentConvolution, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
                self.layers.append(currentConvolution)
                self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                if useBatchNorm:
                    self.intermediateRepresentationFlag.append(False)
                    self.intermediateRepresentationShapes.append(None)
                    self.layers.append(batchNorm(num_features = currentNumberOfChannels))
                    self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(True)
                    self.intermediateRepresentationShapes.append([currentNumberOfChannels,]+currentSpatialDimensions)
                else:
                    self.intermediateRepresentationFlag.append(True)
                    self.intermediateRepresentationShapes.append([currentNumberOfChannels,]+currentSpatialDimensions)
            elif residualMode == 'depthwiseSeparable':
                currentConvolution = dsResidualBlock(in_channels = currentNumberOfChannels, internalDimension = currentNumberOfChannels*4, kernel_size = kernelSize, bias = useBiasThroughout, useBatchNorm = useBatchNorm)
                if spectralNormalisationSettings['useSpectralNormalisation']:
                    currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                self.layers.append(currentConvolution)
                self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.intermediateRepresentationFlag.append(True)
                self.intermediateRepresentationShapes.append([currentNumberOfChannels,]+currentSpatialDimensions)
            elif residualMode == 'residual':
                currentConvolution = residualBlock(in_channels = currentNumberOfChannels, internalDimension = currentNumberOfChannels, kernel_size = kernelSize, bias = useBiasThroughout, useBatchNorm = useBatchNorm)
                if spectralNormalisationSettings['useSpectralNormalisation']:
                    currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                self.layers.append(currentConvolution)
                self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.intermediateRepresentationFlag.append(True)
                self.intermediateRepresentationShapes.append([currentNumberOfChannels,]+currentSpatialDimensions)
            elif residualMode == 'residualX':
                currentConvolution = residualXBlock(in_channels = currentNumberOfChannels, internalDimension = int(round(currentNumberOfChannels/2)), kernel_size = kernelSize, bias = useBiasThroughout, useBatchNorm = useBatchNorm)
                if spectralNormalisationSettings['useSpectralNormalisation']:
                    currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                self.layers.append(currentConvolution)
                self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.intermediateRepresentationFlag.append(True)
                self.intermediateRepresentationShapes.append([currentNumberOfChannels,]+currentSpatialDimensions)
            else:
                raise ValueError('Unknown residual mode: {}'.format(residualMode))
            
            for intermediateComputationIndex in range(numberOfIntermediateComputationLayers):
                if residualMode == 'none':
                    self.layers.append(nn.LeakyReLU())
                    self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(False)
                    self.intermediateRepresentationShapes.append(None)
                    if useDepthWiseSeparableConvolutions:
                        currentConvolution = dsConvolution(in_channels = currentNumberOfChannels, out_channels = currentNumberOfChannels, kernel_size = kernelSize, stride = 1, padding = padding, bias = useBiasThroughout)
                        if spectralNormalisationSettings['useSpectralNormalisation']:
                            currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                    else:
                        currentConvolution = convolution(in_channels = currentNumberOfChannels, out_channels = currentNumberOfChannels, kernel_size = kernelSize, stride = 1, padding = padding, bias = useBiasThroughout)
                        if spectralNormalisationSettings['useSpectralNormalisation']:
                            currentConvolution = nn.utils.spectral_norm(currentConvolution, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
                    self.layers.append(currentConvolution)
                    self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    if useBatchNorm:
                        self.intermediateRepresentationFlag.append(False)
                        self.intermediateRepresentationShapes.append(None)
                        self.layers.append(batchNorm(num_features = currentNumberOfChannels))
                        self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                        self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                        self.intermediateRepresentationFlag.append(True)
                        self.intermediateRepresentationShapes.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    else:
                        self.intermediateRepresentationFlag.append(True)
                        self.intermediateRepresentationShapes.append([currentNumberOfChannels,]+currentSpatialDimensions)
                elif residualMode == 'depthwiseSeparable':
                    currentConvolution = dsResidualBlock(in_channels = currentNumberOfChannels, internalDimension = currentNumberOfChannels*4, kernel_size = kernelSize, bias = useBiasThroughout, useBatchNorm = useBatchNorm)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                    self.layers.append(currentConvolution)
                    self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(True)
                    self.intermediateRepresentationShapes.append([currentNumberOfChannels,]+currentSpatialDimensions)
                elif residualMode == 'residual':
                    currentConvolution = residualBlock(in_channels = currentNumberOfChannels, internalDimension = currentNumberOfChannels, kernel_size = kernelSize, bias = useBiasThroughout, useBatchNorm = useBatchNorm)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                    self.layers.append(currentConvolution)
                    self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(True)
                    self.intermediateRepresentationShapes.append([currentNumberOfChannels,]+currentSpatialDimensions)
                elif residualMode == 'residualX':
                    currentConvolution = residualXBlock(in_channels = currentNumberOfChannels, internalDimension = int(round(currentNumberOfChannels/2)), kernel_size = kernelSize, bias = useBiasThroughout, useBatchNorm = useBatchNorm)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                    self.layers.append(currentConvolution)
                    self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(True)
                    self.intermediateRepresentationShapes.append([currentNumberOfChannels,]+currentSpatialDimensions)
                else:
                    raise ValueError('Unknown residual mode: {}'.format(residualMode))

            #Downscale
            newNumberOfChannels = int(round(downscalingChannelScaling*currentNumberOfChannels))
            if newNumberOfChannels % 8 > 0:
                newNumberOfChannels += 8 - (newNumberOfChannels % 8) # Make sure that number of channels is divisible by 8 (for cuDNN)

            newNumberOfChannels = newNumberOfChannels 
            if useDepthWiseSeparableConvolutions:
                currentDownscale = dsConvolution(in_channels = currentNumberOfChannels, out_channels = newNumberOfChannels, kernel_size = kernelSize, stride = 2, padding = padding, bias = useBiasThroughout)
                if spectralNormalisationSettings['useSpectralNormalisation']:
                    currentDownscale.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
            else:
                currentDownscale = convolution(in_channels = currentNumberOfChannels, out_channels = newNumberOfChannels, kernel_size = kernelSize, stride = 2, padding = padding, bias = useBiasThroughout)
                if spectralNormalisationSettings['useSpectralNormalisation']:
                    currentDownscale = nn.utils.spectral_norm(currentDownscale, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
            self.layers.append(currentDownscale)
            self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
            currentSpatialDimensions = [math.floor((x+1)/2) for x in currentSpatialDimensions]
            currentNumberOfChannels = newNumberOfChannels
            self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
            self.intermediateRepresentationFlag.append(False)
            self.intermediateRepresentationShapes.append(None)
            if useBatchNorm:
                self.layers.append(batchNorm(num_features = currentNumberOfChannels))
                self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.intermediateRepresentationFlag.append(False)
                self.intermediateRepresentationShapes.append(None)
        outputDimension = numpy.prod(currentSpatialDimensions)*currentNumberOfChannels
        self.layers.append(FlattenLayer())
        self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
        currentNumberOfChannels = outputDimension
        self.outputDimensions.append([currentNumberOfChannels])
        self.intermediateRepresentationFlag.append(False)
        self.intermediateRepresentationShapes.append(None)

        self.layers.append(nn.LeakyReLU())
        self.inputDimensions.append([currentNumberOfChannels])
        self.outputDimensions.append([currentNumberOfChannels])
        self.intermediateRepresentationFlag.append(False)
        self.intermediateRepresentationShapes.append(None)
        newNumberOfChannels = 4*representationDimension
        linearLayer = nn.Linear(currentNumberOfChannels, newNumberOfChannels, bias = useBiasThroughout)
        if spectralNormalisationSettings['useSpectralNormalisation']:
            linearLayer = nn.utils.spectral_norm(linearLayer, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
        self.layers.append(linearLayer)
        self.inputDimensions.append([currentNumberOfChannels])
        currentNumberOfChannels = newNumberOfChannels
        self.outputDimensions.append([currentNumberOfChannels])
        self.intermediateRepresentationFlag.append(False)
        self.intermediateRepresentationShapes.append(None)

        self.layers.append(nn.LeakyReLU())
        self.inputDimensions.append([currentNumberOfChannels])
        self.outputDimensions.append([currentNumberOfChannels])
        self.intermediateRepresentationFlag.append(False)
        self.intermediateRepresentationShapes.append(None)
        newNumberOfChannels = representationDimension
        linearLayer = nn.Linear(currentNumberOfChannels, newNumberOfChannels, bias = True)
        # Do not regularise the final layer
        # if spectralNormalisationSettings['useSpectralNormalisation']:
        #     linearLayer = nn.utils.spectral_norm(linearLayer, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
        self.layers.append(linearLayer)
        self.inputDimensions.append([outputDimension])
        self.outputDimensions.append([representationDimension])
        self.intermediateRepresentationFlag.append(False)
        self.intermediateRepresentationShapes.append(None)
    
    def forward(self, x, returnIntermediateRepresentations = False):
        if self.coordinates is not None:
            x = torch.cat([x, self.coordinates[None,:].repeat(x.shape[0],*[1 for x in x.shape[1:]])], dim = 1)
        intermediateRepresentations = list()
        for layer, inputDimension, outputDimension, intermediateRepresentation in zip(self.layers, self.inputDimensions, self.outputDimensions, self.intermediateRepresentationFlag):
            temp = layer(x)
            if intermediateRepresentation:
                intermediateRepresentations.append(temp)
            else:
                intermediateRepresentations.append(None)
            x = temp
        return x, intermediateRepresentations

class Decoder(nn.Module):
    def __init__(self, representationDimension, initialSpatialDimensions, initialProjectionDimension, outputDimension, numberOfUpscalingOperations, 
                numberOfIntermediateComputationLayers = 0, kernelSize = 3,
                skipConnectionDimensions = None, addCoordinatesToRepresentation = True, useTrigonometricCoordinateEmbedding = True,
                residualMode = 'depthwiseSeparable', useDepthWiseSeparableConvolutions = True,
                spectralNormalisationSettings = dict(useSpectralNormalisation = True, numberOfPowerIterations = 5), useBatchNorm = False, useBiasThroughout = False,
                upscalingChannelScaling = 0.5):
        super().__init__()
        if len(initialSpatialDimensions) == 2:
            convolution = nn.Conv2d
            dsConvolution = DSConv2d
            transposeConvolution = nn.ConvTranspose2d
            dsTransposeConvolution = DSConvTranspose2d
            residualBlock = Residual2dBlock
            residualXBlock = ResidualX2dBlock
            dsResidualBlock = DSResidual2dBlock
            batchNorm = nn.BatchNorm2d
        elif len(initialSpatialDimensions) == 3:
            convolution = nn.Conv3d
            dsConvolution = DSConv3d
            transposeConvolution = nn.ConvTranspose3d
            dsTransposeConvolution = DSConvTranspose3d
            residualBlock = Residual3dBlock
            residualXBlock = ResidualX3dBlock
            dsResidualBlock = DSResidual3dBlock
            batchNorm = nn.BatchNorm3d
        self.initialSpatialDimensions = initialSpatialDimensions
        if addCoordinatesToRepresentation:
            coordinates = generateNormalisedCoordinateGrid(initialSpatialDimensions)
            if useTrigonometricCoordinateEmbedding:
                coordinates = calculateTrigonometricEncoding(coordinates, 0, 5)
            self.register_buffer('coordinates', coordinates)
            coordinateDimension = coordinates.shape[0]
        else:
            self.coordinates = None
            coordinateDimension = 0
        self.layers = nn.ModuleList()
        self.inputDimensions = list()
        self.outputDimensions = list()
        self.intermediateRepresentationFlag = list()
        self.intermediateRepresentationShapes = list()
        self.intermediateRepresentationIndices = list()

        if skipConnectionDimensions is not None:
            skipSpatialSizes = [x[1:] if x is not None else x for x in skipConnectionDimensions]
        
        padding = int(round((kernelSize-1)/2))
        currentSpatialDimensions = initialSpatialDimensions
        currentNumberOfChannels = representationDimension + coordinateDimension
        currentConvolution = convolution(in_channels = currentNumberOfChannels, out_channels = initialProjectionDimension, kernel_size = 1, stride = 1, padding = 0, bias = True)
        # Don't regularise first layer
        # if spectralNormalisationSettings['useSpectralNormalisation']:
        #     currentConvolution = nn.utils.spectral_norm(currentConvolution, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
        self.layers.append(currentConvolution)
        self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
        currentNumberOfChannels = initialProjectionDimension
        self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
        self.intermediateRepresentationFlag.append(False)
        self.intermediateRepresentationShapes.append(None)
        self.intermediateRepresentationIndices.append(None)
        if useBatchNorm:
            self.layers.append(batchNorm(num_features = currentNumberOfChannels))
            self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
            self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
            self.intermediateRepresentationFlag.append(False)
            self.intermediateRepresentationShapes.append(None)
            self.intermediateRepresentationIndices.append(None)
        for upscaleIndex in range(numberOfUpscalingOperations):
            augmentedChannels = currentNumberOfChannels
            try:
                intermediateRepresentationIndex = skipSpatialSizes.index(currentSpatialDimensions)
                augmentedChannels += skipConnectionDimensions[intermediateRepresentationIndex][0]
                self.intermediateRepresentationFlag.append(True)
                self.intermediateRepresentationShapes.append(skipConnectionDimensions[intermediateRepresentationIndex])
                self.intermediateRepresentationIndices.append(intermediateRepresentationIndex)
            except:
                self.intermediateRepresentationFlag.append(False)
                self.intermediateRepresentationShapes.append(None)
                self.intermediateRepresentationIndices.append(None)
            if residualMode == 'none':
                self.layers.append(nn.LeakyReLU())
                self.inputDimensions.append([augmentedChannels,]+currentSpatialDimensions)
                self.outputDimensions.append([augmentedChannels,]+currentSpatialDimensions)
                
                if useDepthWiseSeparableConvolutions:
                    currentConvolution = dsConvolution(in_channels = augmentedChannels, out_channels = currentNumberOfChannels, kernel_size = kernelSize, stride = 1, padding = padding, bias = useBiasThroughout)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                else:
                    currentConvolution = convolution(in_channels = augmentedChannels, out_channels = currentNumberOfChannels, kernel_size = kernelSize, stride = 1, padding = padding, bias = useBiasThroughout)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution = nn.utils.spectral_norm(currentConvolution, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
                self.layers.append(currentConvolution)
                self.inputDimensions.append([augmentedChannels,]+currentSpatialDimensions)
                self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.intermediateRepresentationFlag.append(False)
                self.intermediateRepresentationShapes.append(None)
                self.intermediateRepresentationIndices.append(None)
                if useBatchNorm:
                    self.layers.append(batchNorm(num_features = currentNumberOfChannels))
                    self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(False)
                    self.intermediateRepresentationShapes.append(None)
                    self.intermediateRepresentationIndices.append(None)
            elif residualMode == 'depthwiseSeparable':
                currentConvolution = dsResidualBlock(in_channels = augmentedChannels, internalDimension = currentNumberOfChannels*4, kernel_size = kernelSize, bias = useBiasThroughout, useBatchNorm = useBatchNorm)
                if spectralNormalisationSettings['useSpectralNormalisation']:
                    currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                self.layers.append(currentConvolution)
                self.inputDimensions.append([augmentedChannels,]+currentSpatialDimensions)
                self.outputDimensions.append([augmentedChannels,]+currentSpatialDimensions)
                if augmentedChannels != currentNumberOfChannels:
                    currentConvolution = convolution(in_channels = augmentedChannels, out_channels = currentNumberOfChannels, kernel_size = 1, stride = 1, padding = 0, bias = useBiasThroughout)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution = nn.utils.spectral_norm(currentConvolution, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
                    self.layers.append(currentConvolution)
                    self.inputDimensions.append([augmentedChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(False)
                    self.intermediateRepresentationShapes.append(None)
                    self.intermediateRepresentationIndices.append(None)
                    if useBatchNorm:
                        self.layers.append(batchNorm(num_features = currentNumberOfChannels))
                        self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                        self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                        self.intermediateRepresentationFlag.append(False)
                        self.intermediateRepresentationShapes.append(None)
                        self.intermediateRepresentationIndices.append(None)
            elif residualMode == 'residual':
                currentConvolution = residualBlock(in_channels = augmentedChannels, internalDimension = currentNumberOfChannels, kernel_size = kernelSize, bias = useBiasThroughout, useBatchNorm = useBatchNorm)
                if spectralNormalisationSettings['useSpectralNormalisation']:
                    currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                self.layers.append(currentConvolution)
                self.inputDimensions.append([augmentedChannels,]+currentSpatialDimensions)
                self.outputDimensions.append([augmentedChannels,]+currentSpatialDimensions)
                if augmentedChannels != currentNumberOfChannels:
                    currentConvolution = convolution(in_channels = augmentedChannels, out_channels = currentNumberOfChannels, kernel_size = 1, stride = 1, padding = 0, bias = useBiasThroughout)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution = nn.utils.spectral_norm(currentConvolution, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
                    self.layers.append(currentConvolution)
                    self.inputDimensions.append([augmentedChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(False)
                    self.intermediateRepresentationShapes.append(None)
                    self.intermediateRepresentationIndices.append(None)
                    if useBatchNorm:
                        self.layers.append(batchNorm(num_features = currentNumberOfChannels))
                        self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                        self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                        self.intermediateRepresentationFlag.append(False)
                        self.intermediateRepresentationShapes.append(None)
                        self.intermediateRepresentationIndices.append(None)
            elif residualMode == 'residualX':
                currentConvolution = residualXBlock(in_channels = augmentedChannels, internalDimension = int(round(currentNumberOfChannels/2)), kernel_size = kernelSize, bias = useBiasThroughout, useBatchNorm = useBatchNorm)
                if spectralNormalisationSettings['useSpectralNormalisation']:
                    currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                self.layers.append(currentConvolution)
                self.inputDimensions.append([augmentedChannels,]+currentSpatialDimensions)
                self.outputDimensions.append([augmentedChannels,]+currentSpatialDimensions)
                if augmentedChannels != currentNumberOfChannels:
                    currentConvolution = convolution(in_channels = augmentedChannels, out_channels = currentNumberOfChannels, kernel_size = 1, stride = 1, padding = 0, bias = useBiasThroughout)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution = nn.utils.spectral_norm(currentConvolution, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
                    self.layers.append(currentConvolution)
                    self.inputDimensions.append([augmentedChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(False)
                    self.intermediateRepresentationShapes.append(None)
                    self.intermediateRepresentationIndices.append(None)
                    if useBatchNorm:
                        self.layers.append(batchNorm(num_features = currentNumberOfChannels))
                        self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                        self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                        self.intermediateRepresentationFlag.append(False)
                        self.intermediateRepresentationShapes.append(None)
                        self.intermediateRepresentationIndices.append(None)
            else:
                raise ValueError('Unknown residual mode: {}'.format(residualMode))
            
            for intermediateComputationIndex in range(numberOfIntermediateComputationLayers):
                if residualMode == 'none':
                    self.layers.append(nn.LeakyReLU())
                    self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(False)
                    self.intermediateRepresentationShapes.append(None)
                    self.intermediateRepresentationIndices.append(None)
                    if useDepthWiseSeparableConvolutions:
                        currentConvolution = dsConvolution(in_channels = currentNumberOfChannels, out_channels = currentNumberOfChannels, kernel_size = kernelSize, stride = 1, padding = padding, bias = useBiasThroughout)
                        if spectralNormalisationSettings['useSpectralNormalisation']:
                            currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                    else:
                        currentConvolution = convolution(in_channels = currentNumberOfChannels, out_channels = currentNumberOfChannels, kernel_size = kernelSize, stride = 1, padding = padding, bias = useBiasThroughout)
                        if spectralNormalisationSettings['useSpectralNormalisation']:
                            currentConvolution = nn.utils.spectral_norm(currentConvolution, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
                    self.layers.append(currentConvolution)
                    self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(False)
                    self.intermediateRepresentationShapes.append(None)
                    self.intermediateRepresentationIndices.append(None)
                    if useBatchNorm:
                        self.layers.append(batchNorm(num_features = currentNumberOfChannels))
                        self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                        self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                        self.intermediateRepresentationFlag.append(False)
                        self.intermediateRepresentationShapes.append(None)
                        self.intermediateRepresentationIndices.append(None)
                elif residualMode == 'depthwiseSeparable':
                    currentConvolution = dsResidualBlock(in_channels = currentNumberOfChannels, internalDimension = currentNumberOfChannels*4, kernel_size = kernelSize, bias = useBiasThroughout, useBatchNorm = useBatchNorm)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                    self.layers.append(currentConvolution)
                    self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(False)
                    self.intermediateRepresentationShapes.append(None)
                    self.intermediateRepresentationIndices.append(None)
                elif residualMode == 'residual':
                    currentConvolution = residualBlock(in_channels = currentNumberOfChannels, internalDimension = currentNumberOfChannels, kernel_size = kernelSize, bias = useBiasThroughout, useBatchNorm = useBatchNorm)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                    self.layers.append(currentConvolution)
                    self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(False)
                    self.intermediateRepresentationShapes.append(None)
                    self.intermediateRepresentationIndices.append(None)
                elif residualMode == 'residalX':
                    currentConvolution = residualXBlock(in_channels = currentNumberOfChannels, internalDimension = int(round(currentNumberOfChannels/2)), kernel_size = kernelSize, bias = useBiasThroughout, useBatchNorm = useBatchNorm)
                    if spectralNormalisationSettings['useSpectralNormalisation']:
                        currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
                    self.layers.append(currentConvolution)
                    self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                    self.intermediateRepresentationFlag.append(False)
                    self.intermediateRepresentationShapes.append(None)
                    self.intermediateRepresentationIndices.append(None)
                else:
                    raise ValueError('Unknown residual mode: {}'.format(residualMode))


            #Upscale
            newNumberOfChannels = int(round(upscalingChannelScaling*currentNumberOfChannels))
            if newNumberOfChannels % 8 > 0:
                newNumberOfChannels += 8 - (newNumberOfChannels % 8) # Make sure that number of channels is divisible by 8 (for cuDNN)
            if useDepthWiseSeparableConvolutions:
                currentUpscale = dsTransposeConvolution(in_channels = currentNumberOfChannels, out_channels = newNumberOfChannels, kernel_size = kernelSize, stride = 2, 
                    padding = padding, output_padding = 1, bias = useBiasThroughout)
                if spectralNormalisationSettings['useSpectralNormalisation']:
                    currentUpscale.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
            else:
                currentUpscale = transposeConvolution(in_channels = currentNumberOfChannels, out_channels = newNumberOfChannels, kernel_size = kernelSize, stride = 2, 
                    padding = padding, output_padding = 1, bias = useBiasThroughout)
                if spectralNormalisationSettings['useSpectralNormalisation']:
                    currentUpscale = nn.utils.spectral_norm(currentUpscale, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
            self.layers.append(currentUpscale)
            self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
            currentSpatialDimensions = [2*x for x in currentSpatialDimensions]
            currentNumberOfChannels = newNumberOfChannels
            self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
            self.intermediateRepresentationFlag.append(False)
            self.intermediateRepresentationShapes.append(None)
            self.intermediateRepresentationIndices.append(None)  
            if useBatchNorm:
                self.layers.append(batchNorm(num_features = currentNumberOfChannels))
                self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
                self.intermediateRepresentationFlag.append(False)
                self.intermediateRepresentationShapes.append(None)
                self.intermediateRepresentationIndices.append(None)

        self.layers.append(nn.LeakyReLU())
        self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
        self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)  
        self.intermediateRepresentationFlag.append(False)
        self.intermediateRepresentationShapes.append(None)
        self.intermediateRepresentationIndices.append(None)
        newNumberOfChannels = outputDimension*4
        if useDepthWiseSeparableConvolutions:
            currentConvolution = dsConvolution(in_channels = currentNumberOfChannels, out_channels = newNumberOfChannels, kernel_size = kernelSize, stride = 1, padding = padding, bias = useBiasThroughout)
            if spectralNormalisationSettings['useSpectralNormalisation']:
                currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
        else:
            currentConvolution = convolution(in_channels = currentNumberOfChannels, out_channels = newNumberOfChannels, kernel_size = kernelSize, stride = 1, padding = padding, bias = useBiasThroughout)
            if spectralNormalisationSettings['useSpectralNormalisation']:
                currentConvolution = nn.utils.spectral_norm(currentConvolution, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
        self.layers.append(currentConvolution)
        self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
        currentNumberOfChannels = newNumberOfChannels
        self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
        self.intermediateRepresentationFlag.append(False)
        self.intermediateRepresentationShapes.append(None)
        self.intermediateRepresentationIndices.append(None)

        self.layers.append(nn.LeakyReLU())
        self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
        self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)  
        self.intermediateRepresentationFlag.append(False)
        self.intermediateRepresentationShapes.append(None)
        self.intermediateRepresentationIndices.append(None)
        newNumberOfChannels = outputDimension
        if useDepthWiseSeparableConvolutions:
            # Don't regularise final layer
            currentConvolution = dsConvolution(in_channels = currentNumberOfChannels, out_channels = newNumberOfChannels, kernel_size = 1, stride = 1, padding = 0, bias = True)
            # if spectralNormalisationSettings['useSpectralNormalisation']:
            #     currentConvolution.applySpectralNorm(spectralNormalisationSettings['numberOfPowerIterations'])
        else:
            # Don't regularise final layer
            currentConvolution = convolution(in_channels = currentNumberOfChannels, out_channels = newNumberOfChannels, kernel_size = 1, stride = 1, padding = 0, bias = True)
            # if spectralNormalisationSettings['useSpectralNormalisation']:
            #     currentConvolution = nn.utils.spectral_norm(currentConvolution, n_power_iterations = spectralNormalisationSettings['numberOfPowerIterations'])
        self.layers.append(currentConvolution)
        self.inputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
        currentNumberOfChannels = newNumberOfChannels
        self.outputDimensions.append([currentNumberOfChannels,]+currentSpatialDimensions)
        self.intermediateRepresentationFlag.append(False)
        self.intermediateRepresentationShapes.append(None)
        self.intermediateRepresentationIndices.append(None)
        
    def forward(self, x, intermediateRepresentations = None):
        x = x.reshape(*x.shape, *[1 for x in self.initialSpatialDimensions]).expand(*x.shape, *self.initialSpatialDimensions)
        if self.coordinates is not None:
            x = torch.cat([x, self.coordinates[None,:].repeat(x.shape[0],*[1 for x in x.shape[1:]])], dim = 1)
        for layer, inputDimensions, outputDimensions, intermediateRepresentationFlag, intermediateRepresentationShape, intermediateRepresentationIndex in zip(
                self.layers, self.inputDimensions, self.outputDimensions, self.intermediateRepresentationFlag, self.intermediateRepresentationShapes,
                self.intermediateRepresentationIndices):
            if intermediateRepresentationFlag:
                intermediateRepresentation = intermediateRepresentations[intermediateRepresentationIndex]
                augmentedRepresentation = torch.cat([x, intermediateRepresentation], dim = 1)
                assert (list(augmentedRepresentation.shape[1:]) == inputDimensions)
                x = augmentedRepresentation
            x = layer(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda')
    batchSize = 4
    voxelChannels = 14
    voxelSpatialDimensions = [64,64,32]
    voxelEncoderProjectionDimension = 64
    voxelEncoderDownscaling = 4
    voxelEncodingDimension = 512
    voxelEncoderResidualMode = 'depthwiseSeparable'
    voxelEncoderUseDepthWiseSeparableConvolutions = True
    voxelEncoderSpectralNormalisationSettings = dict(useSpectralNormalisation = True, numberOfPowerIterations = 5)
    encoder = Encoder(inputDimensions = [voxelChannels]+voxelSpatialDimensions, initialProjectionDimension = voxelEncoderProjectionDimension, 
        numberOfDownscalingOperations = voxelEncoderDownscaling, representationDimension = voxelEncodingDimension, 
        residualMode = voxelEncoderResidualMode, useDepthWiseSeparableConvolutions = voxelEncoderUseDepthWiseSeparableConvolutions,
        spectralNormalisationSettings = voxelEncoderSpectralNormalisationSettings)
    encoder = encoder.to(device = device)
    encoderInput = torch.randn(batchSize, voxelChannels, *voxelSpatialDimensions, device = device, requires_grad = True)
    encoderOutput, encoderIntermediateRepresentations = encoder(encoderInput)
    assert list(encoderOutput.shape) == [batchSize, voxelEncodingDimension]
    print('Encoder output mean and standard deviation are {} and {}.'.format(encoderOutput.mean(), encoderOutput.std()))

    voxelDecoderInitialSpatialDimensions = [4,4,2]
    voxelDecoderInitialProjectionDimension = voxelEncodingDimension*2
    voxelDecoderOutputDimension = voxelChannels
    voxelDecoderNumberOfUpscalingOperations = voxelEncoderDownscaling
    voxelDecoderAddCoordinatesToRepresentation = True
    voxelDecoderUseTrigonometricCoordinateEmbedding = True
    voxelDecoderResidualMode = 'depthwiseSeparable'
    voxelDecoderUseDepthWiseSeparableConvolutions = True
    voxelDecoderSpectralNormalisationSettings = dict(useSpectralNormalisation = True, numberOfPowerIterations = 5)
    decoder = Decoder(representationDimension = voxelEncodingDimension, initialSpatialDimensions = voxelDecoderInitialSpatialDimensions, 
        initialProjectionDimension = voxelDecoderInitialProjectionDimension, outputDimension = voxelDecoderOutputDimension, 
        numberOfUpscalingOperations = voxelDecoderNumberOfUpscalingOperations, 
        skipConnectionDimensions = encoder.intermediateRepresentationShapes, addCoordinatesToRepresentation = voxelDecoderAddCoordinatesToRepresentation, 
        useTrigonometricCoordinateEmbedding = voxelDecoderUseTrigonometricCoordinateEmbedding,
        residualMode = voxelDecoderResidualMode, useDepthWiseSeparableConvolutions = voxelDecoderUseDepthWiseSeparableConvolutions,
        spectralNormalisationSettings = voxelDecoderSpectralNormalisationSettings)
    decoder = decoder.to(device = device)
    decoderOutput = decoder(encoderOutput, encoderIntermediateRepresentations)
    assert list(decoderOutput.shape) == [batchSize, voxelChannels, *voxelSpatialDimensions]
    print('Decoder output mean and standard deviation are {} and {}.'.format(decoderOutput.mean(), decoderOutput.std()))

