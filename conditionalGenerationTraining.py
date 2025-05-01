import torch
from torch import nn
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import networkArchitectures
import numpy
import hdf5storage
import os
import sys
import datetime
import voxelVisualisationPyplot
import voxelVisualisation
import voxelVisualisationPytorch3D
import dataSets
import time
import conditionalTrainingConfiguration as configuration
import stabilityEstimationTrainingConfiguration as stabilityConfiguration
import shutil



def setRequiresGrad(module, value):
    for parameter in module.parameters():
        parameter.requires_grad_(value)

def calculateAbsoluteGradientStatistics(moduleParameters):
    averageGrad = torch.stack([x.grad.abs().mean() for x in moduleParameters if hasattr(x,'grad') and x.grad is not None])
    maxGrad = torch.stack([x.grad.abs().max() for x in moduleParameters if hasattr(x,'grad') and x.grad is not None])
    minGrad = torch.stack([x.grad.abs().min() for x in moduleParameters if hasattr(x,'grad') and x.grad is not None])
    return averageGrad, maxGrad, minGrad

def hingeLoss(target, predicted):
    return (1 - target*predicted).clamp_min(0)

def wassersteinLoss(target, predicted):
    return -target*predicted

def klDivergence( logProbabilities1, logProbabilities2):
    divergence = logProbabilities1.exp()*(logProbabilities1 - logProbabilities2)
    return divergence

def jsDivergence( logProbabilities1, logProbabilities2):
    probabilityDifference = logProbabilities1.exp() - logProbabilities2.exp()
    divergence = (probabilityDifference*logProbabilities1 - probabilityDifference*logProbabilities2)/2
    return divergence.sum(dim = 1)

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

def trainDiscriminator( generatorEncoder, generatorDecoder, discriminator, stabilityEstimator, bootstrapEncoder, bootstrapDecoder, 
                        discriminatorDataloader, discriminatorDataIterator1, discriminatorDataIterator2,
                        idealBackground, device, calculateGradientStatistics, discriminatorOptimiser,
                        encoderConfig, decoderConfig, generatorConfig, discriminatorConfig, dataConfig, trainingConfig, stabilityTrainingConfig, 
                        bootstrapEncoderConfig, bootstrapDecoderConfig, bootstrapGeneratorConfig):
    discriminatorData1, discriminatorDataIterator1 = getDataBatch(discriminatorDataloader, discriminatorDataIterator1)
    with torch.no_grad():
        if dataConfig.voxelEmbeddingType == 'surfaceVoxelEmbedding':
            partialScenes = discriminatorData1['surfaceVoxelEmbedding']
        elif dataConfig.voxelEmbeddingType == 'projectionVoxelEmbedding':
            partialScenes = discriminatorData1['projectionVoxelEmbedding']
        partialScenes = partialScenes.to(dtype = torch.float, device = device)

        if generatorConfig.useBootstrap:
            if bootstrapEncoderConfig.stochasticRepresentationDimension > 0:
                bootstrapEncoderStochasticInput = torch.randn(trainingConfig.batchSize, bootstrapEncoderConfig.stochasticRepresentationDimension, 1, 1, 1, device = device, requires_grad = False)
                bootstrapEncoderInput = torch.cat([partialScenes, bootstrapEncoderStochasticInput.expand(-1,-1,*partialScenes.shape[2:])], dim = 1)
                del bootstrapEncoderStochasticInput
            else:
                bootstrapEncoderInput = partialScenes
            if bootstrapGeneratorConfig.useSkipConnections:
                bootstrapPartialSceneEncoding, skipConnections = bootstrapEncoder(bootstrapEncoderInput, returnIntermediateRepresentations = True)
            else:
                bootstrapPartialSceneEncoding, _ = bootstrapEncoder(bootstrapEncoderInput, returnIntermediateRepresentations = False)
                del bootstrapEncoderInput
            if bootstrapDecoderConfig.stochasticRepresentationDimension > 0:
                bootstrapDecoderStochasticInput = torch.randn(trainingConfig.batchSize, bootstrapDecoderConfig.stochasticRepresentationDimension, device = device, requires_grad = False)
                bootstrapDecoderInput = torch.cat([bootstrapPartialSceneEncoding, bootstrapDecoderStochasticInput],dim = 1)
                del bootstrapDecoderStochasticInput
            else:
                bootstrapDecoderInput = bootstrapPartialSceneEncoding
            del bootstrapPartialSceneEncoding
            if bootstrapGeneratorConfig.useSkipConnections:
                bootstrapGeneratedScenes = bootstrapDecoder(bootstrapDecoderInput, skipConnections)
                del skipConnections
            else:
                bootstrapGeneratedScenes = generatorDecoder(bootstrapDecoderInput)
            del bootstrapDecoderInput
            if not bootstrapGeneratorConfig.reconstructBackground:
                bootstrapGeneratedScenes = torch.cat([bootstrapGeneratedScenes, idealBackground[None,None,:,:,:].expand(trainingConfig.batchSize,-1,-1,-1,-1)], dim = 1)

        if encoderConfig.stochasticRepresentationDimension > 0:
            encoderStochasticInput = torch.randn(trainingConfig.batchSize, encoderConfig.stochasticRepresentationDimension, 1, 1, 1, device = device, requires_grad = False)
            encoderInputList = [partialScenes, encoderStochasticInput.expand(-1,-1,*partialScenes.shape[2:])]
            if generatorConfig.useBootstrap:
                encoderInputList.append(bootstrapGeneratedScenes)
                if not generatorConfig.bootstrapResidual:
                    del bootstrapGeneratedScenes
            encoderInput = torch.cat(encoderInputList, dim = 1)
            del encoderStochasticInput
        else:
            encoderInput = partialScenes
        if generatorConfig.useSkipConnections:
            partialSceneEncoding, skipConnections = generatorEncoder(encoderInput, returnIntermediateRepresentations = True)
        else:
            partialSceneEncoding, _ = generatorEncoder(encoderInput, returnIntermediateRepresentations = False)
            del encoderInput
        if decoderConfig.stochasticRepresentationDimension > 0:
            decoderStochasticInput = torch.randn(trainingConfig.batchSize, decoderConfig.stochasticRepresentationDimension, device = device, requires_grad = False)
            decoderInput = torch.cat([partialSceneEncoding, decoderStochasticInput],dim = 1)
            del decoderStochasticInput
        else:
            decoderInput = partialSceneEncoding
        del partialSceneEncoding
        if generatorConfig.useSkipConnections:
            generatedScenes = generatorDecoder(decoderInput, skipConnections)
            del skipConnections
        else:
            generatedScenes = generatorDecoder(decoderInput)
        del decoderInput
        if generatorConfig.useBootstrap and generatorConfig.bootstrapResidual:
            if generatorConfig.reconstructBackground:
                generatedScenes = generatedScenes + bootstrapGeneratedScenes
            else:
                generatedScenes = generatedScenes + bootstrapGeneratedScenes[:,:-1,:,:,:]
                generatedScenes = torch.cat([generatedScenes, idealBackground[None,None,:,:,:].expand(trainingConfig.batchSize,-1,-1,-1,-1)], dim = 1)
            del bootstrapGeneratedScenes
        else:
            if not generatorConfig.reconstructBackground:
                generatedScenes = torch.cat([generatedScenes, idealBackground[None,None,:,:,:].expand(trainingConfig.batchSize,-1,-1,-1,-1)], dim = 1)
        if trainingConfig.useStability:
            stabilityConditioning, _ = stabilityEstimator(generatedScenes)
            if stabilityTrainingConfig.predictionScale == 'scene':
                stabilityConditioning = stabilityConditioning.softmax(dim = 1)
            elif stabilityTrainingConfig.predictionScale == 'object':
                stabilityConditioning = stabilityConditioning.reshape(trainingConfig.batchSize, -1, stabilityTrainingConfig.numberOfQuantisationBins)
                stabilityConditioning = stabilityConditioning.softmax(dim = 2)
                stabilityConditioning = stabilityConditioning.reshape(trainingConfig.batchSize, -1)
            else:
                raise ValueError('Unknown stability estimation scale: {}'.format(stabilityTrainingConfig.predictionScale))
            stabilityConditioning = stabilityConditioning[:,:,None,None,None].expand(-1,-1,*generatedScenes.shape[2:]) - 0.5 #mean zero, in interval [-0.5,0.5]
            generatedInput = torch.cat([partialScenes, generatedScenes, stabilityConditioning], dim = 1)
        else:
            generatedInput = torch.cat([partialScenes, generatedScenes], dim = 1)
    del partialScenes, generatedScenes
    generatedInput = generatedInput.requires_grad_(True)
    if trainingConfig.discriminatorInputNoise > 0:
        generatedInput = generatedInput + trainingConfig.discriminatorInputNoise*torch.randn_like(generatedInput)
    generatedCritique, _ = discriminator(generatedInput)
    if not trainingConfig.useGradientPenalty:
        del generatedInput
    discriminatorData2, discriminatorDataIterator2 = getDataBatch(discriminatorDataloader, discriminatorDataIterator2)
    if dataConfig.voxelEmbeddingType == 'surfaceVoxelEmbedding':
        partialScenes = discriminatorData2['surfaceVoxelEmbedding']
    elif dataConfig.voxelEmbeddingType == 'projectionVoxelEmbedding':
        partialScenes = discriminatorData2['projectionVoxelEmbedding']
    realScenes = discriminatorData2['completeVoxelScene']
    if trainingConfig.useStability:
        with torch.no_grad():
            stabilityConditioning, _ = stabilityEstimator(realScenes.to(dtype = torch.float, device = device))
            if stabilityTrainingConfig.predictionScale == 'scene':
                stabilityConditioning = stabilityConditioning.softmax(dim = 1)
            elif stabilityTrainingConfig.predictionScale == 'object':
                stabilityConditioning = stabilityConditioning.reshape(trainingConfig.batchSize, -1, stabilityTrainingConfig.numberOfQuantisationBins)
                stabilityConditioning = stabilityConditioning.softmax(dim = 2)
                stabilityConditioning = stabilityConditioning.reshape(trainingConfig.batchSize, -1)
            else:
                raise ValueError('Unknown stability estimation scale: {}'.format(stabilityTrainingConfig.predictionScale))
        stabilityConditioning = stabilityConditioning[:,:,None,None,None].expand(-1,-1,*realScenes.shape[2:]) - 0.5
        realInput = torch.cat([partialScenes.to(dtype = torch.float, device = device), realScenes.to(dtype = torch.float, device = device), stabilityConditioning], dim = 1)
    else:
        realInput = torch.cat([partialScenes, realScenes], dim = 1)
        realInput = realInput.to(dtype = torch.float, device = device)
    realInput = realInput.requires_grad_(True)
    if trainingConfig.discriminatorInputNoise > 0:
        realInput = realInput + trainingConfig.discriminatorInputNoise*torch.randn_like(realInput)
    realCritique, _ = discriminator(realInput)
    if not trainingConfig.useGradientPenalty:
        del realInput
    if trainingConfig.lossType == 'hinge':
        discriminatorRealLoss = hingeLoss(1, realCritique)
        discriminatorGeneratedLoss = hingeLoss(-1, generatedCritique)
    elif trainingConfig.lossType == 'wasserstein':
        discriminatorRealLoss = wassersteinLoss(1, realCritique)
        discriminatorGeneratedLoss = wassersteinLoss(-1, generatedCritique)
    else:
        raise ValueError('Unknown loss type: {}'.format(trainingConfig.lossType))
    realCritique = realCritique.detach()
    generatedCritique = generatedCritique.detach()
    discriminatorLoss = discriminatorGeneratedLoss + discriminatorRealLoss
    discriminatorGeneratedLoss = discriminatorGeneratedLoss.detach()
    discriminatorRealLoss = discriminatorRealLoss.detach()
    if trainingConfig.useGradientPenalty:
        gradientPenaltyTime = -time.time()
        gradientPenaltyLoss = 10*calculateInterpolationGradientPenalty(generatedInput, realInput, discriminator)
        gradientPenaltyTime += time.time()
        del generatedInput, realInput
    else:
        gradientPenaltyLoss = torch.zeros(*discriminatorLoss.shape, device = device)
        gradientPenaltyTime = 0
    discriminatorTotalLoss = discriminatorLoss + gradientPenaltyLoss
    discriminatorBackpropagationTime = -time.time()
    discriminatorOptimiser.zero_grad()
    discriminatorTotalLoss.mean().backward()
    discriminatorBackpropagationTime += time.time()
    discriminatorLoss = discriminatorLoss.detach()
    gradientPenaltyLoss = gradientPenaltyLoss.detach()
    discriminatorTotalLoss = discriminatorTotalLoss.detach()
    if calculateGradientStatistics:
        with torch.no_grad():
            discriminatorMeanGradients, discriminatorMaxGradients, discriminatorMinGradients = calculateAbsoluteGradientStatistics(
                list(discriminator.parameters()))
            discriminatorMeanGradients = discriminatorMeanGradients.detach().cpu()
            discriminatorMaxGradients = discriminatorMaxGradients.detach().cpu()
            discriminatorMinGradients = discriminatorMinGradients.detach().cpu()
    else:
        discriminatorMeanGradients = discriminatorMaxGradients = discriminatorMinGradients = torch.as_tensor(0)
    discriminatorOptimiser.step()
    output = dict(  
                discriminatorTotalLoss = discriminatorTotalLoss.mean().item(),
                discriminatorLoss = discriminatorLoss.mean().item(),
                discriminatorRealLoss = discriminatorRealLoss.mean().item(),
                discriminatorGeneratedLoss = discriminatorGeneratedLoss.mean().item(),
                gradientPenaltyLoss = gradientPenaltyLoss.mean().item(),
                discriminatorRealCritique = realCritique.detach().cpu(),
                discriminatorGeneratedCritique = generatedCritique.detach().cpu(),
                discriminatorMeanGradients = discriminatorMeanGradients,
                discriminatorMaxGradients = discriminatorMaxGradients,
                discriminatorMinGradients = discriminatorMinGradients,
                discriminatorBackpropagationTime = discriminatorBackpropagationTime,
                gradientPenaltyTime = gradientPenaltyTime
                )
    return output, discriminatorDataIterator1, discriminatorDataIterator2

def trainRegression( generatorEncoder, generatorDecoder, stabilityEstimator, bootstrapEncoder, bootstrapDecoder, 
                    generatorDataloader, generatorDataIterator, 
                    idealBackground, device, calculateGradientStatistics, generatorOptimiser,
                    encoderConfig, decoderConfig, generatorConfig, dataConfig, trainingConfig, stabilityTrainingConfig, 
                    bootstrapEncoderConfig, bootstrapDecoderConfig, bootstrapGeneratorConfig ):
    generatorData, generatorDataIterator = getDataBatch(generatorDataloader, generatorDataIterator)
    if dataConfig.voxelEmbeddingType == 'surfaceVoxelEmbedding':
        partialScenes = generatorData['surfaceVoxelEmbedding']
    elif dataConfig.voxelEmbeddingType == 'projectionVoxelEmbedding':
        partialScenes = generatorData['projectionVoxelEmbedding']
    partialScenes = partialScenes.to(dtype = torch.float, device = device)

    with torch.no_grad():
        if generatorConfig.useBootstrap:
            if bootstrapEncoderConfig.stochasticRepresentationDimension > 0:
                bootstrapEncoderStochasticInput = torch.randn(trainingConfig.batchSize, bootstrapEncoderConfig.stochasticRepresentationDimension, 1, 1, 1, device = device, requires_grad = False)
                bootstrapEncoderInput = torch.cat([partialScenes, bootstrapEncoderStochasticInput.expand(-1,-1,*partialScenes.shape[2:])], dim = 1)
                del bootstrapEncoderStochasticInput
            else:
                bootstrapEncoderInput = partialScenes
            if bootstrapGeneratorConfig.useSkipConnections:
                bootstrapPartialSceneEncoding, skipConnections = bootstrapEncoder(bootstrapEncoderInput, returnIntermediateRepresentations = True)
            else:
                bootstrapPartialSceneEncoding, _ = bootstrapEncoder(bootstrapEncoderInput, returnIntermediateRepresentations = False)
                del bootstrapEncoderInput
            if bootstrapDecoderConfig.stochasticRepresentationDimension > 0:
                bootstrapDecoderStochasticInput = torch.randn(trainingConfig.batchSize, bootstrapDecoderConfig.stochasticRepresentationDimension, device = device, requires_grad = False)
                bootstrapDecoderInput = torch.cat([bootstrapPartialSceneEncoding, bootstrapDecoderStochasticInput],dim = 1)
                del bootstrapDecoderStochasticInput
            else:
                bootstrapDecoderInput = bootstrapPartialSceneEncoding
            del bootstrapPartialSceneEncoding
            if bootstrapGeneratorConfig.useSkipConnections:
                bootstrapGeneratedScenes = bootstrapDecoder(bootstrapDecoderInput, skipConnections)
                del skipConnections
            else:
                bootstrapGeneratedScenes = generatorDecoder(bootstrapDecoderInput)
            del bootstrapDecoderInput
            if not bootstrapGeneratorConfig.reconstructBackground:
                bootstrapGeneratedScenes = torch.cat([bootstrapGeneratedScenes, idealBackground[None,None,:,:,:].expand(trainingConfig.batchSize,-1,-1,-1,-1)], dim = 1)

    partialScenes = partialScenes.requires_grad_(True)
    realScenes = generatorData['completeVoxelScene'].to(dtype = torch.float, device = device)
    if encoderConfig.stochasticRepresentationDimension > 0:
        encoderStochasticInput = torch.randn(trainingConfig.batchSize, encoderConfig.stochasticRepresentationDimension, 1, 1, 1, device = device, requires_grad = True)
        encoderInputList = [partialScenes, encoderStochasticInput.expand(-1,-1,*partialScenes.shape[2:])]
        if generatorConfig.useBootstrap:
            encoderInputList.append(bootstrapGeneratedScenes)
            if not generatorConfig.bootstrapResidual:
                del bootstrapGeneratedScenes
        encoderInput = torch.cat(encoderInputList, dim = 1)
        del encoderStochasticInput
    else:
        encoderInput = partialScenes
    if generatorConfig.useSkipConnections:
        partialSceneEncoding, skipConnections = generatorEncoder(encoderInput, returnIntermediateRepresentations = True)
    else:
        partialSceneEncoding, _ = generatorEncoder(encoderInput, returnIntermediateRepresentations = False)
    del encoderInput
    if decoderConfig.stochasticRepresentationDimension > 0:
        decoderStochasticInput = torch.randn(trainingConfig.batchSize, decoderConfig.stochasticRepresentationDimension, device = device, requires_grad = True)
        decoderInput = torch.cat([partialSceneEncoding, decoderStochasticInput],dim = 1)
        del decoderStochasticInput
    else:
        decoderInput = partialSceneEncoding
    del partialSceneEncoding
    if generatorConfig.useSkipConnections:
        generatedScenes = generatorDecoder(decoderInput, skipConnections)
        del skipConnections
    else:
        generatedScenes = generatorDecoder(decoderInput)
    del decoderInput
    l1ResidualPenalty = torch.as_tensor(0.0, dtype = torch.float, device = device)
    if generatorConfig.useBootstrap and generatorConfig.bootstrapResidual:
        if trainingConfig.l1ResidualPenaltyWeight > 0:
            l1ResidualPenalty = trainingConfig.l1ResidualPenaltyWeight*generatedScenes.abs().mean()
        if generatorConfig.reconstructBackground:
            generatedScenes = generatedScenes + bootstrapGeneratedScenes
        else:
            generatedScenes = generatedScenes + bootstrapGeneratedScenes[:,:-1,:,:,:]
            generatedScenes = torch.cat([generatedScenes, idealBackground[None,None,:,:,:].expand(trainingConfig.batchSize,-1,-1,-1,-1)], dim = 1)
        del bootstrapGeneratedScenes
    else:
        if not generatorConfig.reconstructBackground:
            generatedScenes = torch.cat([generatedScenes, idealBackground[None,None,:,:,:].expand(trainingConfig.batchSize,-1,-1,-1,-1)], dim = 1)
    voxelLoss = (realScenes - generatedScenes).pow(2).mean()
    if trainingConfig.useStability:
        realStability, _ = stabilityEstimator(realScenes)
        generatedStability, _ = stabilityEstimator(generatedScenes)
        if stabilityTrainingConfig.predictionScale == 'object':
            realStability = realStability.reshape(-1, stabilityTrainingConfig.numberOfQuantisationBins)
            generatedStability = generatedStability.reshape(-1, stabilityTrainingConfig.numberOfQuantisationBins)
        realStability = log_softmax(realStability, dim = 1)
        generatedStability = log_softmax(generatedStability, dim = 1)
        stabilityLoss = jsDivergence(realStability, generatedStability).mean()
        del realStability, generatedStability
    else:
        stabilityLoss = torch.as_tensor(0, device = device)
    del realScenes, generatedScenes
    generatorLoss = voxelLoss + stabilityLoss + l1ResidualPenalty
    generatorBackpropagationTime = -time.time()
    generatorOptimiser.zero_grad()
    generatorLoss.backward()
    generatorBackpropagationTime += time.time()
    generatorLoss = generatorLoss.detach()
    if iterationNumber % distributionLoggingInterval == 0:
        with torch.no_grad():
            generatorMeanGradients, generatorMaxGradients, generatorMinGradients = calculateAbsoluteGradientStatistics(
                list(generatorEncoder.parameters())+list(generatorDecoder.parameters()))
            generatorMeanGradients = generatorMeanGradients.detach().cpu()
            generatorMaxGradients = generatorMaxGradients.detach().cpu()
            generatorMinGradients = generatorMinGradients.detach().cpu()
    else:
        generatorMeanGradients = generatorMaxGradients = generatorMinGradients = torch.as_tensor(0)
    generatorOptimiser.step()
    output = dict(  l1ResidualPenalty = l1ResidualPenalty.mean().item(),
                    voxelLoss = voxelLoss.item(),
                    stabilityLoss = stabilityLoss.item(),
                    generatorLoss = generatorLoss.item(),
                    generatorMeanGradients = generatorMeanGradients,
                    generatorMaxGradients = generatorMaxGradients,
                    generatorMinGradients = generatorMinGradients,
                    generatorBackpropagationTime = generatorBackpropagationTime
                )
    return output, generatorDataIterator

def trainGenerator( generatorEncoder, generatorDecoder, discriminator, stabilityEstimator, bootstrapEncoder, bootstrapDecoder, 
                    generatorDataloader, generatorDataIterator,
                    idealBackground, device, calculateGradientStatistics, generatorOptimiser,
                    encoderConfig, decoderConfig, generatorConfig, dataConfig, trainingConfig, stabilityTrainingConfig,
                    bootstrapEncoderConfig, bootstrapDecoderConfig, bootstrapGeneratorConfig):
    generatorData, generatorDataIterator = getDataBatch(generatorDataloader, generatorDataIterator)
    if dataConfig.voxelEmbeddingType == 'surfaceVoxelEmbedding':
        partialScenes = generatorData['surfaceVoxelEmbedding']
    elif dataConfig.voxelEmbeddingType == 'projectionVoxelEmbedding':
        partialScenes = generatorData['projectionVoxelEmbedding']
    partialScenes = partialScenes.to(dtype = torch.float, device = device)

    with torch.no_grad():
        if generatorConfig.useBootstrap:
            if bootstrapEncoderConfig.stochasticRepresentationDimension > 0:
                bootstrapEncoderStochasticInput = torch.randn(trainingConfig.batchSize, bootstrapEncoderConfig.stochasticRepresentationDimension, 1, 1, 1, device = device, requires_grad = False)
                bootstrapEncoderInput = torch.cat([partialScenes, bootstrapEncoderStochasticInput.expand(-1,-1,*partialScenes.shape[2:])], dim = 1)
                del bootstrapEncoderStochasticInput
            else:
                bootstrapEncoderInput = partialScenes
            if bootstrapGeneratorConfig.useSkipConnections:
                bootstrapPartialSceneEncoding, skipConnections = bootstrapEncoder(bootstrapEncoderInput, returnIntermediateRepresentations = True)
            else:
                bootstrapPartialSceneEncoding, _ = bootstrapEncoder(bootstrapEncoderInput, returnIntermediateRepresentations = False)
                del bootstrapEncoderInput
            if bootstrapDecoderConfig.stochasticRepresentationDimension > 0:
                bootstrapDecoderStochasticInput = torch.randn(trainingConfig.batchSize, bootstrapDecoderConfig.stochasticRepresentationDimension, device = device, requires_grad = False)
                bootstrapDecoderInput = torch.cat([bootstrapPartialSceneEncoding, bootstrapDecoderStochasticInput],dim = 1)
                del bootstrapDecoderStochasticInput
            else:
                bootstrapDecoderInput = bootstrapPartialSceneEncoding
            del bootstrapPartialSceneEncoding
            if bootstrapGeneratorConfig.useSkipConnections:
                bootstrapGeneratedScenes = bootstrapDecoder(bootstrapDecoderInput, skipConnections)
                del skipConnections
            else:
                bootstrapGeneratedScenes = generatorDecoder(bootstrapDecoderInput)
            del bootstrapDecoderInput
            if not bootstrapGeneratorConfig.reconstructBackground:
                bootstrapGeneratedScenes = torch.cat([bootstrapGeneratedScenes, idealBackground[None,None,:,:,:].expand(trainingConfig.batchSize,-1,-1,-1,-1)], dim = 1)

    partialScenes = partialScenes.requires_grad_(True)
    if encoderConfig.stochasticRepresentationDimension > 0:
        encoderStochasticInput = torch.randn(trainingConfig.batchSize, encoderConfig.stochasticRepresentationDimension, 1, 1, 1, device = device, requires_grad = True)
        encoderInputList = [partialScenes, encoderStochasticInput.expand(-1,-1,*partialScenes.shape[2:])]
        if generatorConfig.useBootstrap:
            encoderInputList.append(bootstrapGeneratedScenes)
            if not generatorConfig.bootstrapResidual:
                del bootstrapGeneratedScenes
        encoderInput = torch.cat(encoderInputList, dim = 1)
        del encoderStochasticInput
    else:
        encoderInput = partialScenes
    if generatorConfig.useSkipConnections:
        partialSceneEncoding, skipConnections = generatorEncoder(encoderInput, returnIntermediateRepresentations = True)
    else:
        partialSceneEncoding, _ = generatorEncoder(encoderInput, returnIntermediateRepresentations = False)
    del encoderInput
    if decoderConfig.stochasticRepresentationDimension > 0:
        decoderStochasticInput = torch.randn(trainingConfig.batchSize, decoderConfig.stochasticRepresentationDimension, device = device, requires_grad = True)
        decoderInput = torch.cat([partialSceneEncoding, decoderStochasticInput],dim = 1)
        del decoderStochasticInput
    else:
        decoderInput = partialSceneEncoding
    del partialSceneEncoding
    if generatorConfig.useSkipConnections:
        generatedScenes = generatorDecoder(decoderInput, skipConnections)
        del skipConnections
    else:
        generatedScenes = generatorDecoder(decoderInput)
    del decoderInput
    l1ResidualPenalty = torch.as_tensor(0.0, dtype = torch.float, device = device)
    if generatorConfig.useBootstrap and generatorConfig.bootstrapResidual:
        if trainingConfig.l1ResidualPenaltyWeight > 0:
            l1ResidualPenalty = trainingConfig.l1ResidualPenaltyWeight*generatedScenes.abs().mean()
        if generatorConfig.reconstructBackground:
            generatedScenes = generatedScenes + bootstrapGeneratedScenes
        else:
            generatedScenes = generatedScenes + bootstrapGeneratedScenes[:,:-1,:,:,:]
            generatedScenes = torch.cat([generatedScenes, idealBackground[None,None,:,:,:].expand(trainingConfig.batchSize,-1,-1,-1,-1)], dim = 1)
        del bootstrapGeneratedScenes
    else:
        if not generatorConfig.reconstructBackground:
            generatedScenes = torch.cat([generatedScenes, idealBackground[None,None,:,:,:].expand(trainingConfig.batchSize,-1,-1,-1,-1)], dim = 1)
    if trainingConfig.useStability:
        stabilityConditioning, _ = stabilityEstimator(generatedScenes)
        if stabilityTrainingConfig.predictionScale == 'scene':
            stabilityConditioning = stabilityConditioning.softmax(dim = 1)
        elif stabilityTrainingConfig.predictionScale == 'object':
            stabilityConditioning = stabilityConditioning.reshape(trainingConfig.batchSize, -1, stabilityTrainingConfig.numberOfQuantisationBins)
            stabilityConditioning = stabilityConditioning.softmax(dim = 2)
            stabilityConditioning = stabilityConditioning.reshape(trainingConfig.batchSize, -1)
        else:
            raise ValueError('Unknown stability estimation scale: {}'.format(stabilityTrainingConfig.predictionScale))
        stabilityConditioning = stabilityConditioning[:,:,None,None,None].expand(-1,-1,*generatedScenes.shape[2:])
        generatedInput = torch.cat([partialScenes, generatedScenes, stabilityConditioning], dim = 1)
    else:
        generatedInput = torch.cat([partialScenes, generatedScenes], dim = 1)

    del partialScenes, generatedScenes
    generatedCritique, _ = discriminator(generatedInput)
    del generatedInput
    if trainingConfig.lossType == 'hinge':
        discriminatorLoss = hingeLoss(1, generatedCritique)
    elif trainingConfig.lossType == 'wasserstein':
        discriminatorLoss = wassersteinLoss(1, generatedCritique)
    else:
        raise ValueError('Unknown loss type: {}'.format(trainingConfig.lossType))
    generatorLoss = discriminatorLoss + l1ResidualPenalty
    generatedCritique = generatedCritique.detach()
    generatorBackpropagationTime = -time.time()
    generatorOptimiser.zero_grad()
    generatorLoss.mean().backward()
    generatorBackpropagationTime += time.time()
    generatorLoss = generatorLoss.detach()
    if iterationNumber % distributionLoggingInterval == 0:
        with torch.no_grad():
            generatorMeanGradients, generatorMaxGradients, generatorMinGradients = calculateAbsoluteGradientStatistics(
                list(generatorEncoder.parameters())+list(generatorDecoder.parameters()))
            generatorMeanGradients = generatorMeanGradients.detach().cpu()
            generatorMaxGradients = generatorMaxGradients.detach().cpu()
            generatorMinGradients = generatorMinGradients.detach().cpu()
    else:
        generatorMeanGradients = generatorMaxGradients = generatorMinGradients = torch.as_tensor(0)
    generatorOptimiser.step()
    output = dict(  
                    l1ResidualPenalty = l1ResidualPenalty.mean().item(),
                    generatorLoss = generatorLoss.mean().item(),
                    generatedCritique = generatedCritique.mean().item(),
                    generatorMeanGradients = generatorMeanGradients,
                    generatorMaxGradients = generatorMaxGradients,
                    generatorMinGradients = generatorMinGradients,
                    generatorBackpropagationTime = generatorBackpropagationTime
                    )
    return output, generatorDataIterator

def visualisationAndLogging(  generatorEncoder, generatorDecoder, bootstrapEncoder, bootstrapDecoder, loggingPartialScenes, 
                        loggingEncoderLatents, loggingDecoderLatents, loggingBootstrapEncoderLatents, loggingBootstrapDecoderLatents, loggingCompleteScenes,
                        completeSceneMean, completeSceneStandardDeviation,
                        idealBackground, device, calculateGradientStatistics, logDirectory, logger,
                        logVoxels, logVisualisations, visualisationEngine, visualisationFormat, visualisationVideoFrameRate,
                        visualisationThreshold, maximumFillThreshold, visualisationAngles, voxelSize, xBinnedGridCoordinates, yBinnedGridCoordinates, zBinnedGridCoordinates, classColours,
                        iterationNumber, encoderConfig, decoderConfig, generatorConfig, bootstrapEncoderConfig, bootstrapDecoderConfig, bootstrapGeneratorConfig):
    voxelOutput = dict()

    # generatorEncoder.train()
    # generatorDecoder.train()
    # with torch.no_grad():
    #     if encoderConfig.stochasticRepresentationDimension > 0:
    #         encoderInput = torch.cat([loggingPartialScenes, loggingEncoderLatents.expand(-1,-1,*loggingPartialScenes.shape[2:])], dim = 1).to(device)
    #     else:
    #         encoderInput = loggingPartialScenes.to(device)
    #     if generatorConfig.useSkipConnections:
    #         partialSceneEncoding, skipConnections = generatorEncoder(encoderInput, returnIntermediateRepresentations = True)
    #     else:
    #         partialSceneEncoding, _ = generatorEncoder(encoderInput, returnIntermediateRepresentations = False)
    #     del encoderInput
    #     if decoderConfig.stochasticRepresentationDimension > 0:
    #         decoderInput = torch.cat([partialSceneEncoding, loggingDecoderLatents.to(device)], dim = 1)
    #     else:
    #         decoderInput = partialSceneEncoding
    #     del partialSceneEncoding
    #     if generatorConfig.useSkipConnections:
    #         generatedScenes = generatorDecoder(decoderInput, skipConnections)
    #         del skipConnections
    #     else:
    #         generatedScenes = generatorDecoder(decoderInput)
    #     del decoderInput
    #     if not generatorConfig.reconstructBackground:
    #         generatedScenes = torch.cat([generatedScenes, idealBackground[None,None,:,:,:].expand(generatedScenes.size(0),-1,-1,-1,-1)], dim = 1)
    #     trainVisualisationError = (generatedScenes - loggingCompleteScenes.to(device)).pow(2).sum(dim = list(range(1,generatedScenes.dim()))).sqrt().sum()
    #     trainGeneratedScenes = (generatedScenes*completeSceneStandardDeviation + completeSceneMean).cpu()
    #     del generatedScenes
    #     logger.add_scalar('Visualisation scenes/Train-mode error', scalar_value = trainVisualisationError.item(), global_step = iterationNumber)
    #     voxelOutput['trainModeScenes'] = trainGeneratedScenes.numpy().transpose(0,2,3,4,1)
        
    generatorEncoder.eval()
    generatorDecoder.eval()
    with torch.no_grad():
        
        if generatorConfig.useBootstrap:
            if bootstrapEncoderConfig.stochasticRepresentationDimension > 0:
                bootstrapEncoderInput = torch.cat([loggingPartialScenes, loggingBootstrapEncoderLatents.expand(-1,-1,*loggingPartialScenes.shape[2:])], dim = 1).to(device = device)
            else:
                bootstrapEncoderInput = loggingPartialScenes.to(device = device)
            if bootstrapGeneratorConfig.useSkipConnections:
                bootstrapPartialSceneEncoding, skipConnections = bootstrapEncoder(bootstrapEncoderInput, returnIntermediateRepresentations = True)
            else:
                bootstrapPartialSceneEncoding, _ = bootstrapEncoder(bootstrapEncoderInput, returnIntermediateRepresentations = False)
                del bootstrapEncoderInput
            if bootstrapDecoderConfig.stochasticRepresentationDimension > 0:
                bootstrapDecoderInput = torch.cat([bootstrapPartialSceneEncoding, loggingBootstrapDecoderLatents.to(device = device)],dim = 1)
            else:
                bootstrapDecoderInput = bootstrapPartialSceneEncoding
            del bootstrapPartialSceneEncoding
            if bootstrapGeneratorConfig.useSkipConnections:
                bootstrapGeneratedScenes = bootstrapDecoder(bootstrapDecoderInput, skipConnections)
                del skipConnections
            else:
                bootstrapGeneratedScenes = generatorDecoder(bootstrapDecoderInput)
            del bootstrapDecoderInput
            if not bootstrapGeneratorConfig.reconstructBackground:
                bootstrapGeneratedScenes = torch.cat([bootstrapGeneratedScenes, idealBackground[None,None,:,:,:].expand(bootstrapGeneratedScenes.size(0),-1,-1,-1,-1)], dim = 1)

        if encoderConfig.stochasticRepresentationDimension > 0:
            encoderInputList = [loggingPartialScenes.to(device = device), loggingEncoderLatents.expand(-1,-1,*loggingPartialScenes.shape[2:]).to(device = device)]
            if generatorConfig.useBootstrap:
                encoderInputList.append(bootstrapGeneratedScenes)
            encoderInput = torch.cat(encoderInputList, dim = 1).to(device = device)
        else:
            encoderInput = loggingPartialScenes.to(device = device)
        if generatorConfig.useSkipConnections:
            partialSceneEncoding, skipConnections = generatorEncoder(encoderInput, returnIntermediateRepresentations = True)
        else:
            partialSceneEncoding, _ = generatorEncoder(encoderInput, returnIntermediateRepresentations = False)
        del encoderInput
        if decoderConfig.stochasticRepresentationDimension > 0:
            decoderInput = torch.cat([partialSceneEncoding, loggingDecoderLatents.to(device = device)], dim = 1)
        else:
            decoderInput = partialSceneEncoding
        del partialSceneEncoding
        if generatorConfig.useSkipConnections:
            generatedScenes = generatorDecoder(decoderInput, skipConnections)
            del skipConnections
        else:
            generatedScenes = generatorDecoder(decoderInput)
        del decoderInput
        if generatorConfig.useBootstrap and generatorConfig.bootstrapResidual:
            if generatorConfig.reconstructBackground:
                generatedScenes = generatedScenes + bootstrapGeneratedScenes
            else:
                generatedScenes = generatedScenes + bootstrapGeneratedScenes[:,:-1,:,:,:]
                generatedScenes = torch.cat([generatedScenes, idealBackground[None,None,:,:,:].expand(generatedScenes.size(0),-1,-1,-1,-1)], dim = 1)
            del bootstrapGeneratedScenes
        else:
            if not generatorConfig.reconstructBackground:
                generatedScenes = torch.cat([generatedScenes, idealBackground[None,None,:,:,:].expand(generatedScenes.size(0),-1,-1,-1,-1)], dim = 1)
        evalVisualisationError = (generatedScenes - loggingCompleteScenes.to(device)).pow(2).sum(dim = list(range(1,generatedScenes.dim()))).sqrt().sum()
        evalGeneratedScenes = (generatedScenes*completeSceneStandardDeviation + completeSceneMean).cpu()
        del generatedScenes
        logger.add_scalar('Visualisation scenes/Eval-mode error', scalar_value = evalVisualisationError.item(), global_step = iterationNumber)
        voxelOutput['evalModeScenes'] = evalGeneratedScenes.numpy().transpose(0,2,3,4,1)

    if logVoxels:
        hdf5storage.savemat('{}/scenes_iteration_{}.mat'.format(logDirectory,iterationNumber), voxelOutput)
    if logVisualisations:
        # for sceneIndex, trainScene in enumerate(trainGeneratedScenes):
        #     if visualisationEngine == 'matplotlib':
        #         if visualisationType == 'voxelGrid':
        #             fig, ax = voxelVisualisationPyplot.createVoxelBlockFigure(trainScene.numpy(), visualisationThreshold, xBinnedGridCoordinates, yBinnedGridCoordinates, zBinnedGridCoordinates, classColours)
        #         elif visualisationType == 'isosurface':      
        #             assert not any([x < 2 for x in trainScene.shape[1:]])
        #             fig, ax = voxelVisualisationPyplot.createVoxelIsosurfaceFigure(trainScene.numpy(), visualisationThreshold, xBinnedGridCoordinates, yBinnedGridCoordinates, zBinnedGridCoordinates, voxelSize, classColours)
        #         else:
        #             raise ValueError('Unknown visualisation type: {}'.format(visualisationType))
        #         if visualisationFormat == 'image':
        #             output = voxelVisualisationPyplot.createMultipleViewImage(fig, ax, visualisationAngles)
        #             output = output.astype(numpy.float32).transpose(2,0,1)/255
        #         elif visualisationFormat == 'video':
        #             output = voxelVisualisationPyplot.createMultipleViewVideo(fig, ax, visualisationAngles)
        #             output = output.astype(numpy.float32).transpose(0,3,1,2)/255
        #         else:
        #             raise ValueError('Unknown visualisation format: {}'.format(visualisationFormat))
        #         voxelVisualisationPyplot.closeFigure(fig)
        #     elif visualisationEngine == 'pytorch3d':
        #         images = voxelVisualisationPytorch3D.renderVoxelScene(  trainScene.to(device = device), visualisationThreshold, maximumFillThreshold, torch.as_tensor(classColours[:15,:]), 
        #                                     [2.7]*len(visualisationAngles), visualisationAngles, [45]*len(visualisationAngles), device)
        #         images = torch.stack(images, dim = 0).cpu().numpy()
        #         if visualisationFormat == 'image':
        #             output = voxelVisualisation.createImageGrid(images, 3).transpose(2,0,1)
        #         else:
        #             output = images.transpose(0,3,1,2)
        #     else:
        #         raise ValueError('Unknown visualisation engine: {}'.format(visualisationEngine))
        #     currentLoggingTag = 'scene_{}/train_mode'.format(sceneIndex)
        #     if visualisationFormat == 'image':
        #         logger.add_image(tag = currentLoggingTag, img_tensor = output, global_step = iterationNumber)
        #     elif visualisationFormat == 'video':
        #         logger.add_video(tag = currentLoggingTag, vid_tensor = output, global_step = iterationNumber, fps = visualisationVideoFrameRate)
        #     else:
        #         raise ValueError('Unknown visualisation format: {}'.format(visualisationFormat))
            
        for sceneIndex, evalScene in enumerate(evalGeneratedScenes):
            if visualisationEngine == 'matplotlib':
                if visualisationType == 'voxelGrid':
                    fig, ax = voxelVisualisationPyplot.createVoxelBlockFigure(evalScene.numpy(), visualisationThreshold, xBinnedGridCoordinates, yBinnedGridCoordinates, zBinnedGridCoordinates, classColours)
                elif visualisationType == 'isosurface':      
                    assert not any([x < 2 for x in evalScene.shape[1:]])
                    fig, ax = voxelVisualisationPyplot.createVoxelIsosurfaceFigure(evalScene.numpy(), visualisationThreshold, xBinnedGridCoordinates, yBinnedGridCoordinates, zBinnedGridCoordinates, voxelSize, classColours)
                else:
                    raise ValueError('Unknown visualisation type: {}'.format(visualisationType))
                if visualisationFormat == 'image':
                    output = voxelVisualisationPyplot.createMultipleViewImage(fig, ax, visualisationAngles)
                    output = output.astype(numpy.float32).transpose(2,0,1)/255
                elif visualisationFormat == 'video':
                    output = voxelVisualisationPyplot.createMultipleViewVideo(fig, ax, visualisationAngles)
                    output = output.astype(numpy.float32).transpose(0,3,1,2)/255
                else:
                    raise ValueError('Unknown visualisation format: {}'.format(visualisationFormat))
                voxelVisualisationPyplot.closeFigure(fig)
            elif visualisationEngine == 'pytorch3d':
                images = voxelVisualisationPytorch3D.renderVoxelScene(  evalScene.to(device), visualisationThreshold, maximumFillThreshold, torch.as_tensor(classColours[:15,:]), 
                                            [2.7]*len(visualisationAngles), visualisationAngles, [45]*len(visualisationAngles), device)
                images = torch.stack(images, dim = 0).cpu().numpy()
                if visualisationFormat == 'image':
                    output = voxelVisualisation.createImageGrid(images, 3).transpose(2,0,1)
                else:
                    output = images.transpose(0,3,1,2)
            else:
                raise ValueError('Unknown visualisation engine: {}'.format(visualisationEngine))
            currentLoggingTag = 'scene_{}/eval_mode'.format(sceneIndex)
            if visualisationFormat == 'image':
                logger.add_image(tag = currentLoggingTag, img_tensor = output, global_step = iterationNumber)
            elif visualisationFormat == 'video':
                logger.add_video(tag = currentLoggingTag, vid_tensor = output, global_step = iterationNumber, fps = visualisationVideoFrameRate)
            else:
                raise ValueError('Unknown visualisation format: {}'.format(visualisationFormat))   

if __name__ == '__main__':   
    config = configuration.getDefaultConfiguration()
    config.merge_from_file('defaults.yaml')
    if len(sys.argv) > 1:
        configurationFileName = sys.argv[1]
    else:
        configurationFileName = 'local.yaml'
    config.merge_from_file(configurationFileName)
    config.freeze()
    systemConfig = config.SYSTEM
    dataConfig = config.DATA
    networksConfig = config.NETWORKS
    generatorConfig = networksConfig.GENERATOR
    encoderConfig = generatorConfig.ENCODER
    decoderConfig = generatorConfig.DECODER
    discriminatorConfig = networksConfig.DISCRIMINATOR
    stabilityConfig = networksConfig.STABILITY
    bootstrapConfig = networksConfig.BOOTSTRAP
    trainingConfig = config.TRAINING

    device = torch.device('cuda')
    if systemConfig.disableCuDNN:
        torch.backends.cudnn.enabled = False

    print('Setting up configuration')
    sceneSizes = list(range(1,dataConfig.maximumSceneSize+1))
    firstSceneNumbers = [1 for x in sceneSizes]
    lastSceneNumbers = [800 for x in sceneSizes]
    if trainingConfig.generatorDiscriminatorDataSplit == 'joint':
        generatorFirstSceneNumbers = [1 for x in sceneSizes]
        generatorLastSceneNumbers = [800 for x in sceneSizes]
        discriminatorFirstSceneNumbers = [1 for x in sceneSizes]
        discriminatorLastSceneNumbers = [800 for x in sceneSizes]
    elif trainingConfig.generatorDiscriminatorDataSplit == 'disjoint':
        generatorFirstSceneNumbers = [1 for x in sceneSizes]
        generatorLastSceneNumbers = [400 for x in sceneSizes]
        discriminatorFirstSceneNumbers = [401 for x in sceneSizes]
        discriminatorLastSceneNumbers = [800 for x in sceneSizes]
    else:
        raise ValueError('Unknown data split: {}'.format(dataConfig.generatorDiscriminatorDataSplit))
    validationFirstSceneNumbers = [801 for x in sceneSizes]
    validationLastSceneNumbers = [900 for x in sceneSizes]
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
    if systemConfig.type == 'bear':
        imageDataDirectory = '/rds/projects/2018/leonarda-muri/hector/data/dataFiles_d435_single_instance'
        voxelDataDirectory = '/rds/projects/2018/leonarda-muri/hector/data/dataFiles_d435_single_instance_stability'
        preprocessedDataDirectory = '/rds/projects/2018/leonarda-muri/hector/data/dataFiles_d435_single_instance_preprocessed_v3_{}'.format(preprocessedDataType)
        numberOfScenesToLog = 8
        # visualisationEngine = 'matplotlib'
        visualisationEngine = 'pytorch3d'
    elif systemConfig.type == 'desktop':
        imageDataDirectory = 'F:/data/dataFiles_d435_single_instance'
        voxelDataDirectory = 'F:/data/dataFiles_d435_single_instance_stability'
        preprocessedDataDirectory = 'F:/data/dataFiles_d435_single_instance_preprocessed_v3_{}'.format(preprocessedDataType)
        numberOfScenesToLog = 1
        # visualisationEngine = 'matplotlib'
        visualisationEngine = 'pytorch3d'
    elif systemConfig.type == 'hpc':
        imageDataDirectory = '/home/research/users/basevhra/data/dataFiles_d435_single_instance'
        voxelDataDirectory = '/home/research/users/basevhra/data/dataFiles_d435_single_instance_stability'
        preprocessedDataDirectory = '/home/research/users/basevhra/data/dataFiles_d435_single_instance_preprocessed_v3_{}'.format(preprocessedDataType)
        numberOfScenesToLog = 8
        visualisationEngine = 'matplotlib'
        # visualisationEngine = 'pytorch3d'
    else:
        raise ValueError('Unknown system type: {}'.format(systemConfig.type))

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
        print('Calculating data statistics')
        if dataSetType == 'preprocessed':
            raise RuntimeError('Preprocessed data set requires precomputed scene statistics')
        statisticsDataSet = dataSets.MixedDataset(sceneSizes, firstSceneNumbers, lastSceneNumbers, imageDataDirectory, imageFileTemplate, voxelDataDirectory, voxelFileTemplate,
                voxelGridBounds, voxelSize, cameraIntrinsics, cameraRotationMatrix, cameraTranslationVector, 
                cameraResolution = cameraResolution, voxelBinningPower = dataConfig.voxelBinningPower, classLabels = classLabels, numberOfObjectClasses = numberOfObjectClasses,
                dataMeans = dict(), dataStandardDeviations = dict(), enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
        statisticsDataloader = DataLoader(statisticsDataSet, batch_size = trainingConfig.batchSize, shuffle = False, num_workers = systemConfig.numberOfWorkers, drop_last = False)
        statisticsMeansData, statisticsVariancesData = dataSets.calculateDatasetStatistics(statisticsDataloader)
        statisticsStandardDeviationsData = {key:value.sqrt() for (key, value) in statisticsVariancesData.items()}
        hdf5storage.savemat(sceneStatisticsMeansFileName, {key:value.detach().cpu().numpy() for (key, value) in statisticsMeansData.items()})
        hdf5storage.savemat(sceneStatisticsStandardDeviationsFileName, {key:value.detach().cpu().numpy() for (key, value) in statisticsStandardDeviationsData.items()})
    
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
    shutil.copy2('conditionalGenerationTraining.py','{}/conditionalGenerationTraining.py'.format(logDirectory))
    shutil.copy2('networkArchitectures.py','{}/networkArchitectures.py'.format(logDirectory))
    shutil.copy2('dataSets.py','{}/dataSets.py'.format(logDirectory))
    configuration.saveConfiguration(config, '{}/config.yaml'.format(logDirectory))

    print('Creating data sets')
    completeSceneMean = currentStatisticsMeans['completeVoxelScene'].unsqueeze(dim = 0).to(dtype = torch.float32, device = device)
    completeSceneStandardDeviation = currentStatisticsStandardDeviations['completeVoxelScene'].unsqueeze(dim = 0).to(dtype = torch.float32,device = device)
    visualisationSceneListFiles = ['difficultValidationScenes.npz', 'hiddenValidationScenes.npz']
    visualisationSceneSize = []
    visualisationSceneNumber = []
    for visualisationSceneListFile in visualisationSceneListFiles:
        currentVisualisationData = numpy.load(visualisationSceneListFile)
        visualisationSceneSize.extend(currentVisualisationData['sceneSize'].tolist())
        visualisationSceneNumber.extend(currentVisualisationData['sceneNumber'].tolist())
    if dataSetType == 'runtime':
        generatorDataSet = dataSets.MixedDataset(sceneSizes, generatorFirstSceneNumbers, generatorLastSceneNumbers, imageDataDirectory, imageFileTemplate, voxelDataDirectory, voxelFileTemplate,
                    voxelGridBounds, voxelSize, cameraIntrinsics, cameraRotationMatrix, cameraTranslationVector, 
                    cameraResolution = cameraResolution, voxelBinningPower = dataConfig.voxelBinningPower, classLabels = classLabels, numberOfObjectClasses = numberOfObjectClasses,
                    dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations, enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
        validationDataSet = dataSets.MixedDataset(sceneSizes, validationFirstSceneNumbers, validationLastSceneNumbers, imageDataDirectory, imageFileTemplate, voxelDataDirectory, voxelFileTemplate,
                    voxelGridBounds, voxelSize, cameraIntrinsics, cameraRotationMatrix, cameraTranslationVector, 
                    cameraResolution = cameraResolution, voxelBinningPower = dataConfig.voxelBinningPower, classLabels = classLabels, numberOfObjectClasses = numberOfObjectClasses,
                    dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations, enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
        visualisationDataSet = dataSets.MixedListDataset(visualisationSceneSize, visualisationSceneNumber, imageDataDirectory, imageFileTemplate, voxelDataDirectory, voxelFileTemplate,
                    voxelGridBounds, voxelSize, cameraIntrinsics, cameraRotationMatrix, cameraTranslationVector, 
                    cameraResolution = cameraResolution, voxelBinningPower = dataConfig.voxelBinningPower, classLabels = classLabels, numberOfObjectClasses = numberOfObjectClasses,
                    dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations, enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
    elif dataSetType == 'preprocessed':
        generatorDataSet = dataSets.PreprocessedMixedDataset(sceneSizes, generatorFirstSceneNumbers, generatorLastSceneNumbers, preprocessedDataDirectory, preprocessedFileTemplate, 
                    voxelBinningPower = dataConfig.voxelBinningPower, dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations, 
                    enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
        validationDataSet = dataSets.PreprocessedMixedDataset(sceneSizes, validationFirstSceneNumbers, validationLastSceneNumbers, preprocessedDataDirectory, preprocessedFileTemplate, 
                    voxelBinningPower = dataConfig.voxelBinningPower, dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations, 
                    enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
        visualisationDataSet = dataSets.PreprocessedMixedListDataset(visualisationSceneSize, visualisationSceneNumber, preprocessedDataDirectory, preprocessedFileTemplate, 
                    voxelBinningPower = dataConfig.voxelBinningPower, dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations, 
                    enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
    print('There are {} generator training scenes.'.format(len(generatorDataSet)))
    generatorDataloader = DataLoader(generatorDataSet, batch_size = trainingConfig.batchSize, shuffle = True, num_workers = systemConfig.numberOfWorkers, drop_last = True, pin_memory = True)
    generatorDataIterator = iter(generatorDataloader)
    if trainingConfig.trainingType == 'adversarial':
        if dataSetType == 'runtime':
            discriminatorDataSet = dataSets.MixedDataset(sceneSizes, discriminatorFirstSceneNumbers, discriminatorLastSceneNumbers, imageDataDirectory, imageFileTemplate, voxelDataDirectory, voxelFileTemplate,
                        voxelGridBounds, voxelSize, cameraIntrinsics, cameraRotationMatrix, cameraTranslationVector, 
                        cameraResolution = cameraResolution, voxelBinningPower = dataConfig.voxelBinningPower, classLabels = classLabels, numberOfObjectClasses = numberOfObjectClasses,
                        dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations, enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
        elif dataSetType == 'preprocessed':
            discriminatorDataSet = dataSets.PreprocessedMixedDataset(sceneSizes, discriminatorFirstSceneNumbers, discriminatorLastSceneNumbers, preprocessedDataDirectory, preprocessedFileTemplate, 
                        voxelBinningPower = dataConfig.voxelBinningPower, dataMeans = currentStatisticsMeans, dataStandardDeviations = currentStatisticsStandardDeviations, 
                        enforceBinaryCompleteScenes = enforceBinaryCompleteScenes)
        print('There are {} discriminator training scenes.'.format(len(discriminatorDataSet)))
        discriminatorDataloader = DataLoader(discriminatorDataSet, batch_size = trainingConfig.batchSize, shuffle = True, num_workers = systemConfig.numberOfWorkers, drop_last = True, pin_memory = True)
        discriminatorDataIterator1 = iter(discriminatorDataloader)
        discriminatorDataIterator2 = iter(discriminatorDataloader)

    numberOfSpatialScales = int(math.floor(min([math.log2(x) for x in voxelBinnedSpatialDimensions]))-1)

    partialSceneVoxelChannels = 16
    completeSceneVoxelChannels = 15

    if generatorConfig.useBootstrap:
        print('Constructing bootstrap network')
        bootstrapNetworkConfig = configuration.getDefaultConfiguration()
        bootstrapNetworkConfig.merge_from_file(bootstrapConfig.configFileName)
        bootstrapNetworksConfig = bootstrapNetworkConfig.NETWORKS
        bootstrapGeneratorConfig = bootstrapNetworksConfig.GENERATOR
        bootstrapEncoderConfig = bootstrapGeneratorConfig.ENCODER
        bootstrapDecoderConfig = bootstrapGeneratorConfig.DECODER
        print('Constructing bootstrap encoder')
        bootstrapGeneratorEncoderInputDimension = partialSceneVoxelChannels + bootstrapEncoderConfig.stochasticRepresentationDimension
        bootstrapGeneratorEncoderDownscaling = numberOfSpatialScales
        bootstrapGeneratorEncoderUseTrigonometricCoordinateEmbedding = True
        if bootstrapEncoderConfig.useSpectralNorm:
            bootstrapGeneratorEncoderSpectralNormalisationSettings = dict(useSpectralNormalisation = True, numberOfPowerIterations = 5)
        else:
            bootstrapGeneratorEncoderSpectralNormalisationSettings = dict(useSpectralNormalisation = False)
        bootstrapEncoder = networkArchitectures.Encoder(inputDimensions = [bootstrapGeneratorEncoderInputDimension]+voxelBinnedSpatialDimensions, 
            initialProjectionDimension = bootstrapEncoderConfig.initialProjectionDimension, 
            numberOfDownscalingOperations = bootstrapGeneratorEncoderDownscaling, 
            representationDimension = bootstrapGeneratorConfig.representationVectorDimension, 
            numberOfIntermediateComputationLayers = bootstrapEncoderConfig.numberOfIntermediateComputationLayers, 
            kernelSize = bootstrapEncoderConfig.kernelSize,
            addCoordinatesToRepresentation = bootstrapEncoderConfig.injectCoordinateInput,
            useTrigonometricCoordinateEmbedding = bootstrapGeneratorEncoderUseTrigonometricCoordinateEmbedding,
            residualMode = bootstrapEncoderConfig.residualMode, 
            useDepthWiseSeparableConvolutions = bootstrapEncoderConfig.useDSConvolutions,
            spectralNormalisationSettings = bootstrapGeneratorEncoderSpectralNormalisationSettings, 
            useBatchNorm = bootstrapEncoderConfig.useBatchNorm, 
            useBiasThroughout = bootstrapEncoderConfig.useBiasThroughout,
            downscalingChannelScaling = bootstrapEncoderConfig.downscalingChannelScaling)
        bootstrapEncoder = bootstrapEncoder.to(device = device)
        bootstrapEncoder.load_state_dict(torch.load(bootstrapConfig.encoderWeightFileName, map_location=device))
        bootstrapEncoder.eval()
        setRequiresGrad(bootstrapEncoder, False)

        print('Constructing bootstrap decoder')
        bootstrapRepresentationDimension = bootstrapGeneratorConfig.representationVectorDimension + bootstrapDecoderConfig.stochasticRepresentationDimension
        bootstrapGeneratorDecoderInitialSpatialDimensions = [3,3,2]
        bootstrapGeneratorDecoderOutputDimension = completeSceneVoxelChannels
        if not generatorConfig.reconstructBackground:
            bootstrapGeneratorDecoderOutputDimension -= 1 #Not reconstructing background/table
        bootstrapGeneratorDecoderNumberOfUpscalingOperations = numberOfSpatialScales
        bootstrapGeneratorDecoderUseTrigonometricCoordinateEmbedding = True
        if bootstrapDecoderConfig.useSpectralNorm:
            bootstrapGeneratorDecoderSpectralNormalisationSettings = dict(useSpectralNormalisation = True, numberOfPowerIterations = 5)
        else:
            bootstrapGeneratorDecoderSpectralNormalisationSettings = dict(useSpectralNormalisation = False)
        if bootstrapGeneratorConfig.useSkipConnections:
            skipConnectionDimensions = bootstrapEncoder.intermediateRepresentationShapes
        else:
            skipConnectionDimensions = None
        bootstrapDecoder = networkArchitectures.Decoder(representationDimension = bootstrapRepresentationDimension, 
            initialSpatialDimensions = bootstrapGeneratorDecoderInitialSpatialDimensions, 
            initialProjectionDimension = bootstrapDecoderConfig.initialProjectionDimension, 
            outputDimension = bootstrapGeneratorDecoderOutputDimension, 
            numberOfUpscalingOperations = bootstrapGeneratorDecoderNumberOfUpscalingOperations, 
            numberOfIntermediateComputationLayers = bootstrapDecoderConfig.numberOfIntermediateComputationLayers, 
            kernelSize = bootstrapDecoderConfig.kernelSize,
            skipConnectionDimensions = skipConnectionDimensions, 
            addCoordinatesToRepresentation = bootstrapDecoderConfig.injectCoordinateInput, 
            useTrigonometricCoordinateEmbedding = bootstrapGeneratorDecoderUseTrigonometricCoordinateEmbedding,
            residualMode = bootstrapDecoderConfig.residualMode, 
            useDepthWiseSeparableConvolutions = bootstrapDecoderConfig.useDSConvolutions,
            spectralNormalisationSettings = bootstrapGeneratorDecoderSpectralNormalisationSettings, 
            useBatchNorm = bootstrapDecoderConfig.useBatchNorm, 
            useBiasThroughout = bootstrapDecoderConfig.useBiasThroughout,
            upscalingChannelScaling = bootstrapDecoderConfig.upscalingChannelScaling)
        bootstrapDecoder = bootstrapDecoder.to(device = device)
        bootstrapDecoder.load_state_dict(torch.load(bootstrapConfig.decoderWeightFileName, map_location=device))
        bootstrapDecoder.eval()
        setRequiresGrad(bootstrapDecoder, False)
    else:
        bootstrapNetworkConfig = None
        bootstrapNetworksConfig = None
        bootstrapGeneratorConfig = None
        bootstrapEncoderConfig = None
        bootstrapDecoderConfig = None
        bootstrapEncoder = None
        bootstrapDecoder = None
    
    if trainingConfig.useStability:
        print('Constructing stability estimator')
        stabilityConditioningConfig = stabilityConfiguration.getDefaultConfiguration()
        stabilityConditioningConfig.merge_from_file(stabilityConfig.configFileName)
        stabilityConditioningConfig.freeze()
        stabilityNetworkConfig = stabilityConditioningConfig.NETWORK 
        stabilityTrainingConfig = stabilityConditioningConfig.TRAINING 

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
        stabilityEstimator = networkArchitectures.Encoder(inputDimensions = [completeSceneVoxelChannels]+voxelBinnedSpatialDimensions, 
            initialProjectionDimension = stabilityNetworkConfig.initialProjectionDimension, 
            numberOfDownscalingOperations = stabilityEstimatorDownscaling, representationDimension = stabilityEstimatorOutputDimension, 
            numberOfIntermediateComputationLayers = stabilityNetworkConfig.numberOfIntermediateComputationLayers, kernelSize = stabilityNetworkConfig.kernelSize,
            addCoordinatesToRepresentation = stabilityNetworkConfig.injectCoordinateInput,
            useTrigonometricCoordinateEmbedding = stabilityEstimatorUseTrigonometricCoordinateEmbedding,
            residualMode = stabilityNetworkConfig.residualMode, useDepthWiseSeparableConvolutions = stabilityNetworkConfig.useDSConvolutions,
            spectralNormalisationSettings = stabilityEstimatorSpectralNormalisationSettings, useBatchNorm = False, useBiasThroughout = stabilityNetworkConfig.useBiasThroughout,
            downscalingChannelScaling = stabilityNetworkConfig.downscalingChannelScaling)
        stabilityEstimator = stabilityEstimator.to(device = device)
        stabilityEstimator.load_state_dict(torch.load(stabilityConfig.weightFileName, map_location=device))
        stabilityEstimator.eval()
        setRequiresGrad(stabilityEstimator, False)
    else:
        stabilityEstimator = None
        stabilityNetworkConfig = None
        stabilityTrainingConfig = None

    if trainingConfig.trainingType == 'adversarial':
        print('Constructing discriminator')
        voxelDiscriminatorInputChannels = partialSceneVoxelChannels+completeSceneVoxelChannels
        if trainingConfig.useStability:
            voxelDiscriminatorInputChannels += stabilityEstimatorOutputDimension
        voxelDiscriminatorDownscaling = numberOfSpatialScales
        voxelDiscriminatorDimension = 1
        voxelDiscriminatorUseTrigonometricCoordinateEmbedding = True
        if discriminatorConfig.useSpectralNorm:
            voxelDiscriminatorSpectralNormalisationSettings = dict(useSpectralNormalisation = True, numberOfPowerIterations = 5)
        else:
            voxelDiscriminatorSpectralNormalisationSettings = dict(useSpectralNormalisation = False)
        discriminator = networkArchitectures.Encoder(inputDimensions = [voxelDiscriminatorInputChannels]+voxelBinnedSpatialDimensions, 
            initialProjectionDimension = discriminatorConfig.initialProjectionDimension, 
            numberOfDownscalingOperations = voxelDiscriminatorDownscaling, representationDimension = voxelDiscriminatorDimension, 
            numberOfIntermediateComputationLayers = discriminatorConfig.numberOfIntermediateComputationLayers, kernelSize = discriminatorConfig.kernelSize,
            addCoordinatesToRepresentation = discriminatorConfig.injectCoordinateInput,
            useTrigonometricCoordinateEmbedding = voxelDiscriminatorUseTrigonometricCoordinateEmbedding,
            residualMode = discriminatorConfig.residualMode, useDepthWiseSeparableConvolutions = discriminatorConfig.useDSConvolutions,
            spectralNormalisationSettings = voxelDiscriminatorSpectralNormalisationSettings, useBatchNorm = False, useBiasThroughout = discriminatorConfig.useBiasThroughout,
            downscalingChannelScaling = discriminatorConfig.downscalingChannelScaling)
        discriminator = discriminator.to(device = device)

    print('Constructing generator encoder')
    voxelGeneratorEncoderInputDimension = partialSceneVoxelChannels + encoderConfig.stochasticRepresentationDimension
    if generatorConfig.useBootstrap:
        voxelGeneratorEncoderInputDimension += completeSceneVoxelChannels
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


    numberOfTrainingIterations = 50001

    distributionLoggingInterval = 1000
    visualisationType = 'voxelGrid'
    # visualisationType = 'isosurface'
    visualisationFormat = 'image'
    # visualisationFormat = 'video'
    if visualisationFormat == 'image':
        visualisationAngles = list(range(-60,300,40))
        visualisationVideoFrameRate = 10
        numberOfVisualisationColumns = 3
    elif visualisationFormat == 'video':
        visualisationVideoFrameRate = 10
        visualisationVideoLength = 5
        visualisationAngles = list(range(-60,301,int(round(360.0/visualisationVideoFrameRate/visualisationVideoLength))))
    visualisationThreshold = 0.5
    maximumFillThreshold = 1.0
    classColours = hdf5storage.loadmat('officialColours.mat')['colours'][1:17,:]
    classColours = classColours.astype(numpy.float32)

    print('Creating visualisations')
    numberOfVisualisationScenes = len(visualisationDataSet)
    loggingPartialScenes = []
    loggingCompleteScenes = []
    indicesToLog = list(range(0, numberOfVisualisationScenes, int(math.floor(numberOfVisualisationScenes/numberOfScenesToLog))))
    indicesToLog = indicesToLog[:numberOfScenesToLog]
    idealBackground = None
    for arrayIndex, sceneIndex in enumerate(indicesToLog):
        currentSceneData = visualisationDataSet[sceneIndex]
        if dataConfig.voxelEmbeddingType == 'surfaceVoxelEmbedding':
            currentPartialScene = currentSceneData['surfaceVoxelEmbedding']
        elif dataConfig.voxelEmbeddingType == 'projectionVoxelEmbedding':
            currentPartialScene = currentSceneData['projectionVoxelEmbedding']
        loggingPartialScenes.append(currentPartialScene)

        partialSceneToPlot = currentSceneData['surfaceVoxelEmbedding'].to(dtype = torch.float32)
        partialSceneToPlot = partialSceneToPlot*currentStatisticsStandardDeviations['surfaceVoxelEmbedding'].to(dtype = torch.float32) + currentStatisticsMeans['surfaceVoxelEmbedding'].to(dtype = torch.float32)
        partialSceneToPlot = (partialSceneToPlot > 1e-5 ).to(dtype = torch.float)
        partialSceneToPlot = partialSceneToPlot[:15,:,:,:]
        partialSceneToPlot = torch.cat([partialSceneToPlot[1:,:,:,:],partialSceneToPlot[0:1,:,:,:]], dim = 0)
        if visualisationEngine == 'matplotlib':
            partialSceneToPlot = partialSceneToPlot.numpy()
            if visualisationType == 'voxelGrid':
                fig, ax = voxelVisualisationPyplot.createVoxelBlockFigure(partialSceneToPlot, visualisationThreshold, xBinnedGridCoordinates, yBinnedGridCoordinates, zBinnedGridCoordinates, classColours)
            elif visualisationType == 'isosurface':      
                assert not any([x < 2 for x in partialSceneToPlot.shape[1:]])
                fig, ax = voxelVisualisationPyplot.createVoxelIsosurfaceFigure(partialSceneToPlot, visualisationThreshold, xBinnedGridCoordinates, yBinnedGridCoordinates, zBinnedGridCoordinates, voxelSize, classColours)
            else:
                raise ValueError('Unknown visualisation type: {}'.format(visualisationType))
            if visualisationFormat == 'image':
                output = voxelVisualisationPyplot.createMultipleViewImage(fig, ax, visualisationAngles)
                output = output.astype(numpy.float32).transpose(2,0,1)/255
            elif visualisationFormat == 'video':
                output = voxelVisualisationPyplot.createMultipleViewVideo(fig, ax, visualisationAngles)
                output = output.astype(numpy.float32).transpose(0,3,1,2)/255
            else:
                raise ValueError('Unknown visualisation format: {}'.format(visualisationFormat))
            voxelVisualisationPyplot.closeFigure(fig)
        elif visualisationEngine == 'pytorch3d':
            images = voxelVisualisationPytorch3D.renderVoxelScene(  partialSceneToPlot.to(device = device), visualisationThreshold, maximumFillThreshold, torch.as_tensor(classColours[:15,:]), 
                                        [2.7]*len(visualisationAngles), visualisationAngles, [45]*len(visualisationAngles), device)
            images = torch.stack(images, dim = 0).cpu().numpy()
            if visualisationFormat == 'image':
                output = voxelVisualisation.createImageGrid(images, 3).transpose(2,0,1)
            else:
                output = images.transpose(0,3,1,2)
        else:
            raise ValueError('Unknown visualisation engine: {}'.format(visualisationEngine))
        if visualisationFormat == 'image':
            logger.add_image(tag = 'Data/Partial scene {}'.format(arrayIndex), img_tensor = output, global_step = 0)
        elif visualisationFormat == 'video':
            logger.add_video(tag = 'Data/Partial scene {}'.format(arrayIndex), vid_tensor = output, global_step = 0, fps = visualisationVideoFrameRate)
        else:
            raise ValueError('Unknown visualisation format: {}'.format(visualisationFormat))

        completeSceneToPlot = currentSceneData['completeVoxelScene'].to(dtype = torch.float32)
        loggingCompleteScenes.append(completeSceneToPlot)
        if idealBackground is None:
            idealBackground = completeSceneToPlot[14,:,:,:].to(device)
        completeSceneToPlot = completeSceneToPlot*currentStatisticsStandardDeviations['completeVoxelScene'].to(dtype = torch.float32) + \
            currentStatisticsMeans['completeVoxelScene'].to(dtype = torch.float32)
        if visualisationEngine == 'matplotlib':
            completeSceneToPlot = completeSceneToPlot.numpy()
            if visualisationType == 'voxelGrid':
                fig, ax = voxelVisualisationPyplot.createVoxelBlockFigure(completeSceneToPlot, visualisationThreshold, xBinnedGridCoordinates, yBinnedGridCoordinates, zBinnedGridCoordinates, classColours)
            elif visualisationType == 'isosurface':      
                assert not any([x < 2 for x in completeSceneToPlot.shape[1:]])
                fig, ax = voxelVisualisationPyplot.createVoxelIsosurfaceFigure(completeSceneToPlot, visualisationThreshold, xBinnedGridCoordinates, yBinnedGridCoordinates, zBinnedGridCoordinates, voxelSize, classColours)
            else:
                raise ValueError('Unknown visualisation type: {}'.format(visualisationType))
            if visualisationFormat == 'image':
                output = voxelVisualisationPyplot.createMultipleViewImage(fig, ax, visualisationAngles)
                output = output.astype(numpy.float32).transpose(2,0,1)/255
            elif visualisationFormat == 'video':
                output = voxelVisualisationPyplot.createMultipleViewVideo(fig, ax, visualisationAngles)
                output = output.astype(numpy.float32).transpose(0,3,1,2)/255
            else:
                raise ValueError('Unknown visualisation format: {}'.format(visualisationFormat))
            voxelVisualisationPyplot.closeFigure(fig)
        elif visualisationEngine == 'pytorch3d':
            images = voxelVisualisationPytorch3D.renderVoxelScene(  completeSceneToPlot.to(device = device), visualisationThreshold, maximumFillThreshold, torch.as_tensor(classColours[:15,:]), 
                                        [2.7]*len(visualisationAngles), visualisationAngles, [45]*len(visualisationAngles), device)
            images = torch.stack(images, dim = 0).cpu().numpy()
            if visualisationFormat == 'image':
                output = voxelVisualisation.createImageGrid(images, 3).transpose(2,0,1)
            else:
                output = images.transpose(0,3,1,2)
        else:
            raise ValueError('Unknown visualisation engine: {}'.format(visualisationEngine))
        if visualisationFormat == 'image':
            logger.add_image(tag = 'Data/Complete scene {}'.format(arrayIndex), img_tensor = output, global_step = 0)
        elif visualisationFormat == 'video':
            logger.add_video(tag = 'Data/Complete scene {}'.format(arrayIndex), vid_tensor = output, global_step = 0, fps = visualisationVideoFrameRate)
        else:
            raise ValueError('Unknown visualisation format: {}'.format(visualisationFormat))
    loggingPartialScenes = torch.stack(loggingPartialScenes, dim = 0)    
    loggingPartialScenes = loggingPartialScenes.to(dtype = torch.float).requires_grad_(False)
    loggingCompleteScenes = torch.stack(loggingCompleteScenes, dim = 0)    
    loggingCompleteScenes = loggingCompleteScenes.to(dtype = torch.float).requires_grad_(False)
    if encoderConfig.stochasticRepresentationDimension > 0:
        loggingEncoderLatents = torch.randn(numberOfScenesToLog, encoderConfig.stochasticRepresentationDimension, 1, 1, 1, requires_grad = False)
    if decoderConfig.stochasticRepresentationDimension > 0:
        loggingDecoderLatents = torch.randn(numberOfScenesToLog, decoderConfig.stochasticRepresentationDimension, requires_grad = False)
    
    loggingBootstrapEncoderLatents = None
    loggingBootstrapDecoderLatents = None
    if generatorConfig.useBootstrap:
        if bootstrapEncoderConfig.stochasticRepresentationDimension > 0:
            loggingBootstrapEncoderLatents = torch.randn(numberOfScenesToLog, bootstrapEncoderConfig.stochasticRepresentationDimension, 1, 1, 1, requires_grad = False)
        if bootstrapDecoderConfig.stochasticRepresentationDimension > 0:
            loggingBootstrapDecoderLatents = torch.randn(numberOfScenesToLog, bootstrapDecoderConfig.stochasticRepresentationDimension, requires_grad = False)

    if trainingConfig.trainingType == 'adversarial':
        discriminatorEffectiveLearningRate = 5*trainingConfig.baseLearningRate
        discriminatorLearningRate = discriminatorEffectiveLearningRate/trainingConfig.numberOfDiscriminatorIterationsPerGeneratorIteration
        discriminatorOptimiser = torch.optim.Adam(discriminator.parameters(), lr = discriminatorLearningRate)
    generatorLearningRate = 1*trainingConfig.baseLearningRate
    generatorOptimiser = torch.optim.Adam(list(generatorEncoder.parameters()) + list(generatorDecoder.parameters()), lr = generatorLearningRate)

    iterationToProfile = 2

    # torch.backends.cudnn.deterministic = True
    if systemConfig.cudnnBenchmarking:
        torch.backends.cudnn.benchmark = True

    print('Training')
    iterationStartTime = time.time()
    for iterationNumber in range(numberOfTrainingIterations):
        # print('Iteration {}'.format(iterationNumber))
        torch.cuda.reset_max_memory_allocated()

        if iterationNumber == iterationToProfile:
            print('Profiling iteration {}'.format(iterationNumber))
            profile = torch.autograd.profiler.profile(enabled = True, use_cuda = True, record_shapes = False)
        else:
            profile = torch.autograd.profiler.profile(enabled = False, use_cuda = False, record_shapes = False)
        
        with profile:
            generatorEncoder.train()
            setRequiresGrad(generatorEncoder, True)
            generatorDecoder.train()
            setRequiresGrad(generatorDecoder, True)

            calculateGradientStatistics = iterationNumber % distributionLoggingInterval == 0
            if trainingConfig.trainingType == 'adversarial':
                discriminator.train()
                setRequiresGrad(discriminator, True)

                #Train discriminator first
                discriminatorRealLossList = []
                discriminatorGeneratedLossList = []
                discriminatorRealCritiqueList = []
                discriminatorGeneratedCritiqueList = []
                discriminatorGradientPenaltyList = []
                discriminatorLossList = []
                discriminatorMeanGradientsList = []
                discriminatorMaxGradientsList = []
                discriminatorMinGradientsList = []
                discriminatorTrainingTime = -time.time()
                discriminatorBackpropagationTime = 0
                gradientPenaltyTime = 0
                for discriminatorIterationNumber in range(trainingConfig.numberOfDiscriminatorIterationsPerGeneratorIteration):
                    discriminatorTrainingResults, discriminatorDataIterator1, discriminatorDataIterator2 = trainDiscriminator( 
                            generatorEncoder, generatorDecoder, discriminator, stabilityEstimator,
                            bootstrapEncoder, bootstrapDecoder,
                            discriminatorDataloader, discriminatorDataIterator1, discriminatorDataIterator2,
                            idealBackground, device, calculateGradientStatistics, discriminatorOptimiser,
                            encoderConfig, decoderConfig, generatorConfig, discriminatorConfig, dataConfig, trainingConfig, stabilityTrainingConfig,
                            bootstrapEncoderConfig, bootstrapDecoderConfig, bootstrapGeneratorConfig)

                    discriminatorRealLossList.append(discriminatorTrainingResults['discriminatorRealLoss'])
                    discriminatorGeneratedLossList.append(discriminatorTrainingResults['discriminatorGeneratedLoss'])
                    discriminatorRealCritiqueList.append(discriminatorTrainingResults['discriminatorRealCritique'])
                    discriminatorGeneratedCritiqueList.append(discriminatorTrainingResults['discriminatorGeneratedCritique'])
                    discriminatorLossList.append(discriminatorTrainingResults['discriminatorLoss'])
                    discriminatorGradientPenaltyList.append(discriminatorTrainingResults['gradientPenaltyLoss'])
                    discriminatorBackpropagationTime += discriminatorTrainingResults['discriminatorBackpropagationTime']
                    gradientPenaltyTime += discriminatorTrainingResults['gradientPenaltyTime']
                    if iterationNumber % distributionLoggingInterval == 0:
                        discriminatorMeanGradientsList.append(discriminatorTrainingResults['discriminatorMeanGradients'])
                        discriminatorMaxGradientsList.append(discriminatorTrainingResults['discriminatorMaxGradients'])
                        discriminatorMinGradientsList.append(discriminatorTrainingResults['discriminatorMinGradients'])
                    del discriminatorTrainingResults
                discriminatorTrainingTime += time.time()
                discriminatorRealLoss = sum(discriminatorRealLossList)/trainingConfig.numberOfDiscriminatorIterationsPerGeneratorIteration
                discriminatorGeneratedLoss = sum(discriminatorGeneratedLossList)/trainingConfig.numberOfDiscriminatorIterationsPerGeneratorIteration
                discriminatorRealCritique = torch.cat(discriminatorRealCritiqueList, dim = 0)
                discriminatorGeneratedCritique = torch.cat(discriminatorGeneratedCritiqueList, dim = 0)
                discriminatorLoss = sum(discriminatorLossList)/trainingConfig.numberOfDiscriminatorIterationsPerGeneratorIteration
                discriminatorGradientPenalty = sum(discriminatorGradientPenaltyList)/trainingConfig.numberOfDiscriminatorIterationsPerGeneratorIteration
                if iterationNumber % distributionLoggingInterval == 0:
                    discriminatorMeanGradients = sum(discriminatorMeanGradientsList)/trainingConfig.numberOfDiscriminatorIterationsPerGeneratorIteration
                    discriminatorMaxGradients = sum(discriminatorMaxGradientsList)/trainingConfig.numberOfDiscriminatorIterationsPerGeneratorIteration
                    discriminatorMinGradients = sum(discriminatorMinGradientsList)/trainingConfig.numberOfDiscriminatorIterationsPerGeneratorIteration

                #Train generator
                generatorTrainingTime = -time.time()
                generatorTrainingResults, generatorDataIterator = trainGenerator( generatorEncoder, generatorDecoder, discriminator, stabilityEstimator, 
                        bootstrapEncoder, bootstrapDecoder, generatorDataloader, generatorDataIterator,
                        idealBackground, device, calculateGradientStatistics, generatorOptimiser,
                        encoderConfig, decoderConfig, generatorConfig, dataConfig, trainingConfig, stabilityTrainingConfig,
                        bootstrapEncoderConfig, bootstrapDecoderConfig, bootstrapGeneratorConfig)
                generatorTrainingTime += time.time()
  
                logger.add_scalar('Discriminator training/Loss', scalar_value = discriminatorLoss, global_step = iterationNumber)
                logger.add_scalar('Discriminator training/Negative loss', scalar_value = -discriminatorLoss, global_step = iterationNumber)
                logger.add_scalar('Discriminator training/Real loss', scalar_value = discriminatorRealLoss, global_step = iterationNumber)
                logger.add_scalar('Discriminator training/Generated loss', scalar_value = discriminatorGeneratedLoss, global_step = iterationNumber)
                logger.add_scalar('Discriminator training/GradientPenaltyLoss', scalar_value = discriminatorGradientPenalty, global_step = iterationNumber)
                logger.add_scalar('Discriminator training/Learning rate (absolute)', scalar_value = discriminatorLearningRate, global_step = iterationNumber)
                logger.add_scalar('Discriminator training/Learning rate (effective)', scalar_value = discriminatorEffectiveLearningRate, global_step = iterationNumber)
                logger.add_scalar('Discriminator training/Training time', scalar_value = discriminatorTrainingTime, global_step = iterationNumber)
                logger.add_scalar('Discriminator training/Gradient penalty calculation time', scalar_value = gradientPenaltyTime, global_step = iterationNumber)
                logger.add_scalar('Discriminator training/Backpropagation time', scalar_value = discriminatorBackpropagationTime, global_step = iterationNumber)
                if iterationNumber % distributionLoggingInterval == 0:
                    logger.add_scalar('Training gradients/Discriminator mean grad', scalar_value = discriminatorMeanGradients.mean().item(), global_step = iterationNumber)
                    logger.add_scalar('Training gradients/Discriminator max grad', scalar_value = discriminatorMaxGradients.mean().item(), global_step = iterationNumber)
                    logger.add_scalar('Training gradients/Discriminator min grad', scalar_value = discriminatorMinGradients.mean().item(), global_step = iterationNumber)
                    logger.add_histogram('Training histograms/Real critique', values = discriminatorRealCritique, global_step = iterationNumber)
                    logger.add_histogram('Training histograms/Generation critique', values = discriminatorGeneratedCritique, global_step = iterationNumber)
                    # logger.add_histogram('Training histograms/Generator output', values = generatedScenes.detach()*completeSceneStandardDeviation + completeSceneMean, global_step = iterationNumber)
                    logger.add_histogram('Training histograms/Discriminator mean grad', values = discriminatorMeanGradients, global_step = iterationNumber)
                    logger.add_histogram('Training histograms/Discriminator max grad', values = discriminatorMaxGradients, global_step = iterationNumber)
                    logger.add_histogram('Training histograms/Discriminator min grad', values = discriminatorMinGradients, global_step = iterationNumber)
            elif trainingConfig.trainingType == 'regression':
                generatorTrainingTime = -time.time()
                generatorTrainingResults, generatorDataIterator = trainRegression( generatorEncoder, generatorDecoder, stabilityEstimator, 
                    bootstrapEncoder, bootstrapDecoder, generatorDataloader, generatorDataIterator, 
                    idealBackground, device, calculateGradientStatistics, generatorOptimiser,
                    encoderConfig, decoderConfig, generatorConfig, dataConfig, trainingConfig, stabilityTrainingConfig, 
                    bootstrapEncoderConfig, bootstrapDecoderConfig, bootstrapGeneratorConfig )
                generatorTrainingTime += time.time()
                logger.add_scalar('Generator training/Voxel loss', scalar_value = generatorTrainingResults['voxelLoss'], global_step = iterationNumber)
                logger.add_scalar('Generator training/Stability loss', scalar_value = generatorTrainingResults['stabilityLoss'], global_step = iterationNumber)

        if iterationNumber == iterationToProfile:
            tracePath = '{}/iteration_{}.tracing'.format(logDirectory, iterationToProfile)
            profile.export_chrome_trace(tracePath)

        logger.add_scalar('Generator training/L1 residual penalty', scalar_value = generatorTrainingResults['l1ResidualPenalty'], global_step = iterationNumber)
        logger.add_scalar('Generator training/Loss', scalar_value = generatorTrainingResults['generatorLoss'], global_step = iterationNumber)
        logger.add_scalar('Generator training/Learning rate', scalar_value = generatorLearningRate, global_step = iterationNumber)
        logger.add_scalar('Generator training/Training time', scalar_value = generatorTrainingTime, global_step = iterationNumber)
        logger.add_scalar('Generator training/Backpropagation time', scalar_value = generatorTrainingResults['generatorBackpropagationTime'], global_step = iterationNumber)
        logger.add_scalar('Training/Memory usage', scalar_value = torch.cuda.max_memory_allocated(), global_step = iterationNumber)
        if iterationNumber % distributionLoggingInterval == 0:
            logger.add_scalar('Training gradients/Generator mean grad', scalar_value = generatorTrainingResults['generatorMeanGradients'].mean().item(), global_step = iterationNumber)
            logger.add_scalar('Training gradients/Generator max grad', scalar_value = generatorTrainingResults['generatorMaxGradients'].mean().item(), global_step = iterationNumber)
            logger.add_scalar('Training gradients/Generator min grad', scalar_value = generatorTrainingResults['generatorMinGradients'].mean().item(), global_step = iterationNumber)
            # logger.add_histogram('Training histograms/Generator output', values = generatedScenes.detach()*completeSceneStandardDeviation + completeSceneMean, global_step = iterationNumber)
            logger.add_histogram('Training histograms/Generator mean grad', values = generatorTrainingResults['generatorMeanGradients'], global_step = iterationNumber)
            logger.add_histogram('Training histograms/Generator max grad', values = generatorTrainingResults['generatorMaxGradients'], global_step = iterationNumber)
            logger.add_histogram('Training histograms/Generator min grad', values = generatorTrainingResults['generatorMinGradients'], global_step = iterationNumber)

        logVoxels = trainingConfig.voxelDataLoggingInterval > 0 and (iterationNumber % trainingConfig.voxelDataLoggingInterval == 0)
        logVisualisations = trainingConfig.visualisationLoggingInterval > 0 and (iterationNumber % trainingConfig.visualisationLoggingInterval == 0)
        logNetworkstate = trainingConfig.networkStateLoggingInterval > 0 and (iterationNumber % trainingConfig.networkStateLoggingInterval == 0)
        if logVoxels or logVisualisations:
            if systemConfig.cudnnBenchmarking:
                torch.backends.cudnn.benchmark = False
            visualisationAndLogging(  generatorEncoder, generatorDecoder, bootstrapEncoder, bootstrapDecoder, loggingPartialScenes, loggingEncoderLatents, loggingDecoderLatents, 
                                loggingBootstrapEncoderLatents, loggingBootstrapDecoderLatents, loggingCompleteScenes,
                                completeSceneMean, completeSceneStandardDeviation,
                                idealBackground, device, calculateGradientStatistics, logDirectory, logger,
                                logVoxels, logVisualisations, visualisationEngine, visualisationFormat, visualisationVideoFrameRate,
                                visualisationThreshold, maximumFillThreshold, visualisationAngles, voxelSize, xBinnedGridCoordinates, yBinnedGridCoordinates, zBinnedGridCoordinates, classColours,
                                iterationNumber, encoderConfig, decoderConfig, generatorConfig, bootstrapEncoderConfig, bootstrapDecoderConfig, bootstrapGeneratorConfig )
            if systemConfig.cudnnBenchmarking:         
                torch.backends.cudnn.benchmark = True
        
        if logNetworkstate:
            print('Saving generator encoder state')
            generatorEncoderStateFileName = 'generatorEncoder_state_iteration_{}.pts'.format(iterationNumber)
            generatorEncoderStateFilePath = '{}/{}'.format(logDirectory, generatorEncoderStateFileName)
            torch.save(generatorEncoder.state_dict(), generatorEncoderStateFilePath)
            print('Saving generator decoder state')
            generatorDecoderStateFileName = 'generatorDecoder_state_iteration_{}.pts'.format(iterationNumber)
            generatorDecoderStateFilePath = '{}/{}'.format(logDirectory, generatorDecoderStateFileName)
            torch.save(generatorDecoder.state_dict(), generatorDecoderStateFilePath)
            print('Saving generator optimiser state')
            generatorOptimiserStateFileName = 'generatorOptimiser_state_iteration_{}.pts'.format(iterationNumber)
            generatorOptimiserStateFilePath = '{}/{}'.format(logDirectory, generatorOptimiserStateFileName)
            torch.save(generatorOptimiser.state_dict(), generatorOptimiserStateFilePath)
            if trainingConfig.trainingType == 'adversarial':
                print('Saving discriminator state')
                discriminatorStateFileName = 'discriminator_state_iteration_{}.pts'.format(iterationNumber)
                discriminatorStateFilePath = '{}/{}'.format(logDirectory, discriminatorStateFileName)
                torch.save(discriminator.state_dict(), discriminatorStateFilePath)
                print('Saving discriminator optimiser state')
                discriminatorOptimiserStateFileName = 'discriminatorOptimiser_state_iteration_{}.pts'.format(iterationNumber)
                discriminatorOptimiserStateFilePath = '{}/{}'.format(logDirectory, discriminatorOptimiserStateFileName)
                torch.save(discriminatorOptimiser.state_dict(), discriminatorOptimiserStateFilePath)

        iterationEndTime = time.time()
        logger.add_scalar('Training/Iteration time', scalar_value = iterationEndTime - iterationStartTime, global_step = iterationNumber)
        iterationStartTime = iterationEndTime


