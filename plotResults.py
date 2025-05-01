import numpy
import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib as tpl
import os
import sys

def createBoxPlots(data, quantity, outputFormat, xLabel = None, yLabel = None):
    difficultData = {a:b['difficult'][quantity] for a,b in data.items()}
    difficultDF = pd.DataFrame(difficultData, columns = difficultData.keys())
    ax = difficultDF.plot.box(rot = -45)
    ax.set(xlabel=xLabel, ylabel=yLabel)
    tpl.save(outputFormat.format('difficult'))
    plt.close()

    hiddenData = {a:b['hidden'][quantity] for a,b in data.items()}
    hiddenDF = pd.DataFrame(hiddenData, columns = hiddenData.keys())
    ax = hiddenDF.plot.box(rot = -45)
    ax.set(xlabel=xLabel, ylabel=yLabel)
    tpl.save(outputFormat.format('hidden'))
    plt.close()


if __name__ == '__main__':

    plotDirectory = 'figures'
    if not os.path.exists(plotDirectory):
        try:
            os.mkdir(plotDirectory)
        except:
            if not os.path.exists(plotDirectory):
                raise IOError('Unable to create plot directory: {}.'.format(plotDirectory))

    networkTypes = ['Adversarial_no_stability', 'Adversarial_object_stability', 'Adversarial_scene_stability', 'Regression_no_stability', 'Regression_object_stability', 'Regression_scene_stability']
    nameRemapping = dict(   Adversarial_no_stability = 'Adversarial, no stability',
                            Adversarial_object_stability = 'Adversarial, object stability',
                            Adversarial_scene_stability = 'Adversarial, scene stability',
                            Regression_no_stability = 'Regression, no stability',
                            Regression_object_stability = 'Regression, object stability',
                            Regression_scene_stability = 'Regression, scene stability')
    sceneSetTypes = ['difficult', 'hidden']
    # networkTypes = ['Adversarial_no_stability', 'Adversarial_object_stability']
    # sceneSetTypes = ['difficult', 'hidden']
    
    # Load data
    outputFileName = 'collatedResults/testingResults_{}_{}.npz'
    data = dict()
    for networkType in networkTypes:
        networkData = dict()
        for setType in sceneSetTypes:
            currentFileName = outputFileName.format(networkType, setType)
            currentData = dict(numpy.load(currentFileName))
            networkData[setType] = currentData
        data[nameRemapping[networkType]] = networkData

    # displacementData = {a:numpy.concatenate([d['maxFinalDisplacement'] for c,d in b.items()]) for a,b in data.items()}
    # precisionData = {a:numpy.concatenate([d['averagePrecision'] for c,d in b.items()]) for a,b in data.items()}
    # recallData = {a:numpy.concatenate([d['averageRecall'] for c,d in b.items()]) for a,b in data.items()}
    # depthData = {a:numpy.concatenate([d['meanDepthError'] for c,d in b.items()]) for a,b in data.items()}
    # misclassificationData = {a:numpy.concatenate([d['misclassificationRatio'] for c,d in b.items()]) for a,b in data.items()}
    # fractionalSceneSizeData = {a:numpy.concatenate([d['fractionalSceneSize'] for c,d in b.items()]) for a,b in data.items()}
    # fractionOfObjectsRecoveredData = {a:numpy.concatenate([d['fractionOfCorrectObjectsRecovered'] for c,d in b.items()])for a,b in data.items()}

    createBoxPlots(data, 'maxFinalDisplacement', '{}/{{}}Displacement.tex'.format(plotDirectory), xLabel = None, yLabel = 'Displacement (m)')

    createBoxPlots(data, 'averagePrecision', '{}/{{}}Precision.tex'.format(plotDirectory), xLabel = None, yLabel = 'Precision')

    createBoxPlots(data, 'averageRecall', '{}/{{}}Recall.tex'.format(plotDirectory), xLabel = None, yLabel = 'Recall')

    createBoxPlots(data, 'meanDepthError', '{}/{{}}Depth.tex'.format(plotDirectory), xLabel = None, yLabel = 'Depth error (m)')

    createBoxPlots(data, 'misclassificationRatio', '{}/{{}}Misclassification.tex'.format(plotDirectory), xLabel = None, yLabel = 'Misclassification fraction')

    createBoxPlots(data, 'fractionalSceneSize', '{}/{{}}Size.tex'.format(plotDirectory), xLabel = None, yLabel = 'Fractional scene size')

    createBoxPlots(data, 'fractionOfCorrectObjectsRecovered', '{}/{{}}Recovered.tex'.format(plotDirectory), xLabel = None, yLabel = 'Fraction of correct objects')

    print('Complete')
