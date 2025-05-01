import matplotlib.pyplot as plt
import matplotlib.colors as c
from  skimage import measure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy
import math
import voxelVisualisation


def createVoxelBlockFigure(voxelGrid, threshold, xGridCoordinates, yGridCoordinates, zGridCoordinates, classColours):
    
    booleanGrid = (voxelGrid >= threshold)
    voxelOutline = booleanGrid.any(axis = 0)
    voxelColours = numpy.zeros(voxelOutline.shape+(3,), dtype = numpy.float32)
    colourCoefficients = numpy.clip(voxelGrid,0,1)
    for ii in range(booleanGrid.shape[0]):
        for jj in range(3):
            voxelColours[:,:,:,jj] += colourCoefficients[ii,:,:,:]*classColours[ii,jj]
    voxelColours = numpy.clip(voxelColours,0,1)
    voxelColours = voxelColours.astype(numpy.float32)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(xGridCoordinates, yGridCoordinates, zGridCoordinates, voxelOutline, facecolors=voxelColours)#, edgecolor='k')
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    
    xmin = xGridCoordinates.min()
    xmax = xGridCoordinates.max()
    ymin = yGridCoordinates.min()
    ymax = yGridCoordinates.max()
    zmin = zGridCoordinates.min()
    zmax = zGridCoordinates.max()
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.set_zlim([zmin,zmax])
    #Equal axes
    max_range = numpy.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    return fig, ax

def createVoxelIsosurfaceFigure(voxelGrid, surfaceValue, xGridCoordinates, yGridCoordinates, zGridCoordinates, voxelSize, classColours, alpha = 1):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for ii in range(voxelGrid.shape[0]):
        if voxelGrid[ii,:,:,:].min() >= surfaceValue or voxelGrid[ii,:,:,:].max() <= surfaceValue:
            continue
        verts, faces, normals, values = measure.marching_cubes_lewiner(voxelGrid[ii,:,:,:], level = surfaceValue, spacing = (voxelSize*(voxelGrid.shape[1])/(voxelGrid.shape[1]-1),
                                                                                                                            voxelSize*(voxelGrid.shape[2])/(voxelGrid.shape[2]-1),
                                                                                                                            voxelSize*(voxelGrid.shape[3])/(voxelGrid.shape[3]-1)) )
        verts[:,0] += xGridCoordinates.min()
        verts[:,1] += yGridCoordinates.min()
        verts[:,2] += zGridCoordinates.min()
        mesh = Poly3DCollection(verts[faces])
        mesh.set_facecolor(classColours[ii,:])
        mesh.set_edgecolor(classColours[ii,:])
        mesh.set_alpha(alpha)
        ax.add_collection3d(mesh)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    
    xmin = xGridCoordinates.min()
    xmax = xGridCoordinates.max()
    ymin = yGridCoordinates.min()
    ymax = yGridCoordinates.max()
    zmin = zGridCoordinates.min()
    zmax = zGridCoordinates.max()
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.set_zlim([zmin,zmax])
    #Equal axes
    max_range = numpy.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*numpy.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')
    return fig, ax

def createImagesFromMultipleViewingAngles(figureHandle, axisHandle, viewingAngles):
    canvas = axisHandle.figure.canvas
    viewingAngleImages = []
    for viewingAngle in viewingAngles:
        axisHandle.view_init(elev=45., azim=viewingAngle)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        viewingAngleImage = numpy.fromstring(s, numpy.uint8).reshape((height, width, 4))
        viewingAngleImages.append(viewingAngleImage)
    return viewingAngleImages

def createMultipleViewImage(figureHandle, axisHandle, viewingAngles):
    viewingAngleImages = createImagesFromMultipleViewingAngles(figureHandle, axisHandle, viewingAngles)
    finalImage = voxelVisualisation.createImageGrid(numpy.asarray(viewingAngleImages))
    return finalImage

def createMultipleViewVideo(figureHandle, axisHandle, viewingAngles):
    viewingAngleImages = createImagesFromMultipleViewingAngles(figureHandle, axisHandle, viewingAngles)
    return numpy.asarray(viewingAngleImages)

def closeFigure(figureHandle):
    plt.close(figureHandle)
