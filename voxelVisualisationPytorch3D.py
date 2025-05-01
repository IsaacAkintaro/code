import torch
from torch import nn
import pytorch3d
from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, MeshRenderer, SoftPhongShader, PointLights, OpenGLPerspectiveCameras, look_at_view_transform
from pytorch3d.io import load_objs_as_meshes
import hdf5storage
import numpy
import matplotlib.pyplot as plt

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasteriser, shader):
        super().__init__()
        self.rasteriser = rasteriser
        self.shader = shader

    def forward(self, meshes, **kwargs) -> torch.Tensor:
        fragments = self.rasteriser(meshes, **kwargs)
        images = self.shader(fragments, meshes, **kwargs)
        return images, fragments.zbuf

@torch.no_grad()
def generateVoxelGridMesh(sceneVoxelGrid, threshold, colours, device):
    vertices = []
    faces = []
    textures = []
    totalNumberOfVertices = 0
    objectMeshes = cubify(voxels = sceneVoxelGrid, thresh = threshold, device = device)
    for objectIndex, objectMesh in enumerate(objectMeshes):
        currentVertices, currentFaces = objectMesh.get_mesh_verts_faces(0)
        if currentVertices.size(0) == 0 or currentFaces.size(0) == 0:
            continue
        currentTexture = colours[objectIndex,:].to(device = device)
        currentTexture = currentTexture[None,:].expand_as(currentVertices)
        vertices.append( currentVertices[:,[0,2,1]] )
        faces.append(currentFaces + totalNumberOfVertices)
        textures.append(currentTexture)
        totalNumberOfVertices += currentVertices.size(0)
    vertices = torch.cat(vertices, dim = 0)
    faces = torch.cat(faces, dim = 0)
    textures = torch.cat(textures, dim = 0).unsqueeze(dim = 0)
    textureObject = Textures(verts_rgb=textures)
    mesh = Meshes(verts=[vertices], faces=[faces], textures=textureObject)
    return mesh

@torch.no_grad()
def renderVoxelScene(sceneVoxelGrid, threshold, maximumFillThreshold, colours, distances, azimuths, elevations, device):
    depthMaps = []
    images = []
    for channelIndex, (voxelChannel, colour) in enumerate(zip(sceneVoxelGrid, colours)):
        currentFill = (voxelChannel >= threshold).sum()/voxelChannel.numel()
        if currentFill > maximumFillThreshold:
            continue
        objectMeshes = cubify(voxels = voxelChannel.unsqueeze(dim = 0), thresh = threshold, device = device)
        currentVertices, currentFaces = objectMeshes.get_mesh_verts_faces(0)
        if currentVertices.size(0) == 0 or currentFaces.size(0) == 0:
            continue
        currentVertices = currentVertices[:,[2,0,1]]
        currentTexture = colour.to(device = device)
        currentTexture = currentTexture.unsqueeze(dim = 0).expand_as(currentVertices)
        textureObject = Textures(verts_rgb=currentTexture.unsqueeze(dim = 0))
        objectMesh = Meshes(verts=[currentVertices], faces=[currentFaces], textures=textureObject)
        rasteriserSettings = RasterizationSettings  (
                                                        image_size=640, 
                                                        blur_radius=0.0, 
                                                        faces_per_pixel=1, 
                                                    )
        lights = PointLights(device=device, location=[[1.3, 2.0, -0.0]])
        cameras = OpenGLPerspectiveCameras(device = device)
        shader = SoftPhongShader(
                                    device=device, 
                                    cameras=cameras,
                                    lights=lights
                                )
        rasteriser = MeshRasterizer(cameras = cameras, raster_settings = rasteriserSettings)
        renderer = MeshRendererWithDepth(rasteriser = rasteriser, shader = shader)
        for imageIndex, (distance, azimuth, elevation) in enumerate(zip(distances, azimuths, elevations)):
            R, T = look_at_view_transform(distance, elevation, azimuth) 
            cameras = OpenGLPerspectiveCameras(device = device, R= R, T = T)
            shader.cameras = cameras
            rasteriser.cameras = cameras
            currentImage, currentDepth = renderer(objectMesh)
            currentDepth = currentDepth.squeeze(dim = 3)
            if len(depthMaps) <= imageIndex:
                depth = torch.ones_like(currentDepth)*float('inf')
                depthMaps.append(depth)
            else:
                depth = depthMaps[imageIndex]
            if len(images) <= imageIndex:
                image = torch.ones_like(currentImage)
                images.append(image)
            else:
                image = images[imageIndex]
            updateMask = (currentDepth < depth) & (currentDepth > 0) & (currentDepth < float('inf'))
            depth[updateMask] = currentDepth[updateMask]
            image[updateMask] = currentImage[updateMask]
    return [x[0,160:,:,:3] for x in images]


@torch.no_grad()
def generateImages(sceneVoxelGrid, threshold, colours, device, distances, azimuths, elevations):
    images = []
    sceneMesh = generateVoxelGridMesh(voxelScene, 0.5, classColours, device)
    for distance, azimuth, elevation in zip(distances, elevations, azimuths):
        R, T = look_at_view_transform(distance, azimuth, elevation) 
        cameras = OpenGLPerspectiveCameras(device = device, R= R, T = T)
        rasteriserSettings = RasterizationSettings(
                                                                        image_size=640, 
                                                                        blur_radius=0.0, 
                                                                        faces_per_pixel=1, 
                                                                    )
        lights = PointLights(device=device, location=[[1.3, 2.0, -0.0]])
        shader = SoftPhongShader(
                                        device=device, 
                                        cameras=cameras,
                                        lights=lights
                                    )
        rasteriser = MeshRasterizer(cameras = cameras, raster_settings = rasteriserSettings)
        renderer = MeshRenderer(rasterizer = rasteriser, shader = shader)
        image = renderer(sceneMesh)
        images.append(image[0, ..., :3].transpose(0,1).cpu().flip(dims = [0]).numpy())
    return images


if __name__ == '__main__':
    device = torch.device('cuda')
    classColours = hdf5storage.loadmat('officialColours.mat')['colours'][1:17,:]
    classColours = torch.as_tensor(classColours, dtype = torch.float)
    sceneData = numpy.load('F:\\data\\dataFiles_d435_single_instance_preprocessed_numpy_v2\\Data12Heap162.npz')

    voxelScene = torch.as_tensor(sceneData['completeVoxelScene'], dtype = torch.float, device = device)
    threshold = 0.5
    maximumFillThreshold = 0.1

    azimuths = list(range(0, 360, 40))
    distances = [2.7]*len(azimuths)
    elevations = [45]*len(azimuths)
    torch.cuda.reset_max_memory_allocated()
    memoryConsumption = -torch.cuda.max_memory_allocated()
    images = renderVoxelScene(voxelScene, threshold, maximumFillThreshold, classColours, distances, azimuths, elevations, device)
    memoryConsumption += torch.cuda.max_memory_allocated()
    print('Memory consumption was {}GB'.format(memoryConsumption/1e9))
    for index, image in enumerate(images):
        plt.imsave('Image_{}.png'.format(index), image.cpu().numpy())

    print('Testing complete.')


