import torch
from torch import nn
import hdf5storage
import pytorch3d
from pytorch3d import transforms
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import SfMPerspectiveCameras, RasterizationSettings, MeshRasterizer, HardFlatShader, TexturedSoftPhongShader, PointLights, MeshRenderer, look_at_view_transform
import matplotlib.pyplot as plt
import math
import os
import time

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasteriser, shader):
        super().__init__()
        self.rasteriser = rasteriser
        self.shader = shader

    def forward(self, meshes, **kwargs):
        fragments = self.rasteriser(meshes, **kwargs)
        images = self.shader(fragments, meshes, **kwargs)
        return images, fragments.zbuf

class SceneRenderer(nn.Module):
    def __init__(self, objectRenderer, tableRenderer, objectMeshes, tableMesh):
        super().__init__()
        self.objectRenderer = objectRenderer
        self.tableRenderer = tableRenderer
        self.objectMeshes = objectMeshes
        self.tableMesh = tableMesh
    
    def forward(self, objects):
        occludedRGBImage = None
        depthImage = None
        classImage = None
        unoccludedRGBImages = []
        unoccludedDepthImages = []
        classes = []
        for currentObject in objects:
            currentClass = currentObject['objectClass']
            classes.append(currentClass)
            referenceMesh = self.objectMeshes[currentClass]
            currentTransform = currentObject['transform']
            vertices = [(torch.matmul(referenceMesh.verts_list()[0], currentTransform[:3,:3].transpose(0,1)) + currentTransform[:3,3].unsqueeze(dim = 0)).contiguous()]
            faces = referenceMesh.faces_list()
            textures = referenceMesh.textures
            currentMesh = Meshes(vertices, faces, textures=textures)
            currentImage, currentDepth = self.objectRenderer(currentMesh)
            currentImage = currentImage[:,80:560,:,:]
            currentDepth = currentDepth[:,80:560,:,:]
            if occludedRGBImage is None:
                occludedRGBImage = torch.zeros_like(currentImage)
            if depthImage is None:
                depthImage = torch.ones_like(currentDepth)*float('inf')
            if classImage is None:
                classImage = torch.zeros_like(depthImage)
            unoccludedRGBImages.append(currentImage.cpu())
            unoccludedDepthImages.append(currentDepth.cpu())
            currentMask = (currentDepth > 0 ) & ( currentDepth < depthImage )
            depthImage[currentMask] = currentDepth[currentMask]
            classImage[currentMask] = currentClass + 1
            currentMask = currentMask.expand_as(occludedRGBImage)
            occludedRGBImage[currentMask] = currentImage[currentMask]

        currentImage, currentDepth = self.tableRenderer(self.tableMesh)
        currentClass = -1
        currentImage = currentImage[:,80:560,:,:]
        currentDepth = currentDepth[:,80:560,:,:]
        if occludedRGBImage is None:
            occludedRGBImage = torch.zeros_like(currentImage)
        if depthImage is None:
            depthImage = torch.ones_like(depthImage)*float('inf')
        if classImage is None:
            classImage = torch.zeros_like(depthImage)
        unoccludedRGBImages.append(currentImage.cpu())
        unoccludedDepthImages.append(currentDepth.cpu())
        currentMask = (currentDepth > 0 ) & ( currentDepth < depthImage )
        depthImage[currentMask] = currentDepth[currentMask]
        classImage[currentMask] = currentClass
        currentMask = currentMask.expand_as(occludedRGBImage)
        occludedRGBImage[currentMask] = currentImage[currentMask]
        classes.append(currentClass)
        outputs = dict( occludedRGBImage = occludedRGBImage.cpu(),
                        depthImage = depthImage.cpu(),
                        classImage = classImage.cpu(), 
                        unoccludedRGBImages = [x.cpu() for x in unoccludedRGBImages],
                        unoccludedDepthImages = [x.cpu() for x in unoccludedDepthImages],
                        classes = classes )
        return outputs

class Pytorch3DRenderer(nn.Module):
    def __init__(self, cameraIntrinsics, cameraExtrinsics, lightPositions, objectMeshes, tableMesh, device):
        super().__init__()
        if cameraIntrinsics.dim() == 3:
            cameraIntrinsics = [x for x in cameraIntrinsics]
        elif cameraIntrinsics.dim() == 2:
            cameraIntrinsics = [cameraIntrinsics]
        else:
            raise ValueError('Unknown camera intrinsics shape: {}'.format(cameraIntrinsics.shape))
        focalLength = torch.stack([x.diag()[:2] for x in cameraIntrinsics])
        principalPoint  = torch.stack([x[:2,2] for x in cameraIntrinsics])
        if cameraExtrinsics.dim() == 3:
            cameraExtrinsics = [x for x in cameraExtrinsics]
        elif cameraExtrinsics.dim() == 2:
            cameraExtrinsics = [cameraExtrinsics]
        else:
            raise ValueError('Unknown camera intrinsics shape: {}'.format(cameraExtrinsics.shape))
        R = torch.stack([x[:3,:3] for x in cameraExtrinsics]) #.transpose(0,1)
        T = torch.stack([x[:3,3] for x in cameraExtrinsics])
        self.cameras = SfMPerspectiveCameras(focal_length = focalLength, principal_point = principalPoint, R = R, T = T, device = device)
        self.rasterisationSettings = RasterizationSettings(image_size = 640, blur_radius = 0.0, faces_per_pixel = 1, bin_size = None, max_faces_per_bin = None, perspective_correct = True, cull_backfaces = True)
        self.rasteriser = MeshRasterizer(self.cameras, self.rasterisationSettings)
        self.lights = PointLights(location = lightPositions, device = device)
        self.flatShader = HardFlatShader(device = device, cameras = self.cameras, lights = self.lights)
        self.phongShader = TexturedSoftPhongShader(device = device, cameras = self.cameras, lights = self.lights)
        self.flatRenderer = MeshRendererWithDepth(self.rasteriser, self.flatShader ) 
        self.phongRenderer = MeshRendererWithDepth(self.rasteriser, self.phongShader ) 
        self.sceneRenderer = SceneRenderer(self.phongRenderer, self.flatRenderer, objectMeshes, tableMesh)

    def forward(self, objects):
        return self.sceneRenderer(objects)

def loadObjects(objectDirectory, device = torch.device('cuda')):
    # load camera information
    renderData = hdf5storage.loadmat(os.path.join(objectDirectory, 'renderData.mat')) 
    K = torch.as_tensor(renderData['k'], dtype = torch.float, device = device)
    # Convert to normalised device coordinates
    K[0,0] /= 320
    K[1,1] /= 320
    k0 = 1 - K[0,2]/320
    k1 = 1 - K[1,2]/240
    K[1,2] = k1
    K[0,2] = k0
    K = torch.cat([K,torch.zeros(1, 3, dtype = torch.float, device = device)], dim = 0)
    K[2,2] = 1
    K = K.unsqueeze(0).to(device)
    R = torch.as_tensor(renderData['r'], dtype = torch.float, device = device)
    # Rotate camera to account for Pytorch3D coordinate system
    euler = transforms.matrix_to_euler_angles(R,'ZYX')
    euler[0] += math.pi
    R = transforms.euler_angles_to_matrix(euler,'ZYX')
    R = R.unsqueeze(0).to(device)
    t = torch.as_tensor(renderData['t'], dtype = torch.float, device = device)
    t = t.unsqueeze(0).to(device)
    T = torch.cat([R, t.transpose(1,2)], dim = 2)

    # Remake camera
    # R, t = look_at_view_transform(dist = 0.5, elev = 45*7, azim = 180*0, at = [[0,0,0]], up = [[0,0,1]])
    # T = torch.cat([R, t.transpose(0,1).unsqueeze(dim = 0)], dim = 2)

    # R, t = look_at_view_transform(dist = 0.9, elev = 45)
    # T = torch.cat([R, t.unsqueeze(dim = 2)], dim = 2)
    
    objectTransforms = torch.as_tensor(renderData['objectCentres'], dtype = torch.float, device = device).unsqueeze(0)
    objectInformation = renderData['objectInformation'].squeeze()
    outputs = dict(K = K, R = R, t = t, T = T)
    outputs['objectMeshes'] = []
    for objectIndex, currentObject in enumerate(objectInformation):
        file = os.path.join(objectDirectory,'{}.obj'.format(currentObject['name'][0]))
        currentTransform = objectTransforms[0,objectIndex:objectIndex+1,:]
        mesh = load_objs_as_meshes(files = [file], device = device, load_textures = True)[0]
        # Centre objects
        vertices = [mesh.verts_list()[0] - currentTransform]
        faces = mesh.faces_list()
        textures = mesh.textures
        mesh = Meshes(vertices, faces, textures=textures)
        outputs['objectMeshes'].append(mesh)
    tableFile = os.path.join(objectDirectory,'{}.obj'.format('table'))
    mesh = load_objs_as_meshes(files = [tableFile], device = device, load_textures = True)[0]
    # Convert texture to vertex colour to allow use of flat shader
    vertices = mesh.verts_list()
    faces = mesh.faces_list()
    vertexRGB = torch.ones_like(vertices[0])*175/255
    textures = Textures(verts_rgb=vertexRGB.unsqueeze(dim = 0))
    outputs['tableMesh'] = Meshes(verts=vertices, faces=faces, textures=textures)
    outputs['lightPosition'] = torch.as_tensor([0,0,1.5], dtype = torch.float, device = device).unsqueeze(dim = 0)
    return outputs

if __name__ == '__main__':
    objectDirectory = 'ycbobj'
    device = torch.device('cuda')
    sceneData = loadObjects(objectDirectory, device = device)
    renderer = Pytorch3DRenderer(   cameraIntrinsics = sceneData['K'], cameraExtrinsics = sceneData['T'], lightPositions = sceneData['lightPosition'], 
                                    objectMeshes = sceneData['objectMeshes'], tableMesh = sceneData['tableMesh'], device = device)
    groundTruthData = hdf5storage.loadmat('Images14ObjectHeap854.mat', variable_names = ['heap', 'heapRGBCompositeImage', 'heapDepthImage'])
    heap = groundTruthData['heap'].squeeze()
    objects = []
    for currentObject in heap:
        rotationMatrix = torch.as_tensor(currentObject['rotationMatrix'], dtype = torch.float, device = device)
        translationVector = torch.as_tensor(currentObject['translationVector'], dtype = torch.float, device = device)
        currentInformation = dict(  objectClass = int(currentObject['objectLibraryIndex'].item())-1,
                                    transform = torch.cat([rotationMatrix, translationVector.transpose(1,2)], dim = 2).squeeze())
        objects.append(currentInformation)
    renderingTime = -time.time()
    with torch.no_grad():
        imageData = renderer(objects)
    renderingTime += time.time()
    print('Rendering took {} seconds.'.format(renderingTime))
    renderedRGBImage = imageData['occludedRGBImage'].squeeze()[:,:,:3].cpu().numpy()
    renderedDepthImage = imageData['depthImage'].squeeze()
    groundTruthRGBImage = groundTruthData['heapRGBCompositeImage'].squeeze()
    groundTruthDepthImage = groundTruthData['heapDepthImage'].squeeze()
    # depthMask = (renderedDepthImage < float('inf')) & (groundTruthDepthImage < float('inf'))
    # depthDifference = renderedDepthImage - groundTruthDepthImage
    # temp = depthDifference.clone()
    # temp[~depthMask.to(torch.bool)] = 0
    # temp = temp/temp.abs().max()
    # tempImage = torch.zeros(480,640,3, dtype = torch.float)
    # tempImage[:,:,0] = temp.clamp_min(0)
    # tempImage[:,:,1] = (-temp).clamp_min(0)
    # plt.imshow(tempImage.numpy())
    # plt.show()


    # plt.matshow(depthDifference)
    # plt.colorbar()
    # plt.show()
    
    plt.imshow(renderedRGBImage)
    plt.show()
    print('Execution complete')