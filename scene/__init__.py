import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.mlp_model import DecoderMLPs
from scene.entropy_model import EntropyModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians_models, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians_models
        self.num_TFs = len(self.gaussians)
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
                
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        tf_dirs = [os.path.join(args.source_path, tf_name) for tf_name in args.tf_names]
        scene_info = sceneLoadTypeCallbacks["PVNeRFSynthetic"](tf_dirs, args.white_background, args.eval, ply_paths=ply_path)
        
        if not self.loaded_iter:
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]


        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)))
        else:
            for model_idx in range(len(self.gaussians)): 
                self.gaussians[model_idx].create_from_pcd(scene_info.point_clouds[model_idx], self.cameras_extent)

    def save(self, iteration, mlps=None, entropy_model=None):
        for tf_idx in range(self.num_TFs):
            point_cloud_path = os.path.join(self.model_path, f"TF{tf_idx+1:02d}", "point_cloud/iteration_{}".format(iteration))
            self.gaussians[tf_idx].save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        mlps_path = os.path.join(self.model_path, "mlps", "iteration_{}".format(iteration))
        mlps.save_mlp_checkpoints(mlps_path)
        entropy_model.save_mlp_embeddings(mlps_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    
class CameraStack:
    gaussians : GaussianModel

    def __init__(self, args : ModelParams,   resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        

        self.train_cameras = {}
        self.test_cameras = {}
        

        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        
        self.cameras_extent = scene_info.nerf_normalization["radius"]


        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, type='Blender')
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, type='Blender')

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]