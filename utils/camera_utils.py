from scene.cameras import Camera, MyCamera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
import math
WARNED = False



def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def MyloadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_images_rgb = [PILtoTorch(image, resolution) for image in cam_info.images]
    resized_images_rgb = torch.stack(resized_images_rgb, dim=0)

    gt_images = resized_images_rgb[:, 0:3, ...]
    loaded_masks = None

    if resized_images_rgb.shape[2] == 4:
        loaded_masks = resized_images_rgb[:, 3:4, ...]

    return MyCamera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  images=gt_images, gt_alpha_masks=loaded_masks,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, type='PVNeRFSynthetic'):
    camera_list = []
    if type == 'Blender':
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCam(args, id, c, resolution_scale))
    else:
        for id, c in enumerate(cam_infos):
            camera_list.append(MyloadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def JSON_to_camera(json_cam):
    rot = np.array(json_cam['rotation'])
    pos = np.array(json_cam['position'])
    W2C = np.zeros((4, 4))
    W2C[:3, :3] = rot
    W2C[:3, 3] = pos
    W2C[3, 3] = 1
    Rt = np.linalg.inv(W2C)
    R = Rt[:3, :3].transpose()
    T = Rt[:3, 3]
    H, W = json_cam['height'], json_cam['width']
    if 'cx' not in json_cam:
        if 'fx' in json_cam:
            FovX = focal2fov(json_cam["fx"], W)
            FovY = focal2fov(json_cam["fy"], H)
        else:
            FovX = json_cam["FoVx"]
            FovY = json_cam["FoVy"]
        camera = Camera(colmap_id=0, R=R, T=T, FoVx=FovX, FoVy=FovY, fx=None, fy=None, cx=None, cy=None,
                        image=None, image_name=json_cam['img_name'], uid=json_cam['id'],
                        data_device='cuda', height=H, width=W)
    else:
        camera = Camera(colmap_id=0, R=R, T=T, FoVx=None, FoVy=None, fx=json_cam["fx"], fy=json_cam["fy"],
                        cx=json_cam["cx"], cy=json_cam["cy"], image=None, image_name=json_cam['img_name'],
                        uid=json_cam['id'], data_device='cuda', height=H, width=W)
    return camera