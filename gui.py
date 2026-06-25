from icecream import install
install()
ic.configureOutput(includeContext=True)

import glob
import json
import os
import torchvision.transforms
import dearpygui.dearpygui as dpg
from dearpygui_ext.themes import create_theme_imgui_light
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from gaussian_renderer import InfRender
from scene import GaussianModel, DecoderMLPs, EntropyModel
from utils.general_utils import safe_state
from utils.camera_utils import Camera, JSON_to_camera
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov,fov2focal
from scene.palette_color import LearningPaletteColor
from scene.opacity_trans import LearningOpacityTransform
from scene.light_trans import LearningLightTransform
from pyquaternion import Quaternion
import cv2


def screen_to_arcball(p:np.ndarray):
    dist = np.dot(p, p)
    if dist < 1.:
        return np.array([*p, np.sqrt(1.-dist)])
    else:
        return np.array([*normalize_vec(p), 0.])

def normalize_vec(v: np.ndarray):
    if v is None:
        print("None")
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.all(norm == np.zeros_like(norm)):
        return np.zeros_like(v)
    else:
        return v/norm

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))




def scene_composition(scene_dict: dict, args):
    """Updated to match multi_render.py render_sets model loading approach"""
   
    all_TFs = [s.split('_') if '_' in s else [s] for s in scene_dict['all_model_dirs']]
    num_TFs = 0
    gaussians_models, entropy_models, mlps_models = [], [], []
    all_gaussian_models = []
    
    for tf_names in all_TFs: 
        num_TFs += len(tf_names)
        gaussians = [GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.ratio) for _ in range(len(tf_names))]
        entropy_model = EntropyModel(num_TFs=len(tf_names))
        mlps_model = DecoderMLPs(len(tf_names), dataset.feat_dim, dataset.n_offsets)
        
        model_dir = os.path.join(scene_dict['root_model_dir'], '_'.join(tf_names))
        if not os.path.exists(model_dir):
            print(f"Model dir {model_dir} does not exist! Please check again!")
            exit()
        with torch.no_grad():
            bit_stream_path = os.path.join(model_dir, 'bitstreams')
            os.makedirs(bit_stream_path, exist_ok=True)
            max_iters = searchForMaxIteration(os.path.join(model_dir, 'mlps'))
            mlp_folder = os.path.join(model_dir, 'mlps', f"iteration_{max_iters}")
            entropy_model.load_mlp_embeddings(mlp_folder)
            mlps_model.load_mlp_checkpoints(mlp_folder)
            log_info = entropy_model.conduct_decoding(pre_path_name=bit_stream_path, pcs=gaussians)
            print(log_info)
        all_gaussian_models.append(gaussians)

        if len(gaussians) == 0:
            continue
        
        gaussians_models.append(gaussians)
        entropy_models.append(entropy_model)
        mlps_models.append(mlps_model)
    scene_dict['num_TFs'] = num_TFs
    

    return all_gaussian_models, mlps_models, entropy_model

class ArcBallCamera:
    def __init__(self, W, H, fovy=60, near=0.1, far=10, rot=None, translate=None, center=None):
        self.W = W
        self.H = H
        if translate is None:
            self.radius = 1
            self.original_radius = 1
        else:
            self.radius = np.linalg.norm(translate)
            self.original_radius = np.linalg.norm(translate)
            
        self.radius *= 2
        self.fovy = fovy
        self.near = near
        self.far = far

        if center is None:
            self.center = np.array([0, 0, 0], dtype=np.float32)
        else:
            self.center = center

        if rot is None:
            self.rot = R.from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
            self.original_rot = R.from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))
        else:
            self.rot = R.from_matrix(rot)
            self.original_rot = R.from_matrix(rot)

        self.up = -self.rot.as_matrix()[:3, 1]

    @property
    def pose(self):
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        res[:3, 3] -= self.center
        return res

    @property
    def view(self):
        return np.linalg.inv(self.pose)

    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)
    
    def reset_view(self):
        self.rot = self.original_rot
        self.radius = self.original_radius
        self.radius *= 2

    def orbit(self, lastX, lastY, X, Y):
        def vec_angle(v0: np.ndarray, v1: np.ndarray):
            return np.arccos(np.clip(np.dot(v0, v1)/(np.linalg.norm(v0)*np.linalg.norm(v1)), -1., 1.))
        ball_start = screen_to_arcball(np.array([lastX+1e-6, lastY+1e-6]))
        ball_curr = screen_to_arcball(np.array([X, Y]))
        rot_radians = vec_angle(ball_start, ball_curr)
        rot_axis = normalize_vec(np.cross(ball_start, ball_curr))
        q = Quaternion(axis=rot_axis, radians=rot_radians)
        self.rot = self.rot * R.from_matrix(q.inverse.rotation_matrix)
    
    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])




    



def load_ckpts_paths(args):
    """Updated to match multi_render.py render_sets model loading approach"""
    root_model_dir = args.source_dir
    
    all_model_dirs = os.listdir(root_model_dir)
    
    if len(all_model_dirs) == 0:
        raise ValueError(f"No model directories found in {root_model_dir}")
    
    print(all_model_dirs)
    model_dir_path = []
    mlp_folder = []
    bitstream_folder = []
    all_TFs=[s.split('_') if '_' in s else [s] for s in all_model_dirs]
    
    

    for model_dir_name in all_model_dirs:
        
        model_dir_path = os.path.join(root_model_dir, model_dir_name)
        
        max_iteration = searchForMaxIteration(os.path.join(model_dir_path, "mlps"))
        mlp_folder.append(os.path.join(model_dir_path, "mlps", "iteration_" + str(max_iteration)))
        bitstream_folder.append(os.path.join(model_dir_path, "bitstreams"))

    
    ckpts_transforms = {
        'root_model_dir': root_model_dir,
        'all_model_dirs': all_model_dirs,
        'mlp_folder': mlp_folder,
        'bitstream_folder': bitstream_folder,
        'all_TFs': all_TFs,
    }
    print("load_ckpts_paths(): ",ckpts_transforms)
    return ckpts_transforms

class GUI:
    def __init__(self, H, W, fovy, c2w, center, render_fn, render_kwargs, TFnums, debug=True):
        """
        If the image is hdr, set use_hdr2ldr = True for LDR visualization. [0, 1]
        If the image is hdr, set use_hdr2ldr = False, the range of the image is not [0,1].
        """
        self.ctrlW = 475
        self.widget_indent = 75
        self.widget_top = 150
        self.imgW = W
        self.imgH = H
        self.debug = debug
        rot = c2w[:3, :3]
        translate = c2w[:3, 3] - center
        self.TFnums = TFnums
        self.render_fn = render_fn
        self.render_kwargs = render_kwargs
        self.original_palette_colors = [self.render_kwargs["dict_params"]["palette_colors"][TFidx].palette_color for TFidx in range(TFnums)]
        
        self.cam = ArcBallCamera(self.imgW, self.imgH, fovy=fovy * 180 / np.pi, rot=rot, translate=translate, center=center)

        self.render_buffer = np.zeros((self.imgW, self.imgH, 3), dtype=np.float32)
        self.resize_fn = torchvision.transforms.Resize((self.imgH, self.imgW), antialias=True)
        self.downsample = 1
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        self.prev_mouseX = None
        self.prev_mouseY = None
        self.rotating = False
        
        self.light_elevation = 0
        self.light_angle = 180
        self.useHeadlight = True
        
        self.menu = None
        self.mode = None
        self.step()
        self.mode = "render"
        dpg.create_context()
        
        self.setup_font_theme()
        
        light_theme = create_theme_imgui_light()
        dpg.bind_theme(light_theme)
        self.register_dpg()

    def __del__(self):
        dpg.destroy_context()
    
    def setup_font_theme(self):
        with dpg.font_registry():
            default_font = dpg.add_font("./assets/font/Helvetica.ttf", 16)
        with dpg.theme() as theme_button:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (161, 238, 189))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (174, 255, 204))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (205, 250, 219))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)
        dpg.bind_font(default_font)
        self.theme_button = theme_button

    def get_buffer(self, render_results, mode=None):
        if render_results is None or mode is None:
            output = torch.ones(self.imgH, self.imgW, 3, dtype=torch.float32, device='cuda').detach().cpu().numpy()
        else:
            output = render_results[mode]
            
            if mode == "depth":
                output = (output - output.min()) / (output.max() - output.min())
            elif mode == "num_contrib":
                output = output.clamp_max(1000) / 1000

            if len(output.shape) == 2:
                output = output[None]
            if output.shape[0] == 1:
                output = output.repeat(3, 1, 1)
            if "normal" in mode:
                opacity = render_results["opacity"]
                output = output * 0.5 + 0.5 * opacity
                output = output + (1 - opacity)
            elif mode in ["diffuse_term", "specular_term", "ambient_term"]:
                opacity = render_results["opacity"]
                output = output + (1 - opacity)
            if (self.imgH, self.imgW) != tuple(output.shape[1:]):
                output = self.resize_fn(output)

            output = output.permute(1, 2, 0).contiguous().detach().cpu().numpy()
        return output

    @property
    def custom_cam(self):
        w2c = self.cam.view
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        down = self.downsample
        H, W = self.imgH // down, self.imgW // down
        fovy = self.cam.fovy * np.pi / 180
        fovx = fovy * W / H
        custom_cam = Camera(colmap_id=0, R=R, T=-T,
                            FoVx=fovx, FoVy=fovy,
                            image=torch.zeros(3, H, W), gt_alpha_mask=None, image_name=None, uid=0)
        return custom_cam

    @torch.no_grad()
    def render(self):
        self.step()
        dpg.render_dearpygui_frame()

    def step(self):
        self.start.record()
        render_pkg = self.render_fn(viewpoint_camera=self.custom_cam, **self.render_kwargs)
        self.end.record()
        torch.cuda.synchronize()
        t = self.start.elapsed_time(self.end)

        buffer1 = self.get_buffer(render_pkg, self.mode)
        self.render_buffer = buffer1

        if t == 0:
            fps = 0
        else:
            fps = int(1000 / t)

        if self.menu is None:
            self.menu_map = {"phong": "Blinn-Phong", "normal": "Normal", "diffuse_term": "Diffuse",
                             "specular_term": "Specular", "ambient_term": "Ambient"}
            self.inv_menu_map = {v: k for k, v in self.menu_map.items()}
            self.menu = [self.menu_map[k] for k, v in render_pkg.items() if
                         k not in ["pseudo_normal","render", "num_contrib", "surface_xyz", "diffuse_factor","depth", "shininess", "ambient_factor", "specular_factor", "offset_color", "opacity"] and isinstance(v, torch.Tensor) and np.array(v.shape).prod() % (self.imgH * self.imgW) == 0]
            self.menu = ["Blinn-Phong", "Ambient", "Diffuse", "Specular", "Normal"]
            
        else:
            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({fps} FPS)')
            dpg.set_value("_texture", self.render_buffer)
    
    def add_oneTFSlider(self, TFidx):
        def callback_TF_slider(sender, app_data):
            TFidx = int(sender.replace("_slider_TF", "")) - 1
            with torch.no_grad():
                self.render_kwargs["dict_params"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(app_data, dtype=torch.float32, device="cuda")
            self.need_update = True
        
        def callback_TF_color_edit(sender, app_data):
            TFidx = int(sender.replace("_color_TF", "")) - 1
            with torch.no_grad():
                self.render_kwargs["dict_params"]["palette_colors"][TFidx].palette_color = torch.tensor(app_data[:3], dtype=torch.float32, device="cuda")
            self.need_update = True
        
        slider_tag = "_slider_TF" + str(TFidx+1)
        color_tag = "_color_TF" + str(TFidx+1)
        defualt_color = self.render_kwargs["dict_params"]["palette_colors"][TFidx].palette_color.detach().cpu().numpy()
        defualt_color = (defualt_color * 255).astype(np.uint8).tolist()
        
        indent = 0
        slider_width = (self.ctrlW-10)//self.TFnums
      
        with dpg.group():
            dpg.add_text(f"TF{TFidx+1}",indent=indent+slider_width//4)
            dpg.add_slider_float(
                tag=slider_tag,
                label='',
                default_value=0,
                min_value=0,
                max_value=3.0,
                height=300,
                callback=callback_TF_slider,
                vertical=True,
                width=slider_width, 
                indent=indent
            )
            dpg.add_color_edit(tag=color_tag, default_value=defualt_color, callback=callback_TF_color_edit,
                               no_inputs=True, no_label=True, no_alpha=True, indent=indent+slider_width//4)

            
        

    def register_dpg(self):


        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.imgW, self.imgH, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")


        with dpg.window(tag="_primary_window", width=self.imgW, height=self.imgH):

            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        with dpg.window(label="Control", tag="_control_window", width=self.ctrlW, height=self.imgH, pos=(self.imgW, 0),
                        no_resize=True, no_move=True, no_title_bar=True, no_background=True):

            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            with dpg.group(horizontal=True):
                dpg.add_text("Inference Time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            with dpg.collapsing_header(label="Rendering", default_open=True, leaf=True):

                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True
                    
                with dpg.group(horizontal=True):
                    dpg.add_text("Field of View")
                    dpg.add_slider_int(label="",indent=self.widget_top, min_value=1, max_value=120, format="%d deg",
                                   default_value=self.cam.fovy, callback=callback_set_fovy)
                    
                def callback_set_BG_color(sender, app_data):
                    bg_color = app_data[:3]
                    bg_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    self.render_kwargs["bg_color"] = bg_color
                    self.need_update = True
                  
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Background Color")
                    dpg.add_color_edit(label="", no_alpha=True, default_value=[255, 255, 255],
                                       indent=self.widget_top, callback=callback_set_BG_color) 
                
                def callback_reset_view(sender, app_data):
                    self.cam.reset_view()
                    self.need_update = True
                
                def callback_save_image(sender, app_data):
                    rendered_img = self.render_buffer
                    rendered_img = (rendered_img*255).astype(np.uint8)[...,[2,1,0]]
                    cv2.imwrite(os.path.join("./GUI_results", f'rendered_img.png'), rendered_img)
                    print("Image Saved")
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reset View", tag="_button_reset_view", width=self.ctrlW//2, callback=callback_reset_view)
                    dpg.add_button(label="Save Image", tag="_button_save_image",width=self.ctrlW//2, callback=callback_save_image)
                    dpg.bind_item_theme("_button_reset_view", self.theme_button)
                    dpg.bind_item_theme("_button_save_image", self.theme_button)
                                    
            with dpg.collapsing_header(label="Color & Opacity Editing", default_open=True, leaf=True):
                    with dpg.group(horizontal=True, horizontal_spacing=0):
                        for i in range(self.TFnums):
                            self.add_oneTFSlider(i)
                    def callback_reset_color_opacity(sender, app_data):
                        with torch.no_grad():
                            for TFidx in range(self.TFnums):
                                self.render_kwargs["dict_params"]["opacity_factors"][TFidx].opacity_factor = torch.tensor(1.0, dtype=torch.float32, device="cuda")
                                self.render_kwargs["dict_params"]["palette_colors"][TFidx].palette_color = self.original_palette_colors[TFidx]
                                color_value = [int(x*255) for x in self.original_palette_colors[TFidx].detach().cpu().numpy()]
                                dpg.set_value(f"_slider_TF{TFidx+1}", 1)
                                dpg.set_value(f"_color_TF{TFidx+1}", tuple(color_value))
                        self.need_update = True
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Reset Color & Opacity", tag="_button_reset_color_opacity",width=self.ctrlW-15, callback=callback_reset_color_opacity)
                        dpg.bind_item_theme("_button_reset_color_opacity", self.theme_button)
                            
            with dpg.collapsing_header(label="Light Editing", default_open=True, leaf=True):
                def callback_headlight(sender, app_data):
                    if app_data == False:
                        self.useHeadlight = app_data
                        self.render_kwargs["dict_params"]["light_transform"].set_light_theta_phi(self.light_angle, self.light_elevation)
                    else:
                        self.useHeadlight = app_data
                    self.render_kwargs["dict_params"]["light_transform"].useHeadLight = self.useHeadlight
                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_text("Headlight")
                    dpg.add_checkbox(label="", tag="_checkbox_headlight", callback=callback_headlight, default_value=self.useHeadlight)
                
                def callback_light_angle(sender, app_data):
                    if self.useHeadlight:
                        return
                    if sender == "_slider_light_angle":
                        self.light_angle = app_data
                    else:
                        self.light_elevation = app_data
                    
                    self.render_kwargs["dict_params"]["light_transform"].set_light_theta_phi(self.light_angle, self.light_elevation)
                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_text("Azimuthal")
                    dpg.add_slider_int(label="", tag="_slider_light_angle", indent=self.widget_indent,
                                       default_value=self.light_angle, min_value=-180, max_value=180, callback=callback_light_angle)
                with dpg.group(horizontal=True):
                    dpg.add_text("Polar")
                    dpg.add_slider_int(label="", tag="_slider_light_elevation", indent=self.widget_indent,
                                       default_value=self.light_elevation, min_value=-90, max_value=90, callback=callback_light_angle)
                
                def callback_light_multi(sender, app_data):
                    if sender == "_slider_ambient_multi":
                        self.render_kwargs["dict_params"]["light_transform"].ambient_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    elif sender == "_slider_light_intensity_multi":
                        self.render_kwargs["dict_params"]["light_transform"].light_intensity_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    elif sender == "_slider_specular_multi":
                        self.render_kwargs["dict_params"]["light_transform"].specular_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    elif sender == "_slider_shininess_multi":
                        self.render_kwargs["dict_params"]["light_transform"].shininess_multi = torch.tensor(app_data, dtype=torch.float32, device="cuda")
                    self.need_update = True
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Ambient")
                    dpg.add_slider_float(label="", tag="_slider_ambient_multi", indent=self.widget_indent, default_value=1, min_value=0, max_value=5, callback=callback_light_multi)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Diffuse")
                    dpg.add_slider_float(label="", tag="_slider_light_intensity_multi", indent=self.widget_indent, default_value=1, min_value=0, max_value=5, callback=callback_light_multi)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Specular")
                    dpg.add_slider_float(label="", tag="_slider_specular_multi", indent=self.widget_indent, default_value=1, min_value=0, max_value=5, callback=callback_light_multi)
                
                with dpg.group(horizontal=True):
                    dpg.add_text("Shininess")
                    dpg.add_slider_float(label="", tag="_slider_shininess_multi", indent=self.widget_indent, default_value=1, min_value=1, max_value=5, callback=callback_light_multi)
            
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")

        def callback_camera_start_rotate(sender, app_data):
            self.rotating = True

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return
            MouseX, MouseY = dpg.get_mouse_pos()
            x = -(MouseX/ self.imgW - 0.5) * 2
            y = -(MouseY/ self.imgH - 0.5) * 2

            if(self.prev_mouseX is None or self.prev_mouseY is None):
                self.prev_mouseX = x
                self.prev_mouseY = y
                return
            if (self.rotating):
                self.cam.orbit(self.prev_mouseX, self.prev_mouseY, x, y)
                self.prev_mouseX = x
                self.prev_mouseY = y
            
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))
                
        
        def callback_camera_end_rotate(sender, app_data):
            self.rotating = False
            self.prev_mouseX = None
            self.prev_mouseY = None
        
        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))
        
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_start_rotate)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)

            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_end_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_pan)

        dpg.create_viewport(title='ECoNGS', width=self.imgW+self.ctrlW, height=self.imgH, resizable=False)

        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()
        dpg.show_viewport()

def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict

if __name__ == '__main__':
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('-vo', '--view_config', default=None, required=False, help="the config root")
    parser.add_argument('-so', '--source_dir', default=None, required=True, help="the source ckpts dir")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="resume from checkpoint")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--gui_debug", action="store_true", help="show debug info in GUI")

    args = parser.parse_args()
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    
    
    pbr_kwargs = dict()
    scene_dict = load_ckpts_paths(args)
    palette_color_transforms = []
    opacity_transforms = []
    TFcount=0
    
    
    for tf_dir in scene_dict['mlp_folder']:
        tf_names = tf_dir.split('_') if '_' in tf_dir else [tf_dir]
        palette_color_ckpt_names = sorted([f for f in (os.listdir(os.path.join(tf_dir))) if f.startswith('palette_color')])
        for ckpt_idx, ckpt_name in enumerate(palette_color_ckpt_names):
            palette_color_transform = LearningPaletteColor()
            palette_color_transform.create_from_ckpt(os.path.join( tf_dir,  ckpt_name))
            palette_color_transforms.append(palette_color_transform)
            opacity_factor=0.0 if TFcount not in [] else 1.0
            opacity_transform = LearningOpacityTransform(opacity_factor=opacity_factor)
            opacity_transforms.append(opacity_transform)
            TFcount+=1
        
    
    light_transform = LearningLightTransform(theta=180, phi=0)
    pcs, mlp_model, entropy_model  = scene_composition(scene_dict, args)
    
    
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    render_kwargs = {
        "pcs": pcs,
        "pipe": pipe,
        "bg_color": background,
        "mlps": mlp_model,
        "dict_params": {
            "palette_colors": palette_color_transforms,
            "opacity_factors": opacity_transforms,
            "light_transform": light_transform
        }
    }
    
    
    render_fn = InfRender    
    H, W = 800, 800
    fovx = 30 * np.pi / 180
    fovy = focal2fov(fov2focal(fovx, W), H)
    if args.view_config is None:
        c2w = np.array([
            [0.0, 0.0, -1.0, 2.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    else:
        view_config_file = f"{args.view_config}/transforms_test.json"
        view_dict = load_json_config(view_config_file)
        all_views = view_dict["frames"]
        c2w = np.array(all_views[0]["transform_matrix"]).reshape(4, 4) 
        c2w /= 2
        c2w[:3, 1:3] *= -1
    
    windows = GUI(H, W, fovy,
                  c2w=c2w, center=np.zeros(3),
                  render_fn=render_fn, render_kwargs=render_kwargs, TFnums=scene_dict['num_TFs'], debug=args.gui_debug)
    
    while dpg.is_dearpygui_running():
        windows.render()