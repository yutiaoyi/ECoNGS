import os
import torch

import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import CameraStack
import json
import time
from gaussian_renderer import InfRender
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from os import makedirs
from utils.system_utils import searchForMaxIteration
from scene import Scene, GaussianModel, DecoderMLPs, EntropyModel
from scene.palette_color import LearningPaletteColor
from scene.opacity_trans import LearningOpacityTransform
from scene.light_trans import LearningLightTransform

def model_size_computation(root_model_dir, render_TFs, gaussians_models):
    from pathlib import Path

    def get_folder_size(path):
        return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    
    tf_dirs = [d for d in os.listdir(root_model_dir) if os.path.isdir(os.path.join(root_model_dir, d))]
    total_size = 0
    for tf_idx, tf_dir in enumerate(tf_dirs):
        tf_names = tf_dir.split('_') if '_' in tf_dir else [tf_dir]
        bitstream_path = os.path.join(root_model_dir, tf_dir, 'bitstreams')
        tf_model_size = get_folder_size(bitstream_path)
        total_num_GS = [gm._anchor.shape[0] for gm in gaussians_models[tf_idx]]
        used_num_GS = 0
        for tf_name in tf_names:
            if tf_name in render_TFs:
                used_num_GS += total_num_GS[tf_names.index(tf_name)]
                print(tf_dir, tf_name)
        used_GS_ratio = used_num_GS / sum(total_num_GS)
        if used_num_GS != 0:
            total_size += tf_model_size * used_GS_ratio / (1024.0**2) + 191.7/1024
    print('\n')
    print(f"Estimated total model size for rendering: {total_size:.5f} MB")
    print('\n')



def render_set(output_dir, shading_kwargs, views, gaussians_models, mlps_models, pipeline, background, saving_image=True):
    render_path = output_dir
    makedirs(render_path, exist_ok=True)

    t_list = []
    name_list = []
    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            
            torch.cuda.synchronize();t_start = time.time()
            voxel_visible_mask = None
            render_pkg = InfRender(view, gaussians_models, mlps_models, pipe=pipeline, bg_color=background, dict_params=shading_kwargs, visible_mask=voxel_visible_mask)
            torch.cuda.synchronize();t_end = time.time()
            t_list.append(t_end - t_start)
            if saving_image:
                rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)

                name_list.append('{0:05d}'.format(idx) + ".png")
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    
    return t_list

def render_sets(dataset : ModelParams, shading_kwargs, render_TFs:list, root_model_dir, output_dir, pipeline : PipelineParams, skip_test=False):
    with torch.no_grad():
        all_model_dirs = [d for d in os.listdir(root_model_dir) if os.path.isdir(os.path.join(root_model_dir, d))]
        all_TFs = [s.split('_') if '_' in s else [s] for s in all_model_dirs]
        
        gaussians_models, entropy_models, mlps_models = [], [], []
        all_gaussian_models = []
        for tf_names in all_TFs: 
            gaussians = [GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.ratio) for _ in range(len(tf_names))]
            entropy_model = EntropyModel(num_TFs=len(tf_names))
            mlps_model = DecoderMLPs(len(tf_names), dataset.feat_dim, dataset.n_offsets)
            
            model_dir = os.path.join(root_model_dir, '_'.join(tf_names))
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
            selected_gaussians = []
            remove_embedding_idx = []
            for tf_name in tf_names:
                if tf_name not in render_TFs:
                    remove_embedding_idx.append(tf_names.index(tf_name))
                else:
                    selected_gaussians.append(gaussians[tf_names.index(tf_name)])
            
            mlps_model.remove_unused_embeddings(remove_embedding_idx)
            gaussians = selected_gaussians
            if len(gaussians) == 0:
                continue
            
            gaussians_models.append(gaussians)
            entropy_models.append(entropy_model)
            mlps_models.append(mlps_model)
        
        model_size_computation(root_model_dir, render_TFs, all_gaussian_models)

        cams = CameraStack(dataset)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        render_set(output_dir, shading_kwargs, cams.getTestCameras(), gaussians_models, mlps_models, pipeline, background, saving_image=True)
        t_test_list = render_set(output_dir, shading_kwargs, cams.getTestCameras(), gaussians_models, mlps_models, pipeline, background, saving_image=False)
        test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
        print(f'Test FPS: {test_fps.item():.5f}')


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--root_model_dir", default="", type=str, help="List of model paths to be rendered together.")
    parser.add_argument("--TF_names", nargs='+', default="", type=str, help="List of model paths to be rendered together.")
    parser.add_argument("--output_dir", default="", type=str, help="Path for saving rendered images.")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    args = get_combined_args(parser)
    
    

    safe_state(args.quiet)
    
    root_model_dir = args.root_model_dir
    render_TFs = args.TF_names
    tf_dirs = [d for d in os.listdir(root_model_dir) if os.path.isdir(os.path.join(root_model_dir, d))]
    print(f"Rendering TFs: {render_TFs} in {root_model_dir}")
   
    shading_kwargs = dict()
    palette_color_transforms = []
    
    
    
    for tf_dir in tf_dirs:
        tf_names = tf_dir.split('_') if '_' in tf_dir else [tf_dir]
        palette_color_ckpt_names = sorted([f for f in (os.listdir(os.path.join(root_model_dir, tf_dir, 'mlps', 'iteration_20000'))) if f.startswith('palette_color')])
        
        for ckpt_idx, ckpt_name in enumerate(palette_color_ckpt_names):
            if tf_names[ckpt_idx] in render_TFs:
                print(f"Loading palette color for {tf_names[ckpt_idx]} from {os.path.join(root_model_dir, tf_dir, 'mlps', 'iteration_20000', ckpt_name)}")
                palette_color_transform = LearningPaletteColor()
                palette_color_transform.create_from_ckpt(os.path.join(root_model_dir, tf_dir, 'mlps', 'iteration_20000', ckpt_name))
                palette_color_transforms.append(palette_color_transform)
        
    shading_kwargs["palette_colors"] = palette_color_transforms
    shading_kwargs["opacity_factors"] = [LearningOpacityTransform(1.0) for _ in range(len(render_TFs))]
    shading_kwargs["light_transform"] = LearningLightTransform()

    render_sets(model.extract(args), shading_kwargs, render_TFs, root_model_dir, args.output_dir, pipeline.extract(args), args.skip_test)

    if os.environ.get("ECONGS_TIMING") == "1":
        import json
        from gaussian_renderer.inference import exp7_timing
        n = max(exp7_timing["n"], 1)
        out = {k: exp7_timing[k] / n for k in ["mlp", "shading", "raster"]}
        out["n_views"] = exp7_timing["n"]
        out["n_gauss"] = exp7_timing["n_gauss"] / n
        json.dump(out, open(os.environ.get("ECONGS_TIMING_OUT", "/tmp/exp7.json"), "w"))
        print(f"[EXP7] per-view ms  MLP={out['mlp']:.3f}  shading={out['shading']:.3f}  raster={out['raster']:.3f}  (n={out['n_views']})")
