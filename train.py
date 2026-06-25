import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import subprocess

import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import Myrender
import sys
from scene import Scene, GaussianModel, DecoderMLPs, EntropyModel
from utils.general_utils import safe_state, DecayScheduler
from utils.system_utils import getTFDirs
from scene.palette_color import LearningPaletteColor
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, resize_image, downsample_image, blur_image, save_numpy_image
from utils.encodings import get_binary_vxl_size
from utils.system_utils import searchForMaxIteration
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from fused_ssim import fused_ssim
import auto_debug
from gaussian_renderer import InfRender

import torch.profiler

bit2MB_scale = 8 * 1024 * 1024
run_codec = True


def save_comparison_subplot(gt_images, rendered_images, view_names, save_dir, config_name, tf_idx, iteration):
    """
    Save a comparison subplot showing GT and rendered images side by side
    """
    n_views = len(gt_images)
    if n_views == 0:
        return
    
    fig, axes = plt.subplots(2, n_views, figsize=(4*n_views, 8))
    
    if n_views == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(f'TF {tf_idx} - {config_name.upper()} - Iteration {iteration}', fontsize=16, fontweight='bold')
    
    for i in range(n_views):
        axes[0, i].imshow(np.clip(gt_images[i], 0, 1))
        axes[0, i].set_title(f'GT - {view_names[i]}', fontsize=12)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(np.clip(rendered_images[i], 0, 1))
        axes[1, i].set_title(f'Rendered - {view_names[i]}', fontsize=12)
        axes[1, i].axis('off')
    
    fig.text(0.02, 0.75, 'Ground Truth', rotation=90, fontsize=14, fontweight='bold', ha='center', va='center')
    fig.text(0.02, 0.25, 'Rendered', rotation=90, fontsize=14, fontweight='bold', ha='center', va='center')
    
    save_path = os.path.join(save_dir, f'comparison_TF{tf_idx}_{config_name}_iter{iteration}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Saved comparison image: {save_path}")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')

train_prof = {
    "data_prep": [], "rendering": [], "loss_compute": [], "backward": [], "optimizer": [], "other": [], "total": []
}

def training(dataset, opt, pipe, num_TFs, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None, shading_kwargs=None, verbose=False):
    profile_flag = False
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians_models = [GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.ratio) for _ in range(num_TFs)]
    entropy_model = EntropyModel(num_TFs=num_TFs)
    mlps_model = DecoderMLPs(num_TFs, dataset.feat_dim, dataset.n_offsets)
    scene = Scene(dataset, gaussians_models, ply_path=ply_path, shuffle=False)
    
    for gaussians in gaussians_models:
        gaussians.training_setup(opt)
    mlps_model.training_setup(opt)
    entropy_model.training_setup(opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    loss_item_list = []
    loss_csv_path = os.path.join(scene.model_path, 'loss_curve.csv')
    loss_csv_f = open(loss_csv_path, 'w')
    loss_csv_f.write('iteration,total_loss,l1_loss,ssim_loss,scaling_reg\n')

    training_tic = time.time()
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        
        for gaussians in gaussians_models:
            gaussians.update_learning_rate(iteration)
        mlps_model.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        gt_image = viewpoint_cam.original_image.cuda()
        voxel_visible_mask = None
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = Myrender(viewpoint_cam, gaussians_models, mlps_model, entropy_model, pipe,  background, image_shape=gt_image.shape, visible_mask=voxel_visible_mask, dict_params=shading_kwargs, retain_grad=retain_grad, step=iteration)


        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        num_anchors = render_pkg["total_num_anchors"]
        bit_per_param = render_pkg["bit_per_param"]
        if verbose:
            bit_per_feat_param = render_pkg["bit_per_feat_param"]
            bit_per_scaling_param = render_pkg["bit_per_scaling_param"]
            bit_per_offsets_param = render_pkg["bit_per_offsets_param"]

            if iteration % 1000 == 0 and bit_per_param is not None:

                ttl_size_feat_MB = bit_per_feat_param.item() * num_anchors * gaussians.feat_dim / bit2MB_scale
                ttl_size_scaling_MB = bit_per_scaling_param.item() * num_anchors * 6 / bit2MB_scale
                ttl_size_offsets_MB = bit_per_offsets_param.item() * num_anchors * 3 * gaussians.n_offsets / bit2MB_scale
                ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB

                logger.info("\n----------------------------------------------------------------------------------------")
                logger.info("\n-----[ITER {}] bits info: bit_per_feat_param={}, anchor_num={}, ttl_size_feat_MB={}-----".format(iteration, bit_per_feat_param.item(), num_anchors, ttl_size_feat_MB))
                logger.info("\n-----[ITER {}] bits info: bit_per_scaling_param={}, anchor_num={}, ttl_size_scaling_MB={}-----".format(iteration, bit_per_scaling_param.item(), num_anchors, ttl_size_scaling_MB))
                logger.info("\n-----[ITER {}] bits info: bit_per_offsets_param={}, anchor_num={}, ttl_size_offsets_MB={}-----".format(iteration, bit_per_offsets_param.item(), num_anchors, ttl_size_offsets_MB))
                logger.info("\n-----[ITER {}] bits info: bit_per_param={}, anchor_num={}, ttl_size_MB={}-----".format(iteration, bit_per_param.item(), num_anchors, ttl_size_MB))
                with torch.no_grad():
                    binary_grid_masks_anchor = gaussians.get_mask_anchor.float()
                    mask_1_rate, mask_size_bit, mask_size_MB, mask_numel = get_binary_vxl_size(binary_grid_masks_anchor + 0.0)
                logger.info("\n-----[ITER {}] bits info: 1_rate_mask={}, mask_numel={}, mask_size_MB={}-----".format(iteration, mask_1_rate, mask_numel, mask_size_MB))

        image = image.view(-1, 3, image.shape[-2], image.shape[-1])
        gt_image = gt_image.view(-1, 3, gt_image.shape[-2], gt_image.shape[-1])
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - fused_ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

        if bit_per_param is not None:
            _, bit_hash_grid, MB_hash_grid, _ = get_binary_vxl_size((entropy_model.get_encoding_params()+1)/2)
            denom = num_anchors*(gaussians.feat_dim+6+6*gaussians.n_offsets)
            loss = loss + opt.lmbda * (bit_per_param + bit_hash_grid / denom)

    
        loss.backward()
        loss_csv_f.write(f'{iteration},{loss.item():.6f},{Ll1.item():.6f},{ssim_loss.item():.6f},{scaling_reg.item():.6f}\n')
        if iteration % 100 == 0:
            loss_csv_f.flush()

        if verbose:
            psnr_val = psnr(image, gt_image).mean().item()
            rendering_loss_item = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
            loss_item_list.append(rendering_loss_item.item())
        

        if iteration == testing_iterations[-1]:
            training_toc = time.time()
            logger.info(f"\nTraining time: {training_toc - training_tic:.2f} seconds")
        
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                if verbose:
                    progress_bar.set_postfix({"num of anchor":f"{num_anchors}","PSNR": f"{psnr_val:.{7}f}"})
                else:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                   
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, mlps_model, entropy_model)
                for com_name, component in shading_kwargs.items():
                    if com_name == "palette_colors":
                        for idx, palette_color_transform in enumerate(component):
                            torch.save(palette_color_transform.capture(), os.path.join(scene.model_path, 'mlps', f'iteration_{iteration}', f"{com_name}_{idx}_chkpnt" + ".pth"))
                            print("\n[ITER {}] Saving Checkpoint".format(iteration))
                
            training_report(tb_writer, mlps_model, entropy_model, dataset_name, shading_kwargs, iteration, Ll1, loss, l1_loss, testing_iterations, scene, Myrender, (pipe, background), wandb, logger)

            if iteration < opt.update_until and iteration > opt.start_stat:
                for tf_idx, gaussians in enumerate(gaussians_models):
                    gaussians.training_statis(viewspace_point_tensor[tf_idx], opacity[tf_idx], visibility_filter[tf_idx], offset_selection_mask[tf_idx], None)
                
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    
                    if iteration not in range(3000, 4000):
                        for tf_idx, gaussians in enumerate(gaussians_models):
                            gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)

            elif iteration == opt.update_until:
                for guassians in gaussians_models:
                    try:
                        del guassians.opacity_accum
                        del guassians.offset_gradient_accum
                        del guassians.offset_denom
                    except:
                        pass
                torch.cuda.empty_cache()
                
                    
        if iteration < opt.iterations:
            mlps_model.optimizer.step()
            mlps_model.optimizer.zero_grad(set_to_none = True)
            entropy_model.optimizer.step()
            entropy_model.optimizer.zero_grad(set_to_none = True)
            
            for component in shading_kwargs.values():
                try:
                    if isinstance(component, list):
                        for c in component:
                            c.step()
                    else:
                        component.step()
                except:
                    pass
                
            for gaussians in gaussians_models:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                


    loss_csv_f.close()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    return tb_writer

def training_report(tb_writer, mlps_model, entropy_model, dataset_name, shading_kwargs, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if iteration in testing_iterations:
        mlps_model.eval()
        if iteration == testing_iterations[-1]:
                if run_codec:
                    with torch.no_grad():
                        bit_stream_path = os.path.join(scene.model_path, 'bitstreams')
                        os.makedirs(bit_stream_path, exist_ok=True)
        
                        log_info = entropy_model.conduct_encoding(pre_path_name=bit_stream_path, pcs=scene.gaussians)
                        max_iters = searchForMaxIteration(os.path.join(scene.model_path, 'mlps'))
                        mlp_folder = os.path.join(scene.model_path, 'mlps', f"iteration_{max_iters}")
                        entropy_model.load_mlp_embeddings(mlp_folder)
                        mlps_model.load_mlp_checkpoints(mlp_folder)
                        log_info = entropy_model.conduct_decoding(pre_path_name=bit_stream_path, pcs=scene.gaussians)
                        logger.info(log_info)
                    
        
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        num_TFs = scene.num_TFs
        
        comparison_dir = os.path.join(scene.model_path, 'comparison_images')
        os.makedirs(comparison_dir, exist_ok=True)
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = [0.0 for _ in range(num_TFs)]
                psnr_test = [0.0 for _ in range(num_TFs)]
                lpips_test = [0.0 for _ in range(num_TFs)]
                
                all_gt_images = [[] for _ in range(num_TFs)]
                all_rendered_images = [[] for _ in range(num_TFs)]
                view_names = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = None
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, mlps_model, entropy_model, *renderArgs, dict_params=shading_kwargs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    image = torch.split(image, 3, dim=0)
                    gt_image = torch.split(gt_image, 3, dim=0)
                    
                    if idx < 6:
                        view_names.append(viewpoint.image_name)
                        for i in range(num_TFs):
                            gt_np = gt_image[i].detach().cpu().numpy().transpose(1, 2, 0)
                            render_np = image[i].detach().cpu().numpy().transpose(1, 2, 0)
                            all_gt_images[i].append(gt_np)
                            all_rendered_images[i].append(render_np)
                    
                    if config['name'] == 'test':
                        for i in range(num_TFs):
                            iter_render_dir = os.path.join(scene.model_path, 'test', f'iter{iteration}_TF{i}', 'renders')
                            iter_gt_dir = os.path.join(scene.model_path, 'test', f'iter{iteration}_TF{i}', 'gt')
                            os.makedirs(iter_render_dir, exist_ok=True)
                            os.makedirs(iter_gt_dir, exist_ok=True)
                            torchvision.utils.save_image(image[i], os.path.join(iter_render_dir, f'{idx:05d}.png'))
                            torchvision.utils.save_image(gt_image[i], os.path.join(iter_gt_dir, f'{idx:05d}.png'))
            
                    for i in range(num_TFs):
                        l1_test[i] += l1_loss(image[i], gt_image[i]).mean().double()
                        psnr_test[i] += psnr(image[i], gt_image[i]).mean().double()
                        
                        
                        
                
                    
                psnr_test = [psnr_val / len(config['cameras']) for psnr_val in psnr_test]
                l1_test = [l1_val / len(config['cameras']) for l1_val in l1_test]  
                
                for i in range(num_TFs):
                    logger.info("\n[ITER {}] Evaluating {} for TF {} : L1 {} PSNR {}".format(iteration,  config['name'], i, l1_test[i].item(), psnr_test[i].item()))

        torch.cuda.empty_cache()
        mlps_model.train()



def render_set(model_path, name, shading_kwargs, tf_idx, views, gaussian, mlps_model, entropy_model, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_TF{}".format(tf_idx), "renders")
    error_path = os.path.join(model_path, name, "ours_TF{}".format(tf_idx), "errors")
    gts_path = os.path.join(model_path, name, "ours_TF{}".format(tf_idx), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    
    t_list = []
    name_list = []
    per_view_dict = {}
    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            
            torch.cuda.synchronize();t_start = time.time()
            voxel_visible_mask = None
            render_pkg = Myrender(view, [gaussian], mlps_model, entropy_model, pipe=pipeline, bg_color=background, dict_params=shading_kwargs, visible_mask=voxel_visible_mask, force_tf_idx=tf_idx)
            torch.cuda.synchronize();t_end = time.time()
            t_list.append(t_end - t_start)

            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)

            gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            gt_image = torch.split(gt_image, 3, dim=0)
            gt = gt_image[tf_idx]
            
            errormap = (rendering - gt).abs()
            
            name_list.append('{0:05d}'.format(idx) + ".png")
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    return t_list

def render_sets(dataset : ModelParams, shading_kwargs, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None, num_TFs=1):
    with torch.no_grad():
        gaussians_models = [GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.ratio) for _ in range(num_TFs)]
        entropy_model = EntropyModel(num_TFs=num_TFs)
        mlps_model = DecoderMLPs(num_TFs, dataset.feat_dim, dataset.n_offsets)
        
        scene = Scene(dataset, gaussians_models,shuffle=False)
        
        with torch.no_grad():
            bit_stream_path = os.path.join(scene.model_path, 'bitstreams')
            os.makedirs(bit_stream_path, exist_ok=True)
            max_iters = searchForMaxIteration(os.path.join(scene.model_path, 'mlps'))
            mlp_folder = os.path.join(scene.model_path, 'mlps', f"iteration_{max_iters}")
            entropy_model.load_mlp_embeddings(mlp_folder)
            mlps_model.load_mlp_checkpoints(mlp_folder)

            log_info = entropy_model.conduct_decoding(pre_path_name=bit_stream_path, pcs=scene.gaussians)
            logger.info(log_info)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        for tf_idx in range(num_TFs):
        
            if not skip_train:
                t_train_list  = render_set(dataset.model_path, "train", shading_kwargs, tf_idx, scene.getTrainCameras(), gaussians_models[tf_idx], mlps_model, entropy_model, pipeline, background)
                train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
                logger.info(f'Train FPS: {train_fps.item():.5f}')

            if not skip_test:
                t_test_list = render_set(dataset.model_path, "test", shading_kwargs, tf_idx, scene.getTestCameras(), gaussians_models[tf_idx], mlps_model, entropy_model, pipeline, background)
                test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
                logger.info(f'Test FPS: {test_fps.item():.5f}')
    


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        print("")
    
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    logger.info(f'args: {args}')        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]


    logger.info("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset_args = lp.extract(args)
    num_TFs = len(dataset_args.tf_names)
    tf_dirs = [os.path.join(dataset_args.source_path, tf_name) for tf_name in dataset_args.tf_names]

    shading_kwargs = dict()
    palette_color_transforms = []
    for tf_dir in tf_dirs:
        palette_color_transform = LearningPaletteColor()
        palette_color_transform.load_palette_color(os.path.join(args.source_path, tf_dir, 'points3d.ply'))
        palette_color_transform.training_setup(op)
        palette_color_transforms.append(palette_color_transform)
    shading_kwargs["palette_colors"] = palette_color_transforms
    
    training(lp.extract(args), op.extract(args), pp.extract(args), num_TFs, dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger, shading_kwargs=shading_kwargs, verbose=args.verbose)

    logger.info("\nTraining complete.")

    logger.info(f'\nStarting Rendering~')
    render_sets(lp.extract(args), shading_kwargs, -1, pp.extract(args), wandb=wandb, logger=logger, num_TFs=num_TFs)
    logger.info("\nRendering complete.")

    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
