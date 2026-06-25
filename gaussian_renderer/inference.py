import torch
from einops import repeat
import torch.nn.functional as F

import math
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.loss_utils import ssim, bilateral_smooth_loss, contentrate_loss, sparsity_loss
from utils.encodings import STE_multistep
import time

timing_data = {
    "total_time": 0.0,
    "part1_gen_gaussians": 0.0,
    "part2_render_loop": 0.0,
    "part3_concat": 0.0,
    "iterations": 0
}

import os as _os
EXP7_TIMING = _os.environ.get("ECONGS_TIMING") == "1"
exp7_timing = {"mlp": 0.0, "shading": 0.0, "raster": 0.0, "n": 0, "n_gauss": 0}

def exp7_reset():
    exp7_timing.update({"mlp": 0.0, "shading": 0.0, "raster": 0.0, "n": 0, "n_gauss": 0})

class _cuda_timer:
    def __init__(self, key):
        self.key = key
    def __enter__(self):
        if EXP7_TIMING:
            self.s = torch.cuda.Event(enable_timing=True)
            self.e = torch.cuda.Event(enable_timing=True)
            self.s.record()
        return self
    def __exit__(self, *a):
        if EXP7_TIMING:
            self.e.record(); torch.cuda.synchronize()
            exp7_timing[self.key] += self.s.elapsed_time(self.e)

def r_saturation_ratio(r: torch.Tensor, threshold: float = 0.9):
    """
    统计 tanh(r) 饱和率
    Args:
        r: Tensor, 形状 [N, D] 或 [N]，MLP_q 输出
        threshold: 饱和阈值，默认 0.9
    Returns:
        ratio: 在 [-1, 1] 区间内，|tanh(r)| > threshold 的比例
        mean_tanh: tanh(r) 的均值（可以用来观察偏正还是偏负）
    """
    tanh_r = torch.tanh(r)
    mask = tanh_r.abs() > threshold
    ratio = mask.float().mean().item()
    mean_tanh = tanh_r.mean().item()
    return ratio, mean_tanh

import torch

def MyGenerate_neural_gaussians(viewpoint_camera, pcss, mlps):
    num_achors_each_model = [[pc._anchor.shape[0] for pc in pcs] for pcs in pcss]
    num_anchors_each_mlp = [sum(counts) for counts in num_achors_each_model]
    num_achors_each_TF = [x for sub in num_achors_each_model for x in sub]

    start_end_anchor_idx_without_mask = [(start, start + count) for start, count in zip([0] + list(torch.cumsum(torch.tensor(num_achors_each_TF[:-1]), 0, dtype=torch.int64).tolist()), num_achors_each_TF)]
    start_end_anchor_idx_each_mlp = [(start, start + count) for start, count in zip([0] + list(torch.cumsum(torch.tensor(num_anchors_each_mlp[:-1]), 0, dtype=torch.int64).tolist()), num_anchors_each_mlp)]
    total_num_anchors = start_end_anchor_idx_without_mask[-1][-1]
    feat_dim = pcss[0][0].feat_dim
    n_offsets = pcss[0][0].n_offsets

    
    all_feat = torch.empty((total_num_anchors, feat_dim), dtype=torch.float32, device="cuda")
    all_anchor =  torch.empty((total_num_anchors, 3), dtype=torch.float32, device="cuda")
    all_grid_offsets = torch.empty((total_num_anchors, n_offsets, 3), dtype=torch.float32, device="cuda")
    all_grid_scaling = torch.empty((total_num_anchors, 6), dtype=torch.float32, device="cuda")
    all_binary_grid_mask = torch.empty((total_num_anchors, n_offsets, 1), dtype=torch.float32, device="cuda")
    tf_feats = torch.empty((total_num_anchors, feat_dim), dtype=torch.float32, device="cuda")
    neural_opacity = torch.empty((total_num_anchors, 1*n_offsets), dtype=torch.float32, device="cuda")
    scale_rot = torch.empty((total_num_anchors, 7*n_offsets), dtype=torch.float32, device="cuda")
    ambient_diffuse_specular_shininess_color = torch.empty((total_num_anchors, 7*n_offsets), dtype=torch.float32, device="cuda")
    normals_raw = torch.empty((total_num_anchors, 3*n_offsets), dtype=torch.float32, device="cuda")
    
    pc_idx = 0
    for pcs in pcss:
        for pc in pcs:
            start, end = start_end_anchor_idx_without_mask[pc_idx]
            all_anchor[start:end] = pc.get_anchor
            all_binary_grid_mask[start:end] = pc.get_mask
            all_feat[start:end] = pc._anchor_feat
            all_grid_offsets[start:end] = pc._offset
            all_grid_scaling[start:end] = pc.get_scaling
            pc_idx += 1

                
    ob_view = all_anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    for pcs_idx, num_anchors in enumerate(num_achors_each_model):
        anchor_start, anchor_end = start_end_anchor_idx_each_mlp[pcs_idx]
        tf_indices = torch.arange(len(num_anchors), device='cuda').repeat_interleave(torch.tensor(num_anchors, device='cuda'))
        tf_feats[anchor_start: anchor_end] = mlps[pcs_idx].get_tf_embeddings(tf_indices)  
    

    cat_local_view_wodist = torch.cat([tf_feats, all_feat, ob_view], dim=1)
    
    for pcs_idx, num_anchors in enumerate(num_achors_each_model):
        anchor_start, anchor_end = start_end_anchor_idx_each_mlp[pcs_idx]
        neural_opacity[anchor_start: anchor_end] = mlps[pcs_idx].get_opacity_mlp(cat_local_view_wodist[anchor_start:anchor_end])
        scale_rot[anchor_start: anchor_end] = mlps[pcs_idx].get_cov_mlp(cat_local_view_wodist[anchor_start:anchor_end])
        ambient_diffuse_specular_shininess_color[anchor_start: anchor_end] = mlps[pcs_idx].get_shading_mlp(cat_local_view_wodist[anchor_start:anchor_end])
        normals_raw[anchor_start: anchor_end] = mlps[pcs_idx].get_normal_mlp(cat_local_view_wodist[anchor_start:anchor_end])
        
        
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask_cuda = (neural_opacity>0.0)
    mask_cuda = mask_cuda.view(-1)
    
    mask_counts = torch.zeros(len(start_end_anchor_idx_without_mask), dtype=torch.long, device='cuda')
    for i, (start, end) in enumerate(start_end_anchor_idx_without_mask):
        mask_counts[i] = mask_cuda[start*n_offsets:end*n_offsets].sum()
    
    masked_cumsum = torch.cat([torch.zeros(1, device='cuda'), torch.cumsum(mask_counts, 0)])
    start_end_GS_idx_with_mask = torch.stack([
        masked_cumsum[:-1], masked_cumsum[1:]
    ], dim=1).to(torch.long).tolist() 

    opacity = neural_opacity[mask_cuda]

    scale_rot = scale_rot.reshape([total_num_anchors*pc.n_offsets, 7])
    
    offsets = all_grid_offsets.contiguous().view([-1, 3])

    normals_raw = normals_raw.reshape([total_num_anchors*pc.n_offsets, 3])
    normals = pc.normal_activation(normals_raw)

    ambient_diffuse_specular_shininess_color = ambient_diffuse_specular_shininess_color.reshape([total_num_anchors*pc.n_offsets, 7])

    
    ambient = pc.specular_factor_activation(ambient_diffuse_specular_shininess_color[:, 0:1])
    diffuse = pc.specular_factor_activation(ambient_diffuse_specular_shininess_color[:, 1:2])
    specular = 3*pc.specular_factor_activation(ambient_diffuse_specular_shininess_color[:, 2:3])
    shininess = 30*(pc.shininess_activation(ambient_diffuse_specular_shininess_color[:, 3:4]))+1.0
    color = 0.1*torch.tanh(ambient_diffuse_specular_shininess_color[:, 4:7]).reshape([total_num_anchors*pc.n_offsets, 3])
    
    concatenated = torch.cat([all_grid_scaling, all_anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets, normals, ambient, diffuse, specular, shininess], dim=-1)
    masked = concatenated_all[mask_cuda]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets,\
    normals, ambient, diffuse, specular, shininess= masked.split([6, 3, 3, 7, 3, 3, 1, 1, 1, 1], dim=-1)
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    
    binary_grid_masks_pergaussian = all_binary_grid_mask.view(-1, 1)
    opacity = opacity * binary_grid_masks_pergaussian[mask_cuda]
    scaling = scaling * binary_grid_masks_pergaussian[mask_cuda]
    
    
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets
    all_vars = {
        "xyz": xyz,
        "color_offset": color,
        "opacity": opacity,
        "neural_opacity": neural_opacity,
        "scaling": scaling,
        "rot": rot,
        "normals": normals,
        "ambient": ambient,
        "diffuse": diffuse,
        "specular": specular,
        "shininess": shininess,
        "mask": mask_cuda,
        "start_end_GS_idx_with_mask": start_end_GS_idx_with_mask,
        "start_end_anchor_idx_without_mask": start_end_anchor_idx_without_mask,

    }

    return all_vars




def InfRender(viewpoint_camera, pcs: list, mlps, pipe, bg_color: torch.Tensor, image_shape=None,
             scaling_modifier=1.0, visible_mask=None, dict_params=None, light_pos=None, retain_grad=False, step=0):
    
    with _cuda_timer("mlp"):
        all_vars = MyGenerate_neural_gaussians(viewpoint_camera, pcs, mlps)
    if EXP7_TIMING:
        exp7_timing["n"] += 1
        exp7_timing["n_gauss"] += int(all_vars["xyz"].shape[0])

    scene_params = {
        "palette_color_transforms": dict_params["palette_colors"],
        "opacity_transforms": dict_params["opacity_factors"],
        "light_transform": dict_params["light_transform"],
        "xyz": all_vars["xyz"],
        "color_offset": all_vars["color_offset"],
        "opacity": all_vars["opacity"],
        "scaling": all_vars["scaling"],
        "rot": all_vars["rot"],
        "normals": all_vars["normals"],
        "ambient": all_vars["ambient"],
        "diffuse": all_vars["diffuse"],
        "specular": all_vars["specular"],
        "shininess": all_vars["shininess"],
        "neural_opacity": all_vars["neural_opacity"],
        "image_shape": image_shape,
    }
    render_result = multi_neuralGS_render(all_vars["start_end_GS_idx_with_mask"], viewpoint_camera, pipe, bg_color, scaling_modifier,
                                        light_pos, retain_grad, **scene_params)


    return render_result



    
def multi_neuralGS_render(start_end_GS_idx_with_mask, viewpoint_camera, pipe, bg_color, scaling_modifier, light_pos, retain_grad, palette_color_transforms, opacity_transforms, light_transform, xyz, \
    color_offset, opacity, scaling, rot, normals, ambient, diffuse, specular, shininess, \
    neural_opacity, image_shape):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if image_shape is None:
        image_shape = (3, viewpoint_camera.image_height, viewpoint_camera.image_width)
    
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=image_shape[1],
        image_width=image_shape[2],
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    viewdirs = F.normalize(viewpoint_camera.camera_center - xyz, dim=-1)
    light_pos = light_transform.get_light_dir()
    
    if light_pos is None:
        incidents_dirs = viewdirs.detach().contiguous()
    else:
        incidents_dirs = torch.zeros_like(viewdirs) + light_pos.view(1, 3)
        incidents_dirs = F.normalize(incidents_dirs, dim=-1).detach().contiguous()
    
    with _cuda_timer("shading"):
        brdf_color, extra_results, opacity = multi_BlinnPhong_python(start_end_GS_idx_with_mask, palette_color_transforms, opacity_transforms, light_transform, opacity, color_offset, diffuse, shininess, ambient, specular, normals, viewdirs, incidents_dirs)

    with _cuda_timer("raster"):
        rendered_image, rendered_depth, rendered_alpha, radii, extra = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = brdf_color,
            opacities = opacity,
            scales = scaling,
            rotations = rot,
            cov3Ds_precomp = None)
    

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
            }

def multi_BlinnPhong_python(start_end_GS_idx_with_mask, palette_color_transforms, opacity_transforms, light_transform, opacity, offset_color, diffuse_factor, shininess, ambient_factor, specular_factor, normals, viewdirs, incidents_dirs):
    pbr = torch.zeros_like(offset_color)
    diffuse_intensity = torch.zeros_like(offset_color)
    specular_intensity = torch.zeros_like(offset_color)
    
    palette_colors = []
    opacity_factors = []
    for i in range(len(palette_color_transforms)):
        palette_colors.append(palette_color_transforms[i].palette_color.clamp(0,1))
        if opacity_transforms is not None:
            opacity_factors.append(opacity_transforms[i].opacity_factor)
    
    for tf_idx, (start, end) in enumerate(start_end_GS_idx_with_mask):
        if opacity_transforms is not None:
            opacity[start:end] = opacity[start:end] * opacity_factors[tf_idx].view(1,1)
        pbr[start:end], diffuse_intensity[start:end], specular_intensity[start:end] = BlinnPhong_python(palette_colors[tf_idx].view(1,3), light_transform, offset_color[start:end], diffuse_factor[start:end], shininess[start:end], ambient_factor[start:end], specular_factor[start:end], normals[start:end], viewdirs[start:end], incidents_dirs[start:end])

    extra_results = {
        "diffuse_intensity": diffuse_intensity,
        "specular_intensity": specular_intensity,
    }

    return pbr, extra_results, opacity
    

def BlinnPhong_python(palette_color, light_transform, offset_color, diffuse_factor, shininess, ambient_factor, specular_factor, normals, viewdirs, incidents_dirs):
    """
    Args:
        palette_color: (N, 3)
        opacity: (N, 1)
        offset_color: (N, 3)
        diffuse_factor, shininess, ambient_factor, specular_factor: (N, 1)
        normals, viewdirs, incidents_dirs: (N, 3)
    Returns:
        pbr: (N, 3)
    """

    spcular_multi, diffuse_factor_multi, ambient_multi, shininess_multi,\
        _, _, _, _= light_transform.get_light_transform()
    
    diffuse_factor = diffuse_factor * diffuse_factor_multi
    ambient_factor = ambient_factor * ambient_multi 
    specular_factor = specular_factor * spcular_multi 
    shininess = shininess * shininess_multi 
    
    diffuse_color = offset_color + palette_color.clamp(0, 1)
    cos_l = (normals * incidents_dirs).sum(dim=-1, keepdim=True)
    diffuse_intensity = diffuse_factor * cos_l.abs()
    diffuse_total = (ambient_factor + diffuse_intensity) * diffuse_color

    h = F.normalize(incidents_dirs + viewdirs, dim=-1)
    cos_h = (normals * h).sum(dim=-1, keepdim=True)


    base = torch.clamp(torch.abs(cos_h), min=1e-8)
    spec_term = torch.exp(shininess * torch.log(base))
    specular_intensity = specular_factor*torch.where(cos_l != 0, spec_term, 0.0)

    pbr = diffuse_total + specular_intensity
    pbr = torch.clamp(pbr, 0., 1.)

    return pbr, diffuse_total, specular_intensity



