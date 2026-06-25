import torch
from einops import repeat
import torch.nn.functional as F

import math
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.loss_utils import ssim, bilateral_smooth_loss, contentrate_loss, sparsity_loss
from utils.encodings import STE_multistep
import time
from gaussian_renderer.inference import InfRender

def MyGenerate_neural_gaussians(viewpoint_camera, pcs, mlps, entropy_model, visible_mask=None, is_training=False, step=0, force_tf_idx=None):
    num_achors_each_TF = [pc._anchor.shape[0] for pc in pcs]
    start_end_anchor_idx_without_mask = [(start, start + count) for start, count in zip([0] + list(torch.cumsum(torch.tensor(num_achors_each_TF[:-1]), 0, dtype=torch.int64).tolist()), num_achors_each_TF)]
    total_num_anchors = start_end_anchor_idx_without_mask[-1][-1]
    feat_dim = pcs[0].feat_dim
    n_offsets = pcs[0].n_offsets
    
    step_3000 = 3000
    step_10k = 10_000
    
    bit_per_param = None
    bit_per_feat_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None

    all_feat = torch.empty((total_num_anchors, feat_dim), dtype=torch.float32, device="cuda")
    all_anchor =  torch.empty((total_num_anchors, 3), dtype=torch.float32, device="cuda")
    all_grid_offsets = torch.empty((total_num_anchors, n_offsets, 3), dtype=torch.float32, device="cuda")
    all_grid_scaling = torch.empty((total_num_anchors, 6), dtype=torch.float32, device="cuda")
    all_binary_grid_mask = torch.empty((total_num_anchors, n_offsets, 1), dtype=torch.float32, device="cuda")
    all_grid_mask_anchor = torch.empty((total_num_anchors, 1), dtype=torch.float32, device="cuda")
    
    
    for pc_idx, pc in enumerate(pcs):
        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        start, end = start_end_anchor_idx_without_mask[pc_idx]
        anchor = pc.get_anchor
        feat = pc._anchor_feat
        grid_offsets = pc._offset
        grid_scaling = pc.get_scaling
        grid_mask = pc.get_mask

        if (step <= step_3000) or (step > step_10k):
            all_anchor[start:end] = anchor
            all_binary_grid_mask[start:end] = grid_mask
            all_feat[start:end] = feat
            all_grid_offsets[start:end] = grid_offsets
            all_grid_scaling[start:end] = grid_scaling
            all_grid_mask_anchor[start:end] = pc.get_mask_anchor if not pc.decoded_version else torch.ones((end-start, 1), device="cuda")

        elif (step > step_3000 and step <= step_10k):
            all_anchor[start:end] = anchor
            all_binary_grid_mask[start:end] = grid_mask
            all_feat[start:end] = feat + torch.empty_like(feat, device="cuda").uniform_(-0.5, 0.5) * Q_feat
            all_grid_offsets[start:end] = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets
            all_grid_scaling[start:end] = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling

        if step == step_10k:
            x_bound_min = (torch.min(anchor, dim=0, keepdim=True)[0]).detach()
            x_bound_max = (torch.max(anchor, dim=0, keepdim=True)[0]).detach()
            entropy_model.update_anchor_bound(x_bound_min, x_bound_max)
            
    
            
    if (step > step_10k) and (pcs[0].decoded_version==False):
        tf_indices = torch.arange(len(num_achors_each_TF), device='cuda').repeat_interleave(torch.tensor(num_achors_each_TF, device='cuda'))
        tf_feats_entropy = entropy_model.get_tf_embeddings(tf_indices)

        feat_context_orig = torch.cat([entropy_model.calc_interp_feat(all_anchor), tf_feats_entropy], dim=1)
        feat_context = entropy_model.get_grid_mlp(feat_context_orig)

        mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

        all_grid_scaling_mean = all_grid_scaling.mean()
        all_feat_mean = all_feat.mean()
        all_grid_offsets_mean = all_grid_offsets.mean()

        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

        all_binary_grid_mask = all_binary_grid_mask
        all_feat = all_feat + torch.empty_like(all_feat).uniform_(-0.5, 0.5) * Q_feat
        all_grid_offsets = all_grid_offsets + torch.empty_like(all_grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)
        all_grid_scaling = all_grid_scaling + torch.empty_like(all_grid_scaling).uniform_(-0.5, 0.5) * Q_scaling

        choose_idx = torch.rand_like(all_anchor[:, 0]) <= 0.05
        feat_chosen = all_feat[choose_idx]
        mask_anchor = all_grid_mask_anchor[choose_idx]
        grid_offsets_chosen = all_grid_offsets[choose_idx]
        grid_offsets_chosen = grid_offsets_chosen.view(-1, 3 * pc.n_offsets)
        grid_scaling_chosen = all_grid_scaling[choose_idx]
        binary_grid_masks_chosen = all_binary_grid_mask[choose_idx]
        binary_grid_masks_chosen = binary_grid_masks_chosen.repeat(1, 1, 3).contiguous().view(-1, 3*pc.n_offsets)

        mean = mean[choose_idx]
        scale = scale[choose_idx]
        mean_scaling = mean_scaling[choose_idx]
        scale_scaling = scale_scaling[choose_idx]
        mean_offsets = mean_offsets[choose_idx]
        scale_offsets = scale_offsets[choose_idx]
        Q_feat = Q_feat[choose_idx]
        Q_scaling = Q_scaling[choose_idx]
        Q_offsets = Q_offsets[choose_idx]

        bit_feat = entropy_model.entropy_gaussian.forward(feat_chosen, mean, scale, Q_feat, all_feat_mean)
        bit_feat = bit_feat * mask_anchor
        bit_scaling = entropy_model.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, all_grid_scaling_mean)
        bit_scaling = bit_scaling * mask_anchor
        bit_offsets = entropy_model.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets, all_grid_offsets_mean)
        bit_offsets = bit_offsets * binary_grid_masks_chosen * mask_anchor

        bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel()
        bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel()
        bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel()
        bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets))  / \
                        (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel())
            
                    

    ob_view = all_anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist
    
    if force_tf_idx is None:
        tf_indices = torch.arange(len(num_achors_each_TF), device='cuda').repeat_interleave(torch.tensor(num_achors_each_TF, device='cuda'))
    else:
        tf_indices = torch.ones(total_num_anchors, device='cuda', dtype=torch.long)*force_tf_idx
    tf_feats = mlps.get_tf_embeddings(tf_indices)

    cat_local_view_wodist = torch.cat([tf_feats, all_feat, ob_view], dim=1)
    
    neural_opacity = mlps.get_opacity_mlp(cat_local_view_wodist)
    neural_opacity = neural_opacity.view([-1, 1])
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


    scale_rot = mlps.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.view([total_num_anchors*pc.n_offsets, 7])
    
    offsets = all_grid_offsets.contiguous().view([-1, 3])

    ambient_diffuse_specular_shininess_color = mlps.get_shading_mlp(cat_local_view_wodist)
    ambient_diffuse_specular_shininess_color = ambient_diffuse_specular_shininess_color.view([total_num_anchors*pc.n_offsets, 7])

    normals_raw = mlps.get_normal_mlp(cat_local_view_wodist)
    normals_raw = normals_raw.view([total_num_anchors*pc.n_offsets, 3])
    normals = pc.normal_activation(normals_raw)

    
    ambient = pc.specular_factor_activation(ambient_diffuse_specular_shininess_color[:, 0:1])
    diffuse = pc.specular_factor_activation(ambient_diffuse_specular_shininess_color[:, 1:2])
    specular = 3*pc.specular_factor_activation(ambient_diffuse_specular_shininess_color[:, 2:3])
    shininess = 30*(pc.shininess_activation(ambient_diffuse_specular_shininess_color[:, 3:4]))+1.0
    color = 0.1*torch.tanh(ambient_diffuse_specular_shininess_color[:, 4:7]).view([total_num_anchors*pc.n_offsets, 3])
    
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
        "bit_per_param": bit_per_param,
        "bit_per_feat_param": bit_per_feat_param,
        "bit_per_scaling_param": bit_per_scaling_param,
        "bit_per_offsets_param": bit_per_offsets_param,
    }

    return all_vars




def Myrender(viewpoint_camera, pcs: list, mlps, entropy_model, pipe, bg_color: torch.Tensor, image_shape=None,
             scaling_modifier=1.0, visible_mask=None, dict_params=None, light_pos=None, retain_grad=False, step=0, force_tf_idx=None):
    
    is_training = mlps.get_cov_mlp.training

    all_vars = MyGenerate_neural_gaussians(viewpoint_camera, pcs, mlps, entropy_model, visible_mask, is_training=is_training, step=step, force_tf_idx=force_tf_idx)
    

    start_end_GS_idx_with_mask = all_vars["start_end_GS_idx_with_mask"]
    
    results = {
        "render": [],
        "viewspace_points": [],
        "visibility_filter": [],
        "radii": [],

        "selection_mask": [all_vars["mask"][anchor_start*pcs[0].n_offsets:anchor_end*pcs[0].n_offsets] for (anchor_start, anchor_end) in all_vars["start_end_anchor_idx_without_mask"]],
        "neural_opacity": [all_vars["neural_opacity"][anchor_start*pcs[0].n_offsets:anchor_end*pcs[0].n_offsets] for (anchor_start, anchor_end) in all_vars["start_end_anchor_idx_without_mask"]],

        "scaling": all_vars["scaling"],
        "bit_per_param": all_vars['bit_per_param'],
        "bit_per_feat_param": all_vars['bit_per_feat_param'],
        "bit_per_scaling_param": all_vars['bit_per_scaling_param'],
        "bit_per_offsets_param": all_vars['bit_per_offsets_param'],
    }


    scene_params = {
        "palette_colors": dict_params["palette_colors"],
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
        "image_shape": image_shape,
        "force_tf_idx": force_tf_idx,
        "start_end_GS_idx": start_end_GS_idx_with_mask,
    }
    render_result = neuralGS_render(viewpoint_camera, pipe, bg_color, scaling_modifier,
                                        light_pos, retain_grad, **scene_params)
    results["render"] = render_result["render"]
    results["viewspace_points"] = render_result["viewspace_points"]
    results["visibility_filter"] = [render_result["visibility_filter"][start:end] for (start, end) in all_vars["start_end_GS_idx_with_mask"]]
    results["radii"] = [render_result["radii"][start:end] for (start, end) in all_vars["start_end_GS_idx_with_mask"]]
    results['total_num_anchors'] = all_vars['start_end_anchor_idx_without_mask'][-1][-1]
    
    return results



    
def neuralGS_render(viewpoint_camera, pipe, bg_color, scaling_modifier, light_pos, retain_grad, palette_colors, xyz, \
    color_offset, opacity, scaling, rot, normals, ambient, diffuse, specular, shininess, \
    image_shape, start_end_GS_idx, force_tf_idx):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if image_shape is None:
        image_shape = (3, viewpoint_camera.image_height, viewpoint_camera.image_width)
    numTFs = len(start_end_GS_idx)
    screenspace_points = [torch.zeros_like(xyz[start:end], dtype=xyz.dtype, requires_grad=True, device="cuda") + 0 for start, end in start_end_GS_idx]
    if retain_grad:
        try:
            for tf_idx in range(numTFs):
                screenspace_points[tf_idx].retain_grad()
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
    
    if light_pos is None:
        incidents_dirs = viewdirs.detach().contiguous()
    else:
        incidents_dirs = F.normalize(light_pos, dim=-1).detach().contiguous()
    
    brdf_color, extra_results, opacity = BlinnPhong_python(
        start_end_GS_idx, palette_colors, opacity, color_offset, diffuse, shininess, ambient, specular, normals, viewdirs, incidents_dirs, force_tf_idx)
    
    rendered_image = torch.zeros((3*numTFs, image_shape[1], image_shape[2]), dtype=xyz.dtype, device="cuda")
    radii = torch.zeros((xyz.shape[0]), dtype=xyz.dtype, device="cuda")
    

    for tf_idx, (start, end) in enumerate(start_end_GS_idx):
        rendered_image[3*tf_idx:3*(tf_idx+1)], rendered_depth, rendered_alpha, radii[start:end], extra = rasterizer(
            means3D = xyz[start:end],
            means2D = screenspace_points[tf_idx],
            shs = None,
            colors_precomp = brdf_color[start:end],
            opacities = opacity[start:end],
            scales = scaling[start:end],
            rotations = rot[start:end],
            cov3Ds_precomp = None)
        
    

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            }

def BlinnPhong_python(start_end_GS_idx, palette_colors, opacity, offset_color, diffuse_factor, shininess, ambient_factor, specular_factor, normals, viewdirs, incidents_dirs, force_tf_idx=None):
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
    if force_tf_idx is None:
        numGS_TFs = torch.tensor([end - start for start, end in start_end_GS_idx], device=opacity.device, dtype=torch.long)
        all_palette_colors = torch.stack([palette_colors[i].palette_color.data for i in range(len(palette_colors))], dim=0)    
        palette_color_repeat = torch.repeat_interleave(all_palette_colors, numGS_TFs, dim=0)
    else:
        all_palette_colors = palette_colors[force_tf_idx].palette_color.data
        palette_color_repeat = all_palette_colors.repeat(opacity.shape[0], 1)


    diffuse_color = offset_color + palette_color_repeat.clamp(0, 1)
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

    extra_results = {
        "diffuse_render": diffuse_total,
        "specular_render": specular_intensity,
        "offset_color_norm": (offset_color ** 2).sum(dim=-1, keepdim=True),
    }
    return pbr, extra_results, opacity
