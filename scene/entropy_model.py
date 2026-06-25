import torch
from torch import nn
from utils.encodings import GridEncoder
import torch.nn.functional as nnf
import numpy as np
from torch.distributions.uniform import Uniform
from utils.encodings import use_clamp
from utils.general_utils import get_expon_lr_func
from utils.system_utils import mkdir_p
import matplotlib.pyplot as plt
import time
import os

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    GridEncoder, \
    anchor_round_digits, \
    get_binary_vxl_size

from utils.encodings_cuda import \
    encoder, decoder, \
    encoder_gaussian_chunk, decoder_gaussian_chunk, encoder_gaussian_mixed_chunk, decoder_gaussian_mixed_chunk
from utils.gpcc_utils import compress_gpcc, decompress_gpcc, calculate_morton_order

bit2MB_scale = 8 * 1024 * 1024
MAX_batch_size = 3000

def get_time():
    torch.cuda.synchronize()
    tt = time.time()
    return tt

class EntropyModel(nn.Module):
    def __init__(self,
                 num_TFs,
                 feat_dim=50,
                 n_offsets=10,
                 n_features_per_level: int=2,
                 log2_hashmap_size: int=19,
                 log2_hashmap_size_2D: int=17,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 decoded_version: bool=False):
        print('hash_params:', use_2D, n_features_per_level,
              log2_hashmap_size, resolutions_list,
              log2_hashmap_size_2D, resolutions_list_2D,
              ste_binary, ste_multistep, add_noise)
        super(EntropyModel, self).__init__()
        self.optimizer = None
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version
        
        self.tf_embeddings = nn.Embedding(num_TFs, feat_dim).cuda()
        with torch.no_grad():
            self.tf_embeddings.weight.zero_()
        
        
        if use_2D:
            self.encoding_xyz = mix_3D2D_encoding(
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                resolutions_list_2D=resolutions_list_2D,
                log2_hashmap_size_2D=log2_hashmap_size_2D,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()
        else:
            self.encoding_xyz = GridEncoder(
                num_dim=3,
                n_features=n_features_per_level,
                resolutions_list=resolutions_list,
                log2_hashmap_size=log2_hashmap_size,
                ste_binary=ste_binary,
                ste_multistep=ste_multistep,
                add_noise=add_noise,
                Q=Q,
            ).cuda()
        
        self.mlp_grid = nn.Sequential(
            nn.Linear(self.encoding_xyz.output_dim+feat_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, (feat_dim+6+3*self.n_offsets)*2+1+1+1),
        ).cuda()
        
        encoding_params_num = 0
        for n, p in self.encoding_xyz.named_parameters():
            encoding_params_num += p.numel()
        encoding_MB = encoding_params_num / 8 / 1024 / 1024
        if not ste_binary: encoding_MB *= 32
        print(f'encoding_param_num={encoding_params_num}, size={encoding_MB}MB.')

        self.mlp_deform = Channel_CTX_fea_tiny().cuda()
        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()
        self.EG_mix_prob_2 = Entropy_gaussian_mix_prob_2(Q=1).cuda()

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_deform_mlp(self):
        return self.mlp_deform
    
    @property
    def get_tf_embeddings(self):
        return self.tf_embeddings
    
    
    @torch.no_grad()
    def update_anchor_bound(self, x_bound_min, x_bound_max):
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.x_bound_min = torch.min(self.x_bound_min, x_bound_min)
        self.x_bound_max = torch.max(self.x_bound_min, x_bound_max)
        print('anchor_bound_updated to:', self.x_bound_min, self.x_bound_max)
    
    def calc_interp_feat(self, x):
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)
        features = self.encoding_xyz(x)
        return features
    
    def get_encoding_params(self):
        params = []
        if self.use_2D:
            params.append(self.encoding_xyz.encoding_xyz.params)
            params.append(self.encoding_xyz.encoding_xy.params)
            params.append(self.encoding_xyz.encoding_xz.params)
            params.append(self.encoding_xyz.encoding_yz.params)
        else:
            params.append(self.encoding_xyz.params)
        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def get_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.named_parameters():
            if 'mlp' in n:
                mlp_size += p.numel()*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024
    
    def training_setup(self, training_args):
        l = [
            {'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
            {'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"},
            {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
            {'params': self.tf_embeddings.parameters(), 'lr': training_args.tf_embedding_lr_init, "name": "tf_embeddings"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_lr_init,
                                                    lr_final=training_args.encoding_xyz_lr_final,
                                                    lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
                                                    max_steps=training_args.encoding_xyz_lr_max_steps,
                                                             step_sub=0 if self.ste_binary else 10000,
                                                             )
        self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_grid_lr_init,
                                                    lr_final=training_args.mlp_grid_lr_final,
                                                    lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
                                                    max_steps=training_args.mlp_grid_lr_max_steps,
                                                         step_sub=0 if self.ste_binary else 10000,
                                                         )

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                    lr_final=training_args.mlp_deform_lr_final,
                                                    lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                    max_steps=training_args.mlp_deform_lr_max_steps)
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "encoding_xyz":
                lr = self.encoding_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_grid":
                lr = self.mlp_grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_deform":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr
    
    @torch.no_grad()
    def estimate_final_bits(self, pc):
        pass
    
    def save_mlp_embeddings(self, path, mode='split'):
        mkdir_p(path)
        if mode == 'split':
            self.mlp_grid.eval()
            grid_mlp = torch.jit.trace(self.mlp_grid, (torch.rand(1, self.encoding_xyz.output_dim + self.feat_dim).cuda()))
            grid_mlp.save(os.path.join(path, 'grid_mlp.pt'))
            self.mlp_grid.train()
            
            self.tf_embeddings.eval()
            torch.save(self.tf_embeddings.state_dict(), os.path.join(path, 'tf_embeddings_entropy.pth'))
            self.tf_embeddings.train()
            
        
        elif mode == 'unite':
            torch.save({
                'mlp_grid': self.mlp_grid.state_dict(),
                'tf_embeddings_entropy': self.tf_embeddings.state_dict(),
                }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError
        
    def load_mlp_embeddings(self, path, mode='split'):
        if mode == 'split':
            self.mlp_grid = torch.jit.load(os.path.join(path, 'grid_mlp.pt'))
            self.tf_embeddings.load_state_dict(torch.load(os.path.join(path, 'tf_embeddings_entropy.pth')))
        elif mode == 'unite':
            ckpt = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_grid.load_state_dict(ckpt['mlp_grid'])
            self.tf_embeddings.load_state_dict(ckpt['tf_embeddings_entropy'])
        else:
            raise NotImplementedError
    
    @torch.no_grad()
    def conduct_encoding(self, pre_path_name, pcs):
        num_achors_each_TF = [pc._anchor.shape[0] for pc in pcs]
        start_end_anchor_idx_without_mask = [(start, start + count) for start, count in zip([0] + list(torch.cumsum(torch.tensor(num_achors_each_TF[:-1]), 0, dtype=torch.int64).tolist()), num_achors_each_TF)]
        total_num_anchors = start_end_anchor_idx_without_mask[-1][-1]
        feat_dim = pcs[0].feat_dim
        n_offsets = pcs[0].n_offsets
        voxel_size = pcs[0].voxel_size
        tf_indices = torch.arange(len(num_achors_each_TF), device='cuda').repeat_interleave(torch.tensor(num_achors_each_TF, device='cuda'))
        
        
        all_feat = torch.empty((total_num_anchors, feat_dim), dtype=torch.float32, device="cuda")
        all_anchor =  torch.empty((total_num_anchors, 3), dtype=torch.float32, device="cuda")
        all_grid_offsets = torch.empty((total_num_anchors, n_offsets, 3), dtype=torch.float32, device="cuda")
        all_grid_scaling = torch.empty((total_num_anchors, 6), dtype=torch.float32, device="cuda")
        all_binary_grid_mask = torch.empty((total_num_anchors, n_offsets, 1), dtype=torch.float32, device="cuda")
        mask_anchor = torch.empty((total_num_anchors, 1), dtype=torch.float32, device="cuda")
        tf_feats_entropy = self.get_tf_embeddings(tf_indices)

        for pc_idx, pc in enumerate(pcs):
            start, end = start_end_anchor_idx_without_mask[pc_idx]
            mask_anchor[start:end] = pc.get_mask_anchor
            all_anchor[start:end] = pc.get_anchor
            all_binary_grid_mask[start:end] = pc.get_mask
            all_feat[start:end] = pc._anchor_feat
            all_grid_offsets[start:end] = pc._offset
            all_grid_scaling[start:end] = pc.get_scaling
        
        

        t_codec = 0

        t_total_0 = get_time()
        torch.cuda.synchronize(); t1 = time.time()
        print('Start encoding ...')
        
        torch.save(self.x_bound_min, os.path.join(pre_path_name, 'x_bound_min.pkl'))
        torch.save(self.x_bound_max, os.path.join(pre_path_name, 'x_bound_max.pkl'))
        
        mask_anchor = mask_anchor.to(torch.bool)[:, 0]
        
        all_anchor = all_anchor[mask_anchor]
        all_feat = all_feat[mask_anchor]
        all_grid_offsets = all_grid_offsets[mask_anchor]
        all_grid_scaling = all_grid_scaling[mask_anchor]
        all_binary_grid_mask = all_binary_grid_mask[mask_anchor]
        tf_feats_entropy = tf_feats_entropy[mask_anchor]
        
        tf_indices_masked = tf_indices[mask_anchor] 
        
        num_anchors_each_TF_mask = torch.bincount(
            tf_indices_masked, minlength=len(num_achors_each_TF)
        ).tolist()

        start_end_anchor_idx_without_mask = [
            (start, start + count)
            for start, count in zip(
                [0] + list(torch.cumsum(torch.tensor(num_anchors_each_TF_mask[:-1]), 0, dtype=torch.int64).tolist()),
                num_anchors_each_TF_mask
            )
        ]

        total_num_anchors_mask = start_end_anchor_idx_without_mask[-1][-1]
        bit_anchor = 0
        t_anchor_0 = get_time()

        global_sorted_indices = []

        for tf_id, (start, end) in enumerate(start_end_anchor_idx_without_mask):
            anchors_tf = all_anchor[start:end]
            _anchor_int_tf = torch.round(anchors_tf / voxel_size)
            sorted_indices_tf = calculate_morton_order(_anchor_int_tf)
            _anchor_int_tf = _anchor_int_tf[sorted_indices_tf]

            npz_path = os.path.join(pre_path_name, f'xyz_gpcc_tf{tf_id:02d}.npz')
            means_strings = compress_gpcc(_anchor_int_tf)
            np.savez_compressed(npz_path, voxel_size=voxel_size, means_strings=means_strings)

            all_anchor[start:end] = _anchor_int_tf * voxel_size
            bits_xyz = os.path.getsize(npz_path) * 8
            bit_anchor += bits_xyz
            print(f"TF {tf_id} compressed size: {bits_xyz} bits")

            global_sorted_indices.append(sorted_indices_tf + start)

        t_codec += get_time() - t_anchor_0

        global_sorted_indices = torch.cat(global_sorted_indices, dim=0)

        print(global_sorted_indices)
        print(global_sorted_indices.shape)
        
        
        all_anchor = all_anchor
        all_feat = all_feat[global_sorted_indices]
        all_grid_offsets = all_grid_offsets[global_sorted_indices]
        all_grid_scaling = all_grid_scaling[global_sorted_indices]
        all_binary_grid_mask = all_binary_grid_mask[global_sorted_indices]
        tf_feats_entropy = tf_feats_entropy[global_sorted_indices]

        all_grid_scaling_mean = all_grid_scaling.mean()
        all_feat_mean = all_feat.mean()
        all_grid_offsets_mean = all_grid_offsets.mean()


        self.all_binary_grid_mask = all_binary_grid_mask
        self.all_grid_scaling = all_grid_scaling
        self.all_grid_offsets = all_grid_offsets*all_binary_grid_mask
        self.tf_feats_entropy = tf_feats_entropy
        self.all_feat = all_feat
    
        
        N = all_anchor.shape[0]
        steps = (N // MAX_batch_size) if (N % MAX_batch_size) == 0 else (N // MAX_batch_size + 1)

        bit_feat_list = []
        bit_scaling_list = []
        bit_offsets_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        for s in range(steps):
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N)

            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            anchor_slice = all_anchor[N_start:N_end]
            tf_feats_entropy_slice = tf_feats_entropy[N_start:N_end]
            feat_context = torch.cat([self.calc_interp_feat(anchor_slice), tf_feats_entropy_slice], dim=1)
            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[feat_dim, feat_dim, 6, 6, 3*n_offsets, 3*n_offsets, 1, 1, 1], dim=-1)

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1]).view(-1)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)

            mean = mean.contiguous().view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)

            scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)

            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat = all_feat[N_start:N_end].view(-1)
            feat = STE_multistep.apply(feat, Q_feat, all_feat_mean)
            torch.cuda.synchronize(); t0 = time.time()
            bit_feat = encoder_gaussian_chunk(feat, mean, scale, Q_feat, file_name=feat_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_feat_list.append(bit_feat)

            scaling = all_grid_scaling[N_start:N_end].view(-1)
            scaling = STE_multistep.apply(scaling, Q_scaling, all_grid_scaling_mean)
            torch.cuda.synchronize(); t0 = time.time()
            bit_scaling = encoder_gaussian_chunk(scaling, mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_scaling_list.append(bit_scaling)

            mask = all_binary_grid_mask[N_start:N_end]
            mask = mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets).view(-1).to(torch.bool)

            offsets = all_grid_offsets[N_start:N_end].view(-1, 3*self.n_offsets).view(-1)
            offsets = STE_multistep.apply(offsets, Q_offsets, all_grid_offsets_mean)
            offsets[~mask] = 0.0

            torch.cuda.synchronize(); t0 = time.time()
            bit_offsets = encoder_gaussian_chunk(offsets[mask], mean_offsets[mask], scale_offsets[mask], Q_offsets[mask], file_name=offsets_b_name)
            torch.cuda.synchronize(); t_codec += time.time() - t0

            bit_offsets_list.append(bit_offsets)

            torch.cuda.empty_cache()

        bit_feat = sum(bit_feat_list)
        bit_scaling = sum(bit_scaling_list)
        bit_offsets = sum(bit_offsets_list)

        hash_embeddings = self.get_encoding_params()
        if self.ste_binary:
            bit_hash = encoder(((hash_embeddings.view(-1) + 1) / 2), file_name=hash_b_name)
        else:
            bit_hash = hash_embeddings.numel()*32

        bit_masks = encoder(all_binary_grid_mask, file_name=masks_b_name)

        torch.cuda.synchronize(); t2 = time.time()
        print('encoding time:', t2 - t1)
        print('codec time:', t_codec)

        log_info = f"\nEncoded sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + self.get_mlp_size()[0])/bit2MB_scale, 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"
        print(log_info)
        return log_info
        
    
    @torch.no_grad()
    def conduct_decoding(self, pre_path_name, pcs):
        torch.cuda.synchronize(); t1 = time.time()
        print('Start decoding ...')
        
        self.x_bound_min = torch.load(os.path.join(pre_path_name, 'x_bound_min.pkl'))
        self.x_bound_max = torch.load(os.path.join(pre_path_name, 'x_bound_max.pkl'))
        
        N = 0
        anchor_decoded_list = []
        num_anchors_each_TF = []
        
        xyz_decoded_list = []
        feat_decoded_list = []
        scaling_decoded_list = []
        offsets_decoded_list = []

        hash_b_name = os.path.join(pre_path_name, 'hash.b')
        masks_b_name = os.path.join(pre_path_name, 'masks.b')
        
        npz_paths = sorted([os.path.join(pre_path_name, f) for f in os.listdir(pre_path_name) if f.startswith('xyz_gpcc')])
        for npz_path in npz_paths:
            data_dict = np.load(npz_path)
            voxel_size = float(data_dict['voxel_size'])
            means_strings = data_dict['means_strings'].tobytes()
            _anchor_int_dec = decompress_gpcc(means_strings).to('cuda')
            sorted_indices = calculate_morton_order(_anchor_int_dec)
            _anchor_int_dec = _anchor_int_dec[sorted_indices]
            
            
            anchor_decoded = _anchor_int_dec * voxel_size
            num_anchors_each_TF.append(anchor_decoded.shape[0])
            N += anchor_decoded.shape[0]
                
            anchor_decoded_list.append(anchor_decoded)
        anchor_decoded = torch.cat(anchor_decoded_list, dim=0)
        del anchor_decoded_list
        start_end_anchor_idx_without_mask = [(start, start + count) for start, count in zip([0] + list(torch.cumsum(torch.tensor(num_anchors_each_TF[:-1]), 0, dtype=torch.int64).tolist()), num_anchors_each_TF)]
        tf_indices = torch.arange(len(num_anchors_each_TF), device='cuda').repeat_interleave(torch.tensor(num_anchors_each_TF, device='cuda'))
        tf_feats_entropy = self.get_tf_embeddings(tf_indices) 
        
        N_masked = N
        print(N_masked, 'valid anchors to be decoded.')
        print('voxel_size:', voxel_size)
        print('start_end_anchor', start_end_anchor_idx_without_mask)
        
        
        steps = (N_masked // MAX_batch_size) if (N_masked % MAX_batch_size) == 0 else (N_masked // MAX_batch_size + 1)
        masks_decoded = decoder(N_masked*self.n_offsets, masks_b_name)
        masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)

        if self.ste_binary:
            N_hash = torch.zeros_like(self.get_encoding_params()).numel()
            hash_embeddings = decoder(N_hash, hash_b_name)
            hash_embeddings = (hash_embeddings * 2 - 1).to(torch.float32)
            hash_embeddings = hash_embeddings.view(-1, self.n_features_per_level)
        
        if self.ste_binary:
            if self.use_2D:
                len_3D = self.encoding_xyz.encoding_xyz.params.shape[0]
                len_2D = self.encoding_xyz.encoding_xy.params.shape[0]
                self.encoding_xyz.encoding_xyz.params = nn.Parameter(hash_embeddings[0:len_3D])
                self.encoding_xyz.encoding_xy.params = nn.Parameter(hash_embeddings[len_3D:len_3D+len_2D])
                self.encoding_xyz.encoding_xz.params = nn.Parameter(hash_embeddings[len_3D+len_2D:len_3D+len_2D*2])
                self.encoding_xyz.encoding_yz.params = nn.Parameter(hash_embeddings[len_3D+len_2D*2:len_3D+len_2D*3])
            else:
                self.encoding_xyz.params = nn.Parameter(hash_embeddings)
        
        Q_feat_list = []
        Q_scaling_list = []
        Q_offsets_list = []

        for s in range(steps):

            N_num = min(MAX_batch_size, N_masked - s*MAX_batch_size)
            N_start = s * MAX_batch_size
            N_end = min((s+1)*MAX_batch_size, N_masked)
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{s}.b')
            scaling_b_name = os.path.join(pre_path_name, 'scaling.b').replace('.b', f'_{s}.b')
            offsets_b_name = os.path.join(pre_path_name, 'offsets.b').replace('.b', f'_{s}.b')

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2

            anchor_sort = anchor_decoded[N_start:N_end]
            tf_feats_entropy_slice = tf_feats_entropy[N_start:N_end]
            feat_context = torch.cat([self.calc_interp_feat(anchor_sort), tf_feats_entropy_slice], dim=1)

            mean, scale, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
                torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)

            Q_feat_list.append(Q_feat * (1 + torch.tanh(Q_feat_adj.contiguous())))
            Q_scaling_list.append(Q_scaling * (1 + torch.tanh(Q_scaling_adj.contiguous())))
            Q_offsets_list.append(Q_offsets * (1 + torch.tanh(Q_offsets_adj.contiguous())))

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1]).view(-1)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)

            mean = mean.contiguous().view(-1)
            mean_scaling = mean_scaling.contiguous().view(-1)
            mean_offsets = mean_offsets.contiguous().view(-1)

            scale = torch.clamp(scale.contiguous().view(-1), min=1e-9)
            scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
            scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)

            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

            feat_decoded = decoder_gaussian_chunk(mean, scale, Q_feat, file_name=feat_b_name)
            feat_decoded = feat_decoded.view(N_num, self.feat_dim)
            feat_decoded_list.append(feat_decoded)

            scaling_decoded = decoder_gaussian_chunk(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name)
            scaling_decoded = scaling_decoded.view(N_num, 6)
            scaling_decoded_list.append(scaling_decoded)

            masks_tmp = masks_decoded[N_start:N_end].repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
            offsets_decoded_tmp = decoder_gaussian_chunk(mean_offsets[masks_tmp], scale_offsets[masks_tmp], Q_offsets[masks_tmp], file_name=offsets_b_name)
            offsets_decoded = torch.zeros_like(mean_offsets)
            offsets_decoded[masks_tmp] = offsets_decoded_tmp
            offsets_decoded = offsets_decoded.view(N_num, -1).view(N_num, self.n_offsets, 3)
            offsets_decoded_list.append(offsets_decoded)

            xyz_decoded_list.append(anchor_sort)

            torch.cuda.empty_cache()

        feat_decoded = torch.cat(feat_decoded_list, dim=0)
        scaling_decoded = torch.cat(scaling_decoded_list, dim=0)
        offsets_decoded = torch.cat(offsets_decoded_list, dim=0)

        for pc_idx, pc in enumerate(pcs):
            start, end = start_end_anchor_idx_without_mask[pc_idx]
            pc._anchor = nn.Parameter(anchor_decoded[start:end])
            pc._anchor_feat = nn.Parameter(feat_decoded[start:end])
            pc._offset = nn.Parameter(offsets_decoded[start:end])
            pc._scaling = nn.Parameter(scaling_decoded[start:end])
            pc._mask = nn.Parameter(masks_decoded[start:end], requires_grad=False)
            pc.decoded_version = True

        torch.cuda.synchronize(); t2 = time.time()
        print('decoding time:', t2 - t1)
            
        

        print('Parameters are successfully replaced by decoded ones!')

        log_info = f"\nDecTime {round(t2 - t1, 4)}"

        return log_info

        





class Entropy_gaussian_clamp(nn.Module):
    def __init__(self, Q=1):
        super(Entropy_gaussian_clamp, self).__init__()
        self.Q = Q
    def forward(self, x, mean, scale, Q=None):
        if Q is None:
            Q = self.Q
        if use_clamp:
            x_mean = x.mean()
            x_min = x_mean - 15_000 * Q
            x_max = x_mean + 15_000 * Q
            x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
        scale = torch.clamp(scale, min=1e-9)
        m1 = torch.distributions.normal.Normal(mean, scale)
        lower = m1.cdf(x - 0.5*Q)
        upper = m1.cdf(x + 0.5*Q)
        likelihood = torch.abs(upper - lower)
        likelihood = Low_bound.apply(likelihood)
        bits = -torch.log2(likelihood)
        return bits


import torch
import torch.nn as nn
import math

SQRT2 = math.sqrt(2.0)
LOG2E = 1.4426950408889634

class EntropyGaussianFast(nn.Module):
    def __init__(self, Q=1.0, eps=1e-12, use_clamp=False, clamp_k=15000.0, small_bin_thresh=0.05):
        """
        Q: 默认步长（可被forward的Q覆盖）
        eps: 下界，防止 log2(0)
        use_clamp: 是否对 x 做范围裁剪（以 x_mean ± clamp_k*Q）
        clamp_k: 裁剪系数
        small_bin_thresh: 触发 PDF 近似的阈值，条件是 (Q/scale) < small_bin_thresh
        """
        super().__init__()
        self.Q = float(Q)
        self.eps = float(eps)
        self.use_clamp = bool(use_clamp)
        self.clamp_k = float(clamp_k)
        self.small_bin_thresh = float(small_bin_thresh)

    @torch.no_grad()
    def _maybe_get_xmean(self, x, x_mean):
        if x_mean is None:
            return x.mean()
        return x_mean

    def forward(self, x, mean, scale, Q=None, x_mean=None):
        if Q is None:
            Q = self.Q

        if not torch.is_tensor(Q):
            Q = torch.as_tensor(Q, device=x.device, dtype=x.dtype)

        if self.use_clamp:
            xm = self._maybe_get_xmean(x, x_mean)
            x = x.clamp(min=(xm - self.clamp_k * Q).detach(),
                        max=(xm + self.clamp_k * Q).detach())

        scale = scale.clamp_min(1e-9)

        halfQ = 0.5 * Q
        inv_scale = 1.0 / scale

        z1 = (x - mean - halfQ) * inv_scale
        z2 = (x - mean + halfQ) * inv_scale

        small_bin = (Q * inv_scale) < self.small_bin_thresh

        likelihood = torch.empty_like(x)

        if small_bin.any():
            z = (x - mean) * inv_scale
            log_pdf = -0.5 * z * z - (math.log(math.sqrt(2 * math.pi)) + torch.log(scale))
            ll_small = (torch.log(Q) + log_pdf)
            bits_small = -(ll_small) * LOG2E
            likelihood[small_bin] = torch.exp(ll_small[small_bin])
        else:
            bits_small = None

        if hasattr(torch.special, "ndtr"):
            cdf2 = torch.special.ndtr(z2[~small_bin])
            cdf1 = torch.special.ndtr(z1[~small_bin])
        else:
            cdf2 = 0.5 * (1.0 + torch.erf(z2[~small_bin] / SQRT2))
            cdf1 = 0.5 * (1.0 + torch.erf(z1[~small_bin] / SQRT2))

        like_big = (cdf2 - cdf1).clamp_min(self.eps)
        bits_big = -torch.log2(like_big)

        bits = torch.empty_like(x)
        if small_bin.any():
            bits[small_bin] = bits_small[small_bin] if bits_small is not None else 0.0
        bits[~small_bin] = bits_big

        return bits


class Entropy_gaussian(nn.Module):
    def __init__(self, Q=1):
        super(Entropy_gaussian, self).__init__()
        self.Q = Q
    def forward(self, x, mean, scale, Q=None, x_mean=None):
        if Q is None:
            Q = self.Q
        if use_clamp:
            if x_mean is None:
                x_mean = x.mean()
            x_min = x_mean - 15_000 * Q
            x_max = x_mean + 15_000 * Q
            x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
        scale = torch.clamp(scale, min=1e-9)
        m1 = torch.distributions.normal.Normal(mean, scale)
        lower = m1.cdf(x - 0.5*Q)
        upper = m1.cdf(x + 0.5*Q)
        likelihood = torch.abs(upper - lower)
        likelihood = Low_bound.apply(likelihood)
        bits = -torch.log2(likelihood)
        return bits

class Entropy_gaussian_mix_prob_2(nn.Module):
    def __init__(self, Q=1):
        super(Entropy_gaussian_mix_prob_2, self).__init__()
        self.Q = Q
    def forward(self, x,
                mean1, mean2,
                scale1, scale2,
                probs1, probs2,
                Q=None, x_mean=None, return_lkl=False):
        if Q is None:
            Q = self.Q
        if use_clamp:
            if x_mean is None:
                x_mean = x.mean().detach()
            x_min = x_mean - 15_000 * Q
            x_max = x_mean + 15_000 * Q
            x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())

        scale1 = torch.clamp(scale1, min=1e-9)
        scale2 = torch.clamp(scale2, min=1e-9)

        m1 = torch.distributions.normal.Normal(mean1, scale1)
        m2 = torch.distributions.normal.Normal(mean2, scale2)

        likelihood1 = torch.abs(m1.cdf(x + 0.5*Q) - m1.cdf(x - 0.5*Q))
        likelihood2 = torch.abs(m2.cdf(x + 0.5*Q) - m2.cdf(x - 0.5*Q))

        likelihood = Low_bound.apply(probs1 * likelihood1 + probs2 * likelihood2)

        if return_lkl:
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            return bits

class Entropy_gaussian_mix_prob_3(nn.Module):
    def __init__(self, Q=1):
        super(Entropy_gaussian_mix_prob_3, self).__init__()
        self.Q = Q
    def forward(self, x,
                mean1, mean2, mean3,
                scale1, scale2, scale3,
                probs1, probs2, probs3,
                Q=None, x_mean=None, return_lkl=False):
        if Q is None:
            Q = self.Q
        if use_clamp:
            if x_mean is None:
                x_mean = x.mean().detach()
            x_min = x_mean - 15_000 * Q
            x_max = x_mean + 15_000 * Q
            x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())

        scale1 = torch.clamp(scale1, min=1e-9)
        scale2 = torch.clamp(scale2, min=1e-9)
        scale3 = torch.clamp(scale3, min=1e-9)

        m1 = torch.distributions.normal.Normal(mean1, scale1)
        m2 = torch.distributions.normal.Normal(mean2, scale2)
        m3 = torch.distributions.normal.Normal(mean3, scale3)

        likelihood1 = torch.abs(m1.cdf(x + 0.5*Q) - m1.cdf(x - 0.5*Q))
        likelihood2 = torch.abs(m2.cdf(x + 0.5*Q) - m2.cdf(x - 0.5*Q))
        likelihood3 = torch.abs(m3.cdf(x + 0.5*Q) - m3.cdf(x - 0.5*Q))

        likelihood = Low_bound.apply(probs1 * likelihood1 + probs2 * likelihood2 + probs3 * likelihood3)

        if return_lkl:
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            return bits

class Entropy_bernoulli(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, p):
        p = torch.clamp(p, min=1e-6, max=1 - 1e-6)
        pos_mask = (1 + x) / 2.0
        neg_mask = (1 - x) / 2.0
        pos_prob = p
        neg_prob = 1 - p
        param_bit = -torch.log2(pos_prob) * pos_mask + -torch.log2(neg_prob) * neg_mask
        return param_bit


class Entropy_factorized(nn.Module):
    def __init__(self, channel=32, init_scale=10, filters=(3, 3, 3), likelihood_bound=1e-6,
                 tail_mass=1e-9, optimize_integer_offset=True, Q=1):
        super(Entropy_factorized, self).__init__()
        self.filters = tuple(int(t) for t in filters)
        self.init_scale = float(init_scale)
        self.likelihood_bound = float(likelihood_bound)
        self.tail_mass = float(tail_mass)
        self.optimize_integer_offset = bool(optimize_integer_offset)
        self.Q = Q
        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1")
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
        self._matrices = nn.ParameterList([])
        self._bias = nn.ParameterList([])
        self._factor = nn.ParameterList([])
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix = nn.Parameter(torch.FloatTensor(
                channel, filters[i + 1], filters[i]))
            self.matrix.data.fill_(init)
            self._matrices.append(self.matrix)
            self.bias = nn.Parameter(
                torch.FloatTensor(channel, filters[i + 1], 1))
            noise = np.random.uniform(-0.5, 0.5, self.bias.size())
            noise = torch.FloatTensor(noise)
            self.bias.data.copy_(noise)
            self._bias.append(self.bias)
            if i < len(self.filters):
                self.factor = nn.Parameter(
                    torch.FloatTensor(channel, filters[i + 1], 1))
                self.factor.data.fill_(0.0)
                self._factor.append(self.factor)

    def _logits_cumulative(self, logits, stop_gradient):
        for i in range(len(self.filters) + 1):
            matrix = nnf.softplus(self._matrices[i])
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(matrix, logits)
            bias = self._bias[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self._factor):
                factor = nnf.tanh(self._factor[i])
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * nnf.tanh(logits)
        return logits

    def forward(self, x, Q=None, return_lkl=False):
        if Q is None:
            Q = self.Q
        else:
            if isinstance(Q, torch.Tensor):
                Q = Q.permute(1, 0).contiguous()
                Q = Q.view(Q.shape[0], 1, -1)
        x = x.permute(1, 0).contiguous()
        x = x.view(x.shape[0], 1, -1)
        lower = self._logits_cumulative(x - 0.5*Q, stop_gradient=False)
        upper = self._logits_cumulative(x + 0.5*Q, stop_gradient=False)
        sign = -torch.sign(torch.add(lower, upper))
        sign = sign.detach()
        likelihood = torch.abs(nnf.sigmoid(sign * upper) - nnf.sigmoid(sign * lower))
        if return_lkl:
            likelihood = likelihood.view(likelihood.shape[0], -1).permute(1, 0).contiguous()
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            bits = bits.view(bits.shape[0], -1)
            bits = bits.permute(1, 0).contiguous()
            return bits


class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = np.logical_or(
            x.cpu().numpy() >= 1e-6, g.cpu().numpy() < 0.0)
        t = torch.Tensor(pass_through_if+0.0).cuda()
        return grad1 * t


class UniverseQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        b = 0
        uniform_distribution = Uniform(-0.5*torch.ones(x.size())
                                       * (2**b), 0.5*torch.ones(x.size())*(2**b)).sample().cuda()
        return torch.round(x+uniform_distribution)-uniform_distribution

    @staticmethod
    def backward(ctx, g):

        return g



class mix_3D2D_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list,
            log2_hashmap_size,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.output_dim = self.encoding_xyz.output_dim + \
                          self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))
        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)
        return out_i

class Channel_CTX_fea(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP_d0 = nn.Sequential(
            nn.Linear(50*3+10*0, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(50*3+10*1, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(50*3+10*2, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(50*3+10*3, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(50*3+10*4, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(torch.cat([mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3, mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        if to_dec == 4:
            return mean_d4, scale_d4, prob_d4
        return mean_adj, scale_adj, prob_adj

class Channel_CTX_fea_tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 10]))
        self.MLP_d1 = nn.Sequential(
            nn.Linear(10*1, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(10*2, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(10*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d4 = nn.Sequential(
            nn.Linear(10*4, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):
        NN = fea_q.shape[0]
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2], dim=-1)), chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d4(torch.cat([d0, d1, d2, d3], dim=-1)), chunks=3, dim=-1)
        mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        if to_dec == 4:
            return mean_d4, scale_d4, prob_d4
        return mean_adj, scale_adj, prob_adj
