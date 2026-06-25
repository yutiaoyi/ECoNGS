import torch
import numpy as np
from utils.general_utils import get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p

class DecoderMLPs(nn.Module):
    def __init__(self, num_TFs, feat_dim: int=32, n_offsets: int=10):
        super(DecoderMLPs, self).__init__()
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets

        self.optimizer = None
        self.tf_embeddings = nn.Embedding(num_TFs, feat_dim).cuda()
        with torch.no_grad():
            self.tf_embeddings.weight.zero_()
        
        self.ob_dist_dim = 0
        
        self.mlp_opacity = nn.Sequential(
            nn.Linear(2*feat_dim+3, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.cov_dist_dim = 0        
        self.mlp_cov = nn.Sequential(
            nn.Linear(2*feat_dim+3, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()
        
        self.mlp_shading = nn.Sequential(
            nn.Linear(2*feat_dim+3, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()
        
        self.mlp_normal = nn.Sequential(
            nn.Linear(2*feat_dim+3, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
        ).cuda()
    
    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_normal.eval()
        self.mlp_shading.eval()


    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_normal.train()
        self.mlp_shading.train()
    
    @property
    def get_normal_mlp(self):
        return self.mlp_normal
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_shading_mlp(self):
        return self.mlp_shading
    
    @property
    def get_tf_embeddings(self):
        return self.tf_embeddings
    
    def remove_unused_embeddings(self, remove_idxs):
        if not remove_idxs:
            return
        device = self.tf_embeddings.weight.device
        old_weight = self.tf_embeddings.weight.data
        num_old, feat_dim = old_weight.shape
        keep_idxs = [i for i in range(num_old) if i not in remove_idxs]
        new_weight = old_weight[keep_idxs].clone()

        new_embedding = nn.Embedding(len(keep_idxs), feat_dim).to(device)
        with torch.no_grad():
            new_embedding.weight.copy_(new_weight)

        self.tf_embeddings = new_embedding
        print(f"Removed embeddings at idx {remove_idxs}, new num_TFs={len(keep_idxs)}")
      
    
    
    def training_setup(self, training_args):
        l = [
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_normal.parameters(), 'lr': training_args.mlp_normal_lr_init, "name": "mlp_normal"},
            {'params': self.mlp_shading.parameters(), 'lr': training_args.mlp_shading_lr_init, "name": "mlp_shading"},
            {'params': self.tf_embeddings.parameters(), 'lr': training_args.tf_embedding_lr_init, "name": "tf_embeddings"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_normal_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_normal_lr_init,
                                                    lr_final=training_args.mlp_normal_lr_final,
                                                    lr_delay_mult=training_args.mlp_normal_lr_delay_mult,
                                                    max_steps=training_args.mlp_normal_lr_max_steps)
        
        self.mlp_shading_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_shading_lr_init,
                                                    lr_final=training_args.mlp_shading_lr_final,
                                                    lr_delay_mult=training_args.mlp_shading_lr_delay_mult,
                                                    max_steps=training_args.mlp_shading_lr_max_steps)
        
        self.tf_embeddings_scheduler_args = get_expon_lr_func(lr_init=training_args.tf_embedding_lr_init,
                                                    lr_final=training_args.tf_embedding_lr_final,
                                                    lr_delay_mult=training_args.tf_embedding_lr_delay_mult,
                                                    max_steps=training_args.tf_embedding_lr_max_steps)
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_normal":
                lr = self.mlp_normal_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_shading":
                lr = self.mlp_shading_scheduler_args(iteration)
                param_group['lr'] = lr
    
    def save_mlp_checkpoints(self, path, mode = 'split'):
        mkdir_p(path)
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, 2*self.feat_dim+3).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, 2*self.feat_dim+3).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_normal.eval()
            mlp_normal = torch.jit.trace(self.mlp_normal, (torch.rand(1, 2*self.feat_dim+3).cuda()))
            mlp_normal.save(os.path.join(path, 'normal_mlp.pt'))
            self.mlp_normal.train()
            
            self.mlp_shading.eval()
            shading_mlp = torch.jit.trace(self.mlp_shading, (torch.rand(1, 2*self.feat_dim+3).cuda()))
            shading_mlp.save(os.path.join(path, 'shading_mlp.pt'))
            self.mlp_shading.train()
            
            self.tf_embeddings.eval()
            torch.save(self.tf_embeddings.state_dict(), os.path.join(path, 'tf_embeddings.pth'))
            self.tf_embeddings.train()

        elif mode == 'unite':
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'normal_mlp': self.mlp_normal.state_dict(),
                'shading_mlp': self.mlp_shading.state_dict(),
                'tf_embeddings': self.tf_embeddings.state_dict()
                }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_normal = torch.jit.load(os.path.join(path, 'normal_mlp.pt')).cuda()
            self.mlp_shading = torch.jit.load(os.path.join(path, 'shading_mlp.pt')).cuda()
            self.tf_embeddings.load_state_dict(torch.load(os.path.join(path, 'tf_embeddings.pth')))
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_normal.load_state_dict(checkpoint['normal_mlp'])
            self.mlp_shading.load_state_dict(checkpoint['shading_mlp'])
            self.tf_embeddings.load_state_dict(checkpoint['tf_embeddings'])
        else:
            raise NotImplementedError
    