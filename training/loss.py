'''
Loss implementation
Code adapted from following paper
"Training Generative Adversarial Networks with Limited Data."
See LICENSES/LICENSE_STYLEGAN2_ADA.txt for original license.
'''

import numpy as np
import torch
import copy

from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from metrics import id_loss, moco_loss
from metrics.lpips.lpips import LPIPS

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class TuningLoss(Loss):
    def __init__(self, k_shot, stage_name, device, G_mapping, G_synthesis, D=None, augment_pipe=None, \
        style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, \
        sreg_args=None, loss_lambda_dict=None):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

        self.k_shot = k_shot
        self.stage_name = stage_name
        self.sreg_args = sreg_args

        # Duplicate the modules for seen regularization
        if sreg_args.Greg or sreg_args.Dreg:
            self.Gt_mapping = copy.deepcopy(G_mapping).eval()
            self.Gt_synthesis = copy.deepcopy(G_synthesis).eval()
            self.Gt_mapping.requires_grad = False
            self.Gt_synthesis.requires_grad = False

        # If allow Discriminator's seen regularization
        if self.stage_name == 'LSR' and self.sreg_args.Dreg:
            self.Dt = copy.deepcopy(D).eval()
            self.Dt.requires_grad = False
        
        # Initialize the perceptual loss
        if self.stage_name == 'LSR':
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
            if sreg_args.perc_opts.moco_path is not None:
                self.id_loss = moco_loss.MocoLoss(sreg_args.perc_opts).to(self.device).eval()
            elif sreg_args.perc_opts.id_path is not None:
                self.id_loss = id_loss.IDLoss(sreg_args.perc_opts).to(self.device).eval()

        self.loss_lambda_dict = loss_lambda_dict

    def run_G(self, z, c):
        with misc.ddp_sync(self.G_mapping, False):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, False):
            img = self.G_synthesis(ws)
        return img, ws

    def run_G_magn(self, z, c, m=1):
        with misc.ddp_sync(self.G_mapping, False):
            ws = self.G_mapping(z, c, magnitude=m)
        with misc.ddp_sync(self.G_synthesis, False):
            img = self.G_synthesis(ws)
        return img, ws

    def Gts(self, z, c, all=False):
        seen_ws_t = self.Gt_mapping(z, c)
        seen_ws_s = self.G_mapping(z, c)
        if not all:
            return seen_ws_t, seen_ws_s
        seen_img_t = self.Gt_synthesis(seen_ws_t)
        seen_img_s = self.G_synthesis(seen_ws_s)
        return seen_ws_t, seen_ws_s, seen_img_t, seen_img_s

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, False):
            logits = self.D(img, c)
        return logits
    
    # min-max normalization
    def normalize(self, t, nmin=-1, nmax=1):
        min = t.min()
        max = t.max()
        nt = (t - min) / (max - min) * (nmax - nmin) + nmin
        return nt
    
    def D_conv(self, D, img):
        x = None
        for res in D.block_resolutions:
            block = getattr(D, f'b{res}')
            x, img = block(x, img)

        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if D.b4.architecture == 'skip':
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + D.b4.fromrgb(img)

        # Main layers.
        if D.b4.mbstd is not None:
            x = D.b4.mbstd(x)
        x = D.b4.conv(x)
        x = x.reshape([x.shape[0], -1])
        return x
    
    def Dts_feat(self, img):
        seen_feature_t = self.D_conv(self.Dt, img)
        with misc.ddp_sync(self.D, False):
            seen_feature_s = self.D_conv(self.D, img)
        stack_seen_feature = torch.stack([seen_feature_t, seen_feature_s])
        stack_norm_feature = self.normalize(stack_seen_feature)
        return  stack_norm_feature[0],  stack_norm_feature[1]     

    def accumulate_gradients(self, phase, ureal_img, opt_a, ureal_c, ureal_w, ugen_z, ugen_c, sgen_z, sgen_c, m, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                loss_G = 0

                # For both stages, magnitude restriction is used
                m_sig = torch.sigmoid(m)
                loss_mgt = (m_sig ** 2).mean()
                training_stats.report(f'Loss/G/loss_mgt', loss_mgt)
                loss_G += self.loss_lambda_dict['mgt_lambda'] * loss_mgt
                
                # Save gpu memory
                fs_uopt_a = opt_a[:self.k_shot]
                fs_ureal_img = ureal_img[:self.k_shot]
                fs_ureal_c = ureal_c[:self.k_shot]
                fs_ureal_w = ureal_w[:self.k_shot]
                # Generate from trainable anchors
                fs_ugen_img, fs_ugen_w = self.run_G_magn(fs_uopt_a, fs_ureal_c, m=m_sig)

                if self.stage_name == 'LAL':
                    # Approximation loss for LAL
                    loss_app = torch.nn.functional.mse_loss(fs_ureal_w, fs_ugen_w)
                    training_stats.report('Loss/G/loss_app', loss_app)
                    loss_G += self.loss_lambda_dict['app_lambda'] * loss_app

                if self.stage_name == 'LSR':
                    # Perceptual loss for LSR
                    loss_l2 = torch.nn.functional.mse_loss(fs_ureal_img, fs_ugen_img)
                    loss_lpips = self.lpips_loss(fs_ureal_img, fs_ugen_img)
                    loss_id = self.id_loss(fs_ureal_img, fs_ugen_img)
                    loss_perc = self.loss_lambda_dict['perc']['l2_lambda'] * loss_l2 + \
                        self.loss_lambda_dict['perc']['lpips_lambda'] * loss_lpips + \
                        self.loss_lambda_dict['perc']['id_lambda'] * loss_id
                    training_stats.report('Loss/G/loss_perc', loss_perc)
                    loss_G += self.loss_lambda_dict['perc_lambda'] * loss_perc

                    # Adversarial loss for LSR
                    ugen_img, _ugen_w = self.run_G_magn(ugen_z, ugen_c) # May get synced by Gpl.
                    ugen_logits = self.run_D(ugen_img, ugen_c, sync=False)
                    training_stats.report('Loss/scores/fake', ugen_logits)
                    training_stats.report('Loss/signs/fake', ugen_logits.sign())
                    loss_Gadv = torch.nn.functional.softplus(-ugen_logits) # -log(sigmoid(gen_logits))
                    training_stats.report('Loss/G/loss_adv', loss_Gadv)
                    loss_G += self.loss_lambda_dict['adv_lambda'] * loss_Gadv.mean()
                
                if self.sreg_args.Greg:
                    if self.stage_name == 'LAL':
                        # Regularize the generated seen latents
                        sgen_w_t, sgen_w_s = self.Gts(sgen_z, sgen_c)
                        loss_Greg = torch.nn.functional.mse_loss(sgen_w_t, sgen_w_s)
                        training_stats.report('Loss/G/loss_Greg', loss_Greg)
                        loss_G += self.loss_lambda_dict['sreg_lambda']['G'] * loss_Greg
                    elif self.stage_name == 'LSR':
                        # Regularize the generated seen images
                        _sgen_w_t, _sgen_w_s, sgen_img_t, sgen_img_s  = self.Gts(sgen_z, sgen_c, all=True)
                        loss_Greg = self.loss_lambda_dict['perc']['l2_lambda']  * torch.nn.functional.mse_loss(sgen_img_t, sgen_img_s) + \
                            self.loss_lambda_dict['perc']['lpips_lambda'] * self.lpips_loss(sgen_img_t, sgen_img_s).mean() + \
                            self.loss_lambda_dict['perc']['id_lambda'] * self.id_loss(sgen_img_t, sgen_img_s).mean()
                        training_stats.report('Loss/G/loss_sreg', loss_Greg)
                        loss_G += self.loss_lambda_dict['sreg_lambda']['G'] * loss_Greg

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_G.mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = ugen_z.shape[0] // self.pl_batch_shrink
                ugen_img, gen_ws = self.run_G(ugen_z[:batch_size], ugen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(ugen_img) / np.sqrt(ugen_img.shape[2] * ugen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(ugen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (ugen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                loss_Dgen = 0
                ugen_img, _ugen_w = self.run_G_magn(ugen_z, ugen_c)
                ugen_logits = self.run_D(ugen_img, ugen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', ugen_logits)
                training_stats.report('Loss/signs/fake', ugen_logits.sign())
                loss_Dgen_adv = torch.nn.functional.softplus(ugen_logits) # -log(1 - sigmoid(gen_logits))
                loss_Dgen += loss_Dgen_adv.mean()
                if self.sreg_args.Dreg:
                    fake_img = self.Gt_synthesis(self.Gt_mapping(sgen_z, sgen_c))
                    seen_feature_t, seen_feature_s = self.Dts_feat(fake_img)
                    loss_Dreg = torch.nn.functional.mse_loss(seen_feature_t, seen_feature_s)
                    training_stats.report('Loss/D/loss_Dreg', loss_Dreg)
                    loss_Dgen += self.loss_lambda_dict['sreg_lambda']['D'] * loss_Dreg
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = ureal_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, ureal_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_adv', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + (loss_Dreal + loss_Dr1)).mean().mul(gain).backward()

#----------------------------------------------------------------------------
