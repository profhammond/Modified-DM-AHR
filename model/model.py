import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os

import model.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)

        # =========================================================
        # GENERATOR
        # =========================================================
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # =========================================================
        # EMA MODEL (QUALITY BOOST)
        # =========================================================
        self.use_ema = True
        if self.use_ema:
            self.netG_ema = self.set_device(networks.define_G(opt))
            self.netG_ema.load_state_dict(self.netG.state_dict())
            self.ema_decay = 0.999

        # =========================================================
        # LOSS + NOISE SCHEDULE
        # =========================================================
        self.set_loss()

        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'],
            schedule_phase='train'
        )

        # =========================================================
        # TRAIN MODE
        # =========================================================
        if self.opt['phase'] == 'train':
            self.netG.train()

            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if 'transformer' in k:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(f'Optimizing param: {k}')
            else:
                optim_params = list(self.netG.parameters())

            # =========================================================
            # ADAMW (better than Adam for diffusion models)
            # =========================================================
            self.optG = torch.optim.AdamW(
                optim_params,
                lr=opt['train']["optimizer"]["lr"],
                betas=(0.9, 0.999),
                weight_decay=1e-4
            )

            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()

    # =========================================================
    # EMA UPDATE (NEW)
    # =========================================================
    def update_ema(self):
        if not self.use_ema:
            return

        model = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
        ema_model = self.netG_ema.module if isinstance(self.netG_ema, nn.DataParallel) else self.netG_ema

        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    # =========================================================
    # DATA FEED
    # =========================================================
    def feed_data(self, data):
        self.data = self.set_device(data)

    # =========================================================
    # TRAIN STEP (IMPROVED STABILITY)
    # =========================================================
    def optimize_parameters(self):
        self.optG.zero_grad()

        l_pix = self.netG(self.data)

        # stabilize diffusion loss
        l_pix = l_pix.mean()

        l_pix.backward()

        # gradient clipping (VERY important for DDPM stability)
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 1.0)

        self.optG.step()

        # EMA update
        self.update_ema()

        # log
        self.log_dict['l_pix'] = l_pix.item()

    # =========================================================
    # TEST (USES EMA FOR BETTER QUALITY)
    # =========================================================
    def test(self, continous=False):
        model = self.netG_ema if self.use_ema else self.netG
        model.eval()

        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                self.SR = model.module.super_resolution(
                    self.data['SR'], continous
                )
            else:
                self.SR = model.super_resolution(
                    self.data['SR'], continous
                )

        model.train()

    # =========================================================
    # SAMPLING (EMA MODEL)
    # =========================================================
    def sample(self, batch_size=1, continous=False):
        model = self.netG_ema if self.use_ema else self.netG
        model.eval()

        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                self.SR = model.module.sample(batch_size, continous)
            else:
                self.SR = model.sample(batch_size, continous)

        model.train()

    # =========================================================
    # LOSS SETUP
    # =========================================================
    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    # =========================================================
    # NOISE SCHEDULE
    # =========================================================
    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase

            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device
                )
            else:
                self.netG.set_new_noise_schedule(
                    schedule_opt, self.device
                )

    # =========================================================
    # LOGGING
    # =========================================================
    def get_current_log(self):
        return self.log_dict

    # =========================================================
    # VISUALIZATION OUTPUT
    # =========================================================
    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()

        if sample:
            out_dict['SAM'] = self.SR.detach().cpu()
        else:
            out_dict['SR'] = self.SR.detach().cpu()
            out_dict['INF'] = self.data['SR'].detach().cpu()
            out_dict['HR'] = self.data['HR'].detach().cpu()

            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().cpu()
            else:
                out_dict['LR'] = out_dict['INF']

        return out_dict

    # =========================================================
    # NETWORK PRINT
    # =========================================================
    def print_network(self):
        s, n = self.get_network_description(self.netG)

        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = f"{self.netG.__class__.__name__}-{self.netG.module.__class__.__name__}"
        else:
            net_struc_str = self.netG.__class__.__name__

        logger.info(
            f'Network G structure: {net_struc_str}, params: {n:,}'
        )
        logger.info(s)

    # =========================================================
    # SAVE (INCLUDES EMA MODEL)
    # =========================================================
    def save_network(self, epoch, iter_step):

        gen_path = os.path.join(
            self.opt['path']['checkpoint'],
            f'I{iter_step}_E{epoch}_gen.pth'
        )

        opt_path = os.path.join(
            self.opt['path']['checkpoint'],
            f'I{iter_step}_E{epoch}_opt.pth'
        )

        # main model
        net = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
        state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
        torch.save(state_dict, gen_path)

        # EMA model
        if self.use_ema:
            ema_net = self.netG_ema.module if isinstance(self.netG_ema, nn.DataParallel) else self.netG_ema
            ema_state = {k: v.cpu() for k, v in ema_net.state_dict().items()}
            torch.save(ema_state, gen_path.replace("_gen.pth", "_ema_gen.pth"))

        # optimizer
        opt_state = {
            'epoch': epoch,
            'iter': iter_step,
            'optimizer': self.optG.state_dict()
        }

        torch.save(opt_state, opt_path)

        logger.info(f"Saved model: {gen_path}")

    # =========================================================
    # LOAD (EMA SUPPORT)
    # =========================================================
    def load_network(self):
        load_path = self.opt['path']['resume_state']

        if load_path is not None:
            logger.info(f'Loading model: {load_path}')

            gen_path = f'{load_path}_gen.pth'
            opt_path = f'{load_path}_opt.pth'

            net = self.netG.module if isinstance(self.netG, nn.DataParallel) else self.netG
            net.load_state_dict(
                torch.load(gen_path),
                strict=(not self.opt['model']['finetune_norm'])
            )

            # EMA load
            if self.use_ema:
                ema_path = f'{load_path}_ema_gen.pth'
                if os.path.exists(ema_path):
                    ema_net = self.netG_ema.module if isinstance(self.netG_ema, nn.DataParallel) else self.netG_ema
                    ema_net.load_state_dict(torch.load(ema_path))
                    logger.info("EMA model loaded")

            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
