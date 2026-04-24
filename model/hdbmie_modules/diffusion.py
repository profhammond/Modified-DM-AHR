# Modified Diffusion
import math
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from tqdm import tqdm


# ----------------------------
# Beta Schedule
# ----------------------------

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64
    )
    return betas


def make_beta_schedule(schedule, n_timestep,
                       linear_start=1e-4,
                       linear_end=2e-2,
                       cosine_s=8e-3):

    if schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)

    elif schedule == 'warmup':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.2)

    elif schedule == 'cosine':
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = torch.cos(timesteps / (1 + cosine_s) * math.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)

    else:
        raise NotImplementedError(schedule)

    return betas


# ----------------------------
# Helpers
# ----------------------------

def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device):
    return torch.randn(shape, device=device)


# ----------------------------
# Gaussian Diffusion
# ----------------------------

class GaussianDiffusion(nn.Module):
    def __init__(
        self, 
        model, 
        image_size, 
        channels=3,
        loss_type='l1', 
        conditional=True,
        ):
        super().__init__()

        self.denoise_fn = model
        self.image_size = image_size
        self.channels = channels
        self.conditional = conditional
        self.loss_type = loss_type
        timesteps = 1000

        betas = torch.linspace(1e-4, 2e-2, timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod)
        )
        
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1 - alphas_cumprod)
        )

        

    # ----------------------------
    # LOSS
    # ----------------------------

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(device)

        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(device)

        elif self.loss_type == 'huber':
            self.loss_func = nn.SmoothL1Loss(reduction='mean').to(device)

        else:
            raise NotImplementedError(self.loss_type)

    # ----------------------------
    # SCHEDULE
    # ----------------------------

    def set_new_noise_schedule(self, schedule_opt, device):
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']
        )

        betas = torch.tensor(betas, dtype=torch.float32, device=device)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = betas.shape[0]

        # buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))

    # ----------------------------
    # q(x_t | x_0)
    # ----------------------------

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ----------------------------
    # TRAINING LOSS
    # ----------------------------

    def p_losses(self, x_in):

        x_start = x_in['HR']
        b = x_start.shape[0]

        t = torch.randint(
            0, self.num_timesteps, (b,), device=x_start.device
        ).long()

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        if self.conditional:
            model_input = torch.cat([x_in['SR'], x_noisy], dim=1)
        else:
            model_input = x_noisy

        noise_pred = self.denoise_fn(model_input, t)

        # weighted loss improves diffusion stability
        weight = (t.float() / self.num_timesteps).clamp(0.1, 1.0)
        loss = (self.loss_func(noise_pred, noise) * weight.mean())

        return loss

    # ----------------------------
    # FORWARD
    # ----------------------------

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
