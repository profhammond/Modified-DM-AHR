import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# ----------------------------
# helpers
# ----------------------------

def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


# ----------------------------
# beta schedule
# ----------------------------

def make_beta_schedule(n_timestep, linear_start=1e-4, linear_end=2e-2):
    return np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)


# ----------------------------
# indexing helper
# ----------------------------

def extract(a, t, x_shape):
    b = t.shape[0]
    out = a.gather(0, t)  # FIX: gather along correct dim
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


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
        timesteps=1000
    ):
        super().__init__()

        self.denoise_fn = model
        self.image_size = image_size
        self.channels = channels
        self.conditional = conditional
        self.num_timesteps = timesteps

        # ---------------- LOSS ----------------
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss()
        elif loss_type == 'huber':
            self.loss_func = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # ---------------- SCHEDULE ----------------
        betas = torch.tensor(
            make_beta_schedule(self.num_timesteps),
            dtype=torch.float32
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # ---------------- buffers ----------------
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))

        self.register_buffer("sqrt_recip_alphas",
                             torch.sqrt(1.0 / alphas))

        # FIX: numerical stability
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8)
        )


    # ----------------------------
    # forward diffusion
    # ----------------------------

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


    # ----------------------------
    # training loss
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

        loss = self.loss_func(noise_pred, noise)

        return loss


    # ----------------------------
    # reverse step
    # ----------------------------

    def p_sample(self, x, t, cond=None):

        if self.conditional and cond is not None:
            model_input = torch.cat([cond, x], dim=1)
        else:
            model_input = x

        noise_pred = self.denoise_fn(model_input, t)

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred / (sqrt_one_minus_alphas_cumprod_t + 1e-8)
        )

        # FIX: correct per-sample t == 0 handling
        if (t == 0).all():
            return model_mean

        noise = torch.randn_like(x)
        posterior_var_t = extract(self.posterior_variance, t, x.shape)

        return model_mean + torch.sqrt(posterior_var_t) * noise


    # ----------------------------
    # sampling loop
    # ----------------------------

    @torch.no_grad()
    def sample(self, batch_size=1, cond=None):

        device = next(self.parameters()).device

        x = torch.randn(
            batch_size,
            self.channels,
            self.image_size,
            self.image_size,
            device=device
        )

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, cond)

        return x


    # ----------------------------
    # forward
    # ----------------------------

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
