# -*- coding: utf-8 -*-

import math
import os
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
import yaml

__version__ = "1.0.0"

from evalParams import evalParams

import sys
import numpy as np
from sklearn import metrics

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv1d):

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(nn.Module):

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h=self.heads)
        return self.to_out(out)


class Attention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

class OneDLayerWithLinear(nn.Module):
    def __init__(self, layernum, indexstart, layersLength, d_model):
        super().__init__()
        self.layernum = layernum
        self.indexstart = indexstart
        self.layersLength = layersLength

        lengthset = list(set(layersLength))
        self.linears = nn.ModuleList([nn.Linear(lengthset[i], d_model) for i in range(len(lengthset))])
        self.chooselinear = [lengthset.index(i) for i in layersLength]
        self.lu = nn.GELU()

    def forward(self, x):
        # x.shape: [1, 1, 97094]
        for i in range(self.layernum):
            res = self.linears[self.chooselinear[i]](x[:, :, self.indexstart[i]:self.indexstart[i + 1]])
            # res.shape # [1, 1, 128]
            if i == 0:
                totalres = res
            else:
                totalres = torch.concat((totalres, res), 1)
        # totalres.shape  # torch.Size([1, 34, 128])
        output = self.lu(totalres)
        return output


class Layers2OneDWithLinear(nn.Module):
    def __init__(self, layernum, indexstart, layersLength, d_model):
        super().__init__()
        self.layernum = layernum
        self.indexstart = indexstart
        self.layersLength = layersLength

        lengthset = list(set(layersLength))
        self.linears = nn.ModuleList([nn.Linear(d_model, lengthset[i]) for i in range(len(lengthset))])
        self.chooselinear = [lengthset.index(i) for i in layersLength]

    def forward(self, x):
        # x.shape: [1, 34, 128]
        for i in range(self.layernum):
            res = self.linears[self.chooselinear[i]](x[:, i, :])

            if i == 0:
                totalres = res
            else:
                totalres = torch.concat((totalres, res), 1)
        return totalres.unsqueeze(1)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1
    beta_start = scale * 0.00001 
    beta_end = scale * 0.02 
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion1D(nn.Module):

    def __init__(
            self,
            dinoisingModel,  
            *,
            seq_length, 
            timesteps=1000,
            sampling_timesteps=None, 
            loss_type='l1',  
            objective='pred_noise',  
            beta_schedule='cosine',  
            ddim_sampling_eta=0.,  
            auto_normalize=False  
    ):
        super().__init__()
        self.model = dinoisingModel
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, \
            'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, kgEmb, timeEmb, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False):

        model_output = self.model(x, t, kgEmb, timeEmb, x_self_cond)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, kgEmb, timeEmb, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, kgEmb, timeEmb, x_self_cond)

        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, kgEmb, timeEmb, x_self_cond=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, kgEmb=kgEmb,
                                                                          timeEmb=timeEmb, x_self_cond=x_self_cond,
                                                                          clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, kgEmb, timeEmb):
        batch, device = shape[0], self.betas.device

        img_change = []
        img = torch.randn(shape, device=device)
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, kgEmb, timeEmb, self_cond, clip_denoised=True)
            img_change.append(list(img.cpu().numpy()))

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, emb, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, emb, self_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, kgEmb, timeEmb, batch_size=16):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length), kgEmb, timeEmb)

    @torch.no_grad()
    def interpolate(self, x1, x2, emb, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, emb, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, kgEmb, timeEmb, noise=None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_self_cond = None
        if self.self_condition and random() < 0.5: 
            with torch.no_grad():

                x_self_cond = self.model_predictions(x, t, kgEmb, timeEmb).pred_x_start
                x_self_cond.detach_() 

        model_out = self.model(x, t, kgEmb, timeEmb, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise 
        elif self.objective == 'pred_x0':
            target = x_start 
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v 
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean() 

    def forward(self, mparam, kgEmb, timeEmb, *args, **kwargs):
        b, c, n, device, seq_length, = *mparam.shape, mparam.device, self.seq_length

        assert n == seq_length, f'seq length {n} must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        mparam = self.normalize(mparam)
        return self.p_losses(mparam, t, kgEmb, timeEmb, *args, **kwargs)


class Trainer1D(object):
    def __init__(
            self,
            diffusion_model: GaussianDiffusion1D,
            dataset: Dataset,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=3000,
            num_samples=25,
            results_folder='./ModelSave/exp777',
            amp=False,
            fp16=False,
            split_batches=True,
            logger,
            kgEmb,
            timeEmb,
            genTarget,
            targetDataset,
            scale,
            tbwriter,
            outputpath,
            sampleTimes,
            basemodel,
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        optimal_workers = min(8, cpu_count())
        dl = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=optimal_workers,
            drop_last=True,
            persistent_workers=True if optimal_workers > 0 else False,
            prefetch_factor=4 if optimal_workers > 0 else 2
        )

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=adam_betas, weight_decay=0.01)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.step = 0

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.logger = logger
        self.kgEmb = kgEmb
        self.timeEmb = timeEmb
        self.genTarget = genTarget

        with open('PredictionModel/config.yaml') as f:
            self.config = yaml.full_load(f)

        self.targetDataset = targetDataset
        self.scale = scale

        self.writer = tbwriter
        self.outputpath = outputpath
        self.sampleTimes = sampleTimes
        self.basemodel = basemodel

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        bestmetricsum = 99999999999999
        countBreak = 0
        early_stopping_patience = 6  # Early stop
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    mparam = data[0].to(device, non_blocking=True)
                    kgEmb = data[1].to(device, non_blocking=True)
                    timeEmb = data[2].to(device, non_blocking=True)

                    with self.accelerator.autocast():
                        loss = self.model(mparam, kgEmb, timeEmb)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)
                pbar.set_description(f'loss: {total_loss:.4f}')

                self.writer.add_scalar('Diffusion Loss', total_loss, self.step)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1                
                if accelerator.is_main_process:
                    self.ema.update()
                    # sample
                    if self.step > 0 and (self.step % self.save_and_sample_every == 0):
                        if self.save_and_sample_every == 3000:
                            self.save_and_sample_every = 4000 
                        self.ema.ema_model.eval()
                        self.logger.info('\neval:')
                        with torch.no_grad():

                            sampleRes = None
                            for _ in range(self.sampleTimes):
                                if self.targetDataset == 'sh':
                                    length = len(self.kgEmb)
                                    startNum = [0, length // 2, length]
                                    generateNum = np.diff(startNum)
                                    result = None
                                    for id in range(len(generateNum)):
                                        smallkgEmb = self.kgEmb[startNum[id]:startNum[id + 1]]
                                        smalltimeEmb = self.timeEmb[startNum[id]:startNum[id + 1]]
                                        sampled_seq = self.ema.ema_model.sample(
                                            torch.tensor(smallkgEmb).to(device),
                                            torch.tensor(smalltimeEmb).to(device),
                                            generateNum[id]
                                        )
                                        if result is None:
                                            result = sampled_seq.detach().cpu()
                                        else:
                                            result = np.concatenate((result, sampled_seq.detach().cpu()), axis=0)
                                elif self.targetDataset == 'nc':
                                    length = len(self.kgEmb)
                                    startNum = [0, length // 3, 2 * (length // 3), length]
                                    generateNum = np.diff(startNum)
                                    result = None
                                    for id in range(len(generateNum)):
                                        smallkgEmb = self.kgEmb[startNum[id]:startNum[id + 1]]
                                        smalltimeEmb = self.timeEmb[startNum[id]:startNum[id + 1]]
                                        sampled_seq = self.ema.ema_model.sample(
                                            torch.tensor(smallkgEmb).to(device),
                                            torch.tensor(smalltimeEmb).to(device),
                                            generateNum[id]
                                        )
                                        if result is None:
                                            result = sampled_seq.detach().cpu()
                                        else:
                                            result = np.concatenate((result, sampled_seq.detach().cpu()), axis=0)
                                elif self.targetDataset == 'nj':
                                    length = len(self.kgEmb)
                                    startNum = [0, length // 3, 2 * (length // 3), length]
                                    generateNum = np.diff(startNum)
                                    result = None
                                    for id in range(len(generateNum)):
                                        smallkgEmb = self.kgEmb[startNum[id]:startNum[id + 1]]
                                        smalltimeEmb = self.timeEmb[startNum[id]:startNum[id + 1]]
                                        sampled_seq = self.ema.ema_model.sample(
                                            torch.tensor(smallkgEmb).to(device),
                                            torch.tensor(smalltimeEmb).to(device),
                                            generateNum[id]
                                        )
                                        if result is None:
                                            result = sampled_seq.detach().cpu()
                                        else:
                                            result = np.concatenate((result, sampled_seq.detach().cpu()), axis=0)
                                else:
                                    raise ValueError(f"Unsupported targetDataset: {self.targetDataset}. Only 'sh', 'nc', 'nj' are supported.")

                                if sampleRes is None:
                                    sampleRes = np.expand_dims(result, axis=0)
                                else:
                                    sampleRes = np.concatenate((np.expand_dims(result, axis=0), sampleRes), axis=0)

                            sampleRes = np.average(sampleRes, axis=0)

                            mae = metrics.mean_absolute_error(sampleRes.flatten(), self.genTarget.flatten())
  
                            self.logger.info(f"parameter generation complete | Step {self.step:6d} | MAE: {mae:.6f} | Shape: {sampleRes.shape}")
                            self.writer.add_scalar('gene_params_mae', mae, self.step)

                            metricsum = evalParams(sampleRes * self.scale, self.config, device, self.logger,
                                                   self.targetDataset, self.writer, self.step, self.basemodel)

                            if metricsum < bestmetricsum:
                                improvement = bestmetricsum - metricsum
                                self.logger.info(f'NEW BEST RESULT! | Step {self.step:6d} | MetricSum: {metricsum:.6f} | Improvement: {improvement:.6f}')
                                bestmetricsum = metricsum

                                if hasattr(self, 'best_model_file') and os.path.exists(self.best_model_file):
                                    os.remove(self.best_model_file)

                                self.best_model_file = str(self.results_folder / f'model-{self.step}.pt')
                                self.save(self.step)
                                countBreak = 0 
                            else:
                                countBreak = countBreak + 1
                                self.logger.debug(f"📊 Evaluating | Step {self.step:6d} | MetricSum: {metricsum:.6f} | Number of unimproved iterations: {countBreak}")

                                if countBreak >= early_stopping_patience:
                                    self.logger.info(f"Early Stopping with {countBreak} times")
                                    accelerator.print(f'Early stopping at step {self.step}')
                                    return  

                pbar.update(1)

        accelerator.print('training complete')


