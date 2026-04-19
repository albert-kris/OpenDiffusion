import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999)

def sigmoid_beta_schedule(timesteps, start=0.0001, end=0.02):
    """
    Sigmoid schedule for beta values
    """
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (end - start) + start

def sqrt_beta_schedule(timesteps, start=0.0001, end=0.02):
    """
    Square root schedule for beta values
    """
    return torch.linspace(start**0.5, end**0.5, timesteps)**2

class diffusion(nn.Module):
    def __init__(self,
                 eps_model:nn.Module,  # x:[B,C,H,W] t:[B,]
                 start:float=0.0001,
                 end:float=2e-2,
                 timesteps:int=1000,
                 criterion:nn.Module=nn.MSELoss(),
                 schedule: str = 'linear',  # Beta schedule strategy: 'linear', 'cosine', 'sigmoid', 'sqrt'
                 objective: str = 'pred_noise',  # Prediction target: 'pred_noise', 'pred_x0', or 'pred_v'
                 min_snr_loss_weight: bool = False,  # Whether to use min-SNR loss weighting from https://arxiv.org/abs/2303.09556
                 min_snr_gamma: float = 5,  # Min-SNR loss weight clipping threshold, typically set to 5
                 vp_rf: bool = False,  # Whether to use variance-preserving rescaling factor for beta schedule
                 rescaling_factor: float = 1):  # Rescaling factor for alphas_cumprod calculation, 1 means no rescaling
        super().__init__()
        self.eps_model = eps_model
        self.T = timesteps
        self.criterion = criterion
        self.schedule = schedule
        self.objective = objective
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.vp_rf = vp_rf
        self.rescaling_factor = rescaling_factor
        self.use_self_cond = getattr(self.eps_model, 'use_self_conditioning', False) if eps_model is not None else False
        
        # Validate objective parameter
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, \
            f'objective must be one of {{"pred_noise", "pred_x0", "pred_v"}}, got {objective}'
        
        """
        Compute diffusion schedule parameters
        """
        if schedule == 'linear':
            betas = torch.linspace(start, end, timesteps)
        elif schedule == 'cosine':
            betas = torch.from_numpy(cosine_beta_schedule(timesteps)).float()
        elif schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps, start, end)
        elif schedule == 'sqrt':
            betas = sqrt_beta_schedule(timesteps, start, end)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        # Apply rescaling_factor and vp_rf
        if self.rescaling_factor != 1 or self.vp_rf:
            f = self.rescaling_factor
            alphas_cumprod = alphas_cumprod / (f - (f - 1) * alphas_cumprod)
            alphas_cumprod = torch.clamp(alphas_cumprod, 0, 1)
        
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_alphas = torch.sqrt(alphas)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)
        
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        posterior_log_variance_clipped = torch.log(posterior_variance)
        if posterior_log_variance_clipped.shape[0] > 1:
            posterior_log_variance_clipped[0] = posterior_log_variance_clipped[1]
        
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * sqrt_alphas / (1.0 - alphas_cumprod)
        
        # Compute loss_weight using min_snr_loss_weight and min_snr_gamma
        if self.min_snr_loss_weight:
            snr = alphas_cumprod / (1 - alphas_cumprod)
            snr = torch.clamp(snr, max=self.min_snr_gamma)
            
            if self.objective == 'pred_noise':
                loss_weight = snr / (snr + 1)
            elif self.objective == 'pred_x0':
                loss_weight = snr
            elif self.objective == 'pred_v':
                loss_weight = snr / (snr + 1)
            else:
                loss_weight = torch.ones_like(snr)
        else:
            # 当不使用 min_snr_loss_weight 时，loss_weight 为全 1
            loss_weight = torch.ones(timesteps)
        
        """
        Register parameters as buffers
        Why use buffers?
        - Ensures parameter device consistency during .to(device) calls
        - Only nn.Module parameters move automatically on .to(device)
        - Other self attributes don't move unless registered as buffers
        """
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas", sqrt_alphas)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)
        self.register_buffer("loss_weight", loss_weight)
    def extract(self, a, t, x):
        """
        Extract values from tensor a at indices t, handling different batch sizes.
        Timesteps start from 1 to T.
        t must be a tensor on the correct device.
        t: [batch_size,]
        """
        x_shape = x.shape
        batch_size = t.shape[0]
        out = a.gather(-1, t-1)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    @torch.no_grad()
    def add_noise(self,x_start,t):
        """
        Add noise to x_start at timestep t.
        t starts from 1, adding one noise level per timestep.
        x_start: [batch, channels, height, width]
        t: [batch,]
        """
        noise = torch.randn_like(x_start)  # Generate noise with same shape as x_start
        x_t = self.extract(self.sqrt_alphas_cumprod,t,x_start) * x_start + \
               self.extract(self.sqrt_one_minus_alphas_cumprod,t,x_start) * noise
        return x_t,noise
    
    def _clip_x_start(self, x):
        """Clip predicted x_start to valid range [-1, 1]"""
        return torch.clamp(x, -1.0, 1.0)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """Predict x_start from x_t and predicted noise"""
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t
            - self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t) * eps
        )

    def _predict_eps_from_x0(self, x_t, t, x0):
        """Convert x0 prediction to eps prediction"""
        return (
            (x_t - self.extract(self.sqrt_alphas_cumprod, t, x_t) * x0) /
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        )
    
    def predict_v(self, x_start, t, eps):
        """Predict v parameter from x_start and eps"""
        return (
            self.extract(self.sqrt_alphas_cumprod, t, x_start) * eps -
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        """Predict x_start from x_t and v"""
        return (
            self.extract(self.sqrt_alphas_cumprod, t, x_t) * x_t -
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t) * v
        )

    def _q_posterior(self, x_start, x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t) * x_start
            + self.extract(self.posterior_mean_coef2, t, x_t) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t)
        posterior_log_variance = self.extract(self.posterior_log_variance_clipped, t, x_t)
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def _predict_eps(self, x_t, t, self_cond=None):
        """
        Predict noise with support for different objectives and self-conditioning.
        Converts model predictions to noise prediction based on objective.
        """
        pred = self.eps_model(x_t, t, self_cond=self_cond)
        
        # Convert predictions based on objective
        if self.objective == 'pred_x0':
            # Model predicts x0, convert to eps
            eps = self._predict_eps_from_x0(x_t, t, pred)
            return eps
        elif self.objective == 'pred_v':
            # Model predicts v, convert to eps
            x_start = self.predict_start_from_v(x_t, t, pred)
            eps = self._predict_eps_from_x0(x_t, t, x_start)
            return eps
        else:  # 'pred_noise'
            # Model predicts eps directly
            return pred
    
    @torch.no_grad()
    def denoise_ddpm(self,x_t,t,clip_denoised=True,self_cond=None):
        """
        DDPM denoising step.
        Timesteps are indexed by the state on the right side (consistent for both adding/removing noise).
        x0->x1->...->xT represents T+1 states in total.
        """
        pred_noise = self._predict_eps(x_t, t, self_cond=self_cond)
        x_start = self._predict_xstart_from_eps(x_t, t, pred_noise)
        if clip_denoised:
            x_start = self._clip_x_start(x_start)
        model_mean, _, model_log_variance = self._q_posterior(x_start, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 1).float().view(-1, *((1,) * (x_t.dim() - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @ torch.no_grad()
    def denoise_ddim(self,x_t,t,t_prev,self_cond=None):
        """DDIM denoising step with support for self-conditioning"""
        if t_prev[0] != 0:
            sqrt_alpha_cumprod_prev = torch.sqrt(self.extract(self.alphas_cumprod,t_prev,x_t))
            sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1-self.extract(self.alphas_cumprod,t_prev,x_t))
        else:
            sqrt_alpha_cumprod_prev = torch.tensor(1.,device='cuda')
            sqrt_one_minus_alphas_cumprod_prev = torch.tensor([0.],device='cuda')
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod,t,x_t)
        sqrt_alpha_cumprod = torch.sqrt(self.extract(self.alphas_cumprod,t,x_t))
        
        pred_noise = self._predict_eps(x_t, t, self_cond=self_cond)
        x_t_1 = sqrt_alpha_cumprod_prev*(x_t-sqrt_one_minus_alphas_cumprod_t*pred_noise)\
            /sqrt_alpha_cumprod + sqrt_one_minus_alphas_cumprod_prev*pred_noise
        return x_t_1 
        
    @torch.no_grad()
    def denoise_loop_ddpm(self,
                          noise,
                          bar=True):
        """
        DDPM denoising loop from T to 1.
        t has shape [batch,], reshaped to [batch, 1, 1, 1] after extract.
        Supports self-conditioning by passing previous prediction as conditioning.
        """
        B = noise.shape[0]
        x_change = noise
        self_cond = None
        for step in tqdm(reversed(range(1,self.T+1)), desc='reverse ddpm..', total=self.T,disable=not bar):  # From T -> 1
            t = torch.full((B,), step, device=noise.device,dtype=torch.long)
            args = (x_change,t)
            kwargs = {}
            kwargs['self_cond'] = self_cond if self.use_self_cond else None
            x_change = self.denoise_ddpm(*args,**kwargs)
            # Update self_cond for next step if using self-conditioning
            if self.use_self_cond:
                with torch.no_grad():
                    pred_eps = self._predict_eps(x_change, t, self_cond=None)
                    self_cond = self._predict_xstart_from_eps(x_change, t, pred_eps).detach()
        return x_change

    @torch.no_grad()
    def denoise_loop_ddim(self,
                          noise,
                          step_interval=7,
                          bar=True):
        """DDIM denoising loop with self-conditioning support"""
        B = noise.shape[0]
        x_change = noise
        self_cond = None
        steps = []
        current_step = self.T
        while current_step >= 1:
            steps.append(current_step)
            current_step -= step_interval
        if steps[-1] != 1:
            steps.append(1)
        for i, step in enumerate(tqdm(steps, desc='reverse ddim..',disable=not bar)):
            t = torch.full((B,), step, device=noise.device, dtype=torch.long)
            t_prev = torch.full((B,), steps[i+1], device=noise.device, dtype=torch.long) if i+1 != len(steps) else torch.zeros((B,), device=noise.device, dtype=torch.long)
            args = (x_change, t, t_prev)
            kwargs = {}
            kwargs['self_cond'] = self_cond if self.use_self_cond else None
            x_change = self.denoise_ddim(*args,**kwargs)
            # Update self_cond for next step
            if self.use_self_cond:
                with torch.no_grad():
                    pred_eps = self._predict_eps(x_change, t, self_cond=None)
                    self_cond = self._predict_xstart_from_eps(x_change, t, pred_eps).detach()
        return x_change
          
    def cluster(self,
                mode,
                noise,
                step_interval=7,
                bar=True):
        """
        Clustering function that extracts intermediate features during denoising.
        """
        device = noise.device
        hsy = []
        def prehook(module, input, output):
            hsy.append(output.detach().cpu().shape)
        prehook_handle = self.eps_model.middle_block.register_forward_hook(prehook)
        with torch.no_grad():
            _,*size = noise.shape
            shape = (1,*size)
            noise_tmp = torch.randn(shape,device=device)
            tmp = self.eps_model(noise_tmp,torch.tensor([1],device=device))
        prehook_handle.remove()

        B = noise.shape[0]
        _,*mid_size_withoutbatch = hsy[0]
        def compute_times(T,number):
            if (T-1) % number == 0:
                return (T-1)//number + 1
            else:
                return (T-1)//number + 2

        mid_size = (B,*mid_size_withoutbatch)
        if mode in {'ddpm','pm','p'}:
            mid_block_outputs = torch.empty((self.T,*mid_size), device=device)
        if mode in {'ddim','im','i'}:
            mid_block_outputs = torch.empty((compute_times(self.T,step_interval),*mid_size), device=device)
        class HookState:
            def __init__(self):
                self.idx = 0
                self.call_count = 0
        state = HookState()
        def hook(module, input, output):
            # When using self-conditioning, each timestep calls forward twice
            # Only record on even calls (0, 2, 4, ...) to avoid duplicates
            if self.use_self_cond:
                if state.call_count % 2 == 0:
                    mid_block_outputs[state.idx] = output
                    state.idx += 1
                state.call_count += 1
            else:
                mid_block_outputs[state.idx] = output
                state.idx += 1    
        hook_handle = self.eps_model.middle_block.register_forward_hook(hook)
        if mode in {'ddpm','pm','p'}:
            output = self.denoise_loop_ddpm(noise,
                                            bar=bar)
        if mode in {'ddim','im','i'}:
            output = self.denoise_loop_ddim(noise, 
                                            step_interval=step_interval,
                                            bar=bar)
        hook_handle.remove()
        return mid_block_outputs.permute(1,0,2,3,4), output
    
    def get_z(self, x, t, pure=False):
        """
        Compute feature representations from UNet at timestep t.
        
        Args:
            x: Input image [B, C, H, W]
            t: Timestep [B,] or scalar
            pure: If True, use clean image at t=1; if False, use noised image at t
        
        Returns:
            z: Feature tensor from middle block
        """
        x_t, _ = self.add_noise(x, t)
        t_1 = torch.full((x.shape[0],), 1, device=x.device, dtype=torch.long)
        
        if pure:
            _, z = self.eps_model(x, t_1, return_z=True)
        else:
            _, z = self.eps_model(x_t, t, return_z=True)
        
        return z
    
    def loss(self, x, t, use_consistency=False, consistency_weight=0.01, cot_range=(0, 100)):
        """
        损失函数：支持基础扩散训练 + 可选的 CME 特征一致性损失
        - 自监督条件：50% 概率在训练中使用
        - 多种预测目标：pred_noise, pred_x0, pred_v
        - Min-SNR 损失加权
        - CME 特征一致性损失：约束中间层特征 z 的稳定性（聚类的直接输入）
        
        Args:
            x: 输入图像 [B, C, H, W]
            t: 时间步 [B,]
            use_consistency: 是否使用 CME 特征一致性损失
            consistency_weight: 一致性损失的权重（推荐 0.005-0.02，默认 0.01）
            cot_range: COT 有效区间 (t_min, t_max)，默认 (200, 400)
        """
        # 标准扩散损失（单样本）
        x_t, noise = self.add_noise(x, t)
        
        # 自监督条件：50% 的样本在训练中使用自监督条件
        self_cond = None
        if self.use_self_cond and torch.rand(1).item() < 0.5:
            with torch.no_grad():
                pred_noise_tmp = self.eps_model(x_t, t, self_cond=None)
                self_cond = self._predict_xstart_from_eps(x_t, t, pred_noise_tmp).detach()

        # 获取预测
        pred = self.eps_model(x_t, t, self_cond=self_cond)
        
        # 根据目标确定 target
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x
        elif self.objective == 'pred_v':
            target = self.predict_v(x, t, noise)
        
        # 应用损失加权
        loss_weight = self.extract(self.loss_weight, t, x_t)
        diffusion_loss = F.mse_loss(pred, target, reduction='none')
        diffusion_loss = (diffusion_loss * loss_weight).mean()
        
        # CME 特征一致性损失：约束 z 的稳定性
        if use_consistency:
            # 第二个样本：相同 t，不同噪声
            x_t_2, _ = self.add_noise(x, t)
            
            # 提取两个样本的 CME 特征 z（中间层特征，聚类的直接输入）
            with torch.no_grad():
                _, z1 = self.eps_model(x_t, t, self_cond=self_cond, return_z=True)
                _, z2 = self.eps_model(x_t_2, t, self_cond=self_cond, return_z=True)
            
            # 特征归一化：抑制极端值波动
            z1_norm = F.normalize(z1.view(z1.size(0), -1), p=2, dim=1)
            z2_norm = F.normalize(z2.view(z2.size(0), -1), p=2, dim=1)
            
            # L1 损失：对噪声更鲁棒（vs MSE）
            consistency_loss = F.l1_loss(z1_norm, z2_norm, reduction='none')
            
            # COT 区间权重：只在语义有效区间强约束
            # 中间 t（COT 区间）权重高，晚期 t（噪声主导）权重低
            t_min, t_max = cot_range
            # 高斯权重：以 COT 中心为峰值
            t_center = (t_min + t_max) / 2.0
            t_std = (t_max - t_min) / 4.0  # 标准差覆盖 95% 的 COT 区间
            t_weight = torch.exp(-((t.float() - t_center) ** 2) / (2 * t_std ** 2))
            t_weight = t_weight.view(-1, 1)  # [B, 1]
            
            consistency_loss = (consistency_loss * t_weight).mean()
            
            # 总损失：扩散损失为主，一致性损失为辅
            total_loss = diffusion_loss + consistency_weight * consistency_loss
            
            return total_loss
        else:
            return diffusion_loss