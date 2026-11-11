import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * -(math.log(10000.0) / (half - 1))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def extract(values, timesteps, x_shape):
    out = values.gather(0, timesteps.long() - 1)
    return out.view(-1, 1, 1, 1).expand(x_shape)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)
        self.cond_proj = nn.Conv2d(cond_ch, out_ch, kernel_size=1) if cond_ch > 0 else None

        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, cond_feat):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # FiLM-style modulation
        h = h + self.time_proj(t_emb).view(x.size(0), -1, 1, 1)
        if cond_feat is not None and self.cond_proj is not None:
            if cond_feat.shape[-2:] != h.shape[-2:]:
                cond_feat = F.interpolate(cond_feat, size=h.shape[-2:], mode='bilinear', align_corners=False)
            h = h + self.cond_proj(cond_feat)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class CondEncoder(nn.Module):
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]


class ConditionalUNet(nn.Module):
    def __init__(self, time_dim=256, base_ch=64):
        super().__init__()
        self.in_conv = nn.Conv2d(3, base_ch, kernel_size=3, padding=1)

        self.down1 = ResidualBlock(base_ch, base_ch, time_dim, cond_ch=base_ch)
        self.down2 = ResidualBlock(base_ch, base_ch * 2, time_dim, cond_ch=base_ch * 2)
        self.downsample1 = Downsample(base_ch * 2)

        self.down3 = ResidualBlock(base_ch * 2, base_ch * 4, time_dim, cond_ch=base_ch * 4)
        self.downsample2 = Downsample(base_ch * 4)

        self.mid = ResidualBlock(base_ch * 4, base_ch * 4, time_dim, cond_ch=base_ch * 4)

        self.up3 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim, cond_ch=base_ch * 4)
        self.upsample2 = Upsample(base_ch * 4)

        # up2 sees concat of u3_up (base_ch*4) and skip d3 (base_ch*4) => in_ch=base_ch*8
        self.up2 = ResidualBlock(base_ch * 8, base_ch * 2, time_dim, cond_ch=base_ch * 2)
        self.upsample1 = Upsample(base_ch * 2)

        # up1 sees concat of u2_up (base_ch*2) and skip d1 (base_ch) => in_ch=base_ch*3
        self.up1 = ResidualBlock(base_ch * 3, base_ch, time_dim, cond_ch=base_ch)

        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=base_ch)
        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)

    def forward(self, x, t_emb, cond_feats):
        # cond_feats: list of [1x,1/2x,1/4x] features
        x0 = self.in_conv(x)

        d1 = self.down1(x0, t_emb, cond_feats[0])
        d2 = self.down2(d1, t_emb, cond_feats[1])
        d2_ds = self.downsample1(d2)

        d3 = self.down3(d2_ds, t_emb, cond_feats[2])
        d3_ds = self.downsample2(d3)

        m = self.mid(d3_ds, t_emb, cond_feats[2])

        u3 = self.up3(m, t_emb, cond_feats[2])
        u3_up = self.upsample2(u3)

        u2 = self.up2(torch.cat([u3_up, d3], dim=1), t_emb, cond_feats[1])
        u2_up = self.upsample1(u2)

        u1 = self.up1(torch.cat([u2_up, d1], dim=1), t_emb, cond_feats[0])

        out = self.out_norm(u1)
        out = F.silu(out)
        out = self.out_conv(out)
        out = torch.clamp(out, 0.0, 1.0)
        return out


class DiffusionRefiner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.train_steps = int(config.DIFF_TRAIN_STEPS)
        self.infer_steps = int(config.DIFF_INFER_STEPS)
        self.beta_start = float(config.DIFF_BETA_START)
        self.beta_end = float(config.DIFF_BETA_END)
        self.eta = float(config.DIFF_DDIM_ETA)
        self.time_dim = int(config.DIFF_TIME_EMB_DIM)
        base_ch = int(config.DIFF_BASE_CH)

        betas = torch.linspace(self.beta_start, self.beta_end, self.train_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        self.cond_encoder = CondEncoder(in_ch=7, base_ch=base_ch)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim * 2),
            nn.SiLU(),
            nn.Linear(self.time_dim * 2, self.time_dim),
        )
        self.unet = ConditionalUNet(time_dim=self.time_dim, base_ch=base_ch)

    def timestep_embedding(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        emb = sinusoidal_embedding(t, self.time_dim)
        return self.time_mlp(emb)

    def q_sample(self, x0, t, noise):
        sqrt_ab = extract(torch.sqrt(self.alphas_cumprod), t, x0.shape)
        sqrt_mab = extract(torch.sqrt(1.0 - self.alphas_cumprod), t, x0.shape)
        return sqrt_ab * x0 + sqrt_mab * noise

    @torch.no_grad()
    def ddpm_step(self, x, t, cond_feats, t_prev):
        if t.dim() == 0:
            t_in = t.unsqueeze(0).expand(x.shape[0])
        elif t.numel() == 1:
            t_in = t.expand(x.shape[0])
        else:
            t_in = t
        t_emb = self.timestep_embedding(t_in)
        pred_x0 = self.unet(x, t_emb, cond_feats)

        alpha_t = extract(self.alphas, t, x.shape)
        alpha_cum_t = extract(self.alphas_cumprod, t, x.shape)
        noise = (x - torch.sqrt(alpha_cum_t) * pred_x0) / torch.clamp(torch.sqrt(1 - alpha_cum_t), min=1e-5)

        if t_prev is None:
            return pred_x0

        alpha_cum_prev = extract(self.alphas_cumprod, t_prev, x.shape)
        sigma = self.eta * torch.sqrt((1 - alpha_cum_prev) / (1 - alpha_cum_t) * (1 - alpha_t))
        mean = torch.sqrt(alpha_cum_prev) * pred_x0 + torch.sqrt(1 - alpha_cum_prev - sigma ** 2) * noise
        if self.eta == 0:
            return mean
        return mean + sigma * torch.randn_like(x)

    def forward_train(self, images, shadows, masks, coarse, t_override=None):
        # images: GT; coarse: coarse prediction
        b = images.size(0)
        device = images.device
        cond = torch.cat([shadows, masks, coarse], dim=1)
        cond_feats = self.cond_encoder(cond)

        # optionally reuse externally provided timesteps (e.g., coarse fusion time)
        if t_override is not None:
            t = torch.clamp(t_override.long(), 1, self.train_steps)
        else:
            t = torch.randint(1, self.train_steps + 1, (b,), device=device)
        noise = torch.randn_like(coarse)
        x_t = self.q_sample(coarse, t, noise)

        t_emb = self.timestep_embedding(t)
        pred_x0 = self.unet(x_t, t_emb, cond_feats)
        return pred_x0, t

    @torch.no_grad()
    def forward_sample(self, shadows, masks, coarse):
        device = coarse.device
        cond = torch.cat([shadows, masks, coarse], dim=1)
        cond_feats = self.cond_encoder(cond)

        # choose a sub-sequence for fast sampling
        timesteps = torch.linspace(1, self.train_steps, steps=self.infer_steps, device=device, dtype=torch.long)
        x = coarse
        for idx in reversed(range(len(timesteps))):
            t = timesteps[idx]
            t_prev = timesteps[idx - 1] if idx > 0 else None
            x = self.ddpm_step(x, t, cond_feats, t_prev)
        return x
