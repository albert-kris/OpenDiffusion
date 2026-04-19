import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion.nn import timestep_embedding
from diffusion.unet import TimestepBlock, TimestepEmbedSequential

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.dropout = dropout

        self.in_layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, self.out_channels)
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels)
        )
        
        self.out_layers = nn.Sequential(
            nn.LayerNorm(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.out_channels, self.out_channels)
        )
        
        if channels != self.out_channels:
            self.skip = nn.Linear(channels, self.out_channels)
        else:
            self.skip = nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip(x) + h

class UNet1D(nn.Module):
    def __init__(self, in_dim, embed_dim, mlp_time_embed=True, self_condition=False, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.mlp_time_embed = mlp_time_embed
        self.self_condition = self_condition
        
        # Time Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
            
        # Input Projection
        self.input_proj = nn.Linear(in_dim, embed_dim)
        
        # Encoder
        # Level 1
        self.enc1 = ResBlock(embed_dim, embed_dim, dropout)
        self.down1 = nn.Linear(embed_dim, embed_dim // 2)
        
        # Level 2
        self.enc2 = ResBlock(embed_dim // 2, embed_dim, dropout)
        self.down2 = nn.Linear(embed_dim // 2, embed_dim // 4)
        
        # Level 3 (Bottleneck)
        self.middle_block = TimestepEmbedSequential(
            ResBlock(embed_dim // 4, embed_dim, dropout),
            ResBlock(embed_dim // 4, embed_dim, dropout)
        )
        
        # Decoder
        # Level 2
        self.up2 = nn.Linear(embed_dim // 4, embed_dim // 2)
        self.dec2 = ResBlock(embed_dim // 2 + embed_dim // 2, embed_dim, dropout, out_channels=embed_dim // 2)
        
        # Level 1
        self.up1 = nn.Linear(embed_dim // 2, embed_dim)
        self.dec1 = ResBlock(embed_dim + embed_dim, embed_dim, dropout, out_channels=embed_dim)
        
        # Output Projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, in_dim)
        )

    def forward(self, x, timesteps, y=None, return_z=False, self_cond=None):
        # Time Embed
        t_emb = timestep_embedding(timesteps, self.embed_dim)
        t_emb = self.time_embed(t_emb)
            
        h = self.input_proj(x) # [B, embed_dim]
        
        # Encoder
        h1 = self.enc1(h, t_emb) # [B, embed_dim]
        h2_in = self.down1(h1) # [B, embed_dim//2]
        
        h2 = self.enc2(h2_in, t_emb) # [B, embed_dim//2]
        h3_in = self.down2(h2) # [B, embed_dim//4]
        
        # Bottleneck
        h_mid = self.middle_block(h3_in, t_emb) # [B, embed_dim//4]
        
        # Decoder
        h_up2 = self.up2(h_mid) # [B, embed_dim//2]
        h_dec2_in = torch.cat([h_up2, h2], dim=-1) # [B, embed_dim]
        h_dec2 = self.dec2(h_dec2_in, t_emb) # [B, embed_dim//2]
         
        h_up1 = self.up1(h_dec2) # [B, embed_dim]
        h_dec1_in = torch.cat([h_up1, h1], dim=-1) # [B, 2*embed_dim]
        h_dec1 = self.dec1(h_dec1_in, t_emb) # [B, embed_dim]
        
        out = self.out_proj(h_dec1)
        
        if return_z:
            return out, h_mid
            
        return out
