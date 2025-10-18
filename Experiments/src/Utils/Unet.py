import torch
import math
from torch import nn

# PyTorch implementation of a model similar to Ho et al., 2020
# From https://github.com/spmallick/learnopencv/tree/master/Guide-to-training-DDPMs-from-Scratch
# And https://github.com/lucidrains/denoising-diffusion-pytorch
# And https://huggingface.co/blog/annotated-diffusion


# ===============================================
#               Position embedding
# ===============================================
# Standard sinusoidal position embedding from Vaswani et al., 2017
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, n_steps=1000, dim=128, dim_exp=512):
        super().__init__()
        self.dim = dim
        self.T = n_steps
        self.dim_exp = dim_exp
        
        half_dim = self.dim // 2
    
        # Time embeddings
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        ts = torch.arange(self.T, dtype=torch.float32)
        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        # Time MLP
        self.time_mlp = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=self.dim, out_features=self.dim_exp),
            nn.GELU(),
            nn.Linear(in_features=self.dim_exp, out_features=self.dim_exp),
        )

    def forward(self, time):
        return self.time_mlp(time)


# ===============================================
#               Attention block
# ===============================================
class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=4, groups=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.mhsa = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, _, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, self.dim, H * W).transpose(1, 2)  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]
        h, _ = self.mhsa(h, h, h)  # [B, H*W, C]
        h = h.transpose(2, 1).view(B, self.dim, H, W)  # [B, C, H*W] --> [B, C, H, W]
        return x + h

# ===============================================
#               ResNet block
# ===============================================
class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout_rate=0.1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, dim, dim_out,
                  dropout_rate=0.1, time_emb_dims=512, 
                  groups=8, apply_attention=False):
        super().__init__()
        
        self.act = nn.SiLU()
        
        # Residual blocks
        self.block1 = Block(dim=dim, dim_out=dim_out, 
                            dropout_rate=0.0, groups=groups)
        self.block2 = Block(dim=dim_out, dim_out=dim_out, 
                            dropout_rate=dropout_rate, groups=groups)

        # Time embedding
        self.dense = nn.Linear(time_emb_dims, dim_out)

        # Residual and attention
        self.res_conv =  nn.Conv2d(dim, dim_out, 1, 1) if dim != dim_out else nn.Identity()
        self.attention = Attention(dim=dim_out) if apply_attention else nn.Identity()

    def forward(self, x, t):
        # Group 1
        h = self.block1(x)
        
        # Group 2
        h += self.dense(self.act(t))[:, :, None, None]

        # Group 3
        h = self.block2(h)

        # Residual and attention
        h = h + self.res_conv(x)
        h = self.attention(h)
        return h
    

# ===============================================
#               Downsample and Upsample
# ===============================================
class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv2d(channels, channels, 3, 2, 1)

    def forward(self, x, *args):
        return self.downsample(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        )

    def forward(self, x, *args):
        return self.upsample(x)
    
    
# ===============================================
#               Unet architecture
# ===============================================
# Putting it all together
class UNet(nn.Module):
    def __init__(
        self,
        input_channels=3,       # Number of input channels
        output_channels=3,      # Number of output channels
        num_res_blocks=2,       # Number of resNet block per resolution maps
        base_channels=128,      # Number of base channels
        base_channels_multiples=(1, 2, 4, 8),           # Channels multiple (defines the number of maps)
        apply_attention=(False, True, True, False),     # Attention to maps (must of the same size as base_channels_multiples)
        dropout_rate=0.1,
    ):
        super().__init__()

        if len(base_channels_multiples) != len(apply_attention):
            raise Exception('base_channels_multiples and apply_attention must have the same length')
        
        # Time embedding
        time_emb_dims_exp = base_channels * 4       # Size of time embedding is base*4 (Ho et al.)
        self.time_embeddings = SinusoidalPositionEmbeddings(dim=base_channels, dim_exp=time_emb_dims_exp)
        
        self.init_conv = nn.Conv2d(input_channels, base_channels, 3, 1, 1) # 0


        # Stacking layers
        self.encoder_blocks = nn.ModuleList([])
        self.decoder_blocks = nn.ModuleList([])
        num_resolutions = len(base_channels_multiples) # Number of resolution maps
        
        curr_channels = [base_channels]
        in_channels = base_channels

        # Encoder blocks
        for level in range(num_resolutions):
            is_last = (level >= (num_resolutions - 1))
            
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks):
                block = ResnetBlock(
                    dim=in_channels,
                    dim_out=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )
                self.encoder_blocks.append(block)

                in_channels = out_channels
                curr_channels.append(in_channels)
            
            # Add downsampling at last block
            if not is_last:
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)

        # Bottleneck blocks: 2 residual blocks
        self.bottleneck_blocks = nn.ModuleList(
            (
                ResnetBlock(
                    dim=in_channels,
                    dim_out=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=True,
                ),
                ResnetBlock(
                    dim=in_channels,
                    dim_out=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=False,
                ),
            )
        )
        
        # Decoder blocks
        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks + 1):
                encoder_in_channels = curr_channels.pop()
                block = ResnetBlock(
                    dim=encoder_in_channels + in_channels,
                    dim_out=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                )

                in_channels = out_channels
                self.decoder_blocks.append(block)

            # Upsample if not last block
            if level != 0:
                self.decoder_blocks.append(UpSample(in_channels=in_channels))

        # Final residual block
        self.final_block = Block(in_channels, output_channels, dropout_rate=0.0)

    def forward(self, x, t):
        # Put t in 1D vector with int64 elements
        t = t.flatten().to(torch.long)

        # Time embedding
        time_emb = self.time_embeddings(t)

        h = self.init_conv(x)
        outs = [h]
        
        # Dimension reduction
        for layer in self.encoder_blocks:
            h = layer(h, time_emb)
            outs.append(h)
        
        # Botleneck
        for layer in self.bottleneck_blocks:
            h = layer(h, time_emb)
            
        # Decoding part: restoration + skip connections
        for layer in self.decoder_blocks:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)
            h = layer(h, time_emb)

        h = self.final_block(h)

        return h
   
