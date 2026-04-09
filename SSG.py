import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
import torch


class MLP_Module(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * (x - u) / torch.sqrt(s + self.eps) + self.bias[:, None, None]


class SAP_Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2., drop_path=0., alpha=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.alpha = alpha
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP_Module(dim, int(dim * mlp_ratio))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5

        seq_len = N // 3
        attn[:, :, seq_len:2 * seq_len] += self.alpha * attn[:, :, 0:seq_len]
        attn[:, :, 2 * seq_len:] += self.alpha * attn[:, :, seq_len:2 * seq_len]

        attn = attn.softmax(dim=-1)
        x = x + self.drop_path((attn @ v).transpose(1, 2).reshape(B, N, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TAP_Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=2., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP_Module(dim, int(dim * mlp_ratio))

    def forward(self, x):
        Ep, Em, Ec = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        q = self.q(Em)
        k, v = self.kv(torch.cat([Ep, Ec], dim=1)).chunk(2, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * (q.shape[-1]) ** -0.5
        attn = attn.softmax(dim=-1)
        x = Em + self.drop_path((attn @ v).transpose(1, 2).reshape(x.shape[0], 1, x.shape[-1]))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SSG_Encoder(nn.Module):
    def __init__(self, num_frame=10, embed_dim_ratio=32):
        super().__init__()
        self.T = num_frame
        self.embed_dim = embed_dim_ratio

        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.spatial_embed = nn.Linear(256, self.embed_dim)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 32 * 32, self.embed_dim))

        self.SAP = SAP_Block(self.embed_dim)
        self.TAP = TAP_Block(self.embed_dim)

        self.norm_s = nn.LayerNorm(self.embed_dim)
        self.norm_t = nn.LayerNorm(self.embed_dim)
        trunc_normal_(self.spatial_pos_embed, std=.02)

    def split_triple(self, x):
        triples = []
        for i in range(self.T - 2):
            triples.append(x[i:i + 3])
        return torch.stack(triples)

    def SAP_forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = rearrange(x, 't c h w -> t (h w) c')
        x = self.spatial_embed(x)
        x = x + self.spatial_pos_embed

        triples = self.split_triple(x)
        B, N_tri, N_token, C = triples.shape

        triples_flat = triples.reshape(B, N_tri * N_token, C)
        feat = self.SAP(triples_flat)
        feat = self.norm_s(feat)
        triple_feat = feat.reshape(B, 3, N_token, C)
        frame_feats = torch.zeros(self.T, N_token, C, device=x.device)
        count = torch.zeros(self.T, device=x.device)

        for i in range(8):
            frame_feats[i:i + 3] += triple_feat[i]
            count[i:i + 3] += 1
        frame_feats = frame_feats / count.unsqueeze(1).unsqueeze(2)

        return frame_feats

    def TAP_forward(self, x):
        frame_out = []
        for i in range(self.T - 2):
            triple = x[i:i + 3].unsqueeze(0)
            temp_feat = triple.mean(dim=2)
            t_feat = self.TAP(temp_feat)
            frame_out.append(t_feat.squeeze(1))

        temporal_feat = torch.zeros(self.T, self.embed_dim, device=x.device)
        temporal_feat[1:-1] = torch.cat(frame_out, dim=0)
        temporal_feat[0] = frame_out[0]
        temporal_feat[-1] = frame_out[-1]
        return temporal_feat.unsqueeze(1)

    def forward(self, x):
        spatial_feat = self.SAP_forward(x)
        temporal_feat = self.TAP_forward(spatial_feat)
        out = spatial_feat * temporal_feat
        out = out.reshape(self.T, 32, 32, self.embed_dim)
        return out


class SSG_Decoder(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            LayerNorm2d(256),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, 2), nn.GELU(),
            nn.ConvTranspose2d(64, 16, 2, 2), nn.GELU(),
            nn.ConvTranspose2d(16, 1, 2, 2),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        feat = x.clone()
        x = self.up(x)
        return x, feat


class SSG(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = SSG_Encoder()
        self.dec = SSG_Decoder()

    def forward(self, x):
        feat = self.enc(x)
        out, feat = self.dec(feat)
        return out
