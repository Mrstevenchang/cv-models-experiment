import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed, Block
from utils.pos_embed import get_2d_sincos_pos_embed


class MAE_Encoder(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_chans=3, embed_dim=192, depth=10, num_heads=3, norm_layer=nn.LayerNorm):
        super(MAE_Encoder, self).__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim, num_heads=num_heads)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def init_weight(self):
        pos_embed = get_2d_sincos_pos_embed(self.patch_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)

    def forward(self, x):

        return
