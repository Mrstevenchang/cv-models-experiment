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
            Block(embed_dim, num_heads=num_heads)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.initial_weights()

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape       # batch, length, dim
        print(x.shape)
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)   # 均匀分布  [0, 1]
        print(noise)
        ids_shuffle = torch.argsort(noise, dim=1)       # torch.argsort 按照升序之前的顺序的索引排序 [3, 1, 4, 2, 5] -> [1, 3, 0, 2, 4]
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        print(ids_keep)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def initial_weights(self):
        # initialize position embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w)

        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        print(x.shape)
        x = self.patch_embed(x)     # batch, channel, height, width
        print(x.shape)
        x = x + self.pos_embed[:, 1:, :]        # x.shape -> (batch, num_patch, embed_dim)  pos_embed[:,1:,:] -> (1, num_patch, embed_dim)
        print(self.pos_embed[:, 1:, :].shape)
        print(x.shape)
        x, mask, ids_restore = self.random_masking(x, mask_ratio=0.75)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore
