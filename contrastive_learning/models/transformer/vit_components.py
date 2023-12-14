import torch
import torch.nn as nn

from models.utils import image_to_patches

class PositionalEncodingEmbeddings(nn.Module):
    def __init__(self,
                 batch_size,
                 embedding_dim,
                 patch_size,
                 img_size):
        super(PositionalEncodingEmbeddings, self).__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size**2)//(patch_size**2)

        assert isinstance(self.n_patches, int)

        self.cls_token_embed = nn.Parameter(torch.rand((batch_size, 1, embedding_dim), requires_grad=True))
        self.position_embed = nn.Parameter(torch.rand((1, self.n_patches+1, embedding_dim), requires_grad=True))

    def forward(self, input): # input shape: (B, C, H, W) or (C, H, W)
        patches = image_to_patches(input, self.patch_size)

        B, N, C = patches.shape
        patches = patches + self.position_embed[:, :N]

        cls_tokens = self.cls_token_embed.expand(B, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)

        return patches

class AttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim,
                 attention_dim,
                 nb_heads,
                 dropout):
        super(AttentionBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, nb_heads, dropout)

        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        output_norm1 = self.layer_norm1(input)
        attn_output, _ = self.attention(output_norm1, output_norm1, output_norm1)
        residual_output1 = attn_output + input

        output_norm2 = self.layer_norm2(residual_output1)
        feed_forward_output = self.feedforward(output_norm2)
        residual_output2 = residual_output1 + feed_forward_output
        return residual_output2
