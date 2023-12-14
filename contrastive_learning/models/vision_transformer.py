import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer.vit_components import PositionalEncodingEmbeddings, AttentionBlock
from .utils import ProjectionHead

class VisionTransformer(nn.Module):
    def __init__(self,
                 embedding_dim,
                 batch_size,
                 image_size,
                 attention_dim,
                 nb_heads,
                 nb_layers,
                 patch_size,
                 dropout=0.0):
        super(VisionTransformer, self).__init__()
        self.pos_embed = PositionalEncodingEmbeddings(batch_size, embedding_dim, patch_size, image_size)

        self.attention_layers = nn.ModuleList([
            AttentionBlock(embedding_dim, attention_dim, nb_heads, dropout) for _ in range(nb_layers)
        ])
        
        self.proj_head = ProjectionHead(embedding_dim)

    def one_forward(self, input):
        embedded_input = self.pos_embed(input)
        for attention_layer in self.attention_layers:
            embedded_input = attention_layer(embedded_input)
        representation = embedded_input[:, 0]
        if not self.training:
            return representation
        output = self.proj_head(representation)
        return output

    def forward(self, input1, input2):
        output1 = self.one_forward(input1)
        output2 = self.one_forward(input2)
        return output1, output2