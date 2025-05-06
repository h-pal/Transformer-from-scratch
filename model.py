import torch
from torch import nn
import math


class SelfAtten(nn.Module):
    def __init__(self, input_embd_dim, d = 512, mask = False):
        super().__init__()
        self.WQ = nn.Linear(input_embd_dim, d)
        self.WK = nn.Linear(input_embd_dim, d)
        self.WV = nn.Linear(input_embd_dim, d)
        self.soft = nn.Softmax(dim = -1)
        self.mask = mask

    def forward(self, x):
        q = self.WQ(x)
        k = self.WK(x)
        v = self.WV(x)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        if self.mask:
            mask = torch.zeros(attn_scores.shape, device = attn_scores.device)
            upper_triangular_mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-2)), diagonal=1).bool()
            mask[:, upper_triangular_mask] = float('-inf')
            attn_scores = attn_scores + mask
        scale = torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32, device=attn_scores.device))
        attn_scores = attn_scores / scale

        attn_weights = self.soft(attn_scores)
        x = torch.matmul(attn_weights, v)
        return x

class CrossAtten(nn.Module):
    def __init__(self, input_embd_dim, d = 512):
        super().__init__()
        self.WQ = nn.Linear(input_embd_dim, d)
        self.WK = nn.Linear(input_embd_dim, d)
        self.WV = nn.Linear(input_embd_dim, d)
        self.soft = nn.Softmax(dim = -1)
        self.d = d
    
    def forward(self, x, cross_input):
        q = self.WQ(x)
        k = self.WK(cross_input)
        v = self.WV(cross_input)

        attn_mat = torch.matmul(q, k.transpose(-2,-1))
        # attn_mat = attn_mat / torch.sqrt(torch.tensor(self.d))
        scale = torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32, device=attn_mat.device))
        attn_mat = attn_mat / scale

        attn_mat = self.soft(attn_mat)
        attn_scores = attn_mat @ v
        return attn_scores


class MultiHeadAtten(nn.Module):
    def __init__(self, input_embd_dim, h, d, cross_attention = False, mask = False):
        super().__init__()
        self.h = h
        self.d = d
        self.cross = cross_attention
        self.Wo = nn.Linear(d * h, input_embd_dim)
        if self.cross:
            self.heads = nn.ModuleList([CrossAtten(input_embd_dim, d) for _ in range(h)])
        else:
            self.heads = nn.ModuleList([SelfAtten(input_embd_dim, d, mask) for _ in range(h)])
        
    def forward(self, x, cross_input = None):
        all_vectors = []
        if cross_input is not None:
            for head in self.heads:
                all_vectors.append(head(x, cross_input))
        else:
            for head in self.heads:
                all_vectors.append(head(x))
        x = torch.cat(all_vectors, dim=-1)
        x = self.Wo(x)
        return x

class Encoder_block(nn.Module):
    def __init__(self, input_embd_dim, ffn_embd, h, d):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(input_embd_dim)
        self.layernorm2 = nn.LayerNorm(input_embd_dim)
        self.multih = MultiHeadAtten(input_embd_dim, h, d)
        self.ffn = nn.Sequential(
            nn.Linear(input_embd_dim, ffn_embd),
            nn.ReLU(),
            nn.Linear(ffn_embd, input_embd_dim)
        )

    def forward(self, x):
        attn_x = self.multih(x)
        x = torch.add(x, attn_x)
        x = self.layernorm1(x)
        pointwise_x = self.ffn(x)
        x = torch.add(x, pointwise_x)
        x = self.layernorm2(x)
        return x
        
class DecoderBlock(nn.Module):
    def __init__(self, input_embd_dim, ffn_embd, h, d, mask = True):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(input_embd_dim)
        self.layernorm2 = nn.LayerNorm(input_embd_dim)
        self.layernorm3 = nn.LayerNorm(input_embd_dim)
        self.masked_mhatn = MultiHeadAtten(input_embd_dim, h, d, mask = mask)
        self.cross_atten = MultiHeadAtten(input_embd_dim, h,  d, cross_attention = True, mask = False)
        self.ffn = nn.Sequential(
            nn.Linear(input_embd_dim, ffn_embd),
            nn.ReLU(),
            nn.Linear(ffn_embd, input_embd_dim)
        )

    def forward(self, x, cross_input):
        attn_x = self.masked_mhatn(x)
        x = x + attn_x
        x = self.layernorm1(x)
        cross_x = self.cross_atten(x, cross_input)
        x = x + cross_x
        x = self.layernorm2(x)
        pointwise_x = self.ffn(x)
        x = x + pointwise_x
        x = self.layernorm3(x)
        return x
        
class Encoder(nn.Module):
    def __init__(self, num_encoder_block, input_embd_dim, ffn_embd, h, d):
        super().__init__()
        self.num_encoders = num_encoder_block
        self.encoders = nn.ModuleList([ Encoder_block(input_embd_dim, ffn_embd, h, d) for _ in range(num_encoder_block)])
    
    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_decoder_block, input_embd_dim, ffn_embd, h, d, mask = True):
        super().__init__()
        self.num_decoders = num_decoder_block
        self.decoders = nn.ModuleList([DecoderBlock(input_embd_dim, ffn_embd, h, d) for _ in range(num_decoder_block)])
    
    def forward(self, x, encoder_out):
        for decoder in self.decoders:
            x = decoder(x, encoder_out)
        return x

class Transformers(nn.Module):
    def __init__(self, num_encoders, num_decoders, vocab_size, input_embd_dim, ffn_embd, h, d):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, input_embd_dim)
        self.position_embd = SinusoidalPositionalEmbedding(input_embd_dim)
        self.encoder = Encoder(num_encoders, input_embd_dim, ffn_embd, h, d)
        self.decoder = Decoder(num_decoders, input_embd_dim, ffn_embd, h, d)
        self.final_layer = nn.Linear(input_embd_dim, vocab_size)
    
    def forward(self, x, y):
        x = self.embed(x)
        y = self.embed(y)
        position_embd_x = self.position_embd(x)
        position_embd_y = self.position_embd(y)
        x = x + position_embd_x
        y = y + position_embd_y
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(y, encoder_out)
        out = self.final_layer(decoder_out)
        return out


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return self.pe[:, :x.size(1)]  # [1, seq_len, d_model]

# batch_size = 64
# seq_len = 4
# input_dim = 256
# ffn_dim = 2048
# num_encoder_blocks = 6
# num_decoder_block = 6

# x = torch.randn((batch_size, seq_len, input_dim))

# y = torch.randn((batch_size, 6, input_dim))
# model = Transformers(num_encoder_blocks, num_decoder_block, input_dim, ffn_dim, h=84, d = 786)
# preds = model(x, y)

# print(preds)
# print(preds.shape)
