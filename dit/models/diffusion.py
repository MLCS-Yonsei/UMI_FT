# Copyright (c) Sudeep Dasari, 2023
# Heavy inspiration taken from DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# and DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos

class _PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        positional_encoding = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float)
            * -(np.log(10000.0) / model_dim)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("positional_encoding", positional_encoding)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, model_dim)

        Returns:
            Tensor of shape (seq_len, batch_size, model_dim) with positional encodings added
        """
        pe = self.positional_encoding[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()
    
class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad = learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 1
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return self.out_net(x)
    
class _SelfAttentionEncoder(nn.Module):
    def __init__(self, model_dim, nhead=8, feedforward_dim=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(model_dim, nhead, dropout = dropout)
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        q = k = _with_pos_embed(src, pos)
        src2, _ = self.self_attention(q, k, value=src, need_weights=False)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))

        src = src + self.dropout3(src2)
        src = self.norm2(src)

        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.activation = nn.SiLU() # Sigmoid Linear Unit
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)
    
    def forward(self, x, conditioning_vector):
        c = self.activation(conditioning_vector)
        return x * self.scale(c)[None] + self.shift(c)[None]
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)

class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.activation = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.activation(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)

class _DiffusionTransformerDecoder(nn.Module):
    def __init__(self, model_dim, nhead, feedforward_dim = 2048, dropout = 0.1, activation = "gelu"):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(model_dim, nhead, dropout = dropout)
        self.linear1 = nn.Linear(model_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, model_dim)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.attention_modulation_layer1 = _ShiftScaleMod(model_dim)
        self.attention_modulation_layer2 = _ZeroScaleMod(model_dim)

        self.mlp_modulation_layer1 = _ShiftScaleMod(model_dim)
        self.mlp_modulation_layer2 = _ZeroScaleMod(model_dim)
    
    def forward(self, x, t, conditioning_vector):
        '''
            stage 1: self attention
            stage 2: MLP
        '''
        conditioning_vector = torch.mean(conditioning_vector, axis = 0)
        conditioning_vector = conditioning_vector + t
        
        # stage 1
        # layer norm -> shift&scale with conditioning vector
        x2 = self.attention_modulation_layer1(self.norm1(x), conditioning_vector) 
        
        # multi-head self attention
        x2, _ = self.self_attention(x2, x2, x2, need_weights=False)

        # scale -> residual
        x = self.attention_modulation_layer2(self.dropout1(x2), conditioning_vector) + x

        # stage 2
        # MLP scale & shift
        x2 = self.mlp_modulation_layer1(self.norm2(x), conditioning_vector)

        # MLP
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))

        # MLP scale
        x2 = self.mlp_modulation_layer2(self.dropout3(x2), conditioning_vector)

        return x + x2
    
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attention_modulation_layer1, self.attention_modulation_layer2, self.mlp_modulation_layer1, self.mlp_modulation_layer2):
            s.reset_parameters()

class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaptive_LayerNorm_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    
    def forward(self, x, t, conditioning_vector):
        conditioning_vector = torch.mean(conditioning_vector, axis = 0)
        conditioning_vector = conditioning_vector + t

        shift, scale = self.adaptive_LayerNorm_modulation(conditioning_vector).chunk(2, dim=1)

        x = x * scale[None] + shift[None]
        x = self.linear(x)

        return x.transpose(0, 1)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)

class _TransformerEncoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )
        
        for l in self.layers:
            l.reset_parameters()
    
    def forward(self, src, pos):
        x, outputs = src, []
        for layer in self.layers:
            x = layer(x, pos)
            outputs.append(x)
        return outputs

class _TransformerDecoder(_TransformerEncoder):
    def forward(self, src, t, all_conds):
        x = src
        for layer, cond in zip(self.layers, all_conds):
            x = layer(x, t, cond)
        return x

class _DiffusionTransformerNoiseNetwork(nn.Module):
    def __init__(self,
                 action_dim,
                 action_chunk,
                 time_dim = 256,
                 hidden_dim = 512,
                 num_blocks = 6,
                 dropout = 0.1,
                 dim_feedforward = 2048,
                 nhead = 8,
                 activation = 'gelu' 
                 ):
        
        super().__init__()

        # positional encoding blocks
        self.encoder_pos = _PositionalEncoding(hidden_dim)
        self.register_parameter(
            "decoder_pos",
            nn.Parameter(torch.empty(action_chunk, 1, hidden_dim), requires_grad = True),
        )
        nn.init.xavier_uniform_(self.decoder_pos.data)

        # input encoder mlps
        self.time_network = _TimeNetwork(time_dim, hidden_dim)
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, hidden_dim),
        )

        # encoder blocks
        encoder_module = _SelfAttentionEncoder(
            hidden_dim,
            nhead=nhead,
            feedforward_dim=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = _TransformerEncoder(encoder_module, num_blocks)
        
        # decoder blocks
        decoder_module = _DiffusionTransformerDecoder(
            hidden_dim,
            nhead=nhead,
            feedforward_dim=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        self.eps_out = _FinalLayer(hidden_dim, action_dim)

        print(
            "number of diffusion parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(self, noise_actions, time, encoder_obs, encoder_cache=None):
        if encoder_cache is None:
            encoder_cache = self.forward_encoder(encoder_obs)
        return encoder_cache, self.forward_decoder(noise_actions, time, encoder_cache)

    def forward_encoder(self, encoder_obs):
        encoder_obs = encoder_obs.transpose(0, 1)
        pos = self.encoder_pos(encoder_obs)
        encoder_cache = self.encoder(encoder_obs, pos)
        return encoder_cache
    
    def forward_decoder(self, noise_actions, time, encoder_cache):
        encoder_time = self.time_network(time)

        action_tokens = self.action_projection(noise_actions)
        action_tokens = action_tokens.transpose(0, 1)
        decoder_input = action_tokens + self.decoder_pos

        decoder_output = self.decoder(decoder_input, encoder_time, encoder_cache)

        return self.eps_out(decoder_output, encoder_time, encoder_cache[-1])
    



