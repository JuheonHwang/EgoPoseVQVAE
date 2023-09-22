"""
From T2M-GPT by Mael-zys
models/vqvae.py
"""

import torch.nn as nn
from model.embedder import get_embedder
from model.encdec import Encoder, Decoder
from model.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset


class VQVAE_251(nn.Module):
    def __init__(self,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 fourier_features='none'):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = 'ema_reset'
        
        if self.quant == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim)
        elif self.quant == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif self.quant == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim)
        elif self.quant == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim)
        
        if fourier_features == 'positional':
            self.fourier_feature_transform, channels = get_embedder(10) # fft_scale is input
            self.encoder = Encoder(channels, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        elif fourier_features == 'none':
            self.fourier_feature_transform = None
            self.encoder = Encoder(3, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        
        
        self.decoder = Decoder(3, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)


    def preprocess(self, x):
        # (bs, J, 3) -> (bs, 3, J)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, 3, J) -> (bs, J, 3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        N, T, _ = x.shape
        if self.fourier_feature_transform is not None:
            x = self.fourier_feature_transform(x)
        x = self.postprocess(x)
        x_encoder = self.encoder(x)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx


    def forward(self, x):
        if self.fourier_feature_transform is not None:
            x = self.fourier_feature_transform(x)
        x = self.postprocess(x)

        # Encode
        x_encoder = self.encoder(x)
        
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out



class PoseVQVAE(nn.Module):
    def __init__(self,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 fourier_features='none'):
        
        super().__init__()
        
        # Human36M: 'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist' -> except 'Pelvis', 'Torso', 'Nose', 'Nose'
        self.nb_joints = 14
        self.vqvae = VQVAE_251(nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, fourier_features=fourier_features)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x):

        x_out, loss, perplexity = self.vqvae(x)
        
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
