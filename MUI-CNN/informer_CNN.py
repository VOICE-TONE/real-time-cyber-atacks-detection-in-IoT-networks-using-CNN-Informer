from libs import *
from data_embedding import *
from encoder import *
from decoder import *
from attn import *


# =====================================================
# CNN Feature Extraction
# =====================================================

class CNN_FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CNN_FeatureExtractor, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2
        )
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # Informer input:  [Batch, Seq_Len, Features]
        # Conv1d expects: [Batch, Features, Seq_Len]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        # Back to: [Batch, Seq_Len, Features]
        return x.permute(0, 2, 1)

# =====================================================
# Informer Layer
# =====================================================
# =====================================================
# Informer Layer (Standard)
# =====================================================
import torch
import torch.nn as nn
# Assuming other custom imports (Encoder, Decoder, etc.) are available in your environment
# from libs import * ...
class Informer(nn.Module):
    def __init__(self, 
                 enc_in_num, enc_in_cat,
                 dec_in_num, dec_in_cat,
                 c_out, seq_len, label_len, out_len, 
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                 dropout=0.0, attn='prob', embed='fixed', freq='s', activation='gelu', 
                 output_attention = False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # 1. CNN Front-End for Numerical Features
        self.enc_cnn = CNN_FeatureExtractor(enc_in_num, enc_in_num)
        self.dec_cnn = CNN_FeatureExtractor(dec_in_num, dec_in_num)

        # 2. Encoding Layers
        self.enc_embedding = DataEmbedding(enc_in_num, enc_in_cat, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in_num, dec_in_cat, d_model, embed, freq, dropout)
        
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        # 3. Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for l in range(e_layers)
            ],
            [ConvLayer(d_model) for l in range(e_layers-1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # 4. Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation,
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_num_enc, x_cat_enc, x_mark_enc, x_num_dec, x_cat_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # Apply CNN to Numerical inputs before Embedding
        x_num_enc = self.enc_cnn(x_num_enc)
        x_num_dec = self.dec_cnn(x_num_dec)

        enc_out = self.enc_embedding(x_num=x_num_enc, x_cat=x_cat_enc, x_mark=x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_num=x_num_dec, x_cat=x_cat_dec, x_mark=x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:]


# =====================================================
# InformerStack Layer
# =====================================================
class InformerStack(nn.Module):
    def __init__(self, 
                 enc_in_num, enc_in_cat, 
                 dec_in_num, dec_in_cat, 
                 c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding: Pass both counts
        self.enc_embedding = DataEmbedding(enc_in_num, enc_in_cat, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in_num, dec_in_cat, d_model, embed, freq, dropout)
        
        Attn = ProbAttention if attn=='prob' else FullAttention

        inp_lens = list(range(len(e_layers))) 
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model, d_ff, dropout=dropout, activation=activation
                    ) for l in range(el)
                ],
                [ConvLayer(d_model) for l in range(el-1)] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        
        self.encoder = EncoderStack(encoders, inp_lens)
        
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation,
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_num_enc, x_cat_enc, x_mark_enc, x_num_dec, x_cat_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        enc_out = self.enc_embedding(x_num=x_num_enc, x_cat=x_cat_enc, x_mark=x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_num=x_num_dec, x_cat=x_cat_dec, x_mark=x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:]