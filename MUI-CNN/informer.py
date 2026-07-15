from libs import *
from data_embedding import *
from encoder import *
from decoder import *
from attn import *

# =====================================================
# Informer Layer
# =====================================================
# =====================================================
# Informer Layer (Standard)
# =====================================================
class Informer(nn.Module):
    def __init__(self, 
                 enc_in_num, 
                 enc_in_cat,
                 dec_in_num, 
                 dec_in_cat,
                 c_out, 
                 seq_len, 
                 label_len, 
                 out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='s', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding: Pass both numerical and categorical feature counts
        self.enc_embedding = DataEmbedding(enc_in_num, enc_in_cat, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in_num, dec_in_cat, d_model, embed, freq, dropout)
        
        # Attention setup
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        # Encoder
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
        
        # Decoder
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


# =====================================================
# CNN Feature Extraction (Kept from your code)
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
        x = x.permute(0, 2, 1) # [B, L, D] -> [B, D, L]
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        return x.permute(0, 2, 1) # [B, D, L] -> [B, L, D]

# =====================================================
# Informer Classifier (Modified)
# =====================================================
class InformerClassifier(nn.Module):
    def __init__(self, 
                 enc_in_num, enc_in_cat,
                 num_classes, # Replaced c_out with num_classes
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512, 
                 dropout=0.1, attn='prob', embed='fixed', freq='t', activation='gelu', 
                 distil=True, device=torch.device('cuda:0')):
        super().__init__()
        
        # 1. CNN Front-End
        self.enc_cnn = CNN_FeatureExtractor(enc_in_num, enc_in_num)

        # 2. Encoding Layers
        self.enc_embedding = DataEmbedding(enc_in_num, enc_in_cat, d_model, embed, freq, dropout)
        
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        # 3. Encoder (The Feature Extractor)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for l in range(e_layers)
            ],
            # Distillation is KEY for classification as it compresses the sequence
            [ConvLayer(d_model) for l in range(e_layers-1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # 4. Classification Head (The Prediction Layer)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x_num_enc, x_cat_enc, x_mark_enc, enc_self_mask=None):
        # A. Apply CNN Feature Extraction
        x_num_enc = self.enc_cnn(x_num_enc)

        # B. Embed and Encode
        enc_out = self.enc_embedding(x_num=x_num_enc, x_cat=x_cat_enc, x_mark=x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)

        # C. Global Average Pooling 
        # Instead of picking a specific time step, we average the "energy" 
        # across the sequence to get a global feature vector.
        # enc_out shape: [Batch, Sequence_Length, d_model]
        # out = torch.mean(enc_out, dim=1) # [Batch, d_model]
        out,_ = torch.max(enc_out, dim=1) # [Batch, d_model]

        # D. Project to Classes
        out = self.projection(out) # [Batch, num_classes]
        
        return out
    


class UniversalInformer(nn.Module):
    def __init__(self, config, num_classes, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.config = config
        
        # 1. Numerical Projection
        self.num_proj = nn.Linear(len(config['num_idx']), d_model) if config['num_idx'] else None
        
        # 2. Dynamic Categorical Embeddings
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(dim, d_model) for dim in config['cat_dims']
        ])
        
        # 3. Dynamic Temporal Embeddings
        self.time_embeds = nn.ModuleDict({
            k: nn.Embedding(v, d_model) for k, v in config['time_dims'].items()
        })

        # Positional Encoding
        self.pos_emb = nn.Parameter(torch.zeros(1, 64, d_model))
        
        # Transformer Core
        # We now use the arguments passed during initialization
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: [Batch, 64, Total_Features]
        batch_size, seq_len, _ = x.shape
        
        # Initialize base embedding tensor on the correct device
        embeddings = torch.zeros(batch_size, seq_len, 128, device=x.device)
        
        # Add Numerical Features
        if self.num_proj:
            embeddings = embeddings + self.num_proj(x[:, :, self.config['num_idx']])
            
        # Add Categorical Features
        for i, idx in enumerate(self.config['cat_idx']):
            embeddings = embeddings + self.cat_embeds[i](x[:, :, idx].long())
            
        # Add Temporal Features
        for i, (name, layer) in enumerate(self.time_embeds.items()):
            idx = self.config['time_idx'][i]
            embeddings = embeddings + layer(x[:, :, idx].long())
            
        # Add Position and apply Transformer
        x = self.dropout(embeddings + self.pos_emb)
        x = self.transformer(x)
        
        # Global Average Pooling (take the mean across the sequence length)
        return self.fc(x.mean(dim=1))



class UniversalInformerLinear(nn.Module):
    def __init__(self, config, num_classes, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.config = config
        self.d_model = d_model
        
        # 1. Numerical Projection
        self.num_proj = nn.Linear(len(config['num_idx']), d_model) if config['num_idx'] else None
        
        # 2. Dynamic Categorical Embeddings
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(dim, d_model) for dim in config['cat_dims']
        ])
        
        # 3. UPDATED: Swapped out legacy integer lookups for a continuous temporal projector
        self.time_proj = nn.Linear(config['time_dim_channels'], d_model) if config['time_dim_channels'] > 0 else None

        # Positional Encoding 
        # (Updated parameter depth to match your true sequence timeline footprint safely)
        self.pos_emb = nn.Parameter(torch.zeros(1, 5958, d_model))
        
        # Transformer Core
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Initialize base embedding tensor dynamically matching d_model configuration
        embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        
        # Add Numerical Features
        if self.num_proj is not None:
            embeddings = embeddings + self.num_proj(x[:, :, self.config['num_idx']])
            
        # Add Categorical Features
        for i, idx in enumerate(self.config['cat_idx']):
            embeddings = embeddings + self.cat_embeds[i](x[:, :, idx].long())
            
        # UPDATED: Add Continuous Cyclical Temporal Features
        if self.time_proj is not None and self.config['time_idx']:
            # Pull continuous chunks across your 12 sin/cos float dimensions
            time_features = x[:, :, self.config['time_idx']].float()
            embeddings = embeddings + self.time_proj(time_features)
            
        # Dynamic slice on position matrices to protect tensor layout boundaries
        pos_slice = self.pos_emb[:, :seq_len, :]
        
        # Add Position and apply Transformer
        x = self.dropout(embeddings + pos_slice)
        x = self.transformer(x)
        
        # Global Average Pooling (take the mean across the sequence length)
        return self.fc(x.mean(dim=1))

class UniversalInformerMask(nn.Module):
    def __init__(self, config, num_classes, d_model=128, nhead=8, num_layers=3, 
                 dropout=0.1, masking_method='mad', threshold=3.0):
        super().__init__()
        self.config = config
        self.masking_method = masking_method
        self.threshold = threshold
        self.d_model = d_model
        
        # 1. Numerical Projection
        self.num_proj = nn.Linear(len(config['num_idx']), d_model) if config['num_idx'] else None
        
        # 2. Dynamic Categorical Embeddings
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(dim, d_model) for dim in config['cat_dims']
        ])
        
        # 3. Dynamic Temporal Embeddings
        self.time_embeds = nn.ModuleDict({
            k: nn.Embedding(v, d_model) for k, v in config['time_dims'].items()
        })

        # Positional Encoding (Initialized with small random values instead of zeros)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        
        # Transformer Core
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def apply_masking(self, x):
        """
        Detects and zero-masks outliers across the sequence dimension (dim 1).
        """
        if self.masking_method == 'zscore':
            mean = torch.mean(x, dim=1, keepdim=True)
            std = torch.std(x, dim=1, keepdim=True) + 1e-6
            scores = torch.abs((x - mean) / std)
            mask = scores > self.threshold
            
        elif self.masking_method == 'mad':
            median = torch.median(x, dim=1, keepdim=True).values
            mad = torch.median(torch.abs(x - median), dim=1, keepdim=True).values + 1e-6
            modified_z_score = 0.6745 * torch.abs(x - median) / mad
            mask = modified_z_score > self.threshold
        else:
            return x

        return x.masked_fill(mask, 0.0)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Initialize base embedding tensor
        embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        
        # Add Numerical Features
        if self.num_proj:
            embeddings = embeddings + self.num_proj(x[:, :, self.config['num_idx']])
            
        # Add Categorical Features
        for i, idx in enumerate(self.config['cat_idx']):
            embeddings = embeddings + self.cat_embeds[i](x[:, :, idx].long())
            
        # Add Temporal Features
        for i, (name, layer) in enumerate(self.time_embeds.items()):
            idx = self.config['time_idx'][i]
            embeddings = embeddings + layer(x[:, :, idx].long())
            
        # --- Masking Step ---
        if self.masking_method:
            embeddings = self.apply_masking(embeddings)
            
        # Add Position and apply Transformer
        x = self.dropout(embeddings + self.pos_emb)
        x = self.transformer(x)
        
        # Global Average Pooling
        return self.fc(x.mean(dim=1))
    

class UniversalInformerCNN(nn.Module):
    def __init__(self, config, num_classes, d_model=128, nhead=8, num_layers=3, 
                 dropout=0.1, masking_method=None, threshold=None, kernel_size=3):
        super().__init__()
        self.config = config
        self.masking_method = masking_method
        self.threshold = threshold
        self.d_model = d_model
        
        # 1. Projections (Numerical, Categorical, Temporal)
        self.num_proj = nn.Linear(len(config['num_idx']), d_model) if config['num_idx'] else None
        self.cat_embeds = nn.ModuleList([nn.Embedding(dim, d_model) for dim in config['cat_dims']])
        self.time_embeds = nn.ModuleDict({k: nn.Embedding(v, d_model) for k, v in config['time_dims'].items()})

        # 2. CNN Feature Extractor
        self.feature_cnn = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            padding='same'
        )
        self.cnn_norm = nn.LayerNorm(d_model)

        # 3. Positional Encoding
        self.pos_emb = nn.Parameter(torch.zeros(1, 64, d_model))
        
        # 4. Transformer Core
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def apply_masking(self, x):
        """
        Applies masking only if a method and threshold are provided.
        """
        if self.masking_method is None or self.threshold is None:
            return x

        if self.masking_method == 'zscore':
            mean = torch.mean(x, dim=1, keepdim=True)
            std = torch.std(x, dim=1, keepdim=True) + 1e-6
            mask = (torch.abs((x - mean) / std)) > self.threshold
            return x.masked_fill(mask, 0.0)
            
        elif self.masking_method == 'mad':
            median = torch.median(x, dim=1, keepdim=True).values
            mad = torch.median(torch.abs(x - median), dim=1, keepdim=True).values + 1e-6
            mask = (0.6745 * torch.abs(x - median) / mad) > self.threshold
            return x.masked_fill(mask, 0.0)
        elif self.masking_method == 'rdmd':
            # x shape: [batch, seq_len, d_model]
            # We treat the sequence length as the distribution population
            mu = torch.mean(x, dim=1, keepdim=True) # [batch, 1, d_model]
            diff = x - mu # [batch, seq_len, d_model]
            
            # Calculate Covariance Matrix (Robustly)
            # Using a small shrinkage (1e-6) to ensure the matrix is invertible
            cov = torch.matmul(diff.transpose(-2, -1), diff) / (x.size(1) - 1)
            inv_cov = torch.linalg.inv(cov + torch.eye(self.d_model, device=x.device) * 1e-6)
            
            # Mahalanobis Distance squared: (x-mu)^T * Inv_Cov * (x-mu)
            left_term = torch.matmul(diff, inv_cov) # [batch, seq_len, d_model]
            mahal_dist = torch.sum(left_term * diff, dim=-1) # [batch, seq_len]
            
            # Thresholding based on Chi-Squared distribution or a static hyperparameter
            mask = mahal_dist > self.threshold
            return x.masked_fill(mask.unsqueeze(-1), 0.0)
        return x

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        
        # Aggregate Projections
        if self.num_proj:
            embeddings += self.num_proj(x[:, :, self.config['num_idx']])
        for i, idx in enumerate(self.config['cat_idx']):
            embeddings += self.cat_embeds[i](x[:, :, idx].long())
        for i, (name, layer) in enumerate(self.time_embeds.items()):
            idx = self.config['time_idx'][i]
            embeddings += layer(x[:, :, idx].long())
            
        # Optional Masking Step
        embeddings = self.apply_masking(embeddings)

        # CNN Feature Extraction
        cnn_features = embeddings.permute(0, 2, 1) 
        cnn_features = F.relu(self.feature_cnn(cnn_features))
        embeddings = cnn_features.permute(0, 2, 1) 
        embeddings = self.cnn_norm(embeddings)
            
        # Attention & Classification
        x = self.dropout(embeddings + self.pos_emb)
        x = self.transformer(x)
        
        return self.fc(x.mean(dim=1))
    


    import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalInformerCNNLinear(nn.Module):
    def __init__(self, config, num_classes, d_model=128, nhead=8, num_layers=3, 
                 dropout=0.1, masking_method=None, threshold=None, kernel_size=3):
        super().__init__()
        self.config = config
        self.masking_method = masking_method
        self.threshold = threshold
        self.d_model = d_model
        
        # 1. Projections (Numerical, Categorical, Continuous Cyclical Temporal)
        self.num_proj = nn.Linear(len(config['num_idx']), d_model) if config['num_idx'] else None
        self.cat_embeds = nn.ModuleList([nn.Embedding(dim, d_model) for dim in config['cat_dims']])
        
        # UPDATED: Swapped out legacy nn.ModuleDict lookup maps for a fast continuous linear projector
        self.time_proj = nn.Linear(config['time_dim_channels'], d_model) if config['time_dim_channels'] > 0 else None

        # 2. CNN Feature Extractor
        self.feature_cnn = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            padding='same'
        )
        self.cnn_norm = nn.LayerNorm(d_model)

        # 3. Dynamic Positional Tracking Space 
        # (Using a dynamic runtime parameter size or fixed slice matching your sequence shape context)
        self.pos_emb = nn.Parameter(torch.zeros(1, 5958, d_model)) # Match actual data timeline shape
        
        # 4. Transformer Core
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def apply_masking(self, x):
        """
        Applies mathematical anomaly truncation / mask suppression options.
        """
        if self.masking_method is None or self.threshold is None:
            return x

        if self.masking_method == 'zscore':
            mean = torch.mean(x, dim=1, keepdim=True)
            std = torch.std(x, dim=1, keepdim=True) + 1e-6
            mask = (torch.abs((x - mean) / std)) > self.threshold
            return x.masked_fill(mask, 0.0)
            
        elif self.masking_method == 'mad':
            median = torch.median(x, dim=1, keepdim=True).values
            mad = torch.median(torch.abs(x - median), dim=1, keepdim=True).values + 1e-6
            mask = (0.6745 * torch.abs(x - median) / mad) > self.threshold
            return x.masked_fill(mask, 0.0)
            
        elif self.masking_method == 'rdmd':
            mu = torch.mean(x, dim=1, keepdim=True) 
            diff = x - mu 
            
            cov = torch.matmul(diff.transpose(-2, -1), diff) / (x.size(1) - 1)
            inv_cov = torch.linalg.inv(cov + torch.eye(self.d_model, device=x.device) * 1e-6)
            
            left_term = torch.matmul(diff, inv_cov) 
            mahal_dist = torch.sum(left_term * diff, dim=-1) 
            
            mask = mahal_dist > self.threshold
            return x.masked_fill(mask.unsqueeze(-1), 0.0)
            
        return x

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        
        # --- Aggregate Feature Engineering Blocks ---
        
        # A. Project continuous network metric coordinates
        if self.num_proj is not None:
            embeddings = embeddings + self.num_proj(x[:, :, self.config['num_idx']])
            
        # B. Accumulate categorical identity codes
        for i, idx in enumerate(self.config['cat_idx']):
            embeddings = embeddings + self.cat_embeds[i](x[:, :, idx].long())
            
        # C. UPDATED: Project all 12 cyclical sin/cos float decimals simultaneously
        if self.time_proj is not None and self.config['time_idx']:
            # Pull continuous chunks cleanly across sequence dimensions
            time_features = x[:, :, self.config['time_idx']].float()
            embeddings = embeddings + self.time_proj(time_features)
            
        # --- Context Sequence Distillation & Processing Blocks ---
            
        # Optional Outlier Filtering Mask Step
        embeddings = self.apply_masking(embeddings)

        # CNN Feature Extraction Local Filtering
        cnn_features = embeddings.permute(0, 2, 1) 
        cnn_features = F.relu(self.feature_cnn(cnn_features))
        embeddings = cnn_features.permute(0, 2, 1) 
        embeddings = self.cnn_norm(embeddings)
            
        # Global Multi-Head Attention & Linear Head Output Calculation
        # Ensure position matrix embeddings matches runtime shape context smoothly
        pos_slice = self.pos_emb[:, :seq_len, :]
        
        x = self.dropout(embeddings + pos_slice)
        x = self.transformer(x)
        
        return self.fc(x.mean(dim=1))