from libs import *


## Class Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular'
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_in',
                    nonlinearity='leaky_relu'
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x_mark):
        return self.emb(x_mark).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='t'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        weekend_size = 2

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)

        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
        self.weekend_embed = Embed(weekend_size, d_model)

    def forward(self, x):
        x = x.long()

        # weekend_x = self.weekend_embed(x[:, :, 5])
        # minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        # return hour_x + weekday_x + day_x + month_x + minute_x + weekend_x
        # return hour_x + weekday_x + day_x + month_x + minute_x
        return hour_x + weekday_x + day_x + month_x
    
class CyclicalTemporalEmbedding(nn.Module):
    
    def __init__(self, d_model, time_channels=12):
        super(CyclicalTemporalEmbedding, self).__init__()
        # Maps your 12 sin/cos float columns directly into the d_model space
        self.linear_projection = nn.Linear(time_channels, d_model)

    def forward(self, x):
        # x shape: [Batch, Sequence_Length, 12] 
        # Forces the input to remain a float tensor so decimals aren't erased
        return self.linear_projection(x.float())

# class CategoricalEmbedding(nn.Module):
#     def __init__(self, cat_cardinalities, d_model):
#         super().__init__()
#         # If it's a list [10, 5, 20], it works. If it's a dict, use .values()
#         cards = cat_cardinalities.values() if isinstance(cat_cardinalities, dict) else cat_cardinalities
        
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(int(c), d_model) for c in cards
#         ])

#     def forward(self, x_cat):
#         if x_cat is None: return 0
#         x_cat = x_cat.long() # Embeddings require LongTensor
#         out = 0
#         for i, emb in enumerate(self.embeddings):
#             out = out + emb(x_cat[:, :, i])
#         return out


class CategoricalEmbedding(nn.Module):
    def __init__(self, cat_cardinalities, d_model):
        super().__init__()
        # Ensure we have a list of cardinalities
        self.cards = list(cat_cardinalities.values()) if isinstance(cat_cardinalities, dict) else list(cat_cardinalities)
        
        # We add +1 to each cardinality as a 'buffer' for unknown/out-of-range values.
        # This prevents the "Index out of range" crash.
        self.embeddings = nn.ModuleList([
            nn.Embedding(int(c) + 1, d_model) for c in self.cards
        ])

    def forward(self, x_cat):
        if x_cat is None or len(self.embeddings) == 0:
            return 0
        
        x_cat = x_cat.long() 
        batch_size, seq_len, num_cat_cols = x_cat.shape
        
        out = 0
        # Iterate through the embedding layers we actually have defined
        for i, emb in enumerate(self.embeddings):
            # 1. Extract the specific categorical column
            column_data = x_cat[:, :, i]
            
            # 2. SAFETY CLIP: 
            # If an attacker sends a value like 9999, or Dataset 1 has a value 
            # higher than our cardinality, we 'clamp' it to the last index (the buffer).
            # The .clamp(max=...) ensures indices are always within [0, cardinality]
            safe_column_data = torch.clamp(column_data, min=0, max=self.cards[i])
            
            # 3. Add to the cumulative embedding representation
            out = out + emb(safe_column_data)
            
        return out
        

# class DataEmbedding(nn.Module):
#     def __init__(self, c_in, cat_cardinalities, d_model, embed_type='fixed', freq='t', dropout=0.1):
#         super(DataEmbedding, self).__init__()
        
#         # 1. Numerical: Only init if we have numerical features
#         self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model) if c_in > 0 else None
        
#         # 2. Categorical: Only init if cardinalities list is not empty
#         self.cat_embedding = None
#         if cat_cardinalities and len(cat_cardinalities) > 0:
#             self.cat_embedding = CategoricalEmbedding(cat_cardinalities, d_model)
            
#         self.position_embedding = PositionalEmbedding(d_model=d_model)
#         self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, x_num=None, x_cat=None, x_mark=None):
#         x = 0
        
#         if self.value_embedding is not None and x_num is not None:
#             x = x + self.value_embedding(x_num)
            
#         if self.cat_embedding is not None and x_cat is not None:
#             x = x + self.cat_embedding(x_cat)
            
#         if x_mark is not None:
#             x = x + self.temporal_embedding(x_mark)
        
#         # Positional embedding needs a reference tensor for sequence length
#         ref = x_num if x_num is not None else x_cat
#         if ref is not None:
#             x = x + self.position_embedding(ref)
            
#         return self.dropout(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, cat_cardinalities, d_model, time_dim_channels=12, dropout=0.1):
        """
        Updated DataEmbedding Layer for Continuous Cyclical Temporal Inputs.
        
        Args:
            c_in (int): Number of continuous numerical features.
            cat_cardinalities (list): Cardinalities (unique counts) of categorical features.
            d_model (int): Hidden dimension size of the Informer model.
            time_dim_channels (int): Number of cyclical time float dimensions (default: 12).
            dropout (float): Dropout probability rate.
        """
        super(DataEmbedding, self).__init__()
        
        # 1. Numerical Features Embedding
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model) if c_in > 0 else None
        
        # 2. Categorical Features Embedding
        self.cat_embedding = None
        if cat_cardinalities and len(cat_cardinalities) > 0:
            self.cat_embedding = CategoricalEmbedding(cat_cardinalities, d_model)
            
        # 3. Positional Sequence Tracking Embedding
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        # 4. UPDATED: Swapped out old integer lookup for the continuous Cyclical Temporal projector
        self.temporal_embedding = CyclicalTemporalEmbedding(d_model=d_model, time_channels=time_dim_channels)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_num=None, x_cat=None, x_mark=None):
        """
        Fuses continuous features, categorical flags, cyclical coordinates, and 
        sequence orders into a combined multidimensional representation matrix.
        """
        x = 0
        
        # Add continuous numerical token projections
        if self.value_embedding is not None and x_num is not None:
            x = x + self.value_embedding(x_num)
            
        # Add categorical category embeddings
        if self.cat_embedding is not None and x_cat is not None:
            x = x + self.cat_embedding(x_cat)
            
        # Add continuous cyclical temporal floats projection
        if x_mark is not None:
            # x_mark represents your 12 sin/cos float tensor channels
            x = x + self.temporal_embedding(x_mark)
        
        # Add fixed positional sequences to preserve temporal order
        ref = x_num if x_num is not None else x_cat
        if ref is not None:
            x = x + self.position_embedding(ref)
            
        return self.dropout(x)