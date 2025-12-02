import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, adj_mask=None):
        # x: [batch, num_nodes, d_model]
        batch_size, num_nodes, d_model = x.size()
        
        Q = self.W_q(x).view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)  
        V = self.W_v(x).view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if adj_mask is not None:
            scores = scores.masked_fill(adj_mask == 0, -1e9)
        
        spatial_attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(spatial_attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, d_model)
        return self.W_o(context)


class TemporalAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        temporal_attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(temporal_attention, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(context)


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.spatial_attention = SpatialAttention(d_model, num_heads)
        self.temporal_attention = TemporalAttention(d_model, num_heads)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)  
        )
    
    def forward(self, x, adj_mask=None):

        batch_size, num_nodes, seq_len, d_model = x.size()

        x_spatial = x.view(-1, num_nodes, d_model)  # [batch*seq_len, num_nodes, d_model]
        x_spatial = self.norm1(x_spatial + self.spatial_attention(x_spatial, adj_mask))
        x_spatial = x_spatial.view(batch_size, num_nodes, seq_len, d_model)

        x_temporal = x_spatial.transpose(1, 2).contiguous().view(-1, seq_len, d_model)  # [batch*num_nodes, seq_len, d_model]
        x_temporal = self.norm2(x_temporal + self.temporal_attention(x_temporal))
        x_temporal = x_temporal.view(batch_size, seq_len, num_nodes, d_model).transpose(1, 2)

        x_out = x_temporal.contiguous().view(-1, d_model)
        x_out = self.norm3(x_out + self.ffn(x_out))
        x_out = x_out.view(batch_size, num_nodes, seq_len, d_model)
        
        return x_out


class v_GraphTransformer(nn.Module):

    def __init__(self, model_args, task_args):
        super(v_GraphTransformer, self).__init__()
        self.model_args = model_args
        self.task_args = task_args

        self.message_dim = model_args['message_dim']
        self.his_num = task_args['his_num'] 
        self.pred_num = task_args['pred_num']  

        self.d_model = model_args.get('d_model', 16)    
        self.num_heads = model_args.get('num_heads', 1)   
        self.num_layers = model_args.get('num_layers', 1) 
        self.dropout = model_args.get('dropout', 0.2) 

        self.build()

    def build(self):

        self.input_projection = nn.Linear(self.message_dim, self.d_model, bias=False)

        self.pos_encoding = nn.Parameter(torch.randn(self.his_num, self.d_model))

        self.layers = nn.ModuleList([
            GraphTransformerBlock(self.d_model, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model * self.his_num, self.d_model, bias=False),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.d_model, self.pred_num, bias=False)
        )

        self.init_weights()

    def init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, data, A_hat):

        X = data.x  # [batch, num_nodes, seq_len, feature_dim]
        batch_size, num_nodes, seq_len, feature_dim = X.shape

        X = self.input_projection(X)  # [batch, num_nodes, seq_len, d_model]
        X = X + self.pos_encoding.unsqueeze(0).unsqueeze(0) 

        if A_hat.dim() == 3:  # [batch, num_nodes, num_nodes]
            adj_mask = (A_hat[0] > 0).float().unsqueeze(0).unsqueeze(0)  # [1, 1, num_nodes, num_nodes]
        else:  # [num_nodes, num_nodes]
            adj_mask = (A_hat > 0).float().unsqueeze(0).unsqueeze(0)    # [1, 1, num_nodes, num_nodes]

        for layer in self.layers:
            X = layer(X, adj_mask)

        X = X.view(batch_size, num_nodes, -1)  # [batch, num_nodes, seq_len*d_model]
        out = self.prediction_head(X)          # [batch, num_nodes, pred_len]

        return out, None


def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)