import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import warnings

#ignore  过滤告警信息
warnings.filterwarnings('ignore')

# 方法1：使用系统中已安装的中文字体
# Windows系统常见中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class PositionalEncoding(nn.Module):
    """位置编码模块，为序列中的每个位置提供唯一的位置信息"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 创建位置索引 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 创建除数项 [1, 1/100, 1/10000, ...]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引使用cos
        
        # 注册为缓冲区（不作为模型参数）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """添加位置编码到输入张量
        Args:
            x: 输入张量 [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :].unsqueeze(1)
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力机制，允许模型关注不同位置的不同特征"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        
        # 创建Q, K, V的线性层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性变换
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        
        # 分割为多头
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 应用softmax
        attention = torch.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attention, v)
        
        # 重新整形
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终线性层
        return self.out_linear(output)

class FeedForward(nn.Module):
    """前馈神经网络，用于在注意力层后处理每个位置的特征"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力 + Add & Norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力 + Add & Norm
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 编码器-解码器注意力 + Add & Norm
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络 + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    """完整的Transformer模型，包含编码器和解码器"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_seq_length=100, dropout=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.d_model = d_model
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器部分
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        enc_output = src
        
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        # 解码器部分
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        dec_output = tgt
        
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
        output = self.fc_out(dec_output)
        return output

def print_model_summary(model, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff):
    """打印模型结构概述"""
    print("Model: \"Transformer\"")
    print("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓")
    print("┃ Layer (type)                    ┃ Output Shape            ┃       Param # ┃")
    print("┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩")
    
    # 输入层
    print("│ Input                           │ (batch, seq_len)        │             0 │")
    
    # 嵌入层
    embedding_params = src_vocab_size * d_model
    print(f"├─────────────────────────────────┼─────────────────────────┼───────────────┤")
    print(f"│ Embedding (Source)              │ (batch, seq_len, {d_model}) │ {embedding_params:13,d} │")
    
    # 位置编码
    print(f"├─────────────────────────────────┼─────────────────────────┼───────────────┤")
    print(f"│ Positional Encoding             │ (batch, seq_len, {d_model}) │             0 │")
    
    # 编码器层
    '''
    这计算的是多头注意力(Multi-Head Attention)机制中的参数数量
    有4个线性变换层：Q(查询)、K(键)、V(值)和Output(输出)线性层
    每个线性层的参数量为 d_model × d_model (输入维度 × 输出维度)
    因此总共有 4 × d_model × d_model 个参数
    '''
    encoder_mha_params = 4 * d_model * d_model  # Q,K,V,Out 共 4个线性层
    '''
    这计算的是前馈神经网络(Feed Forward)层的参数数量
    前馈网络有两个线性层：
    第一层：从 d_model 维度映射到 d_ff 维度 (参数量: d_model × d_ff)
    第二层：从 d_ff 维度映射回 d_model 维度 (参数量: d_ff × d_model)
    总参数量 = d_model × d_ff + d_ff × d_model = 2 × d_model × d_ff
    '''
    encoder_ff_params = d_model * d_ff * 2      # 前馈网络两个线性层
    '''
    这计算的是层归一化(Layer Normalization)层的参数数量
    编码器中有两个LayerNorm层(一个在注意力层之后，一个在前馈网络之后)
    每个LayerNorm层有 d_model 个参数(用于缩放)和 d_model 个偏置参数
    但代码只计算了 d_model 个参数，可能是只考虑了缩放参数或对代码进行了简化
    '''
    encoder_norm_params = d_model * 2           # 两个LayerNorm
    #这计算一个完整编码器层的总参数量，即上述三个组件参数量之和
    encoder_layer_params = encoder_mha_params + encoder_ff_params + encoder_norm_params
    
    for i in range(num_layers):
        print(f"├─────────────────────────────────┼─────────────────────────┼───────────────┤")
        print(f"│ Encoder Block {i+1}                 │                         │               │")
        print(f"│  ├─Multi-Head Attention         │ (batch, seq_len, {d_model}) │ {encoder_mha_params:13,d} │")
        print(f"│  ├─Add & Norm                   │ (batch, seq_len, {d_model}) │ {d_model:13,d} │")
        print(f"│  ├─Feed Forward                 │ (batch, seq_len, {d_model}) │ {encoder_ff_params:13,d} │")
        print(f"│  └─Add & Norm                   │ (batch, seq_len, {d_model}) │ {d_model:13,d} │")
    
    # 解码器嵌入层
    decoder_embedding_params = tgt_vocab_size * d_model
    print(f"├─────────────────────────────────┼─────────────────────────┼───────────────┤")
    print(f"│ Embedding (Target)              │ (batch, seq_len, {d_model}) │ {decoder_embedding_params:13,d} │")
    
    # 解码器层
    '''
    解码器含有两个不同的多头注意力层：
    自注意力层(Masked Multi-Head Attention)：解码器自身的注意力计算
    交叉注意力层(Cross Multi-Head Attention)：处理编码器输出和解码器的关系
    每个注意力层都有4个线性变换(Q, K, V, Output)
    每个线性变换参数量为 d_model × d_model
    因此总参数量 = 4 × d_model × d_model × 2个注意力层
    '''
    decoder_mha_params = 4 * d_model * d_model * 2  # 两个多头注意力  两个不同的多头注意力层
    '''
    这计算的是前馈神经网络层的参数量，与编码器计算相同
    两个线性层：
    第一层：d_model → d_ff (参数量: d_model × d_ff)
    第二层：d_ff → d_model (参数量: d_ff × d_model)
    总参数量 = d_model × d_ff × 2
    '''
    decoder_ff_params = d_model * d_ff * 2          # 前馈网络 两个线性层
    '''
    解码器包含三个LayerNorm层：
    第一个在自注意力层之后
    第二个在交叉注意力层之后
    第三个在前馈网络之后
    每个LayerNorm层关注的是每个d_model维度上的特征
    每个LayerNorm有d_model个缩放参数(和d_model个偏置参数，但似乎代码简化只计算了缩放参数)
    总参数量 = d_model × 3
    '''
    decoder_norm_params = d_model * 3               # 三个LayerNorm
    decoder_layer_params = decoder_mha_params + decoder_ff_params + decoder_norm_params
    
    for i in range(num_layers):
        print(f"├─────────────────────────────────┼─────────────────────────┼───────────────┤")
        print(f"│ Decoder Block {i+1}                 │                         │               │")
        print(f"│  ├─Masked Multi-Head Attention  │ (batch, seq_len, {d_model}) │ {encoder_mha_params:13,d} │")
        print(f"│  ├─Add & Norm                   │ (batch, seq_len, {d_model}) │ {d_model:13,d} │")
        print(f"│  ├─Cross Multi-Head Attention   │ (batch, seq_len, {d_model}) │ {encoder_mha_params:13,d} │")
        print(f"│  ├─Add & Norm                   │ (batch, seq_len, {d_model}) │ {d_model:13,d} │")
        print(f"│  ├─Feed Forward                 │ (batch, seq_len, {d_model}) │ {encoder_ff_params:13,d} │")
        print(f"│  └─Add & Norm                   │ (batch, seq_len, {d_model}) │ {d_model:13,d} │")
    
    # 输出层
    output_params = d_model * tgt_vocab_size
    print(f"├─────────────────────────────────┼─────────────────────────┼───────────────┤")
    print(f"│ Linear Output                   │ (batch, seq_len, {tgt_vocab_size}) │ {output_params:13,d} │")
    print("└─────────────────────────────────┴─────────────────────────┴───────────────┘")
    
    # 计算总参数
    encoder_total = num_layers * encoder_layer_params + embedding_params
    decoder_total = num_layers * decoder_layer_params + decoder_embedding_params
    total_params = encoder_total + decoder_total + output_params
    trainable_params = total_params
    
    print(f" Total params: {total_params:,d} ({total_params*4/1024/1024:.2f} MB)")
    print(f" Trainable params: {trainable_params:,d} ({trainable_params*4/1024/1024:.2f} MB)")

def create_masks(src, tgt):
    """创建训练Transformer时所需的掩码"""
    # 源序列填充掩码
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    
    # 目标序列填充和后续位置掩码
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    
    return src_mask, tgt_mask

def visualize_attention(model, src, tgt):
    """可视化Transformer中的注意力权重"""
    plt.figure(figsize=(15, 8))
    plt.title("Transformer注意力权重可视化")
    
    # 这里应该实现注意力权重可视化
    # 注意：实际实现需要修改模型以返回注意力权重
    
    plt.tight_layout()
    plt.show()

def test_transformer():
    """测试Transformer模型"""
    # 参数设置
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    dropout = 0.1
    
    # 创建模型
    transformer = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout
    )
    
    # 打印模型摘要
    print_model_summary(transformer, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # 创建示例输入
    src = torch.randint(1, src_vocab_size, (2, 10))  # 批量大小2，序列长度10
    tgt = torch.randint(1, tgt_vocab_size, (2, 8))   # 批量大小2，序列长度8
    
    # 创建掩码
    src_mask, tgt_mask = create_masks(src, tgt)
    
    # 前向传播
    output = transformer(src, tgt, src_mask, tgt_mask)
    print(f"\n输出形状: {output.shape}")
    print("Transformer模型测试成功！")

if __name__ == "__main__":
    test_transformer()