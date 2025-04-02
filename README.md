# Transformer Model Architecture

一个从零开始实现的完整 Transformer 模型架构，包含详细注释和结构可视化功能。此项目旨在帮助理解 Transformer 的内部工作原理，适合学习和教学使用。

## 项目简介

Transformer 是一种基于自注意力机制的神经网络架构，在自然语言处理、计算机视觉等领域有广泛应用。本项目提供了 Transformer 的完整实现，包括：

- 位置编码（Positional Encoding）
- 多头注意力机制（Multi-Head Attention）
- 编码器-解码器架构（Encoder-Decoder Architecture）
- 掩码机制（Masking）
- 模型结构可视化

## 不懂 Transformer？没关系！

如果你对 Transformer 不熟悉，这个项目恰好可以帮助你理解其工作原理：

- **Transformer 的基本思想**：捕捉序列中元素之间的关系，而不依赖于递归结构
- **注意力机制**：允许模型关注输入序列的不同部分
- **编码器-解码器结构**：编码器处理输入序列，解码器生成输出序列

## 安装指南

### 环境要求

- Python 3.7 或更高版本
- PyTorch 1.8.0 或更高版本

### 安装步骤

1. 克隆此仓库：
   ```bash
   git clone https://github.com/yourusername/transformer-model-architecture.git
   cd transformer-model-architecture
   ```

2. 安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```

## 运行指南

运行主脚本来测试 Transformer 模型：

```bash
python transformer-model-architecture.py
```

这将：
1. 创建一个 Transformer 模型实例
2. 打印模型结构和参数统计
3. 使用随机输入测试模型的前向传播
4. 显示运行成功信息

## 代码结构

- `PositionalEncoding`: 为序列中的每个位置提供唯一的位置信息
- `MultiHeadAttention`: 实现多头自注意力机制
- `FeedForward`: 实现前馈神经网络
- `EncoderLayer`: Transformer 的编码器层
- `DecoderLayer`: Transformer 的解码器层
- `Transformer`: 完整的 Transformer 模型
- `print_model_summary`: 打印模型结构概览
- `create_masks`: 创建训练所需的掩码
- `test_transformer`: 测试函数

## 自定义与扩展

你可以修改 `test_transformer()` 函数中的参数来尝试不同配置：

```python
# 参数设置
src_vocab_size = 5000    # 源词汇表大小
tgt_vocab_size = 5000    # 目标词汇表大小
d_model = 512            # 模型维度
num_heads = 8            # 注意力头数
num_layers = 6           # 编码器/解码器层数
d_ff = 2048              # 前馈网络维度
max_seq_length = 100     # 最大序列长度
dropout = 0.1            # Dropout比率
```

## 下一步拓展

- 实现完整的训练循环
- 添加实际的翻译示例
- 增强注意力权重可视化
- 实现预训练模型的加载功能

## 注意事项

- 此实现主要用于教育目的，强调代码可读性
- 对于生产环境，建议使用优化的框架如 Hugging Face Transformers

## 许可证

[MIT](LICENSE)