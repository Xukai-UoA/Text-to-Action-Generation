"""
Enhanced Seq2Seq训练脚本

改进特性：
1. Bahdanau Attention机制
2. 多层双向LSTM编码器（2-3层）+ 多层单向LSTM解码器（2-3层）
3. 残差连接（从第2层开始）
4. Layer Normalization
5. Scheduled Sampling with linear decay

使用方法:
    python seq2seq_enhanced_pretrain.py

配置说明：
    - 默认使用2层编码器和2层解码器
    - hidden_size从300提升到512
    - attention_size设置为256
    - 启用Scheduled Sampling（30%-70% epoch线性衰减）
"""

import os
import sys
import numpy as np
import scipy.io as scio
import torch

# 添加模型路径
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from model.seq2seq_enhanced_trainer import EnhancedSeq2SeqTrainer
from model.seq2seq_enhanced_structure import EnhancedSeq2SeqModel


def load_metadata(metadata_path):
    """加载预处理的metadata"""
    npzfile = np.load(metadata_path, allow_pickle=True)

    train_action = npzfile['arr_0']
    train_script = npzfile['arr_1']
    train_length = npzfile['arr_2']
    sentence_steps = int(npzfile['arr_3'])

    return train_action, train_script, train_length, sentence_steps


def main():
    # ==================== 配置 ====================
    # GPU设置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("Enhanced Seq2Seq Model Training")
    print("=" * 80)
    print(f"\n使用设备: {device}")

    # 数据路径
    metadata_path = './data/metadata.npz'
    mean_pose_path = './data/mean_pose.mat'
    model_dir = './seq2seq_enhanced_model'

    # ==================== 加载数据 ====================
    print("\n" + "=" * 80)
    print("加载数据")
    print("=" * 80)

    train_action, train_script, train_length, sentence_steps = load_metadata(metadata_path)

    # 加载初始pose
    init_pose = scio.loadmat(mean_pose_path)['mean_vector']

    num_data_train = train_action.shape[0]

    print(f"  训练数据: {num_data_train} 样本")
    print(f"  动作形状: {train_action.shape} (num_data, dim_action=24, action_steps=32)")
    print(f"  文本形状: {train_script.shape} (num_data, dim_sentence=300, sentence_steps)")
    print(f"  最大句子长度: {sentence_steps}")
    print(f"  初始pose形状: {init_pose.shape}")

    # ==================== 超参数 ====================
    print("\n" + "=" * 80)
    print("超参数配置")
    print("=" * 80)

    # 模型架构参数
    dim_sentence = 300  # Word2Vec embedding维度
    dim_char_enc = 512  # 编码器hidden size（从300提升到512）
    dim_gen = 512       # 解码器hidden size（从300提升到512）
    attention_size = 256  # Attention MLP维度
    dim_random = 10     # 随机噪声维度
    action_steps = 32   # 动作序列长度

    # 多层LSTM配置
    num_encoder_layers = 2  # 编码器层数（2-3层推荐）
    num_decoder_layers = 2  # 解码器层数（2-3层推荐）

    # Layer Normalization和Residual连接
    use_layer_norm = True
    use_residual = True

    # 训练参数
    batch_size = 32
    max_epoch = 500
    save_stride = 5
    learning_rate = 0.00005

    # Scheduled Sampling配置
    use_scheduled_sampling = True
    ss_start_ratio = 0.3    # 30% epoch后开始衰减
    ss_end_ratio = 0.7      # 70% epoch后保持最小值
    ss_min_epsilon = 0.5    # 最小teacher forcing ratio

    # 恢复训练设置
    restore = 0
    restore_path = ''
    restore_step = 0

    print("\n模型架构:")
    print(f"  - 编码器: {num_encoder_layers}层 双向LSTM, hidden_size={dim_char_enc}")
    print(f"  - 解码器: {num_decoder_layers}层 单向LSTM, hidden_size={dim_gen}")
    print(f"  - Attention: Bahdanau, size={attention_size}")
    print(f"  - Layer Normalization: {'启用' if use_layer_norm else '禁用'}")
    print(f"  - 残差连接: {'启用 (第2层开始)' if use_residual else '禁用'}")

    print("\n训练策略:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Max epochs: {max_epoch}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Scheduled Sampling: {'启用' if use_scheduled_sampling else '禁用'}")
    if use_scheduled_sampling:
        print(f"    · 阶段1 (0-{int(max_epoch * ss_start_ratio)}): 完全Teacher Forcing (ε=1.0)")
        print(f"    · 阶段2 ({int(max_epoch * ss_start_ratio)}-{int(max_epoch * ss_end_ratio)}): 线性衰减")
        print(f"    · 阶段3 ({int(max_epoch * ss_end_ratio)}+): 混合采样 (ε={ss_min_epsilon})")

    # ==================== 创建模型 ====================
    print("\n" + "=" * 80)
    print("创建增强版模型")
    print("=" * 80)

    model = EnhancedSeq2SeqModel(
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        attention_size=attention_size,
        use_layer_norm=use_layer_norm,
        use_residual=use_residual
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型统计:")
    print(f"  - 总参数: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 参数占用内存: ~{total_params * 4 / 1024 / 1024:.2f} MB (fp32)")

    # ==================== 创建训练器 ====================
    print("\n" + "=" * 80)
    print("创建训练器")
    print("=" * 80)

    train_module = EnhancedSeq2SeqTrainer(
        model=model,
        train_script=train_script,
        train_script_len=train_length,
        train_action=train_action,
        init_pose=init_pose,
        num_data=num_data_train,
        batch_size=batch_size,
        model_dir=model_dir,
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random,
        restore=restore,
        restore_path=restore_path,
        restore_step=restore_step,
        max_epoch=max_epoch,
        save_stride=save_stride,
        learning_rate=learning_rate,
        use_scheduled_sampling=use_scheduled_sampling,
        ss_start_ratio=ss_start_ratio,
        ss_end_ratio=ss_end_ratio,
        ss_min_epsilon=ss_min_epsilon,
        device=device
    )

    # ==================== 开始训练 ====================
    print("\n" + "=" * 80)
    print("开始训练 Enhanced Seq2Seq 模型")
    print("=" * 80)

    try:
        train_module.train()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"模型保存在: {model_dir}")
    print(f"训练历史保存在: {os.path.join(model_dir, 'training_history.npz')}")

    # ==================== 模型改进总结 ====================
    print("\n" + "=" * 80)
    print("模型改进总结")
    print("=" * 80)
    print("""
相比原始模型的改进：

1. ✅ Bahdanau Attention
   - 在decoder step之前计算attention
   - 更符合"对齐后解码"的直觉
   - 公式: score = v^T × tanh(W1×h + W2×encoder_outputs)

2. ✅ 多层双向LSTM编码器
   - 编码器: 2层双向LSTM (hidden_size=512)
   - 更强的文本表示能力
   - 双向信息流捕获上下文

3. ✅ 多层单向LSTM解码器
   - 解码器: 2层单向LSTM (hidden_size=512)
   - 从第2层开始添加残差连接
   - 缓解梯度消失问题

4. ✅ Layer Normalization
   - 在每层LSTM后应用LayerNorm
   - 稳定训练过程
   - 加速收敛

5. ✅ Scheduled Sampling
   - 课程学习策略，缓解exposure bias
   - 线性衰减: 1.0 → 0.5 (epoch 30%-70%)
   - 提高模型鲁棒性

预期效果：
  - 更好的动作生成质量
  - 更快的收敛速度
  - 更稳定的训练过程
  - 更强的泛化能力
""")


if __name__ == "__main__":
    main()
