"""
Enhanced Seq2Seq Trainer with Scheduled Sampling support

支持特性：
1. Linear decay scheduled sampling
2. 课程学习策略（epsilon衰减）
3. 兼容增强版模型的所有特性
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os


class ScheduledSamplingScheduler:
    """
    Scheduled Sampling调度器

    实现线性衰减策略：
    - Epoch 0-30%: epsilon=1.0 (完全teacher forcing)
    - Epoch 30-70%: 线性衰减
    - Epoch 70-100%: epsilon=0.5 (保持混合)
    """

    def __init__(self, max_epoch, start_ratio=0.3, end_ratio=0.7, min_epsilon=0.5):
        """
        Args:
            max_epoch: 总训练轮数
            start_ratio: 开始衰减的epoch比例 (默认0.3 = 30%)
            end_ratio: 结束衰减的epoch比例 (默认0.7 = 70%)
            min_epsilon: 最小epsilon值 (默认0.5 = 50% teacher forcing)
        """
        self.max_epoch = max_epoch
        self.start_epoch = int(max_epoch * start_ratio)
        self.end_epoch = int(max_epoch * end_ratio)
        self.min_epsilon = min_epsilon

    def get_epsilon(self, current_epoch):
        """
        根据当前epoch计算teacher forcing ratio

        Args:
            current_epoch: 当前训练轮数 (0-based)

        Returns:
            epsilon: teacher forcing probability [0.5, 1.0]
        """
        if current_epoch < self.start_epoch:
            # 阶段1: 完全teacher forcing
            return 1.0
        elif current_epoch >= self.end_epoch:
            # 阶段3: 保持最小值
            return self.min_epsilon
        else:
            # 阶段2: 线性衰减
            progress = (current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            epsilon = 1.0 - progress * (1.0 - self.min_epsilon)
            return epsilon

    def get_stage(self, current_epoch):
        """获取当前训练阶段"""
        if current_epoch < self.start_epoch:
            return "Stage 1: Full Teacher Forcing"
        elif current_epoch >= self.end_epoch:
            return "Stage 3: Mixed Sampling"
        else:
            return "Stage 2: Linear Decay"


class EnhancedSeq2SeqTrainer:
    """
    增强版Seq2Seq训练器

    新特性：
    1. Scheduled Sampling with linear decay
    2. 更详细的训练日志
    3. 兼容增强版模型（多层LSTM + Attention + LayerNorm + Residual）
    """

    def __init__(self, model, train_script, train_script_len, train_action, init_pose,
                 num_data, batch_size, model_dir, sentence_steps, action_steps,
                 dim_sentence, dim_char_enc, dim_gen, dim_random,
                 restore=0, restore_path='', restore_step=0,
                 max_epoch=500, save_stride=5, learning_rate=0.00005,
                 use_scheduled_sampling=True, ss_start_ratio=0.3, ss_end_ratio=0.7, ss_min_epsilon=0.5,
                 device='cuda'):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.model = model.to(self.device)
        self.train_script = train_script  # [num_data, dim_sentence, sentence_steps]
        self.train_script_len = train_script_len
        self.train_action = train_action  # [num_data, dim_action, action_steps]
        self.init_pose = init_pose
        self.num_data = num_data
        self.batch_size = batch_size

        # 准备batch初始pose
        self.batch_init = np.transpose(np.tile(self.init_pose, (1, batch_size)), [1, 0])

        self.num_batch = num_data // batch_size
        self.model_dir = model_dir
        self.sentence_steps = sentence_steps
        self.action_steps = action_steps
        self.dim_sentence = dim_sentence
        self.dim_char_enc = dim_char_enc
        self.dim_gen = dim_gen
        self.dim_random = dim_random

        self.restore = restore
        self.restore_path = restore_path
        self.restore_step = restore_step

        self.max_epoch = max_epoch
        self.save_stride = save_stride
        self.learning_rate = learning_rate

        # Scheduled Sampling配置
        self.use_scheduled_sampling = use_scheduled_sampling
        if use_scheduled_sampling:
            self.ss_scheduler = ScheduledSamplingScheduler(
                max_epoch=max_epoch,
                start_ratio=ss_start_ratio,
                end_ratio=ss_end_ratio,
                min_epsilon=ss_min_epsilon
            )
            print(f"\n启用Scheduled Sampling:")
            print(f"  - 阶段1 (Epoch 0-{self.ss_scheduler.start_epoch}): 完全Teacher Forcing (ε=1.0)")
            print(f"  - 阶段2 (Epoch {self.ss_scheduler.start_epoch}-{self.ss_scheduler.end_epoch}): 线性衰减")
            print(f"  - 阶段3 (Epoch {self.ss_scheduler.end_epoch}+): 混合采样 (ε={ss_min_epsilon})")
        else:
            self.ss_scheduler = None
            print("\n未启用Scheduled Sampling (使用完全Teacher Forcing)")

        # 创建模型目录
        os.makedirs(self.model_dir, exist_ok=True)

        # 设置optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 训练统计
        self.train_losses = []
        self.epoch_losses = []

        # 最优模型追踪
        self.best_loss = float('inf')
        self.best_epoch = 0

    def _save_training_history(self):
        """保存训练历史到npz文件"""
        if len(self.epoch_losses) == 0:
            return

        history_path = os.path.join(self.model_dir, 'training_history.npz')
        np.savez(history_path,
                 epochs=[d['epoch'] for d in self.epoch_losses],
                 total_losses=[d['total_loss'] for d in self.epoch_losses],
                 action_losses=[d['action_loss'] for d in self.epoch_losses],
                 char_losses=[d['char_loss'] for d in self.epoch_losses],
                 epsilons=[d['epsilon'] for d in self.epoch_losses])

    def train(self):
        """主训练循环"""

        # 如果需要恢复训练
        if self.restore == 1:
            checkpoint = torch.load(self.restore_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 恢复最优loss记录
            if 'best_loss' in checkpoint:
                self.best_loss = checkpoint['best_loss']
                self.best_epoch = checkpoint.get('best_epoch', 0)
            print(f'✓ 从检查点恢复: {self.restore_path}')
            if self.best_loss < float('inf'):
                print(f'  当前最优loss: {self.best_loss:.6f} (Epoch {self.best_epoch})')

        # 转换numpy数组为torch tensors
        train_script_tensor = torch.FloatTensor(self.train_script).to(self.device)
        train_script_len_tensor = torch.LongTensor(self.train_script_len)
        train_action_tensor = torch.FloatTensor(self.train_action).to(self.device)

        print(f"\n开始训练...")
        print(f"  - 训练样本: {self.num_data}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Batches/epoch: {self.num_batch}")
        print(f"  - 总轮数: {self.max_epoch}")
        print("=" * 80)

        for epoch in range(self.max_epoch):
            # 计算当前epoch的teacher forcing ratio
            if self.use_scheduled_sampling:
                current_epsilon = self.ss_scheduler.get_epsilon(epoch + self.restore_step)
                current_stage = self.ss_scheduler.get_stage(epoch + self.restore_step)
            else:
                current_epsilon = 1.0
                current_stage = "Full Teacher Forcing"

            # Epoch开始信息
            if epoch % 10 == 0 or epoch == 0:
                print(f"\nEpoch {epoch + self.restore_step}/{self.max_epoch}")
                print(f"  策略: {current_stage}")
                print(f"  Teacher forcing ratio (ε): {current_epsilon:.3f}")

            # Shuffle数据
            batch_shuffle = list(range(self.num_data))
            random.shuffle(batch_shuffle)

            epoch_loss_sum = 0.0
            epoch_action_loss_sum = 0.0
            epoch_char_loss_sum = 0.0

            for i in range(self.num_batch):
                # 获取batch索引
                batch_idx = batch_shuffle[i * self.batch_size:(i + 1) * self.batch_size]

                # 准备batch数据
                script_batch = train_script_tensor[batch_idx].transpose(1, 2)
                length_batch = train_script_len_tensor[batch_idx]
                action_batch = train_action_tensor[batch_idx].transpose(1, 2)

                # 准备输入
                curr_action_init = torch.FloatTensor(self.batch_init).to(self.device)
                curr_char_init = torch.zeros(self.batch_size, self.dim_sentence).to(self.device)

                # 使用零随机噪声
                curr_random_c2a = torch.zeros(self.batch_size, self.sentence_steps, self.dim_random).to(self.device)
                curr_random_a2c = torch.zeros(self.batch_size, self.action_steps, self.dim_random).to(self.device)

                # 前向传播
                self.optimizer.zero_grad()

                # 1. 编码文本
                char_enc_out = self.model.char_encoder(script_batch, length_batch)

                # 2. 从文本生成动作（应用scheduled sampling）
                action_gen_out, action_enc_out = self.model.char2action(
                    char_enc_out, curr_action_init, curr_random_c2a, self.batch_size,
                    teacher_forcing_ratio=current_epsilon
                )

                # 3. 从动作重建文本（不使用scheduled sampling）
                char_recon_out = self.model.action2char(
                    action_enc_out, curr_char_init, curr_random_a2c, self.batch_size
                )

                # 4. 计算损失
                total_loss, action_loss, char_loss = self.model.seq2seq_loss(
                    action_gen_out, action_batch, char_recon_out, script_batch
                )

                # 反向传播
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                # 更新权重
                self.optimizer.step()

                # 累积损失
                epoch_loss_sum += total_loss.item()
                epoch_action_loss_sum += action_loss.item()
                epoch_char_loss_sum += char_loss.item()

                # 打印进度
                if i % 100 == 0:
                    print(f'  Batch {i}/{self.num_batch} | '
                          f'Loss: {total_loss.item():.6f} | '
                          f'Action: {action_loss.item():.6f} | '
                          f'Char: {char_loss.item():.6f}')

            # Epoch结束统计
            avg_epoch_loss = epoch_loss_sum / self.num_batch
            avg_action_loss = epoch_action_loss_sum / self.num_batch
            avg_char_loss = epoch_char_loss_sum / self.num_batch
            self.epoch_losses.append({
                'epoch': epoch + self.restore_step,
                'total_loss': avg_epoch_loss,
                'action_loss': avg_action_loss,
                'char_loss': avg_char_loss,
                'epsilon': current_epsilon
            })

            print(f"\nEpoch {epoch + self.restore_step} 完成:")
            print(f"  平均损失: {avg_epoch_loss:.6f} "
                  f"(Action: {avg_action_loss:.6f}, Char: {avg_char_loss:.6f})")
            print(f"  Teacher forcing ratio: {current_epsilon:.3f}")

            # 检查是否是最优模型
            is_best = avg_epoch_loss < self.best_loss
            if is_best:
                self.best_loss = avg_epoch_loss
                self.best_epoch = epoch + self.restore_step
                print(f"  🌟 新的最优模型! Loss: {self.best_loss:.6f}")

            # 保存定期checkpoint
            if (epoch + 1) % self.save_stride == 0:
                checkpoint_path = os.path.join(
                    self.model_dir,
                    f'model_epoch_{epoch + 1 + self.restore_step}.pth'
                )
                torch.save({
                    'epoch': epoch + 1 + self.restore_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                    'action_loss': avg_action_loss,
                    'char_loss': avg_char_loss,
                    'epsilon': current_epsilon,
                    'best_loss': self.best_loss,
                    'best_epoch': self.best_epoch,
                    'train_losses': self.epoch_losses
                }, checkpoint_path)
                print(f'  ✓ 定期检查点已保存: {checkpoint_path}')

            # 保存最优模型
            if is_best:
                best_model_path = os.path.join(self.model_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + self.restore_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                    'action_loss': avg_action_loss,
                    'char_loss': avg_char_loss,
                    'epsilon': current_epsilon,
                    'best_loss': self.best_loss,
                    'best_epoch': self.best_epoch,
                    'train_losses': self.epoch_losses
                }, best_model_path)
                print(f'  ✓ 最优模型已保存: {best_model_path}')

            # 每个epoch后保存训练历史（用于实时监控）
            self._save_training_history()

        print("\n" + "=" * 80)
        print("训练完成!")
        print("=" * 80)

        # 最终保存训练历史
        self._save_training_history()
        history_path = os.path.join(self.model_dir, 'training_history.npz')
        print(f"最终训练历史已保存: {history_path}")

        # 打印最优模型信息
        print(f"\n最优模型:")
        print(f"  Epoch: {self.best_epoch}")
        print(f"  Loss: {self.best_loss:.6f}")
        print(f"  保存路径: {os.path.join(self.model_dir, 'best_model.pth')}")


if __name__ == "__main__":
    # 测试训练器
    from seq2seq_enhanced_structure import EnhancedSeq2SeqModel

    print("=" * 80)
    print("测试增强版Seq2Seq训练器")
    print("=" * 80)

    # 超参数
    sentence_steps = 30
    action_steps = 32
    dim_sentence = 300
    dim_char_enc = 512
    dim_gen = 512
    dim_random = 10
    attention_size = 256
    batch_size = 32

    # 生成测试数据
    num_data = 1000
    train_script = np.random.randn(num_data, dim_sentence, sentence_steps).astype(np.float32)
    train_script_len = np.random.randint(10, sentence_steps, size=(num_data,))
    train_action = np.random.randn(num_data, 24, action_steps).astype(np.float32)
    init_pose = np.random.randn(24, 1).astype(np.float32)

    # 创建增强模型
    model = EnhancedSeq2SeqModel(
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random,
        num_encoder_layers=2,
        num_decoder_layers=2,
        attention_size=attention_size,
        use_layer_norm=True,
        use_residual=True
    )

    # 创建训练器
    trainer = EnhancedSeq2SeqTrainer(
        model=model,
        train_script=train_script,
        train_script_len=train_script_len,
        train_action=train_action,
        init_pose=init_pose,
        num_data=num_data,
        batch_size=batch_size,
        model_dir='./checkpoints_enhanced',
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random,
        max_epoch=20,
        save_stride=5,
        learning_rate=0.00005,
        use_scheduled_sampling=True,
        ss_start_ratio=0.3,
        ss_end_ratio=0.7,
        ss_min_epsilon=0.5,
        device='cpu'
    )

    # 开始训练
    print("\n开始训练...")
    trainer.train()
    print("训练完成!")
