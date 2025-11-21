"""
训练损失可视化工具

功能：
1. 可视化训练过程中的损失曲线（总损失、动作损失、文本损失）
2. 可视化Scheduled Sampling的epsilon变化
3. 标注最优模型位置
4. 支持实时监控训练中的模型
5. 自动保存高分辨率图片

使用方法：
    # 从训练历史文件生成可视化
    python visualize_training.py --history_path ./seq2seq_enhanced_model/training_history.npz

    # 实时监控训练（自动刷新）
    python visualize_training.py --history_path ./seq2seq_enhanced_model/training_history.npz --live

    # 指定输出路径
    python visualize_training.py --history_path ./seq2seq_enhanced_model/training_history.npz --output ./plots/
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import time


def load_training_history(history_path):
    """
    加载训练历史数据

    Args:
        history_path: training_history.npz文件路径

    Returns:
        dict with keys: epochs, total_losses, action_losses, char_losses, epsilons
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"训练历史文件不存在: {history_path}")

    data = np.load(history_path)

    return {
        'epochs': data['epochs'],
        'total_losses': data['total_losses'],
        'action_losses': data['action_losses'],
        'char_losses': data['char_losses'],
        'epsilons': data['epsilons']
    }


def find_best_model(model_dir):
    """
    从best_model.pth中读取最优模型信息

    Args:
        model_dir: 模型保存目录

    Returns:
        dict with keys: epoch, loss, action_loss, char_loss (如果文件存在)
        None (如果文件不存在)
    """
    best_model_path = os.path.join(model_dir, 'best_model.pth')

    if not os.path.exists(best_model_path):
        return None

    try:
        import torch
        checkpoint = torch.load(best_model_path, map_location='cpu')
        return {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', 0),
            'action_loss': checkpoint.get('action_loss', 0),
            'char_loss': checkpoint.get('char_loss', 0)
        }
    except Exception as e:
        print(f"警告: 无法加载best_model.pth: {e}")
        return None


def plot_training_curves(history, best_model_info=None, output_dir=None, show=True):
    """
    绘制完整的训练曲线

    Args:
        history: 训练历史数据字典
        best_model_info: 最优模型信息字典（可选）
        output_dir: 输出目录（可选，如果提供则保存图片）
        show: 是否显示图片
    """
    epochs = history['epochs']
    total_losses = history['total_losses']
    action_losses = history['action_losses']
    char_losses = history['char_losses']
    epsilons = history['epsilons']

    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)

    # 配色方案
    color_total = '#2E86AB'
    color_action = '#A23B72'
    color_char = '#F18F01'
    color_epsilon = '#06A77D'
    color_best = '#FF6B6B'

    # ==================== 子图1: 总损失曲线 ====================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epochs, total_losses, color=color_total, linewidth=2, label='Total Loss', alpha=0.8)
    ax1.fill_between(epochs, total_losses, alpha=0.2, color=color_total)

    # 标注最优模型
    if best_model_info is not None:
        best_epoch = best_model_info['epoch']
        best_loss = best_model_info['loss']
        ax1.scatter([best_epoch], [best_loss], color=color_best, s=200, zorder=5,
                   marker='*', edgecolors='white', linewidths=2, label='Best Model')
        ax1.axvline(best_epoch, color=color_best, linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.annotate(f'Best: {best_loss:.4f}\nEpoch {best_epoch}',
                    xy=(best_epoch, best_loss),
                    xytext=(10, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc=color_best, alpha=0.7, edgecolor='white'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=color_best, lw=2),
                    fontsize=10, color='white', weight='bold')

    ax1.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax1.set_ylabel('Total Loss', fontsize=12, weight='bold')
    ax1.set_title('Training Loss Curve (Total)', fontsize=14, weight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)

    # ==================== 子图2: 动作损失 ====================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(epochs, action_losses, color=color_action, linewidth=2, label='Action Loss', alpha=0.8)
    ax2.fill_between(epochs, action_losses, alpha=0.2, color=color_action)

    if best_model_info is not None:
        best_epoch = best_model_info['epoch']
        best_action_loss = best_model_info['action_loss']
        ax2.scatter([best_epoch], [best_action_loss], color=color_best, s=150, zorder=5,
                   marker='*', edgecolors='white', linewidths=1.5)

    ax2.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax2.set_ylabel('Action Loss', fontsize=11, weight='bold')
    ax2.set_title('Action Reconstruction Loss', fontsize=12, weight='bold', pad=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)

    # ==================== 子图3: 文本损失 ====================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(epochs, char_losses, color=color_char, linewidth=2, label='Char Loss', alpha=0.8)
    ax3.fill_between(epochs, char_losses, alpha=0.2, color=color_char)

    if best_model_info is not None:
        best_epoch = best_model_info['epoch']
        best_char_loss = best_model_info['char_loss']
        ax3.scatter([best_epoch], [best_char_loss], color=color_best, s=150, zorder=5,
                   marker='*', edgecolors='white', linewidths=1.5)

    ax3.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax3.set_ylabel('Char Loss (×5)', fontsize=11, weight='bold')
    ax3.set_title('Text Reconstruction Loss', fontsize=12, weight='bold', pad=10)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(left=0)

    # ==================== 子图4: Epsilon变化（Scheduled Sampling） ====================
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(epochs, epsilons, color=color_epsilon, linewidth=2.5, label='Teacher Forcing Ratio (ε)', alpha=0.9)
    ax4.fill_between(epochs, epsilons, alpha=0.3, color=color_epsilon)

    # 标注阶段
    max_epoch = epochs[-1]
    unique_epsilons = np.unique(epsilons)

    # 找到epsilon变化的拐点
    if len(unique_epsilons) > 1:
        # 找到开始衰减的epoch
        start_decay_idx = np.where(np.diff(epsilons) < 0)[0]
        if len(start_decay_idx) > 0:
            start_decay = epochs[start_decay_idx[0]]
            ax4.axvline(start_decay, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
            ax4.text(start_decay, 0.5, 'Decay Start', rotation=90, va='bottom', ha='right',
                    fontsize=9, alpha=0.7)

        # 找到衰减结束的epoch
        end_decay_idx = np.where(np.diff(epsilons) == 0)[0]
        if len(end_decay_idx) > 0 and len(start_decay_idx) > 0:
            # 找到第一个在start_decay之后的稳定点
            end_decay_candidates = end_decay_idx[end_decay_idx > start_decay_idx[0]]
            if len(end_decay_candidates) > 0:
                end_decay = epochs[end_decay_candidates[0]]
                ax4.axvline(end_decay, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
                ax4.text(end_decay, 0.5, 'Decay End', rotation=90, va='bottom', ha='left',
                        fontsize=9, alpha=0.7)

    ax4.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax4.set_ylabel('Epsilon (ε)', fontsize=11, weight='bold')
    ax4.set_title('Scheduled Sampling Strategy', fontsize=12, weight='bold', pad=10)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(left=0)
    ax4.set_ylim([0, 1.05])

    # 添加水平参考线
    ax4.axhline(1.0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    # ==================== 总标题 ====================
    fig.suptitle('Enhanced Seq2Seq Training Visualization', fontsize=16, weight='bold', y=0.995)

    # ==================== 添加统计信息文本框 ====================
    stats_text = f"""
    Training Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━━
    Total Epochs: {len(epochs)}
    Final Loss: {total_losses[-1]:.4f}
    Min Loss: {np.min(total_losses):.4f}
    """

    if best_model_info is not None:
        stats_text += f"""Best Model (Epoch {best_model_info['epoch']}):
      Total: {best_model_info['loss']:.4f}
      Action: {best_model_info['action_loss']:.4f}
      Char: {best_model_info['char_loss']:.4f}
    """

    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            verticalalignment='bottom')

    plt.tight_layout(rect=[0, 0.05, 1, 0.99])

    # 保存图片
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ 可视化图片已保存: {output_path}")

    # 显示图片
    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_comparison(history, output_dir=None, show=True):
    """
    绘制损失对比图（单独的简洁版本）

    Args:
        history: 训练历史数据字典
        output_dir: 输出目录
        show: 是否显示图片
    """
    epochs = history['epochs']
    total_losses = history['total_losses']
    action_losses = history['action_losses']
    char_losses = history['char_losses']

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(epochs, total_losses, label='Total Loss', linewidth=2, color='#2E86AB', alpha=0.9)
    ax.plot(epochs, action_losses, label='Action Loss', linewidth=2, color='#A23B72', alpha=0.9)
    ax.plot(epochs, char_losses, label='Char Loss', linewidth=2, color='#F18F01', alpha=0.9)

    ax.set_xlabel('Epoch', fontsize=12, weight='bold')
    ax.set_ylabel('Loss', fontsize=12, weight='bold')
    ax.set_title('Training Loss Comparison', fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)

    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'loss_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ 损失对比图已保存: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def monitor_training(history_path, model_dir, refresh_interval=10):
    """
    实时监控训练过程

    Args:
        history_path: 训练历史文件路径
        model_dir: 模型保存目录
        refresh_interval: 刷新间隔（秒）
    """
    print("=" * 80)
    print("实时训练监控模式")
    print("=" * 80)
    print(f"监控文件: {history_path}")
    print(f"刷新间隔: {refresh_interval}秒")
    print("按 Ctrl+C 退出")
    print("=" * 80)

    plt.ion()  # 开启交互模式

    try:
        while True:
            try:
                # 加载最新数据
                history = load_training_history(history_path)
                best_model_info = find_best_model(model_dir)

                # 清空当前图形
                plt.clf()

                # 绘制
                plot_training_curves(history, best_model_info, output_dir=None, show=False)

                # 刷新显示
                plt.pause(refresh_interval)

                # 打印最新状态
                last_epoch = history['epochs'][-1]
                last_loss = history['total_losses'][-1]
                print(f"[{time.strftime('%H:%M:%S')}] Epoch {last_epoch} | Loss: {last_loss:.6f}", end='\r')

            except FileNotFoundError:
                print(f"等待训练历史文件生成...")
                time.sleep(refresh_interval)
            except Exception as e:
                print(f"\n警告: {e}")
                time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n监控已停止")
        plt.ioff()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Seq2Seq训练可视化工具')
    parser.add_argument('--history_path', type=str, default='./seq2seq_enhanced_model/training_history.npz',
                       help='训练历史文件路径')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='模型保存目录（用于查找best_model.pth，默认从history_path推断）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出目录（如果提供则保存图片）')
    parser.add_argument('--live', action='store_true',
                       help='实时监控模式（自动刷新）')
    parser.add_argument('--refresh_interval', type=int, default=10,
                       help='实时监控刷新间隔（秒），默认10秒')
    parser.add_argument('--no_show', action='store_true',
                       help='不显示图片（仅保存）')
    parser.add_argument('--comparison_only', action='store_true',
                       help='仅绘制损失对比图')

    args = parser.parse_args()

    # 推断模型目录
    if args.model_dir is None:
        args.model_dir = os.path.dirname(args.history_path)

    # 实时监控模式
    if args.live:
        monitor_training(args.history_path, args.model_dir, args.refresh_interval)
        return

    # 加载数据
    print("加载训练历史数据...")
    try:
        history = load_training_history(args.history_path)
        print(f"✓ 成功加载 {len(history['epochs'])} 个epoch的数据")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 查找最优模型
    best_model_info = find_best_model(args.model_dir)
    if best_model_info is not None:
        print(f"✓ 找到最优模型: Epoch {best_model_info['epoch']}, Loss {best_model_info['loss']:.6f}")
    else:
        print("未找到best_model.pth")

    # 绘制图形
    print("\n生成可视化...")
    if args.comparison_only:
        plot_loss_comparison(history, args.output, show=not args.no_show)
    else:
        plot_training_curves(history, best_model_info, args.output, show=not args.no_show)

    print("\n完成!")


if __name__ == "__main__":
    main()
