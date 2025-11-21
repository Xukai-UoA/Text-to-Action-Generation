"""
简单的命令行训练监控工具

功能：
1. 实时显示训练进度
2. 显示当前epoch的loss
3. 显示最优模型信息
4. 显示预估剩余时间

使用方法：
    python monitor_training.py --model_dir ./seq2seq_enhanced_model
"""

import os
import sys
import argparse
import numpy as np
import time
from datetime import datetime, timedelta


def load_training_history(history_path):
    """加载训练历史"""
    if not os.path.exists(history_path):
        return None

    try:
        data = np.load(history_path)
        return {
            'epochs': data['epochs'],
            'total_losses': data['total_losses'],
            'action_losses': data['action_losses'],
            'char_losses': data['char_losses'],
            'epsilons': data['epsilons']
        }
    except Exception as e:
        return None


def load_best_model_info(model_dir):
    """加载最优模型信息"""
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
    except:
        return None


def format_time(seconds):
    """格式化时间"""
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        return f"{int(seconds // 60)}分{int(seconds % 60)}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分"


def print_progress_bar(current, total, width=50):
    """打印进度条"""
    progress = current / total
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    percentage = progress * 100
    return f"[{bar}] {percentage:.1f}%"


def display_training_status(model_dir, max_epoch=500, refresh_interval=5):
    """
    显示训练状态

    Args:
        model_dir: 模型保存目录
        max_epoch: 最大训练轮数
        refresh_interval: 刷新间隔（秒）
    """
    history_path = os.path.join(model_dir, 'training_history.npz')

    print("=" * 80)
    print("训练监控工具".center(80))
    print("=" * 80)
    print(f"模型目录: {model_dir}")
    print(f"刷新间隔: {refresh_interval}秒")
    print(f"按 Ctrl+C 退出")
    print("=" * 80)

    start_time = time.time()
    last_epoch = 0
    epoch_times = []

    try:
        while True:
            # 清屏（Windows: cls, Linux/Mac: clear）
            os.system('clear' if os.name == 'posix' else 'cls')

            print("=" * 80)
            print("训练监控 - Enhanced Seq2Seq Model".center(80))
            print("=" * 80)
            print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

            # 加载训练历史
            history = load_training_history(history_path)

            if history is None:
                print("\n⏳ 等待训练开始...")
                print("   训练历史文件尚未生成")
                print(f"\n   监控路径: {history_path}")
                time.sleep(refresh_interval)
                continue

            # 获取当前状态
            current_epoch = int(history['epochs'][-1])
            current_loss = history['total_losses'][-1]
            current_action_loss = history['action_losses'][-1]
            current_char_loss = history['char_losses'][-1]
            current_epsilon = history['epsilons'][-1]

            # 计算平均epoch时间
            if current_epoch > last_epoch:
                elapsed = time.time() - start_time
                epoch_times.append(elapsed / (current_epoch - last_epoch))
                last_epoch = current_epoch
                start_time = time.time()

            avg_epoch_time = np.mean(epoch_times[-10:]) if epoch_times else 0

            # 加载最优模型信息
            best_model_info = load_best_model_info(model_dir)

            # 显示进度
            print(f"\n📊 训练进度")
            print("-" * 80)
            progress_bar = print_progress_bar(current_epoch, max_epoch, width=60)
            print(f"   {progress_bar}")
            print(f"   Epoch: {current_epoch} / {max_epoch}")

            # 预估剩余时间
            if avg_epoch_time > 0:
                remaining_epochs = max_epoch - current_epoch
                estimated_time = remaining_epochs * avg_epoch_time
                print(f"   预估剩余时间: {format_time(estimated_time)}")
                print(f"   平均epoch时间: {format_time(avg_epoch_time)}")

            # 显示当前损失
            print(f"\n📉 当前损失 (Epoch {current_epoch})")
            print("-" * 80)
            print(f"   总损失:     {current_loss:.6f}")
            print(f"   动作损失:   {current_action_loss:.6f}")
            print(f"   文本损失:   {current_char_loss:.6f}")
            print(f"   Teacher Forcing Ratio (ε): {current_epsilon:.3f}")

            # 显示最优模型
            if best_model_info is not None:
                print(f"\n⭐ 最优模型")
                print("-" * 80)
                print(f"   Epoch:      {best_model_info['epoch']}")
                print(f"   总损失:     {best_model_info['loss']:.6f}")
                print(f"   动作损失:   {best_model_info['action_loss']:.6f}")
                print(f"   文本损失:   {best_model_info['char_loss']:.6f}")

                # 显示改进百分比
                if current_loss > best_model_info['loss']:
                    improvement = (current_loss - best_model_info['loss']) / best_model_info['loss'] * 100
                    print(f"   📈 当前loss比最优高 {improvement:.2f}%")
                else:
                    print(f"   🎉 这是新的最优模型!")

            # 显示损失趋势
            if len(history['epochs']) > 1:
                print(f"\n📈 损失趋势 (最近10个epoch)")
                print("-" * 80)

                recent_losses = history['total_losses'][-10:]
                if len(recent_losses) >= 2:
                    trend = recent_losses[-1] - recent_losses[0]
                    if trend < 0:
                        print(f"   ✅ 下降趋势: {abs(trend):.6f}")
                    elif trend > 0:
                        print(f"   ⚠️  上升趋势: {trend:.6f}")
                    else:
                        print(f"   ➡️  稳定")

                    # 简单的ASCII趋势图
                    print(f"\n   ")
                    max_loss = max(recent_losses)
                    min_loss = min(recent_losses)
                    range_loss = max_loss - min_loss if max_loss > min_loss else 1

                    for i, loss in enumerate(recent_losses):
                        normalized = (loss - min_loss) / range_loss
                        bar_len = int(normalized * 40)
                        bar = '█' * bar_len
                        print(f"   -{len(recent_losses) - i:2d}: {bar} {loss:.4f}")

            # 显示保存的模型
            print(f"\n💾 已保存的模型")
            print("-" * 80)
            saved_models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
            print(f"   检查点数量: {len(saved_models)}")
            if 'best_model.pth' in saved_models:
                print(f"   ✓ best_model.pth (Epoch {best_model_info['epoch']})")

            print("\n" + "=" * 80)
            print(f"下次更新: {refresh_interval}秒后 | 按 Ctrl+C 退出")
            print("=" * 80)

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n监控已停止")


def main():
    parser = argparse.ArgumentParser(description='训练监控工具')
    parser.add_argument('--model_dir', type=str, default='./seq2seq_enhanced_model',
                       help='模型保存目录')
    parser.add_argument('--max_epoch', type=int, default=500,
                       help='最大训练轮数')
    parser.add_argument('--refresh', type=int, default=5,
                       help='刷新间隔（秒）')

    args = parser.parse_args()

    display_training_status(args.model_dir, args.max_epoch, args.refresh)


if __name__ == "__main__":
    main()
