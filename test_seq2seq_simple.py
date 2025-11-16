"""
__init__.py 让目录变成 Python 包
简单的测试脚本 - 快速测试seq2seq预训练模型
使用方法: python -m test_seq2seq_simple

"""

import numpy as np
import torch
import scipy.io as scio
from model.seq2seq_structure import Seq2SeqModel
from model.seq2seq_tester import Seq2SeqTester, visualize_action_sequence, save_action_to_file
from utils.my_functions import load_w2v
import os


def test_with_sentence(tester, w2v_model, sentence, output_dir='./test_results'):
    """
    使用一个句子测试模型

    Args:
        tester: Seq2SeqTester对象
        w2v_model: Word2Vec模型
        sentence: 测试句子
        output_dir: 输出目录
    """
    print(f"\n{'=' * 60}")
    print(f"测试句子: '{sentence}'")
    print(f"{'=' * 60}")

    # 转换句子为embedding
    words = sentence.lower().split()
    test_script = np.zeros((1, 300, 30))  # [1, dim_sentence, sentence_steps]

    for i, word in enumerate(words[:30]):
        if word in w2v_model:
            test_script[0, :, i] = w2v_model[word]

    test_script_len = np.array([min(len(words), 30)])

    # 生成动作
    generated_action = tester.test(test_script, test_script_len)

    # 打印统计信息
    print(f"\n生成的动作序列:")
    print(f"  形状: {generated_action.shape}")
    print(f"  均值: {generated_action.mean():.4f}")
    print(f"  标准差: {generated_action.std():.4f}")
    print(f"  范围: [{generated_action.min():.4f}, {generated_action.max():.4f}]")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)

    # 文件名使用句子的前几个词
    filename_prefix = '_'.join(words[:3])

    # 保存numpy文件
    save_path = os.path.join(output_dir, f'{filename_prefix}_action.npy')
    save_action_to_file(generated_action, save_path)

    # 可视化
    viz_path = os.path.join(output_dir, f'{filename_prefix}_viz.png')
    visualize_action_sequence(
        generated_action,
        title=f"Generated: '{sentence}'",
        save_path=viz_path
    )

    return generated_action


def main():
    """主测试函数"""
    print("=" * 60)
    print("Seq2Seq预训练模型测试")
    print("=" * 60)

    # ==================== 配置 ====================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 模型参数
    sentence_steps = 30
    action_steps = 32
    dim_sentence = 300
    dim_char_enc = 300
    dim_gen = 300
    dim_random = 10

    # 路径
    model_path = './seq2seq_model/model_epoch_500.pth'
    mean_pose_path = './data/mean_pose.mat'
    w2v_path = './data/GoogleNews-vectors-negative300.bin'
    output_dir = './test_results'

    # ==================== 检查文件 ====================
    print("\n检查必要文件...")
    required_files = [
        (model_path, "训练好的模型"),
        (mean_pose_path, "平均pose"),
        (w2v_path, "Word2Vec模型")
    ]

    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {description}: {file_path}")
        else:
            print(f"  ✗ {description}未找到: {file_path}")
            print(f"    请确保文件存在后再运行测试")
            return

    # ==================== 加载模型 ====================
    print("\n" + "=" * 60)
    print("1. 加载模型")
    print("=" * 60)

    # 创建模型
    model = Seq2SeqModel(
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random
    )

    # 加载初始pose
    init_pose = scio.loadmat(mean_pose_path)['mean_vector']

    # 创建测试器
    tester = Seq2SeqTester(
        model=model,
        init_pose=init_pose,
        model_path=model_path,
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random,
        device=device
    )

    # 加载Word2Vec
    print("\n加载Word2Vec模型...")
    w2v_model = load_w2v(w2v_path)

    # ==================== 测试多个句子 ====================
    print("\n" + "=" * 60)
    print("2. 生成动作序列")
    print("=" * 60)

    # 定义测试句子
    test_sentences = [
        "a woman is dancing",
        "a man is lifting weights",
        "a person is waving hands",
        "someone is throwing a ball",
        "a girl is jumping"
    ]

    print(f"\n将测试 {len(test_sentences)} 个句子...")

    # 测试每个句子
    results = {}
    for sentence in test_sentences:
        try:
            action = test_with_sentence(tester, w2v_model, sentence, output_dir)
            results[sentence] = action
        except Exception as e:
            print(f"  ✗ 测试失败: {e}")
            continue

    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print(f"\n成功生成 {len(results)} 个动作序列")
    print(f"结果保存在: {output_dir}")
    print("\n生成的文件:")
    print(f"  - *.npy: 动作序列数据（可用于后续GAN训练）")
    print(f"  - *_viz.png: 动作可视化图像")

    print("\n下一步:")
    print("  1. 检查生成的可视化图像，看动作是否合理")
    print("  2. 如果效果好，可以开始GAN训练")
    print("  3. 使用这个预训练模型初始化GAN的Generator")


if __name__ == "__main__":
    main()