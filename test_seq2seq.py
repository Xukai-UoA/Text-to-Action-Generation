"""
优化版测试脚本
改进：
1. 使用完整句子作为文件名（空格替换为下划线）
2. 彩色骨架渲染（头部红色，手臂绿色，颈部蓝色）
使用方法: python enhanced_test_optimized.py
"""

import numpy as np
import torch
import scipy.io as scio
from model.seq2seq_structure import Seq2SeqModel
from model.seq2seq_tester import Seq2SeqTester
from utils.my_functions import load_w2v
from model.seq2seq_enhanced_visualization import visualize_action_complete
import os


def test_with_enhanced_visualization(tester, w2v_model, sentence, output_dir='./test_results'):
    """
    使用增强可视化测试模型（优化版）

    Args:
        tester: Seq2SeqTester对象
        w2v_model: Word2Vec模型
        sentence: 测试句子
        output_dir: 输出目录
    """
    print(f"\n{'='*70}")
    print(f"测试句子: '{sentence}'")
    print(f"{'='*70}")

    # 1. 转换句子为embedding
    print("\n[1/4] 编码句子...")
    words = sentence.lower().split()
    test_script = np.zeros((1, 300, 30))

    found_words = 0
    for i, word in enumerate(words[:30]):
        if word in w2v_model:
            test_script[0, :, i] = w2v_model[word]
            print(f"  ✓ '{word}'")
            found_words += 1
        else:
            print(f"  ⚠️  '{word}' (不在词汇表中，使用零向量)")

    test_script_len = np.array([min(len(words), 30)])
    print(f"  编码完成: {found_words}/{len(words)} 个词在词汇表中")

    # 2. 生成动作
    print(f"\n[2/4] 生成动作序列...")
    generated_action = tester.test(test_script, test_script_len)

    print(f"  ✓ 动作形状: {generated_action.shape}")
    print(f"  ✓ 均值: {generated_action.mean():.4f}")
    print(f"  ✓ 标准差: {generated_action.std():.4f}")
    print(f"  ✓ 范围: [{generated_action.min():.4f}, {generated_action.max():.4f}]")

    # 3. 保存.npy文件（使用完整句子名）
    print(f"\n[3/4] 保存动作数据...")
    os.makedirs(output_dir, exist_ok=True)

    # 使用完整句子，替换空格为下划线
    filename_prefix = sentence.lower().replace(' ', '_')
    npy_path = os.path.join(output_dir, f'{filename_prefix}_action.npy')

    np.save(npy_path, generated_action)
    print(f"  ✓ 动作数据保存到: {npy_path}")

    # 4. 生成所有可视化（彩色版）
    print(f"\n[4/4] 生成彩色可视化...")
    files = visualize_action_complete(generated_action, sentence, output_dir)

    return generated_action, files


def main():
    """主函数"""
    print("="*70)
    print("优化版Seq2Seq测试 - 完整文件名 + 彩色骨架")
    print("="*70)
    print("\n改进点:")
    print("  ✨ 文件名使用完整句子（用下划线连接）")
    print("  🎨 头部关键点: 红色")
    print("  🎨 手臂/手部关键点: 绿色")
    print("  🎨 颈部关键点: 蓝色")

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
            print(f"\n请确保以下文件存在:")
            print(f"  - {model_path}")
            print(f"  - {mean_pose_path}")
            print(f"  - {w2v_path}")
            return

    # ==================== 加载模型 ====================
    print("\n" + "="*70)
    print("加载模型和Word2Vec")
    print("="*70)

    print("\n创建模型结构...")
    model = Seq2SeqModel(
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random
    )

    print("加载初始pose...")
    init_pose = scio.loadmat(mean_pose_path)['mean_vector']

    print("创建测试器...")
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

    print("加载Word2Vec模型...")
    w2v_model = load_w2v(w2v_path)

    # ==================== 测试句子 ====================
    print("\n" + "="*70)
    print("开始测试")
    print("="*70)

    # 定义测试句子（可以是更长的描述）
    test_sentences = [
        "a woman is dancing",
        "a man is lifting weights",
        "a person is waving hands",
        "someone is throwing a ball",
        "a girl is jumping",
        "a man playing drums"
    ]

    print(f"\n将测试 {len(test_sentences)} 个句子")
    print("\n每个句子将生成 4 个文件:")
    print("  1. [完整句子]_8frames.png        - 8个关键帧（彩色）")
    print("  2. [完整句子]_animation.gif      - 32帧动画（彩色）")
    print("  3. [完整句子]_32frames_grid.png - 32帧网格图（彩色）")
    print("  4. [完整句子]_action.npy        - 动作数据")

    print("\n文件名示例:")
    for sent in test_sentences[:1]:
        example_name = sent.lower().replace(' ', '_')
        print(f"  '{sent}'")
        print(f"    → {example_name}_8frames.png")
        print(f"    → {example_name}_animation.gif")
        print(f"    → {example_name}_32frames_grid.png")
        print(f"    → {example_name}_action.npy")

    # 测试每个句子
    all_results = {}
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n\n{'#'*70}")
        print(f"测试 {i}/{len(test_sentences)}")
        print(f"{'#'*70}")

        try:
            action, files = test_with_enhanced_visualization(
                tester, w2v_model, sentence, output_dir
            )
            all_results[sentence] = {'action': action, 'files': files}
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ==================== 总结 ====================
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)

    print(f"\n✓ 成功生成 {len(all_results)} 个动作序列")
    print(f"✓ 结果保存在: {output_dir}")

    print("\n生成的文件:")
    total_files = 0
    for sentence, result in all_results.items():
        print(f"\n  '{sentence}':")
        for file_type, path in result['files'].items():
            if os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024
                filename = os.path.basename(path)
                print(f"    ✓ {filename} ({size_kb:.1f} KB)")
                total_files += 1
            else:
                print(f"    ✗ {file_type}: 未生成")
        # 显示.npy文件
        npy_name = sentence.lower().replace(' ', '_') + '_action.npy'
        npy_path = os.path.join(output_dir, npy_name)
        if os.path.exists(npy_path):
            size_kb = os.path.getsize(npy_path) / 1024
            print(f"    ✓ {npy_name} ({size_kb:.1f} KB)")
            total_files += 1

    print(f"\n总共生成: {total_files} 个文件")

    print("\n" + "="*70)
    print("文件名优化说明")
    print("="*70)
    print("✨ 旧版文件名 (只用前3个词):")
    print("   a_woman_is_8frames.png")
    print("   a_woman_is_animation.gif")
    print("\n✨ 新版文件名 (使用完整句子):")
    print("   a_woman_is_dancing_gracefully_8frames.png")
    print("   a_woman_is_dancing_gracefully_animation.gif")
    print("\n优点:")
    print("  ✓ 文件名更具描述性")
    print("  ✓ 避免不同句子文件名冲突")
    print("  ✓ 更容易管理和查找")

    print("\n" + "="*70)
    print("颜色编码说明")
    print("="*70)
    print("🔴 红色: 头部关键点 (joint 7)")
    print("🟢 绿色: 手臂和手部关键点 (joints 1-6)")
    print("🔵 蓝色: 颈部关键点 (joint 0, 基准点)")
    print("\n优点:")
    print("  ✓ 更容易识别不同身体部位")
    print("  ✓ 视觉上更直观")
    print("  ✓ 便于分析动作质量")

    print("\n" + "="*70)
    print("使用建议")
    print("="*70)
    print("1. 查看 *_8frames.png 快速预览动作")
    print("2. 播放 *_animation.gif 观察流畅度")
    print("3. 检查 *_32frames_grid.png 分析细节")
    print("4. 注意观察:")
    print("   - 红色头部是否移动合理")
    print("   - 绿色手臂动作是否自然")
    print("   - 蓝色颈部作为稳定的参考点")
    print("\n5. 如果效果好，可以开始GAN训练！")


if __name__ == "__main__":
    main()
