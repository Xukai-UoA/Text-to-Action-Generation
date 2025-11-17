"""
GAN模型增强测试脚本
功能：
1. 使用完整句子作为文件名（空格替换为下划线）
2. 彩色骨架渲染（头部红色，手臂绿色，颈部蓝色）
3. 支持随机噪声生成多样化动作
使用方法: python test_gan_enhanced.py
"""

import numpy as np
import torch
import scipy.io as scio
from model.gan_structure import GANModel
from utils.my_functions import load_w2v
from model.seq2seq_enhanced_visualization import visualize_action_complete
import os


class GANTester:
    """
    GAN测试器 - 用于加载训练好的GAN模型并生成动作序列
    """

    def __init__(self, model, init_pose, model_path,
                 sentence_steps, action_steps, dim_sentence,
                 dim_char_enc, dim_gen, dim_random,
                 device='cuda', use_random_noise=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.action_steps = action_steps
        self.dim_action = 24
        self.num_data = 1  # 测试时batch_size=1

        self.init_pose = init_pose
        self.batch_init = np.transpose(np.tile(self.init_pose, (1, self.num_data)), [1, 0])

        self.model_path = model_path
        self.sentence_steps = sentence_steps
        self.dim_sentence = dim_sentence
        self.dim_char_enc = dim_char_enc
        self.dim_gen = dim_gen
        self.dim_random = dim_random
        self.use_random_noise = use_random_noise

        # 加载训练好的模型
        self._load_model()

    def _load_model(self):
        """加载训练好的GAN模型权重"""
        print(f"Loading GAN model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # 设置为评估模式
        print(f"GAN model loaded successfully! (Epoch: {checkpoint.get('epoch', 'unknown')})")

    def test(self, test_script, test_script_len):
        """
        测试函数 - 从文本生成动作序列

        Args:
            test_script: [1, dim_sentence, sentence_steps] 输入文本embedding
            test_script_len: [1] 文本长度

        Returns:
            test_esti: [1, dim_action, action_steps] 生成的动作序列
        """
        with torch.no_grad():  # 测试时不需要梯度
            # 转换为torch tensor
            script_tensor = torch.FloatTensor(test_script).to(self.device)
            length_tensor = torch.LongTensor(test_script_len)

            # 转置以匹配模型输入格式 [batch, sentence_steps, dim_sentence]
            script_batch = script_tensor.transpose(1, 2)

            # 准备初始输入
            curr_init_input = torch.FloatTensor(self.batch_init).to(self.device)

            # 根据配置选择使用随机噪声或零噪声
            if self.use_random_noise:
                # GAN训练时使用随机噪声，可以生成多样化的动作
                curr_random = torch.randn(self.num_data, self.sentence_steps,
                                          self.dim_random).to(self.device)
                print("  使用随机噪声生成（多样化输出）")
            else:
                # 使用零噪声，生成确定性的结果
                curr_random = torch.zeros(self.num_data, self.sentence_steps,
                                          self.dim_random).to(self.device)
                print("  使用零噪声生成（确定性输出）")

            # 1. 编码文本
            char_enc_out = self.model.char_encoder(script_batch, length_tensor)

            # 2. 从文本生成动作（使用GAN的generator）
            action_gen_list = self.model.char2action(
                char_enc_out, curr_init_input, curr_random, self.num_data
            )

            # 将list转换为tensor [batch, action_steps, dim_action]
            action_gen_out = torch.stack(action_gen_list, dim=1)

            # 转换回numpy并转置为 [1, dim_action, action_steps]
            test_esti = action_gen_out.cpu().numpy().transpose(0, 2, 1)

        return test_esti


def test_with_enhanced_visualization(tester, w2v_model, sentence, output_dir='./test_results'):
    """
    使用增强可视化测试GAN模型

    Args:
        tester: GANTester对象
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
    print("GAN模型测试 - 完整文件名 + 彩色骨架")
    print("="*70)
    print("\n特性:")
    print("  ✨ 文件名使用完整句子（用下划线连接）")
    print("  🎨 头部关键点: 红色")
    print("  🎨 手臂/手部关键点: 绿色")
    print("  🎨 颈部关键点: 蓝色")
    print("  🎲 可选: 使用随机噪声生成多样化动作")

    # ==================== 配置 ====================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 模型参数
    sentence_steps = 30
    action_steps = 32
    dim_sentence = 300
    dim_char_enc = 300
    dim_gen = 300
    dim_dis = 300
    dim_random = 10

    # 路径
    model_path = './gan_model/model_epoch_500.pth'  # GAN模型路径
    mean_pose_path = './data/mean_pose.mat'
    w2v_path = './data/GoogleNews-vectors-negative300.bin'
    output_dir = './test_results'

    # 测试配置
    use_random_noise = False  # 设为True可以生成多样化的动作

    # ==================== 检查文件 ====================
    print("\n检查必要文件...")
    required_files = [
        (model_path, "训练好的GAN模型"),
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
    print("加载GAN模型和Word2Vec")
    print("="*70)

    print("\n创建GAN模型结构...")
    model = GANModel(
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_dis=dim_dis,
        dim_random=dim_random
    )

    print("加载初始pose...")
    init_pose = scio.loadmat(mean_pose_path)['mean_vector']

    print("创建GAN测试器...")
    tester = GANTester(
        model=model,
        init_pose=init_pose,
        model_path=model_path,
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random,
        device=device,
        use_random_noise=use_random_noise
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

    if use_random_noise:
        print("\n⚠️  注意: 使用随机噪声模式，每次运行会生成不同的动作!")
    else:
        print("\n✓ 使用确定性模式，每次运行结果一致")

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
    print("GAN模型特性说明")
    print("="*70)
    print("✨ 相比Seq2Seq模型的优势:")
    print("   ✓ 通过对抗训练提高动作质量")
    print("   ✓ 生成的动作更加自然和多样化")
    print("   ✓ 支持随机噪声生成不同变体")
    print("\n🎲 随机噪声模式:")
    print("   - 设置 use_random_noise=True 可以为同一句子生成多个不同的动作")
    print("   - 适合需要动作多样性的应用场景")
    print("   - 当前设置: use_random_noise=" + str(use_random_noise))

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
    print("4. 对比Seq2Seq和GAN的结果:")
    print("   - GAN生成的动作应该更加平滑")
    print("   - GAN生成的动作应该更符合物理规律")
    print("   - GAN生成的动作应该有更好的语义对齐")
    print("\n5. 如果想生成多个变体:")
    print("   - 设置 use_random_noise=True")
    print("   - 多次运行脚本获得不同结果")


if __name__ == "__main__":
    main()
