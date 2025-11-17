"""
数据集质量检查工具

功能：
1. 读取 .mat 格式的动作数据和 .txt 格式的文本描述
2. 可视化每个动作序列（生成.gif动画）
3. 检查文本-动作对的质量
4. 生成HTML报告方便查看

使用方法:
    # 检查所有数据
    python data_check.py --mode all --max_samples 10

    # 检查特定样本
    python data_check.py --mode single --sample_id 0001

    # 随机采样检查
    python data_check.py --mode random --num_samples 5
"""

import numpy as np
import scipy.io as scio
import os
import argparse
from pathlib import Path
from model.seq2seq_enhanced_visualization import visualize_action_complete
import glob
from tqdm import tqdm


class DatasetChecker:
    """数据集质量检查器"""

    def __init__(self, pose_dir='./data/pose', script_dir='./data/script',
                 output_dir='./data_check_results'):
        """
        Args:
            pose_dir: pose文件目录
            script_dir: script文件目录
            output_dir: 输出目录
        """
        self.pose_dir = pose_dir
        self.script_dir = script_dir
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # 获取所有pose文件
        self.pose_files = sorted(glob.glob(os.path.join(pose_dir, 'pose_*.mat')))
        print(f"✓ 找到 {len(self.pose_files)} 个pose文件")

        # 获取所有script文件
        self.script_files = sorted(glob.glob(os.path.join(script_dir, 'script_*.txt')))
        print(f"✓ 找到 {len(self.script_files)} 个script文件")

    def extract_id_from_filename(self, filename):
        """从文件名中提取ID (例如: pose_0001.mat -> 0001)"""
        basename = os.path.basename(filename)
        # 提取数字部分
        id_str = basename.split('_')[1].split('.')[0]
        return id_str

    def load_pose(self, pose_file):
        """
        加载pose文件

        Args:
            pose_file: .mat文件路径

        Returns:
            pose_data: [dim_action, action_steps] 动作数据
        """
        try:
            mat_data = scio.loadmat(pose_file)
            # 尝试常见的字段名
            for key in ['pred_vector', 'pose_vector', 'action_vector', 'pose', 'action']:
                if key in mat_data:
                    pose_data = mat_data[key]
                    return pose_data

            # 如果没有找到，打印所有可用的key
            print(f"⚠️  未找到标准pose字段，可用字段: {list(mat_data.keys())}")
            # 返回第一个非元数据字段
            for key in mat_data.keys():
                if not key.startswith('__'):
                    pose_data = mat_data[key]
                    print(f"  使用字段: {key}")
                    return pose_data

            return None
        except Exception as e:
            print(f"✗ 加载pose文件失败: {pose_file}")
            print(f"  错误: {e}")
            return None

    def load_scripts(self, script_file):
        """
        加载script文件

        Args:
            script_file: .txt文件路径

        Returns:
            scripts: 文本描述列表
        """
        try:
            with open(script_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 清理每行文本
            scripts = [line.strip() for line in lines if line.strip()]
            return scripts
        except Exception as e:
            print(f"✗ 加载script文件失败: {script_file}")
            print(f"  错误: {e}")
            return []

    def check_single_sample(self, sample_id):
        """
        检查单个样本

        Args:
            sample_id: 样本ID (例如: '0001')
        """
        print(f"\n{'='*70}")
        print(f"检查样本 ID: {sample_id}")
        print(f"{'='*70}")

        # 查找对应的文件
        pose_file = os.path.join(self.pose_dir, f'pose_{sample_id}.mat')
        script_file = os.path.join(self.script_dir, f'script_{sample_id}.txt')

        # 检查文件是否存在
        if not os.path.exists(pose_file):
            print(f"✗ Pose文件不存在: {pose_file}")
            return None

        if not os.path.exists(script_file):
            print(f"✗ Script文件不存在: {script_file}")
            return None

        print(f"✓ Pose文件: {pose_file}")
        print(f"✓ Script文件: {script_file}")

        # 加载pose数据
        print(f"\n[1/3] 加载动作数据...")
        pose_data = self.load_pose(pose_file)
        if pose_data is None:
            return None

        print(f"  ✓ 动作形状: {pose_data.shape}")
        print(f"  ✓ 均值: {pose_data.mean():.4f}")
        print(f"  ✓ 标准差: {pose_data.std():.4f}")
        print(f"  ✓ 范围: [{pose_data.min():.4f}, {pose_data.max():.4f}]")

        # 检查数据维度
        if pose_data.ndim == 2:
            # 期望格式: [dim_action, action_steps]
            print(f"  ✓ 数据维度正确: [dim_action={pose_data.shape[0]}, action_steps={pose_data.shape[1]}]")
        else:
            print(f"  ⚠️  数据维度异常: {pose_data.shape}")

        # 加载文本描述
        print(f"\n[2/3] 加载文本描述...")
        scripts = self.load_scripts(script_file)
        print(f"  ✓ 找到 {len(scripts)} 条文本描述:")
        for i, script in enumerate(scripts, 1):
            print(f"    {i}. {script}")

        # 可视化每个文本-动作对
        print(f"\n[3/3] 生成可视化...")
        sample_output_dir = os.path.join(self.output_dir, f'sample_{sample_id}')
        os.makedirs(sample_output_dir, exist_ok=True)

        results = []
        for i, script in enumerate(scripts):
            print(f"\n  生成第 {i+1}/{len(scripts)} 个可视化: '{script}'")

            # 确保pose_data是3D: [1, dim_action, action_steps]
            if pose_data.ndim == 2:
                pose_data_3d = pose_data[np.newaxis, :, :]
            else:
                pose_data_3d = pose_data

            # 生成可视化文件
            try:
                files = visualize_action_complete(
                    pose_data_3d,
                    f"{sample_id}_{i+1}_{script}",
                    sample_output_dir
                )
                results.append({
                    'sample_id': sample_id,
                    'script': script,
                    'files': files,
                    'pose_shape': pose_data.shape,
                    'pose_stats': {
                        'mean': float(pose_data.mean()),
                        'std': float(pose_data.std()),
                        'min': float(pose_data.min()),
                        'max': float(pose_data.max())
                    }
                })
            except Exception as e:
                print(f"  ✗ 可视化失败: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*70}")
        print(f"✓ 样本 {sample_id} 检查完成")
        print(f"  输出目录: {sample_output_dir}")
        print(f"{'='*70}")

        return results

    def check_all_samples(self, max_samples=None):
        """
        检查所有样本

        Args:
            max_samples: 最多检查的样本数（None表示检查所有）
        """
        print(f"\n{'#'*70}")
        print(f"批量检查数据集")
        print(f"{'#'*70}")

        # 提取所有pose文件的ID
        sample_ids = [self.extract_id_from_filename(f) for f in self.pose_files]

        if max_samples:
            sample_ids = sample_ids[:max_samples]
            print(f"✓ 将检查前 {len(sample_ids)} 个样本")
        else:
            print(f"✓ 将检查所有 {len(sample_ids)} 个样本")

        all_results = []

        # 使用tqdm显示进度
        for sample_id in tqdm(sample_ids, desc="检查样本"):
            try:
                results = self.check_single_sample(sample_id)
                if results:
                    all_results.extend(results)
            except Exception as e:
                print(f"\n✗ 样本 {sample_id} 检查失败: {e}")
                continue

        # 生成汇总报告
        self.generate_summary_report(all_results)

        return all_results

    def check_random_samples(self, num_samples=5):
        """
        随机检查若干样本

        Args:
            num_samples: 检查的样本数
        """
        print(f"\n{'#'*70}")
        print(f"随机检查 {num_samples} 个样本")
        print(f"{'#'*70}")

        # 提取所有pose文件的ID
        sample_ids = [self.extract_id_from_filename(f) for f in self.pose_files]

        # 随机选择
        if len(sample_ids) > num_samples:
            import random
            selected_ids = random.sample(sample_ids, num_samples)
        else:
            selected_ids = sample_ids

        print(f"✓ 随机选择的样本ID: {selected_ids}")

        all_results = []
        for sample_id in selected_ids:
            try:
                results = self.check_single_sample(sample_id)
                if results:
                    all_results.extend(results)
            except Exception as e:
                print(f"\n✗ 样本 {sample_id} 检查失败: {e}")
                continue

        # 生成汇总报告
        self.generate_summary_report(all_results)

        return all_results

    def generate_summary_report(self, all_results):
        """
        生成HTML汇总报告

        Args:
            all_results: 所有检查结果
        """
        print(f"\n{'='*70}")
        print(f"生成汇总报告")
        print(f"{'='*70}")

        report_path = os.path.join(self.output_dir, 'summary_report.html')

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>数据集质量检查报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #666;
            margin-top: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .sample {{
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            border-left: 4px solid #2196F3;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .script {{
            color: #1976D2;
            font-weight: bold;
            font-size: 16px;
            margin: 10px 0;
        }}
        .stats {{
            background: #f9f9f9;
            padding: 10px;
            border-radius: 3px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 14px;
        }}
        .visualization {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }}
        .viz-item {{
            flex: 1;
            min-width: 300px;
        }}
        .viz-item img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .viz-item .caption {{
            text-align: center;
            margin-top: 5px;
            font-size: 12px;
            color: #666;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <h1>📊 数据集质量检查报告</h1>

    <div class="summary">
        <h2>总体统计</h2>
        <p><strong>检查样本数:</strong> {len(all_results)}</p>
        <p><strong>生成时间:</strong> {self._get_current_time()}</p>
        <p><strong>输出目录:</strong> {self.output_dir}</p>
    </div>
"""

        # 添加每个样本的详细信息
        for i, result in enumerate(all_results, 1):
            sample_id = result['sample_id']
            script = result['script']
            pose_shape = result['pose_shape']
            stats = result['pose_stats']
            files = result['files']

            html_content += f"""
    <div class="sample">
        <h2>样本 #{i}: {sample_id}</h2>
        <div class="script">📝 "{script}"</div>

        <div class="stats">
            <strong>动作数据统计:</strong><br>
            形状: {pose_shape}<br>
            均值: {stats['mean']:.4f}<br>
            标准差: {stats['std']:.4f}<br>
            范围: [{stats['min']:.4f}, {stats['max']:.4f}]
        </div>

        <div class="visualization">
"""

            # 添加可视化文件
            for file_type, file_path in files.items():
                if os.path.exists(file_path):
                    rel_path = os.path.relpath(file_path, self.output_dir)
                    file_size = os.path.getsize(file_path) / 1024

                    if file_type == 'animation':
                        # GIF动画
                        html_content += f"""
            <div class="viz-item">
                <img src="{rel_path}" alt="{file_type}">
                <div class="caption">🎬 动画 ({file_size:.1f} KB)</div>
            </div>
"""
                    elif file_type == 'key_frames':
                        # 关键帧
                        html_content += f"""
            <div class="viz-item">
                <img src="{rel_path}" alt="{file_type}">
                <div class="caption">🖼️ 8个关键帧 ({file_size:.1f} KB)</div>
            </div>
"""

            html_content += """
        </div>
    </div>
"""

        html_content += """
</body>
</html>
"""

        # 保存HTML文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✓ 汇总报告已生成: {report_path}")
        print(f"  请在浏览器中打开查看")

    def _get_current_time(self):
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据集质量检查工具')
    parser.add_argument('--mode', type=str, default='random',
                       choices=['all', 'single', 'random'],
                       help='检查模式: all(所有), single(单个), random(随机)')
    parser.add_argument('--sample_id', type=str, default='0001',
                       help='单个样本模式下的样本ID (例如: 0001)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='随机模式下检查的样本数')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='all模式下最多检查的样本数')
    parser.add_argument('--pose_dir', type=str, default='./data/pose',
                       help='pose文件目录')
    parser.add_argument('--script_dir', type=str, default='./data/script',
                       help='script文件目录')
    parser.add_argument('--output_dir', type=str, default='./data_check_results',
                       help='输出目录')

    args = parser.parse_args()

    print("="*70)
    print("数据集质量检查工具")
    print("="*70)
    print(f"模式: {args.mode}")
    print(f"Pose目录: {args.pose_dir}")
    print(f"Script目录: {args.script_dir}")
    print(f"输出目录: {args.output_dir}")

    # 检查目录是否存在
    if not os.path.exists(args.pose_dir):
        print(f"\n✗ Pose目录不存在: {args.pose_dir}")
        print(f"请确保数据目录存在，或使用 --pose_dir 参数指定正确路径")
        return

    if not os.path.exists(args.script_dir):
        print(f"\n✗ Script目录不存在: {args.script_dir}")
        print(f"请确保数据目录存在，或使用 --script_dir 参数指定正确路径")
        return

    # 创建检查器
    checker = DatasetChecker(
        pose_dir=args.pose_dir,
        script_dir=args.script_dir,
        output_dir=args.output_dir
    )

    # 根据模式执行检查
    if args.mode == 'single':
        print(f"\n单个样本检查模式: {args.sample_id}")
        checker.check_single_sample(args.sample_id)

    elif args.mode == 'all':
        print(f"\n所有样本检查模式")
        checker.check_all_samples(max_samples=args.max_samples)

    elif args.mode == 'random':
        print(f"\n随机样本检查模式: {args.num_samples} 个样本")
        checker.check_random_samples(num_samples=args.num_samples)

    print("\n" + "="*70)
    print("检查完成！")
    print("="*70)
    print(f"\n查看结果:")
    print(f"  1. 打开 {os.path.join(args.output_dir, 'summary_report.html')} 查看汇总报告")
    print(f"  2. 检查 {args.output_dir} 目录下的各个样本文件夹")
    print(f"\n提示:")
    print(f"  - 🎬 观看 *_animation.gif 查看动作流畅度")
    print(f"  - 🖼️ 查看 *_8frames.png 快速预览关键帧")
    print(f"  - 🔍 检查文本描述是否与动作匹配")
    print(f"  - 📊 注意动作数据的统计信息（均值、标准差、范围）")


if __name__ == "__main__":
    main()
