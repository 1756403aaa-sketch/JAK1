#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取smiles.xlsx文件中的分子，使用AiZynthFinder搜索合成路径
（修复版本，正确保存总路线和可合成路线数量）
"""

import pandas as pd
import subprocess
import json
import os
from pathlib import Path
import sys
import re
import time

# 配置参数
EXCEL_FILE = "smiles.xlsx"
CONFIG_FILE = "my_data_folder/config.yml"
OUTPUT_DIR = "synthesis_results"


def read_smiles_from_excel(excel_file):
    """读取Excel文件中的SMILES"""
    print(f"📖 正在读取 {excel_file}...")

    try:
        df = pd.read_excel(excel_file, sheet_name='Sheet1')
        print(f"✅ 成功读取 {len(df)} 个分子")
        print("\n分子列表：")
        # 限制显示宽度，避免输出过长
        pd.set_option('display.max_colwidth', 50)
        print(df.to_string(index=False))
        pd.set_option('display.max_colwidth', None)  # 重置
        return df
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {excel_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 读取Excel文件时出错：{e}")
        sys.exit(1)


def clean_smiles(smiles):
    """清理SMILES字符串，去除特殊字符"""
    if pd.isna(smiles):
        return ""
    smiles = str(smiles).replace('\n', '').replace('\r', '').strip()
    return smiles


def parse_aizynth_output(output_text):
    """解析aizynthcli的输出文本，提取统计信息"""
    stats = {}
    try:
        if "search time:" in output_text:
            match = re.search(r'search time:\s*([\d.]+)', output_text)
            if match:
                stats['search_time'] = float(match.group(1))

        if "number of routes:" in output_text:
            match = re.search(r'number of routes:\s*(\d+)', output_text)
            if match:
                stats['number_of_routes'] = int(match.group(1))

        if "number of solved routes:" in output_text:
            match = re.search(r'number of solved routes:\s*(\d+)', output_text)
            if match:
                stats['number_of_solved_routes'] = int(match.group(1))

        if "top score:" in output_text:
            match = re.search(r'top score:\s*([\d.]+)', output_text)
            if match:
                stats['top_score'] = float(match.group(1))

        if "is solved:" in output_text:
            match = re.search(r'is solved:\s*(\w+)', output_text)
            if match:
                stats['is_solved'] = match.group(1).lower() == 'true'

        if "number of steps:" in output_text:
            match = re.search(r'number of steps:\s*(\d+)', output_text)
            if match:
                stats['min_number_of_steps'] = int(match.group(1))

    except Exception as e:
        print(f"   ⚠️ 解析统计信息时出错：{e}")

    return stats


def analyze_json_results(json_file):
    """分析JSON结果文件，提取路径信息"""
    if not os.path.exists(json_file):
        return None

    try:
        with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            return None

        # 找到得分最高的路径
        highest_score = -float('inf')
        highest_score_route = None

        for route in data:
            score = route.get('scores', {}).get('state score', 0)
            if score > highest_score:
                highest_score = score
                highest_score_route = route

        if highest_score_route is None:
            return None

        # 提取路径信息
        path_info = {
            'highest_score': highest_score,
            'number_of_steps': 0,
            'number_of_precursors': 0,
            'precursors_in_stock': 0
        }

        # 从scores对象中提取信息
        scores = highest_score_route.get('scores', {})
        if scores:
            path_info['number_of_steps'] = scores.get('number of reactions', 0)
            path_info['number_of_precursors'] = scores.get('number of pre-cursors', 0)
            path_info['precursors_in_stock'] = scores.get('number of pre-cursors in stock', 0)

        # 添加调试信息
        print(f"   📊 路径分析调试信息:")
        print(f"     • 路径得分: {path_info['highest_score']:.4f}")
        print(f"     • 反应数量: {path_info['number_of_steps']}")
        print(f"     • 前体数量: {path_info['number_of_precursors']}")
        print(f"     • 库存前体: {path_info['precursors_in_stock']}")

        return path_info

    except Exception as e:
        print(f"   ⚠️ 分析JSON文件时出错：{e}")
        print(f"   错误详情: {str(e)}")
        return None


def run_aizynthcli(smiles, mol_id, output_path):
    """为单个分子运行AiZynthFinder"""
    print(f"\n{'=' * 60}")
    print(f"🔬 正在处理分子: {mol_id}")
    smiles_display = smiles[:60] + '...' if len(smiles) > 60 else smiles
    print(f"🧪 SMILES: {smiles_display}")
    print(f"{'=' * 60}")

    # 清理SMILES
    smiles_clean = clean_smiles(smiles)

    if not smiles_clean:
        print(f"❌ SMILES为空，跳过此分子")
        return False

    # 构建命令
    cmd = [
        "aizynthcli",
        "--config", CONFIG_FILE,
        "--smiles", smiles_clean,
        "--output", output_path,
        "--log_to_file"
    ]

    try:
        # 运行命令，使用UTF-8编码并忽略错误
        print("🔄 正在搜索合成路径...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # 替换无法解码的字符
            timeout=600  # 10分钟超时
        )

        # 同时也打印原始错误（如果有）
        if result.returncode != 0:
            print(f"❌ 搜索失败（退出码: {result.returncode}）")
            print(f"标准输出:\n{result.stdout}")
            print(f"错误输出:\n{result.stderr}")
            return False
        else:
            print(f"✅ 命令执行成功")

        # 解析输出中的统计信息
        stats = parse_aizynth_output(result.stdout)

        # 显示提取的统计信息
        if stats:
            print(f"\n📊 结果摘要:")
            if 'search_time' in stats:
                print(f"   • 搜索时间: {stats['search_time']:.2f}秒")
            if 'number_of_routes' in stats:
                print(f"   • 总路线数量: {stats['number_of_routes']}")
            if 'number_of_solved_routes' in stats:
                print(f"   • 可合成路线数量: {stats['number_of_solved_routes']}")
            if 'top_score' in stats:
                print(f"   • 最佳得分: {stats['top_score']:.4f}")
            if 'min_number_of_steps' in stats:
                print(f"   • 最少步数: {stats['min_number_of_steps']}")
            if 'is_solved' in stats:
                solved_status = "✅ 已解决" if stats['is_solved'] else "❌ 未完全解决"
                print(f"   • 解决状态: {solved_status}")

        # 分析JSON文件获取详细路径信息
        path_info = analyze_json_results(output_path)
        if path_info:
            print(f"\n📊 路径分析:")
            print(f"   • 最高路径得分: {path_info['highest_score']:.4f}")
            print(f"   • 最高路径步数: {path_info['number_of_steps']}")
            print(f"   • 最高路径底物数量: {path_info['number_of_precursors']}")
            print(f"   • 底物在库中数量: {path_info['precursors_in_stock']}")
            print(
                f"   • 底物在库中比例: {path_info['precursors_in_stock']}/{path_info['number_of_precursors'] if path_info['number_of_precursors'] > 0 else 0}")

        # 尝试读取JSON文件（可选，用于更详细的分析）
        try:
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # 尝试读取文件内容，但不强制解析JSON
                    content = f.read()
                    print(f"   📄 结果文件大小: {len(content)} 字节")
            else:
                print(f"   ⚠️ 结果文件未生成: {output_path}")
        except Exception as e:
            print(f"   ⚠️ 读取结果文件时出错：{e}")

        return True

    except subprocess.TimeoutExpired:
        print(f"⏰ 超时：搜索时间超过10分钟")
        return False
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        import traceback
        traceback.print_exc()
        return False


def generate_summary_report(results_df, output_dir):
    """生成汇总报告"""
    print(f"\n{'=' * 60}")
    print("📋 生成汇总报告...")
    print(f"{'=' * 60}\n")

    report_path = os.path.join(output_dir, "summary_report.txt")

    # 计算统计信息
    total = len(results_df)
    successful = len(results_df[results_df['success']])
    failed = total - successful
    success_rate = (successful / total * 100) if total > 0 else 0

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("AiZynthFinder 合成路径搜索汇总报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"处理时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置文件: {CONFIG_FILE}\n")
        f.write(f"输入文件: {EXCEL_FILE}\n")
        f.write(f"输出目录: {output_dir}\n\n")

        f.write("-" * 70 + "\n")
        f.write("统计摘要:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  总分子数:     {total}\n")
        f.write(f"  成功处理:     {successful}\n")
        f.write(f"  失败处理:     {failed}\n")
        f.write(f"  成功率:       {success_rate:.1f}%\n\n")

        f.write("-" * 70 + "\n")
        f.write("详细结果:\n")
        f.write("-" * 70 + "\n\n")

        for idx, row in results_df.iterrows():
            mol_id = row['ID']
            smiles = row['SMILES']
            success = row['success']
            result_file = row['result_file']
            number_of_routes = row['number_of_routes']
            number_of_solved_routes = row['number_of_solved_routes']
            highest_score = row['highest_score']
            number_of_steps = row['number_of_steps']
            number_of_precursors = row['number_of_precursors']
            precursors_in_stock = row['precursors_in_stock']

            status = "✅ 成功" if success else "❌ 失败"
            f.write(f"{mol_id}:\n")
            f.write(f"  SMILES: {smiles[:80]}\n")
            if len(smiles) > 80:
                f.write(f"          ...{smiles[-20:]}\n")
            f.write(f"  状态: {status}\n")
            f.write(f"  结果文件: {result_file}\n")
            f.write(f"  总路线数量: {number_of_routes}\n")
            f.write(f"  可合成路线数量: {number_of_solved_routes}\n")
            f.write(f"  最高路径得分: {highest_score:.4f}\n")
            f.write(f"  最高路径步数: {number_of_steps}\n")
            f.write(f"  最高路径底物数量: {number_of_precursors}\n")
            f.write(f"  底物在库中数量: {precursors_in_stock}\n\n")

        f.write("=" * 70 + "\n")
        f.write("报告结束\n")
        f.write("=" * 70 + "\n")

    print(f"📄 汇总报告已保存: {report_path}")

    # 打印到控制台
    with open(report_path, 'r', encoding='utf-8') as f:
        print(f.read())

    return report_path


def save_path_analysis_to_csv(results_df, output_dir):
    """将路径分析结果保存为CSV"""
    print(f"\n{'=' * 60}")
    print("💾 保存路径分析结果到CSV...")
    print(f"{'=' * 60}\n")

    csv_path = os.path.join(output_dir, "path_analysis.csv")

    # 选择需要的列
    analysis_df = results_df[[
        'ID', 'SMILES', 'number_of_routes', 'number_of_solved_routes',
        'highest_score', 'number_of_steps', 'number_of_precursors', 'precursors_in_stock'
    ]].copy()

    # 保存为CSV
    analysis_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"✅ 路径分析结果已保存到: {csv_path}")

    # 打印前几行查看
    print("\n📊 路径分析结果预览:")
    print(analysis_df.head().to_string(index=False))

    return csv_path


def main():
    """主函数"""
    print("🚀 AiZynthFinder 批量分子处理工具（修复版本）")
    print("=" * 60)

    # 检查配置文件
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ 错误：找不到配置文件 {CONFIG_FILE}")
        print("请确保已经运行 download_public_data 下载数据文件")
        sys.exit(1)

    # 读取Excel文件
    df = read_smiles_from_excel(EXCEL_FILE)

    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    print(f"\n📁 结果将保存到: {OUTPUT_DIR}/")

    # 添加结果列，设置默认值
    df['success'] = False
    df['result_file'] = ''
    df['number_of_routes'] = 0
    df['number_of_solved_routes'] = 0
    df['highest_score'] = 0.0
    df['number_of_steps'] = 0
    df['number_of_precursors'] = 0
    df['precursors_in_stock'] = 0

    # 记录开始时间
    start_time = time.time()

    # 处理每个分子
    total_molecules = len(df)
    for idx, row in df.iterrows():
        mol_id = row['ID']
        smiles = row['SMILES']

        print(f"\n📍 进度: [{idx + 1}/{total_molecules}]")

        # 为每个分子创建唯一的结果文件名
        safe_id = mol_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        output_file = os.path.join(OUTPUT_DIR, f"{safe_id}.json")

        # 运行搜索
        success = run_aizynthcli(smiles, mol_id, output_file)

        # 更新结果
        df.at[idx, 'success'] = success
        df.at[idx, 'result_file'] = output_file if success else 'FAILED'

        # 如果搜索成功，分析JSON文件获取路径信息
        if success and os.path.exists(output_file):
            path_info = analyze_json_results(output_file)
            if path_info:
                # 安全地更新数据，使用get方法获取值
                df.at[idx, 'number_of_solved_routes'] = path_info.get('number_of_solved_routes', 0)
                df.at[idx, 'highest_score'] = path_info.get('highest_score', 0.0)
                df.at[idx, 'number_of_steps'] = path_info.get('number_of_steps', 0)
                df.at[idx, 'number_of_precursors'] = path_info.get('number_of_precursors', 0)
                df.at[idx, 'precursors_in_stock'] = path_info.get('precursors_in_stock', 0)

        # 从输出文本中提取总路线数量
        if success:
            # 重新解析输出文本获取总路线数量
            with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                stats = parse_aizynth_output(content)
                if stats and 'number_of_routes' in stats:
                    df.at[idx, 'number_of_routes'] = stats['number_of_routes']

        # 计算已用时间
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / (idx + 1)
        remaining_time = avg_time * (total_molecules - idx - 1)

        print(f"⏱️  已用时间: {elapsed_time / 60:.1f}分钟, 预计剩余: {remaining_time / 60:.1f}分钟")

        # 短暂暂停，避免系统过载
        if idx < total_molecules - 1:  # 不是最后一个
            print(f"⏸️  等待2秒后处理下一个分子...")
            time.sleep(2)

    # 计算总时间
    total_time = time.time() - start_time
    print(f"\n🏁 总处理时间: {total_time / 60:.2f}分钟")

    # 生成汇总报告
    report_path = generate_summary_report(df, OUTPUT_DIR)

    # 保存路径分析结果到CSV
    csv_path = save_path_analysis_to_csv(df, OUTPUT_DIR)

    # 保存结果Excel
    result_excel = os.path.join(OUTPUT_DIR, "results_summary.xlsx")
    try:
        df.to_excel(result_excel, index=False, engine='openpyxl')
        print(f"📊 结果摘要已保存到Excel: {result_excel}")
    except Exception as e:
        print(f"⚠️ 保存Excel文件时出错: {e}")

    print("\n" + "=" * 60)
    print("🎉 所有分子处理完成！")
    print("=" * 60)

    # 提供后续建议
    successful_count = len(df[df['success']])
    if successful_count > 0:
        print(f"\n💡 提示:")
        print(f"   • 成功处理了 {successful_count}/{total_molecules} 个分子")
        print(f"   • 使用Web界面可视化结果:")
        print(f"     aizynthapp --config {CONFIG_FILE}")
        print(f"   • 查看详细报告: {report_path}")
        print(f"   • 路径分析结果: {csv_path}")
        print(f"   • 结果文件保存在: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
