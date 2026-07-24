"""
KPGT 回归微调一键执行脚本
===========================
使用前请确保：
1. 已激活 KPGT 的 conda 环境
2. 在 KPGT 仓库根目录下运行
3. 已将 train.csv, val.csv, test.csv 放入 raw_data/ 文件夹
"""

import sys
import os
import subprocess
import pandas as pd
import numpy as np

# ==========================================
#  项目根目录（自动获取绝对路径）
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==========================================
#  配置区（根据需要修改）
# ==========================================
DATASET_NAME = "mydata"          # 数据集名称（会创建在 datasets/ 下）
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
PRETRAIN_MODEL = os.path.join(PROJECT_ROOT, "src", "model", "pretrained", "base", "base.pth")
SPLIT_NAME = "my_split"

# ============ 训练超参数（已优化防过拟合） ============
N_EPOCHS = 100                # 减少轮次，避免后期过拟合
BATCH_SIZE = 32              # 减小 batch size，增加梯度噪声（隐式正则化）
LR = 3e-5                    # 降低学习率，平缓收敛
WEIGHT_DECAY = 1e-4          # 加大权重衰减，抑制参数过大
DROPOUT = 0.05                # 添加 10% dropout，防止神经元共适应

# L2SP 正则化（惩罚偏离预训练权重的程度，有效防止过拟合）
USE_L2SP = True
L2SP_WEIGHT = 0.001           # 正则化权重（越大越约束，可试 0.001 / 0.01 / 0.1）

# ============ 路径配置 ============
SAVE_DIR = os.path.join(PROJECT_ROOT, "outputs", f"{DATASET_NAME}_regression")
PREPROCESS_LOG = os.path.join(SAVE_DIR, "preprocess.log")
FINETUNE_LOG = os.path.join(SAVE_DIR, "finetune.log")
RESULT_FILE = os.path.join(SAVE_DIR, "result_summary.txt")


def run_command(cmd, description, log_file=None):
    """执行 shell 命令，实时逐行打印输出（进度条可见），同时保存到日志文件"""
    print(f"\n{'='*60}")
    print(f"⚡ {description}")
    print(f"{'='*60}")
    print(f"📋 命令: {cmd}")
    print("-" * 60)

    log_fh = None
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_fh = open(log_file, "w", encoding="utf-8")

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    while True:
        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
            break
        if line:
            line = line.rstrip("\n")
            print(line)
            if log_fh:
                log_fh.write(line + "\n")
                log_fh.flush()

    if log_fh:
        log_fh.close()

    if process.returncode != 0:
        print(f"\n❌ {description} 失败！请检查上方错误信息。")
        if log_file:
            print(f"📄 详细日志已保存到: {log_file}")
        sys.exit(1)

    print(f"\n✅ {description} 完成！")
    if log_file:
        print(f"📄 日志已保存到: {log_file}")
    return process.returncode


def check_preprocess_done():
    """检查预处理产物是否已全部生成，已存在则跳过"""
    dataset_dir = os.path.join(DATASETS_DIR, DATASET_NAME)
    required_files = [
        os.path.join(dataset_dir, f"{DATASET_NAME}_5.pkl"),
        os.path.join(dataset_dir, "rdkfp1-7_512.npz"),
        os.path.join(dataset_dir, "molecular_descriptors.npz"),
    ]
    all_exist = all(os.path.isfile(f) for f in required_files)
    if all_exist:
        print("\n⏭️  检测到预处理文件已存在，跳过预处理步骤：")
        for f in required_files:
            size = os.path.getsize(f)
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"    ✅ {os.path.basename(f)}  ({size_str})")
        return True
    else:
        missing = [os.path.basename(f) for f in required_files if not os.path.isfile(f)]
        print(f"\n🔍 检测到缺少预处理文件: {missing}")
        print("   将重新执行预处理...")
        return False


def step1_prepare_data():
    """第一步：合并数据并生成 KPGT 格式的划分文件"""
    data_dir = os.path.join(DATASETS_DIR, DATASET_NAME)
    splits_dir = os.path.join(data_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(RAW_DATA_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(RAW_DATA_DIR, "test.csv"))

    print(f"  原始数据量: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    for df in [train_df, val_df, test_df]:
        df.columns = df.columns.str.lower()

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    csv_path = os.path.join(data_dir, f"{DATASET_NAME}.csv")
    all_df.to_csv(csv_path, index=False)

    task_cols = all_df.columns.drop("smiles").tolist()
    print(f"  合并完成: 共 {len(all_df)} 条")
    print(f"  标签列: {task_cols}")

    n_train = len(train_df)
    n_val   = len(val_df)

    train_idx = all_df.index[:n_train].values
    val_idx   = all_df.index[n_train : n_train + n_val].values
    test_idx  = all_df.index[n_train + n_val :].values

    splits = np.array([train_idx, val_idx, test_idx], dtype=object)
    npy_path = os.path.join(splits_dir, f"{SPLIT_NAME}.npy")
    np.save(npy_path, splits)

    print(f"  划分文件已保存: {npy_path}")
    print(f"  Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")


def step2_preprocess():
    """第二步：预处理分子，已存在则跳过"""
    if check_preprocess_done():
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    cmd = (
        f"{sys.executable} {os.path.join(PROJECT_ROOT, 'scripts', 'preprocess_downstream_dataset.py')} "
        f"--data_path {DATASETS_DIR} "
        f"--dataset {DATASET_NAME} "
        f"--path_length 5 "
        f"--n_jobs 8"
    )
    run_command(cmd, "预分子处理（生成图缓存、分子指纹、分子描述符）", log_file=PREPROCESS_LOG)

    dataset_dir = os.path.join(DATASETS_DIR, DATASET_NAME)
    print("\n📁 预处理生成的文件：")
    for root, dirs, files in os.walk(dataset_dir):
        level = root.replace(dataset_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}📂 {os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for file in sorted(files):
            fpath = os.path.join(root, file)
            size = os.path.getsize(fpath)
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"{sub_indent}📄 {file}  ({size_str})")


def step3_finetune():
    """第三步：启动回归微调"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    cmd = (
        f"{sys.executable} {os.path.join(PROJECT_ROOT, 'scripts', 'finetune.py')} "
        f"--config base "
        f"--model_path {PRETRAIN_MODEL} "
        f"--dataset {DATASET_NAME} "
        f"--data_path {DATASETS_DIR} "
        f"--dataset_type regression "
        f"--metric rmse "
        f"--split {SPLIT_NAME} "
        f"--n_epochs {N_EPOCHS} "
        f"--batch_size {BATCH_SIZE} "
        f"--lr {LR} "
        f"--weight_decay {WEIGHT_DECAY} "
        f"--dropout {DROPOUT} "
        f"--save_dir {SAVE_DIR}"
    )
    if USE_L2SP:
        cmd += f" --use_l2sp --l2sp_weight {L2SP_WEIGHT}"

    run_command(cmd, "回归微调训练（pIC50 预测）", log_file=FINETUNE_LOG)


def save_result_summary():
    """保存结果摘要"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  KPGT 回归微调 结果摘要\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"项目根目录:   {PROJECT_ROOT}\n")
        f.write(f"数据集名称:   {DATASET_NAME}\n")
        f.write(f"数据目录:     {os.path.join(DATASETS_DIR, DATASET_NAME)}\n")
        f.write(f"预训练模型:   {PRETRAIN_MODEL}\n")
        f.write(f"划分文件:     {os.path.join(DATASETS_DIR, DATASET_NAME, 'splits', f'{SPLIT_NAME}.npy')}\n\n")
        f.write(f"训练超参数:\n")
        f.write(f"  Epochs:        {N_EPOCHS}\n")
        f.write(f"  Batch Size:    {BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {LR}\n")
        f.write(f"  Weight Decay:  {WEIGHT_DECAY}\n")
        f.write(f"  Dropout:       {DROPOUT}\n")
        f.write(f"  L2SP:          {USE_L2SP} (weight={L2SP_WEIGHT})\n\n")

        if os.path.exists(FINETUNE_LOG):
            with open(FINETUNE_LOG, "r", encoding="utf-8") as log:
                lines = log.readlines()
                for line in reversed(lines):
                    if "Training:" in line and "Validation:" in line and "Test:" in line:
                        f.write(f"最终结果:\n")
                        f.write(f"  {line.strip()}\n")
                        break

        f.write(f"\n生成的文件:\n")
        f.write(f"  预处理日志:       {PREPROCESS_LOG}\n")
        f.write(f"  微调日志:         {FINETUNE_LOG}\n")
        f.write(f"  结果摘要:         {RESULT_FILE}\n")
        f.write(f"  最佳模型权重:     {os.path.join(SAVE_DIR, 'best_model.pth')}\n")
        f.write(f"  完整模型(可推理):  {os.path.join(SAVE_DIR, 'full_model.pt')}\n")
        f.write(f"  训练配置:         {os.path.join(SAVE_DIR, 'train_config.json')}\n")

    print(f"\n📄 结果摘要已保存到: {RESULT_FILE}")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║      KPGT 回归微调 一键执行脚本                     ║")
    print("║      任务: pIC50 回归预测                           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  Python:     {sys.executable}")
    print(f"  输出目录:   {SAVE_DIR}")
    print(f"\n  训练配置:")
    print(f"    Epochs={N_EPOCHS}, BatchSize={BATCH_SIZE}, LR={LR}")
    print(f"    WeightDecay={WEIGHT_DECAY}, Dropout={DROPOUT}")
    print(f"    L2SP={USE_L2SP} (weight={L2SP_WEIGHT})")

    required_files = ["train.csv", "val.csv", "test.csv"]
    for fname in required_files:
        filepath = os.path.join(RAW_DATA_DIR, fname)
        if not os.path.exists(filepath):
            print(f"\n❌ 找不到文件: {filepath}")
            print(f"   请将 train.csv, val.csv, test.csv 放入 {RAW_DATA_DIR}/ 文件夹中")
            sys.exit(1)

    if not os.path.exists(PRETRAIN_MODEL):
        print(f"\n❌ 找不到预训练模型: {PRETRAIN_MODEL}")
        print("   请先从 README 中下载预训练权重并放入对应路径")
        sys.exit(1)

    os.makedirs(SAVE_DIR, exist_ok=True)

    step1_prepare_data()
    step2_preprocess()
    step3_finetune()
    save_result_summary()

    print("\n" + "=" * 60)
    print("🎉 全部流程执行完毕！")
    print(f"\n📁 输出目录: {SAVE_DIR}/")
    print(f"   ├── preprocess.log        预处理日志")
    print(f"   ├── finetune.log          微调训练日志")
    print(f"   ├── result_summary.txt    结果摘要")
    print(f"   ├── best_model.pth        最佳模型权重")
    print(f"   ├── full_model.pt         完整模型（可直接推理）")
    print(f"   └── train_config.json     训练配置（含标签归一化参数）")
    print(f"\n🔬 推理时加载模型示例:")
    print(f"   checkpoint = torch.load('{os.path.join(SAVE_DIR, 'full_model.pt')}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
