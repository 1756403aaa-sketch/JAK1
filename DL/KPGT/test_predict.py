#!/usr/bin/env python3
"""
KPGT Inference Script
=====================
Usage:
    python predict.py --input new_molecules.csv \
                      --model outputs/mydata_regression/full_model.pt \
                      --output predictions.csv
"""
import sys
import os
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.finetune_dataset import MoleculeDataset
from src.data.collator import Collator_tune
from src.model.light import LiGhTPredictor as LiGhT
from src.model_config import config_dict

# 导入评估和绘图库
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()


def get_predictor(d_input, n_tasks, n_layers, dropout, device, d_hidden=None):
    if n_layers == 1:
        predictor = nn.Linear(d_input, n_tasks)
    else:
        layers = [nn.Linear(d_input, d_hidden), nn.Dropout(dropout), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(d_hidden, d_hidden), nn.Dropout(dropout), nn.GELU()]
        layers.append(nn.Linear(d_hidden, n_tasks))
        predictor = nn.Sequential(*layers)
    predictor.apply(init_params)
    return predictor.to(device)


def setup_dataset(input_csv, dataset_name, datasets_dir, path_length=5, n_jobs=8):
    """
    将输入 CSV 转换为 KPGT 数据集格式:
      1) 保存 CSV
      2) 创建 split 文件（全部作为 test）
      3) 运行预处理（如需要）
    """
    dataset_dir = os.path.join(datasets_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # ---------- 读取并整理 CSV ----------
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip().str.lower()

    # 找 SMILES 列
    smiles_col = None
    for c in df.columns:
        if c in ['smiles', 'smile', 'canonical_smiles', 'canonicalsmiles']:
            smiles_col = c
            break
    if smiles_col is None:
        raise ValueError(f"CSV 中未找到 SMILES 列！当前列: {df.columns.tolist()}")

    if smiles_col != 'smiles':
        df = df.rename(columns={smiles_col: 'smiles'})

    # 保留原始顺序和额外列
    extra_cols = [c for c in df.columns if c != 'smiles']

    # 加 dummy label（MoleculeDataset 回归模式需要标签列，值不影响预测）
    if 'pic50' not in df.columns:
        df['pic50'] = np.random.randn(len(df))

    csv_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV 已保存: {csv_path}  ({len(df)} 条)")

    # ---------- 创建 split（全部归入 test） ----------
    splits_dir = os.path.join(dataset_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    splits = np.array([
        np.array([], dtype=int),
        np.array([], dtype=int),
        np.arange(len(df)),
    ], dtype=object)
    np.save(os.path.join(splits_dir, "pred_split.npy"), splits)

    # ---------- 预处理 ----------
    pkl_path = os.path.join(dataset_dir, f"{dataset_name}_{path_length}.pkl")
    fp_path = os.path.join(dataset_dir, "rdkfp1-7_512.npz")
    md_path = os.path.join(dataset_dir, "molecular_descriptors.npz")

    if not (os.path.exists(pkl_path) and os.path.exists(fp_path) and os.path.exists(md_path)):
        print("  正在预处理分子（首次运行较慢）...")
        preprocess_script = os.path.join(PROJECT_ROOT, "scripts", "preprocess_downstream_dataset.py")
        cmd = (
            f"{sys.executable} {preprocess_script} "
            f"--data_path {datasets_dir} --dataset {dataset_name} "
            f"--path_length {path_length} --n_jobs {n_jobs}"
        )
        proc = subprocess.run(cmd, shell=True)
        if proc.returncode != 0:
            raise RuntimeError("预处理失败！")
        print("  预处理完成。")
    else:
        print("  预处理文件已存在，跳过。")

    return df, extra_cols


def predict(args):
    datasets_dir = os.path.join(PROJECT_ROOT, "datasets")

    # ==============================
    # Step 1: 准备数据集
    # ==============================
    print("=" * 60)
    print("  KPGT pIC50 预测")
    print("=" * 60)
    print(f"\n[Step 1/3] 准备数据集")
    print(f"  输入: {args.input}")
    df, extra_cols = setup_dataset(args.input, args.dataset_name, datasets_dir)

    # ==============================
    # Step 2: 加载模型
    # ==============================
    print(f"\n[Step 2/3] 加载模型")
    print(f"  模型: {args.model}")

    checkpoint = torch.load(args.model, map_location="cpu")

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        train_cfg = checkpoint['config']
    else:
        model_state = checkpoint
        # 从同目录读取 train_config.json
        config_path = os.path.join(os.path.dirname(args.model), "train_config.json")
        with open(config_path) as f:
            train_cfg = json.load(f)

    config = config_dict[train_cfg['config']]
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collator = Collator_tune(config['path_length'])

    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_fp_feats=train_cfg['d_fp_feats'],
        d_md_feats=train_cfg['d_md_feats'],
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size,
    ).to(device)

    model.predictor = get_predictor(
        d_input=config['d_g_feats'] * 3,
        n_tasks=train_cfg['n_tasks'],
        n_layers=train_cfg['n_predictor_layers'],
        dropout=0,
        device=device,
        d_hidden=train_cfg['d_predictor_hidden'],
    )
    del model.md_predictor
    del model.fp_predictor
    del model.node_predictor

    model.load_state_dict(model_state)
    model.eval()
    print(f"  设备: {device}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 标签归一化参数（推理时反归一化必需）
    label_mean = np.array(train_cfg['label_mean'])
    label_std = np.array(train_cfg['label_std'])

    # ==============================
    # Step 3: 预测
    # ==============================
    print(f"\n[Step 3/3] 预测中...")

    dataset = MoleculeDataset(
        root_path=datasets_dir, dataset=args.dataset_name,
        dataset_type="regression", split_name="pred_split", split="test"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False, collate_fn=collator)

    all_preds = []
    total = 0
    with torch.no_grad():
        for batch_idx, batched_data in enumerate(loader):
            smiles_list, g, ecfp, md, label = batched_data
            pred = model.forward_tune(g.to(device), ecfp.to(device), md.to(device))
            all_preds.append(pred.cpu().numpy())
            total += len(pred)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) * args.batch_size >= total:
                print(f"  进度: {total}/{len(dataset)}")

    preds = np.concatenate(all_preds, axis=0)

    # 反归一化: pIC50_real = pred * std + mean
    preds_real = preds * label_std + label_mean

    # ==============================
    # 保存结果
    # ==============================
    result_df = pd.DataFrame({'smiles': df['smiles'].values[:len(preds_real)]})
    result_df['pic50_predicted'] = preds_real.flatten()

    # 保留原始 CSV 中的额外列
    for col in extra_cols:
        if col != 'pic50':
            result_df[col] = df[col].values[:len(preds_real)]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result_df.to_csv(args.output, index=False)

    print(f"\n{'=' * 60}")
    print(f"  预测完成！")
    print(f"{'=' * 60}")
    print(f"  输入文件:   {args.input}")
    print(f"  分子数量:   {len(preds_real)}")
    print(f"  输出文件:   {args.output}")
    print(f"  pIC50 范围: [{preds_real.min():.2f}, {preds_real.max():.2f}]")
    print(f"  pIC50 均值: {preds_real.mean():.2f}")
    print(f"  pIC50 中位: {np.median(preds_real):.2f}")

    # ==============================
    # Step 4: 评估与绘图 (如果原始CSV有真实标签)
    # ==============================
    # 重新读取原始输入，判断是否有真实标签（排除 setup_dataset 中可能生成的 dummy label）
    raw_df = pd.read_csv(args.input)
    raw_df.columns = raw_df.columns.str.strip().str.lower()

    if 'pic50' in raw_df.columns:
        y_true = raw_df['pic50'].values[:len(preds_real)]
        y_pred = preds_real.flatten()

        # 计算指标
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        print(f"\n{'=' * 60}")
        print(f"  模型评估指标 (基于原始真实标签)")
        print(f"{'=' * 60}")
        print(f"  R²:   {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"{'=' * 60}")

        # 绘制散点图
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5, s=40)

        # 绘制 y=x 参考线
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y = x')

        plt.xlabel('True pIC50', fontsize=14)
        plt.ylabel('Predicted pIC50', fontsize=14)
        plt.title(f'True vs Predicted pIC50\nR²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存图像
        plot_path = args.output.replace('.csv', '_scatter.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n🎨 散点图已保存: {plot_path}")
    else:
        print("\n⚠️  原始输入 CSV 中未检测到 'pic50' 列，跳过评估指标计算和绘图。")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KPGT pIC50 Prediction")
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file (must contain SMILES column)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to full_model.pt or best_model.pth")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output CSV file path")
    parser.add_argument("--dataset_name", type=str, default="predict_input",
                        help="Temporary dataset name (default: predict_input)")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    predict(args)
