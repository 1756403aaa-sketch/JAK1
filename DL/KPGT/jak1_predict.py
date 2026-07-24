#!/usr/bin/env python3
"""
KPGT Inference Script
=====================
Usage:
    python jak1_predict.py --input Data_processed.csv \
                           --model outputs/mydata_regression/full_model.pt \
                           --output jak1_predictions.csv \
                           --n_jobs 64
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

from rdkit import Chem


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


def validate_smiles(smiles_list):
    """
    验证 SMILES 列表，返回有效索引和无效索引。
    """
    valid_idxs = []
    invalid_idxs = []
    invalid_smiles = []
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid_idxs.append(i)
        else:
            invalid_idxs.append(i)
            invalid_smiles.append(s[:80])
    return valid_idxs, invalid_idxs, invalid_smiles


def setup_dataset(input_csv, dataset_name, datasets_dir, path_length=5, n_jobs=32):
    """
    将输入 CSV 转换为 KPGT 数据集格式:
      1) 过滤无效 SMILES
      2) 保存 CSV（只保留 smiles + dummy pic50）
      3) 创建 split 文件（全部作为 test）
      4) 运行预处理（如需要）
    """
    dataset_dir = os.path.join(datasets_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # ---------- 读取原始 CSV（保留原始列名和顺序） ----------
    raw_df = pd.read_csv(input_csv)

    # 找 SMILES 列（不区分大小写）
    smiles_col = None
    for c in raw_df.columns:
        if c.strip().lower() in ['smiles', 'smile', 'canonical_smiles', 'canonicalsmiles']:
            smiles_col = c
            break
    if smiles_col is None:
        raise ValueError(f"CSV 中未找到 SMILES 列！当前列: {raw_df.columns.tolist()}")

    # ---------- 清洗 SMILES ----------
    smiles_series = raw_df[smiles_col].astype(str).str.strip().str.strip('"').str.replace('\r', '').str.replace('\n', '')

    # ---------- 验证 SMILES ----------
    print("  正在验证 SMILES 有效性...")
    valid_idxs, invalid_idxs, invalid_smiles = validate_smiles(smiles_series.tolist())

    if invalid_idxs:
        print(f"  ⚠️  发现 {len(invalid_idxs)} 条无效 SMILES，将跳过（预测值填 NaN）")
        if len(invalid_idxs) <= 20:
            for idx, smi in zip(invalid_idxs, invalid_smiles):
                print(f"      行{idx}: {smi}")
        else:
            for idx, smi in zip(invalid_idxs[:10], invalid_smiles[:10]):
                print(f"      行{idx}: {smi}")
            print(f"      ... 还有 {len(invalid_idxs) - 10} 条无效 SMILES")
    print(f"  有效 SMILES: {len(valid_idxs)} / {len(raw_df)}")

    valid_idxs = np.array(valid_idxs)
    invalid_idxs = np.array(invalid_idxs)

    # ---------- 为 KPGT 内部处理：只保留有效 SMILES + pic50 ----------
    kpgt_df = pd.DataFrame()
    kpgt_df['smiles'] = smiles_series.iloc[valid_idxs].values
    kpgt_df['pic50'] = np.random.randn(len(valid_idxs))  # dummy label

    csv_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
    kpgt_df.to_csv(csv_path, index=False)
    print(f"  CSV 已保存: {csv_path}  ({len(kpgt_df)} 条)")

    # ---------- 创建 split（全部归入 test） ----------
    splits_dir = os.path.join(dataset_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    splits = np.array([
        np.array([], dtype=int),
        np.array([], dtype=int),
        np.arange(len(kpgt_df)),
    ], dtype=object)
    np.save(os.path.join(splits_dir, "pred_split.npy"), splits)

    # ---------- 预处理 ----------
    pkl_path = os.path.join(dataset_dir, f"{dataset_name}_{path_length}.pkl")
    fp_path = os.path.join(dataset_dir, "rdkfp1-7_512.npz")
    md_path = os.path.join(dataset_dir, "molecular_descriptors.npz")

    need_preprocess = False
    if not (os.path.exists(pkl_path) and os.path.exists(fp_path) and os.path.exists(md_path)):
        need_preprocess = True
    else:
        try:
            import scipy.sparse as sp
            old_fp = sp.load_npz(fp_path)
            if old_fp.shape[0] != len(kpgt_df):
                print(f"  ⚠️ 缓存数量({old_fp.shape[0]})与当前数据量({len(kpgt_df)})不匹配，重新预处理")
                need_preprocess = True
        except Exception:
            need_preprocess = True

    if need_preprocess:
        print(f"  正在预处理分子（n_jobs={n_jobs}）...")
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

    return raw_df, smiles_col, valid_idxs, invalid_idxs


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
    raw_df, smiles_col, valid_idxs, invalid_idxs = setup_dataset(
        args.input, args.dataset_name, datasets_dir, n_jobs=args.n_jobs
    )
    n_molecules = len(raw_df)
    n_valid = len(valid_idxs)

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
            print(f"  进度: {total}/{n_valid}  ({total / n_valid * 100:.1f}%)")

    preds = np.concatenate(all_preds, axis=0)
    preds_real = preds * label_std + label_mean

    # ==============================
    # 保存结果：原始列 + 预测值列（无效 SMILES 填 NaN）
    # ==============================
    pic50_col = np.full(n_molecules, np.nan)
    pic50_col[valid_idxs] = preds_real.flatten()[:n_valid]

    insert_pos = list(raw_df.columns).index(smiles_col) + 1
    raw_df.insert(insert_pos, 'pIC50_predicted', pic50_col)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    raw_df.to_csv(args.output, index=False)

    n_invalid = len(invalid_idxs)
    print(f"\n{'='*60}")
    print(f"  预测完成！")
    print(f"{'='*60}")
    print(f"  输入文件:   {args.input}")
    print(f"  分子总数:   {n_molecules}")
    print(f"  有效预测:   {n_valid}")
    print(f"  无效跳过:   {n_invalid}  (pIC50_predicted = NaN)")
    print(f"  输出文件:   {args.output}")
    print(f"  pIC50 范围: [{np.nanmin(pic50_col):.2f}, {np.nanmax(pic50_col):.2f}]")
    print(f"  pIC50 均值: {np.nanmean(pic50_col):.2f}")
    print(f"  pIC50 中位: {np.nanmedian(pic50_col):.2f}")
    print(f"{'='*60}")

    return raw_df


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
    parser.add_argument("--n_jobs", type=int, default=32,
                        help="Number of CPU cores for parallel preprocessing (default: 32)")
    args = parser.parse_args()

    predict(args)
