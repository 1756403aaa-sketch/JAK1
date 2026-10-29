#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化版分子对接流水线
- 使用 Vina Python API
- 每个 worker 只计算一次受体网格地图（共享）
- 完善的错误处理与隔离（不依赖自动居中）
"""

import os
import shutil
import time
import traceback
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from tqdm import tqdm
import multiprocessing as mp
import warnings
from vina import Vina

warnings.filterwarnings('ignore')


# ============================================================
# 辅助函数
# ============================================================
def read_config(config_file):
    """读取对接盒子配置文件"""
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = float(value.strip())
    center = [config['center_x'], config['center_y'], config['center_z']]
    box_size = [config['size_x'], config['size_y'], config['size_z']]
    return center, box_size


# ============================================================
# Step 1: SMILES → 3D Conformer (SDF)
# ============================================================
def generate_3d_conformer_task(args):
    idx, smiles, output_sdf_dir = args
    result = {'index': idx, 'smiles': smiles, 'sdf_path': '', 'status': ''}

    if pd.isna(smiles) or str(smiles).strip() == '':
        result['status'] = 'Error: Empty SMILES'
        return result

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result['status'] = 'Error: Invalid SMILES'
        return result

    mol = Chem.AddHs(mol)
    success = False
    for attempt in range(3):
        AllChem.EmbedMolecule(mol, randomSeed=42 + attempt)
        if mol.GetNumConformers() > 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol)
                success = True
                break
            except Exception:
                continue

    if not success:
        result['status'] = 'Error: Conformation failed'
        return result

    sdf_path = os.path.join(output_sdf_dir, f"ligand_{idx}.sdf")
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()

    result['sdf_path'] = sdf_path
    result['status'] = 'OK'
    return result


def step1_generate_conformers(df, smiles_col, output_dir, num_workers):
    print("\n" + "=" * 70)
    print("  Step 1 / 3 : SMILES → 3D Conformer (.sdf)")
    print("=" * 70)

    sdf_dir = os.path.join(output_dir, "step1_sdf")
    os.makedirs(sdf_dir, exist_ok=True)

    tasks = [(idx, row[smiles_col], sdf_dir) for idx, row in df.iterrows()]

    t0 = time.time()
    results = []
    with mp.Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap(generate_3d_conformer_task, tasks),
                        total=len(tasks), desc="  生成 3D 构象"):
            results.append(res)

    elapsed = time.time() - t0
    ok_count = sum(1 for r in results if r['status'] == 'OK')
    fail_count = len(results) - ok_count

    summary_path = os.path.join(output_dir, "step1_summary.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)

    print(f"  ✔ 完成 | 成功: {ok_count}, 失败: {fail_count} | 耗时: {elapsed:.1f}s")
    print(f"  ✔ SDF 文件目录 : {sdf_dir}")
    print(f"  ✔ 汇总文件     : {summary_path}")
    return results


# ============================================================
# Step 2: SDF → PDBQT
# ============================================================
def sdf_to_pdbqt_task(args):
    idx, sdf_path, output_pdbqt_dir = args
    result = {'index': idx, 'pdbqt_path': '', 'status': ''}

    if not sdf_path or not os.path.exists(sdf_path):
        result['status'] = 'Error: SDF not found'
        return result

    supplier = Chem.SDMolSupplier(sdf_path)
    mol = next(iter(supplier), None)
    if mol is None:
        result['status'] = 'Error: Failed to read SDF'
        return result

    mol = Chem.AddHs(mol, addCoords=True)

    try:
        prep = MoleculePreparation()
        prep.prepare(mol)
        pdbqt_string = prep.write_pdbqt_string()
    except Exception as e:
        result['status'] = f'Error: PDBQT conversion failed ({e})'
        return result

    pdbqt_path = os.path.join(output_pdbqt_dir, f"ligand_{idx}.pdbqt")
    with open(pdbqt_path, 'w') as f:
        f.write(pdbqt_string)

    result['pdbqt_path'] = pdbqt_path
    result['status'] = 'OK'
    return result


def step2_convert_to_pdbqt(step1_results, output_dir, num_workers):
    print("\n" + "=" * 70)
    print("  Step 2 / 3 : 3D Conformer (.sdf) → PDBQT (.pdbqt)")
    print("=" * 70)

    pdbqt_dir = os.path.join(output_dir, "step2_pdbqt")
    os.makedirs(pdbqt_dir, exist_ok=True)

    tasks = [(r['index'], r['sdf_path'], pdbqt_dir)
             for r in step1_results if r['status'] == 'OK']

    t0 = time.time()
    results = []
    with mp.Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap(sdf_to_pdbqt_task, tasks),
                        total=len(tasks), desc="  转换 PDBQT"):
            results.append(res)

    elapsed = time.time() - t0
    ok_count = sum(1 for r in results if r['status'] == 'OK')
    fail_count = len(results) - ok_count

    summary_path = os.path.join(output_dir, "step2_summary.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)

    print(f"  ✔ 完成 | 成功: {ok_count}, 失败: {fail_count} | 耗时: {elapsed:.1f}s")
    print(f"  ✔ PDBQT 文件目录 : {pdbqt_dir}")
    print(f"  ✔ 汇总文件       : {summary_path}")
    return results


# ============================================================
# Step 3: 分子对接（共享网格地图优化版，无自动居中）
# ============================================================
# 全局变量（每个 worker 独立）
_vina_shared = None


def init_worker(receptor_pdbqt, center, box_size):
    """每个 worker 启动时执行一次：加载受体、预计算网格地图"""
    global _vina_shared
    _vina_shared = Vina(sf_name='vina', cpu=1, seed=0)
    _vina_shared.set_receptor(rigid_pdbqt_filename=receptor_pdbqt)
    _vina_shared.compute_vina_maps(center=center, box_size=box_size)


def dock_single_ligand_shared(lig_pdbqt, exhaustiveness=8, num_modes=9):
    """复用全局 _vina_shared 对象进行对接（不自动居中）"""
    global _vina_shared
    if _vina_shared is None:
        raise RuntimeError("Worker not initialized")

    try:
        # 直接设置配体，不移至中心
        _vina_shared.set_ligand_from_file(lig_pdbqt)
        _vina_shared.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)
        affinity = _vina_shared.score()[0]  # 取总结合能
        return affinity, None
    except RuntimeError as e:
        # 捕获配体超出盒子等运行时错误
        error_msg = f"RuntimeError: {e}"
        return None, error_msg
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg


def docking_task_shared(args):
    idx, pdbqt_path, docked_output_dir, exhaustiveness, num_modes = args
    result = {
        'index': idx,
        'pdbqt_path': pdbqt_path,
        'binding_affinity': None,
        'status': ''
    }

    if not pdbqt_path or not os.path.exists(pdbqt_path):
        result['status'] = 'Error: PDBQT not found'
        return result

    affinity, error = dock_single_ligand_shared(
        pdbqt_path,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes
    )

    if affinity is not None:
        result['binding_affinity'] = affinity
        result['status'] = 'OK'
        docked_path = os.path.join(docked_output_dir, f"docked_{idx}.pdbqt")
        # ========== 修改开始 ==========
        # 删除原来的 shutil.copy2(pdbqt_path, docked_path)
        # 改为保存对接后的构象（最佳姿势）
        global _vina_shared
        if _vina_shared is not None:
            _vina_shared.write_poses(docked_path, n_poses=1)   # 只保存最佳姿势
        # ========== 修改结束 ==========
    else:
        result['status'] = f'Error: {error}'
        # 保存详细错误日志
        error_log_dir = os.path.join(docked_output_dir, "errors")
        os.makedirs(error_log_dir, exist_ok=True)
        with open(os.path.join(error_log_dir, f"error_{idx}.log"), 'w') as f:
            f.write(error)

    return result


def step3_docking_optimized(step2_results, config_params, output_dir, num_workers):
    print("\n" + "=" * 70)
    print("  Step 3 / 3 : Molecular Docking (Optimized: Shared Grid Maps)")
    print("=" * 70)

    docked_dir = os.path.join(output_dir, "step3_docked_pdbqt")
    os.makedirs(docked_dir, exist_ok=True)

    tasks = [
        (r['index'], r['pdbqt_path'], docked_dir,
         config_params['exhaustiveness'],
         config_params['num_modes'])
        for r in step2_results if r['status'] == 'OK'
    ]

    if not tasks:
        print("  警告：没有可用的 PDBQT 文件，跳过对接步骤")
        return []

    t0 = time.time()
    with mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(config_params['receptor_pdbqt'],
                  config_params['center'],
                  config_params['box_size'])
    ) as pool:
        results = list(tqdm(pool.imap(docking_task_shared, tasks),
                            total=len(tasks), desc="  分子对接 (共享地图)"))
    elapsed = time.time() - t0

    ok_count = sum(1 for r in results if r['status'] == 'OK')
    fail_count = len(results) - ok_count

    summary_path = os.path.join(output_dir, "step3_summary.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)

    print(f"  ✔ 完成 | 成功: {ok_count}, 失败: {fail_count} | 耗时: {elapsed:.1f}s")
    if fail_count > 0:
        print(f"  ⚠ 失败的对接详情请查看 {docked_dir}/errors/ 目录下的错误日志")
    print(f"  ✔ 对接输出目录 : {docked_dir}")
    print(f"  ✔ 汇总文件     : {summary_path}")
    return results


# ============================================================
# 主流程
# ============================================================
def main():
    # 配置参数（请根据实际情况修改）
    csv_file = "ADMET.csv"
    receptor_pdbqt = "protein.pdbqt"
    config_file = "config.txt"
    output_csv = "docking_results_ADMET.csv"
    output_dir = "docking_workspace"
    num_cpus = 36   # 可根据 CPU 核心数调整

    print(f"使用 CPU 核心数: {num_cpus}")
    print(f"工作目录: {os.path.abspath(output_dir)}")

    # 预检文件
    required_files = [csv_file, receptor_pdbqt, config_file]
    for f in required_files:
        if not os.path.exists(f):
            print(f"错误：找不到文件 '{f}'")
            return

    # 读取对接盒子
    center, box_size = read_config(config_file)
    print(f"对接盒子中心: {center}")
    print(f"对接盒子尺寸: {box_size}")

    # 读取输入数据
    df = pd.read_csv(csv_file)
    if 'SMILES' in df.columns:
        smiles_col = 'SMILES'
    else:
        smiles_col = df.columns[1]
        print(f"警告：未找到 'SMILES' 列，将使用第二列 '{smiles_col}' 作为 SMILES 列")
    print(f"输入文件包含 {len(df)} 个分子，SMILES 列：{smiles_col}")

    os.makedirs(output_dir, exist_ok=True)

    config_params = {
        'receptor_pdbqt': receptor_pdbqt,
        'center': center,
        'box_size': box_size,
        'exhaustiveness': 8,
        'num_modes': 9
    }

    total_t0 = time.time()

    # Step 1
    step1_results = step1_generate_conformers(df, smiles_col, output_dir, num_cpus)

    # Step 2
    step2_results = step2_convert_to_pdbqt(step1_results, output_dir, num_cpus)

    # Step 3 (优化版，无自动居中)
    step3_results = step3_docking_optimized(step2_results, config_params, output_dir, num_cpus)

    total_elapsed = time.time() - total_t0

    # 合并结果
    step1_df = pd.DataFrame(step1_results)[['index', 'smiles', 'sdf_path', 'status']]
    step1_df.rename(columns={'status': 'step1_status'}, inplace=True)

    step2_df = pd.DataFrame(step2_results)[['index', 'pdbqt_path', 'status']]
    step2_df.rename(columns={'status': 'step2_status'}, inplace=True)

    step3_df = pd.DataFrame(step3_results)[['index', 'binding_affinity', 'status']]
    step3_df.rename(columns={'status': 'step3_status'}, inplace=True)

    df_reset = df.reset_index().rename(columns={'index': 'original_index'})
    final_df = df_reset.merge(step1_df, left_on='original_index', right_on='index', how='left') \
                       .merge(step2_df, left_on='original_index', right_on='index', how='left') \
                       .merge(step3_df, left_on='original_index', right_on='index', how='left')
    final_df.drop(columns=['index_x', 'index_y', 'index'], errors='ignore', inplace=True)
    final_df.rename(columns={'original_index': 'index'}, inplace=True)

    # 将 binding_affinity 移至最后一列
    cols = [c for c in final_df.columns if c != 'binding_affinity'] + ['binding_affinity']
    final_df = final_df[cols]
    final_df.to_csv(output_csv, index=False)

    # 统计输出
    success_vals = [r['binding_affinity'] for r in step3_results
                   if isinstance(r['binding_affinity'], (int, float))]
    print("\n" + "=" * 70)
    print("  全部流程结束")
    print("=" * 70)
    print(f"  总耗时                : {total_elapsed:.1f}s")
    print(f"  总分子数              : {len(df)}")
    print(f"  Step1 成功 / 失败     : "
          f"{sum(1 for r in step1_results if r['status']=='OK')} / "
          f"{sum(1 for r in step1_results if r['status']!='OK')}")
    print(f"  Step2 成功 / 失败     : "
          f"{sum(1 for r in step2_results if r['status']=='OK')} / "
          f"{sum(1 for r in step2_results if r['status']!='OK')}")
    print(f"  Step3 成功 / 失败     : {len(success_vals)} / "
          f"{len(step3_results) - len(success_vals)}")
    if success_vals:
        print(f"  最好亲和力            : {min(success_vals):.2f} kcal/mol")
        print(f"  最差亲和力            : {max(success_vals):.2f} kcal/mol")
        print(f"  平均亲和力            : {sum(success_vals)/len(success_vals):.2f} kcal/mol")
    else:
        print("  警告：没有成功对接的分子，请检查错误日志。")

    print(f"\n  最终结果文件 : {output_csv}")
    print(f"  中间文件目录 : {os.path.abspath(output_dir)}/")
    print(f"    ├── step1_sdf/           (3D 构象 SDF 文件)")
    print(f"    ├── step1_summary.csv    (Step1 汇总)")
    print(f"    ├── step2_pdbqt/         (PDBQT 文件)")
    print(f"    ├── step2_summary.csv    (Step2 汇总)")
    print(f"    ├── step3_docked_pdbqt/  (对接后 PDBQT 文件)")
    print(f"    │   └── errors/          (对接失败的错误日志)")
    print(f"    └── step3_summary.csv    (Step3 汇总)")


if __name__ == "__main__":
    main()
