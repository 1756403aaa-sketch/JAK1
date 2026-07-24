#!/usr/bin/env python3
"""
批量处理 PDB 文件：
1. 使用 OpenBabel 补全氢原子（生成全原子结构）
2. 自动执行 ORCA 优化 + 单点能 + molden 转换
用法: python run_orca_batch.py --pdb_dir ./pdbs --orca_path orca
"""

import os
import sys
import subprocess
import argparse
import re
from pathlib import Path

# ========== 配置参数 ==========
DEFAULT_ORCA = "orca"                # ORCA 可执行文件
DEFAULT_OBABEL = "obabel"            # OpenBabel 可执行文件
DEFAULT_MAXCORE = 8000
DEFAULT_NPROCS = 8                   # 并行核数，可根据需要调整
DEFAULT_CHARGE = 0
DEFAULT_SPIN = 1
SOLVENT = "CPCM(WATER)"
OPT_BASIS = "def2-SVP"
SP_BASIS   = "def2-TZVP"

# ========== 辅助函数 ==========

def get_full_path(cmd_name):
    """在 PATH 中查找可执行文件的完整路径"""
    path = shutil.which(cmd_name)
    if path is None:
        return None
    return path

def add_hydrogens(pdb_in, pdb_out, obabel_cmd="obabel"):
    """
    使用 OpenBabel 给 PDB 文件补全氢原子
    返回 True/False
    """
    # 检查 obabel 是否可用
    if shutil.which(obabel_cmd) is None:
        raise RuntimeError(f"OpenBabel 命令 '{obabel_cmd}' 未找到，请安装或指定正确路径。")
    cmd = [obabel_cmd, "-ipdb", str(pdb_in), "-opdb", "-O", str(pdb_out), "--h"]
    with open(os.devnull, 'w') as null:
        proc = subprocess.Popen(cmd, stdout=null, stderr=subprocess.STDOUT)
        proc.wait()
    return proc.returncode == 0

def write_opt_input(pdb_file, out_dir, charge, spin, maxcore, nprocs):
    """生成几何优化的 ORCA 输入文件（从 PDB 读取坐标）"""
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atom = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = re.sub(r'[0-9]', '', atom)
                coords.append(f"{elem:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    if not coords:
        raise ValueError(f"未从 {pdb_file} 中找到原子坐标")

    xyz_block = "\n".join(coords)
    pal_line = f"%pal nprocs {nprocs} end" if nprocs > 1 else ""
    content = f"""! B3LYP D3 {OPT_BASIS} OPT FREQ {SOLVENT} TIGHTOPT PRINTGAP

%maxcore {maxcore}
{pal_line}

%scf
  MaxIter 500
  THRESH 1e-7
end

%geom
  Trust -0.1
  MaxIter 200
end

* xyz {charge} {spin}
{xyz_block}
*
"""
    inp_path = out_dir / "opt.inp"
    with open(inp_path, 'w') as f:
        f.write(content)
    return inp_path

def write_sp_input(xyz_file, out_dir, charge, spin, maxcore, nprocs):
    """生成单点能的 ORCA 输入文件"""
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    coord_lines = [line.strip() for line in lines[2:] if line.strip()]
    xyz_block = "\n".join(coord_lines)

    pal_line = f"%pal nprocs {nprocs} end" if nprocs > 1 else ""
    content = f"""! B3LYP D3 {SP_BASIS} {SOLVENT} PRINTGAP

%maxcore {maxcore}
{pal_line}

* xyz {charge} {spin}
{xyz_block}
*
"""
    inp_path = out_dir / "sp.inp"
    with open(inp_path, 'w') as f:
        f.write(content)
    return inp_path

def run_orca(inp_file, out_file, orca_full):
    """运行 ORCA 计算"""
    cmd = [orca_full, str(inp_file)]
    with open(out_file, 'w') as outf:
        proc = subprocess.Popen(cmd, stdout=outf, stderr=subprocess.STDOUT)
        proc.wait()
    return proc.returncode == 0

def check_convergence(out_file):
    """检查 ORCA 几何优化是否真正收敛（改进版）"""
    try:
        # 使用 errors='ignore' 避免因偶发字符编码问题导致读取失败
        with open(out_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # 1. 首先检查程序是否正常退出
        if "****ORCA TERMINATED NORMALLY****" not in text:
            return False, "程序未正常结束（可能崩溃或被中断）"

        # 2. 检查是否真正收敛（ORCA 标准输出）
        # "HURRAY" 是最可靠的收敛标志
        if "HURRAY" in text or "THE OPTIMIZATION HAS CONVERGED" in text:
            return True, "几何优化成功收敛"

        # 3. 检查是否因达到最大循环数而停止（未收敛）
        if "did not converge but reached the maximum number" in text:
            return False, "达到最大循环数仍未收敛"

        # 4. 兜底判断：如果频率计算已开始，则优化必然已完成
        if "VIBRATIONAL FREQUENCIES" in text:
            return True, "几何优化成功（频率分析已进行）"

        # 5. 无法判断状态
        return False, "无法识别收敛状态（可能仍在运行或日志不完整）"

    except Exception as e:
        return False, str(e)


def convert_to_molden(gbw_file, out_dir, orca_full):
    """使用 orca_2mkl 转换 .gbw 为 .molden.input"""
    orca_dir = os.path.dirname(orca_full)
    if not orca_dir:
        orca_dir = "."
    orca_2mkl = os.path.join(orca_dir, "orca_2mkl")
    if not os.path.exists(orca_2mkl):
        orca_2mkl = "orca_2mkl"
    base = gbw_file.stem
    cmd = [orca_2mkl, str(gbw_file.parent / base), "-molden"]
    with open(out_dir / "convert.log", 'w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        proc.wait()
    return proc.returncode == 0

def process_pdb(pdb_path, orca_full, obabel_cmd, maxcore, nprocs, charge, spin):
    """处理单个 PDB 文件：补氢 → ORCA 优化 → 单点能 → molden"""
    base_name = pdb_path.stem
    work_dir = pdb_path.parent / f"{base_name}_orca"
    work_dir.mkdir(exist_ok=True)

    # ---- 1. 补氢 ----
    hydro_pdb = work_dir / f"{base_name}_h.pdb"
    print(f"[{base_name}] 补氢...")
    if not add_hydrogens(pdb_path, hydro_pdb, obabel_cmd):
        print(f"[{base_name}] 补氢失败，跳过该分子。")
        return
    # 可选：将补氢后的坐标也保存一份（已保存为 hydro_pdb）
    print(f"[{base_name}] 补氢完成，保存至 {hydro_pdb}")

    # ---- 2. 生成 ORCA 优化输入 ----
    opt_inp = write_opt_input(hydro_pdb, work_dir, charge, spin, maxcore, nprocs)
    opt_out = work_dir / "opt.out"

    print(f"[{base_name}] 开始几何优化...")
    success = run_orca(opt_inp, opt_out, orca_full)
    if not success:
        print(f"[{base_name}] 优化任务运行失败，跳过后续步骤。")
        return

    converged, msg = check_convergence(opt_out)
    if not converged:
        print(f"[{base_name}] 优化未成功: {msg}")
        return
    print(f"[{base_name}] 几何优化成功！")

    # ---- 3. 单点能计算 ----
    xyz_file = work_dir / "opt.xyz"
    if not xyz_file.exists():
        print(f"[{base_name}] 未找到优化后的 opt.xyz，跳过单点能。")
        return

    sp_inp = write_sp_input(xyz_file, work_dir, charge, spin, maxcore, nprocs)
    sp_out = work_dir / "sp.out"

    print(f"[{base_name}] 开始单点能计算...")
    success = run_orca(sp_inp, sp_out, orca_full)
    if not success:
        print(f"[{base_name}] 单点能任务运行失败，跳过转换。")
        return

    with open(sp_out, 'r') as f:
        if "****ORCA TERMINATED NORMALLY****" not in f.read():
            print(f"[{base_name}] 单点能未正常结束，跳过转换。")
            return
    print(f"[{base_name}] 单点能计算成功！")

    # ---- 4. 转换 .gbw 为 .molden.input ----
    gbw_file = work_dir / "sp.gbw"
    if not gbw_file.exists():
        print(f"[{base_name}] sp.gbw 不存在，跳过转换。")
        return

    print(f"[{base_name}] 转换为 molden.input...")
    if convert_to_molden(gbw_file, work_dir, orca_full):
        print(f"[{base_name}] 转换成功！")
    else:
        print(f"[{base_name}] 转换失败，请检查 orca_2mkl 是否可用。")

    # ---- 5. 提取能隙 ----
    try:
        with open(sp_out, 'r') as f:
            text = f.read()
        gap_match = re.search(r"HOMO-LUMO-GAP\s*\(eV\)\s*:\s*([\d.]+)", text)
        if gap_match:
            gap = gap_match.group(1)
            print(f"[{base_name}] HOMO-LUMO 能隙 = {gap} eV")
            with open(work_dir / "summary.txt", 'w') as sf:
                sf.write(f"PDB: {pdb_path.name}\n")
                sf.write(f"Hydrogenated PDB: {hydro_pdb.name}\n")
                sf.write(f"HOMO-LUMO gap (eV): {gap}\n")
    except:
        pass

    print(f"[{base_name}] 所有步骤完成。结果保存在 {work_dir}\n")

def main():
    parser = argparse.ArgumentParser(description="批量处理 PDB：补氢 + ORCA 计算")
    parser.add_argument("--pdb_dir", type=str, default=".", help="包含 PDB 文件的文件夹 (默认当前目录)")
    parser.add_argument("--orca", type=str, default=DEFAULT_ORCA, help="ORCA 可执行文件路径")
    parser.add_argument("--obabel", type=str, default=DEFAULT_OBABEL, help="OpenBabel 可执行文件路径 (默认 obabel)")
    parser.add_argument("--maxcore", type=int, default=DEFAULT_MAXCORE, help="最大内存 (MB)")
    parser.add_argument("--nprocs", type=int, default=DEFAULT_NPROCS, help="并行核心数 (1 为单核)")
    parser.add_argument("--charge", type=int, default=DEFAULT_CHARGE, help="分子净电荷 (默认 0)")
    parser.add_argument("--spin", type=int, default=DEFAULT_SPIN, help="自旋多重度 (默认 1)")
    args = parser.parse_args()

    # 检查 ORCA 路径
    if not os.path.exists(args.orca) and shutil.which(args.orca) is None:
        print(f"错误：找不到 ORCA 可执行文件 '{args.orca}'")
        sys.exit(1)
    orca_full = args.orca if os.path.exists(args.orca) else shutil.which(args.orca)

    # 检查 OpenBabel 路径
    if shutil.which(args.obabel) is None:
        print(f"错误：找不到 OpenBabel 可执行文件 '{args.obabel}'")
        print("请安装 OpenBabel 或使用 --obabel 指定完整路径。")
        sys.exit(1)
    obabel_cmd = args.obabel

    pdb_dir = Path(args.pdb_dir)
    if not pdb_dir.exists():
        print(f"错误：路径 {pdb_dir} 不存在")
        sys.exit(1)

    pdb_files = list(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        print(f"在 {pdb_dir} 中未找到任何 .pdb 文件")
        sys.exit(1)

    print(f"找到 {len(pdb_files)} 个 PDB 文件。")
    for pdb_path in pdb_files:
        process_pdb(
            pdb_path,
            orca_full,
            obabel_cmd,
            args.maxcore,
            args.nprocs,
            args.charge,
            args.spin
        )
    print("所有任务完成！")

if __name__ == "__main__":
    import shutil   # 在顶部导入，但为清晰在这里再导入一次
    # 由于我们使用了 shutil.which，需要在顶部导入，但我们已在前面导入，此处不再重复。
    main()