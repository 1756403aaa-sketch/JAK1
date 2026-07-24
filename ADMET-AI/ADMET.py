import pandas as pd
from admet_ai import ADMETModel


def predict_and_filter_admet(input_csv, output_csv, smiles_column="SMILES"):
    """
    修正版：修正了 Caco2_Wang 的单位误解，将其作为回归值处理。
    收紧了 CYP、DILI、溶解性和 Caco2 阈值。
    最终输出仅保留：原CSV列 + 用于筛选的ADMET列 + 对应的百分位列。
    """
    try:
        # 记录原始列名，用于后续过滤
        original_cols = set(pd.read_csv(input_csv, nrows=1).columns)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_csv}")
        return

    if smiles_column not in original_cols:
        print(f"错误: CSV中找不到列名 '{smiles_column}'")
        return

    # 重新读取完整数据
    df = pd.read_csv(input_csv)

    # ================= 1. ADMET 预测 =================
    admet_cols_needed = ['HIA_Hou', 'Caco2_Wang', 'DILI']
    needs_prediction = not all(col in df.columns for col in admet_cols_needed)

    if needs_prediction:
        print("检测到缺少ADMET列，开始预测...")
        df = df.dropna(subset=[smiles_column])
        smiles_list = df[smiles_column].unique().tolist()

        model = ADMETModel()
        preds_df = model.predict(smiles=smiles_list)
        preds_df = preds_df.reset_index().rename(columns={"index": smiles_column})

        df = pd.merge(df, preds_df, on=smiles_column, how="left")
        print("ADMET 性质预测完成！")
    else:
        print("检测到已有ADMET列，直接进行筛选。")

    # ================= 2. 口服药筛选 (收紧阈值) =================
    print("\n开始按【口服药标准】进行纯筛选（已修正Caco2单位，收紧阈值）...")
    initial_count = len(df)

    # --- 2.1 类药性 (增加防御性检查) ---
    if 'molecular_weight' in df.columns:
        df = df[df['molecular_weight'] <= 500]
    if 'logP' in df.columns:
        df = df[df['logP'] <= 5]
    if 'hydrogen_bond_acceptors' in df.columns:
        df = df[df['hydrogen_bond_acceptors'] <= 10]
    if 'hydrogen_bond_donors' in df.columns:
        df = df[df['hydrogen_bond_donors'] <= 5]
    if 'tpsa' in df.columns:
        df = df[df['tpsa'] <= 140]
    print(f"1. Lipinski & TPSA 筛选后剩余: {len(df)}")

    # --- 2.2 吸收与渗透性 ---
    if 'HIA_Hou' in df.columns:
        df = df[df['HIA_Hou'] >= 0.5]  # 分类概率：> 50% 吸收
    if 'Bioavailability_Ma' in df.columns:
        df = df[df['Bioavailability_Ma'] >= 0.5]  # 分类概率：> 50% 生物利用度
    if 'Caco2_Wang' in df.columns:
        df = df[df['Caco2_Wang'] >= -6.0]  # 回归值：> -6.0 log(cm/s) (严格：仅保留高渗透)
    print(f"2. 吸收与渗透性 (Caco2>=-6.0) 筛选后剩余: {len(df)}")

    # --- 2.3 溶解性 (收紧) ---
    if 'HydrationFreeEnergy_FreeSolv' in df.columns:
        df = df[df['HydrationFreeEnergy_FreeSolv'] <= 0]  # 严格：必须自发溶解
    print(f"3. 溶解性 筛选后剩余: {len(df)}")

    # --- 2.4 CYP抑制 (收紧) ---
    cyp_cols = ['CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Veith',
                'CYP2D6_Veith', 'CYP3A4_Veith']
    for col in cyp_cols:
        if col in df.columns:
            df = df[df[col] < 0.5]  # 收紧至 < 0.5，标准抑制剂红线
    print(f"4. CYP抑制 (<0.5) 筛选后剩余: {len(df)}")

    # --- 2.5 毒性红线 (收紧DILI) ---
    if 'AMES' in df.columns:
        df = df[df['AMES'] < 0.5]  # 严格
    if 'hERG' in df.columns:
        df = df[df['hERG'] < 0.5]  # 严格
    if 'DILI' in df.columns:
        df = df[df['DILI'] < 0.5]  # 收紧至 < 0.5，与其他毒性对齐
    if 'Carcinogens_Lagunin' in df.columns:
        df = df[df['Carcinogens_Lagunin'] < 0.5]  # 严格
    if 'ClinTox' in df.columns:
        df = df[df['ClinTox'] < 0.5]  # 严格
    print(f"5. 毒性红线 筛选后剩余: {len(df)}")

    # ================= 3. 列过滤与保存 =================
    if len(df) > 0:
        # 定义用于筛选的 19 个性质列
        selected_props = [
            'molecular_weight', 'logP', 'hydrogen_bond_acceptors', 'hydrogen_bond_donors', 'tpsa',
            'HIA_Hou', 'Bioavailability_Ma', 'Caco2_Wang', 'HydrationFreeEnergy_FreeSolv',
            'CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
            'AMES', 'hERG', 'DILI', 'Carcinogens_Lagunin', 'ClinTox'
        ]

        # 定义对应的百分位列名
        selected_percentiles = [f"{prop}_drugbank_approved_percentile" for prop in selected_props]

        # 构建最终保留的列列表
        cols_to_keep = []
        for col in df.columns:
            # 1. 保留SMILES列
            if col == smiles_column:
                cols_to_keep.append(col)
            # 2. 保留原始CSV自带的列 (排除掉ADMET-AI可能覆盖的同名列)
            elif col in original_cols:
                cols_to_keep.append(col)
            # 3. 保留用于筛选的性质列
            elif col in selected_props:
                cols_to_keep.append(col)
            # 4. 保留用于筛选的百分位列
            elif col in selected_percentiles:
                cols_to_keep.append(col)

        # 过滤掉可能不存在的列（防御性编程）
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]

        # 提取需要保留的列
        df_final = df[cols_to_keep]

        df_final.to_csv(output_csv, index=False)
        print(f"\n✅ 筛选完成！共保留 {len(df_final)} 个分子，已保存至: {output_csv}")
    else:
        print("\n⚠️ 警告: 所有分子均被过滤。")
        print("请检查输入数据是否整体存在高风险（如 hERG > 0.5 或 CYP > 0.5）。")


if __name__ == "__main__":
    INPUT_FILE = "input.csv"
    OUTPUT_FILE = "filtered_oral_drugs.csv"
    SMILES_COL = "SMILES"

    predict_and_filter_admet(INPUT_FILE, OUTPUT_FILE, SMILES_COL)
