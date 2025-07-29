import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
print("加载数据...")
df = pd.read_csv("/Users/ruihan/Downloads/IWFP-main/data/Untitled2.csv")
print(f"原始数据形状: {df.shape}")

# 过滤数据 (a7 == 1)
df_filtered = df[df['a7'] == 1]
print(f"过滤后数据形状: {df_filtered.shape}")

print("\n=== 原始数据缺失值检查 ===")

# 1. 检查每列的缺失值数量
print("各列缺失值情况:")
missing_counts = df_filtered.isnull().sum()
missing_percent = (df_filtered.isnull().sum() / len(df_filtered)) * 100

missing_info = pd.DataFrame({
    '缺失值数量': missing_counts,
    '缺失值百分比': missing_percent
}).sort_values('缺失值数量', ascending=False)

# 只显示有缺失值的列
missing_columns = missing_info[missing_info['缺失值数量'] > 0]
if len(missing_columns) > 0:
    print(f"发现 {len(missing_columns)} 列有缺失值:")
    print(missing_columns.head(20))  # 显示前20列
else:
    print("没有发现缺失值！")

print(f"\n总缺失值数量: {df_filtered.isnull().sum().sum()}")
print(f"总数据点数量: {df_filtered.size}")
print(f"总缺失值比例: {(df_filtered.isnull().sum().sum() / df_filtered.size) * 100:.4f}%")

# 2. 检查每行的缺失值数量
print("\n每行缺失值数量分布:")
row_missing = df_filtered.isnull().sum(axis=1)
print(row_missing.value_counts().sort_index().head(10))

# 3. 显示完全没有缺失值的行数
complete_rows = (row_missing == 0).sum()
print(f"\n完整行数（无缺失值）: {complete_rows}")
print(f"完整行比例: {complete_rows/len(df_filtered)*100:.2f}%")

# 4. 检查关键变量的缺失值情况
print("\n=== 关键变量缺失值检查 ===")
key_original_vars = ['a4', 'a5', 'a7', 'a9', 'a10', 'a18', 'a20', 'a48_a49', 'a50_a51', 
                    'a51_a52', 'a52_a53', 'a131_a119', 'a130_a118', 'a132_a120']

for var in key_original_vars:
    if var in df_filtered.columns:
        missing_count = df_filtered[var].isnull().sum()
        missing_percent = (missing_count / len(df_filtered)) * 100
        if missing_count > 0:
            print(f"{var}: {missing_count} 个缺失值 ({missing_percent:.2f}%)")
        else:
            print(f"{var}: 无缺失值")

# 5. 检查一些具体的缺失值模式
print("\n=== 缺失值模式分析 ===")
rows_with_missing = df_filtered.isnull().any(axis=1)
print(f"有缺失值的行数: {rows_with_missing.sum()}")
print(f"有缺失值的行比例: {rows_with_missing.sum()/len(df_filtered)*100:.2f}%")

if rows_with_missing.sum() > 0:
    print("\n前5行有缺失值的数据索引:")
    missing_indices = df_filtered[rows_with_missing].index[:5]
    print(missing_indices.tolist())
    
    # 检查这些行中哪些列有缺失值
    print("\n这些行中的缺失值分布:")
    for idx in missing_indices:
        missing_cols = df_filtered.loc[idx].isnull()
        if missing_cols.any():
            missing_col_names = missing_cols[missing_cols].index.tolist()
            print(f"行 {idx}: {len(missing_col_names)} 个缺失值在列: {missing_col_names[:10]}...")  # 只显示前10个列名

print("\n检查完成！") 