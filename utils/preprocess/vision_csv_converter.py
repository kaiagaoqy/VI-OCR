import pandas as pd
import re
import numpy as np

# 读取原始CSV
df = pd.read_csv("/cis/home/qgao14/my_documents/VIOCR_infer_models/data/human/human_measured_vision.csv")

# 拆分 Age/Sex 字段
df[['Age', 'Sex']] = df['Age/Sex'].str.extract(r'(\d+)\s*,\s*([MF])')

# 提取 VA 信息
def extract_va(text):
    result = {'VA_OD': None, 'VA_OS': None, 'VA_OU': None}
    if pd.isna(text):
        return result
    text = str(text)
    od = re.search(r'OD\s*([\w/+-]+)', text)
    os = re.search(r'OS\s*([\w/+-]+)', text)
    ou = re.search(r'OU\s*([\w/+-]+)', text)
    if od: result['VA_OD'] = od.group(1)
    if os: result['VA_OS'] = os.group(1)
    if ou: result['VA_OU'] = ou.group(1)
    return result

va_df = df['VA'].apply(extract_va).apply(pd.Series)
df = pd.concat([df, va_df], axis=1)

# 提取 CS 信息
def extract_cs(text):
    result = {'CS_OD': None, 'CS_OS': None, 'CS_OU': None}
    if pd.isna(text):
        return result
    text = str(text)
    od = re.search(r'OD\s*([\d.]+)', text)
    os = re.search(r'OS\s*([\d.]+)', text)
    ou = re.search(r'OU\s*([\d.]+)', text)
    if od: result['CS_OD'] = od.group(1)
    if os: result['CS_OS'] = os.group(1)
    if ou: result['CS_OU'] = ou.group(1)
    return result

cs_df = df['CS'].apply(extract_cs).apply(pd.Series)
df = pd.concat([df, cs_df], axis=1)

# 将VA转换为logMAR
def va_to_logmar(va):
    """
    将视觉表值（例如 20/40）转换为 logMAR。
    NLP, CF, HM, LP 等特殊情况也进行近似转换。
    """
    if pd.isna(va):
        return np.nan
    va = str(va).strip().upper()
    if "NLP" in va:
        return 2.9
    elif "LP" in va:
        return 2.7
    elif "HM" in va:
        return 2.3
    elif "CF" in va:
        return 1.9
    match = re.match(r'(\d+)[/](\d+)', va)
    if match:
        num, den = map(float, match.groups())
        return round(np.log10(den / num), 2)
    return np.nan

df['VA_OD_logMAR'] = df['VA_OD'].apply(va_to_logmar)
df['VA_OS_logMAR'] = df['VA_OS'].apply(va_to_logmar)
df['VA_OU_logMAR'] = df['VA_OU'].apply(va_to_logmar)

# 只保留需要的列
result = df[['SubID', 'Name', 'Age', 'Sex', 'Occular Hx',
             'VA_OD', 'VA_OD_logMAR',
             'VA_OS', 'VA_OS_logMAR',
             'VA_OU', 'VA_OU_logMAR',
             'CS_OD', 'CS_OS', 'CS_OU']]

# 输出结果
result.to_csv("/cis/home/qgao14/my_documents/VIOCR_infer_models/data/human/human_measured_vision_cleaned.csv", index=False)
