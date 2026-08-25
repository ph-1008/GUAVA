import pandas as pd

# 讀取 words
with open('chineese_values.txt', 'r', encoding='utf-8') as f:
    words = {line.strip() for line in f}  # 使用 set 儲存唯一值

# 讀取 Excel 檔案
file_path = r"C:\Users\Dreamy\Documents\work\paper\output2.xlsx"
df = pd.read_excel(file_path)

# 過濾掉 '是否採用' 欄位包含 '@' 的資料
df = df[~df['是否採用'].astype(str).str.contains('@', na=False)]

# 提取 ws 欄位的資料，將所有詞彙合併成一個 set
df['ws'] = df['ws'].apply(eval)  # 假設 ws 欄位的值是字串形式的列表
all_words = {word for sublist in df['ws'] for word in sublist}

# 找出不在 words 裡的詞
missing_words = all_words - words

# 存成 CSV
otput_file = "omit.csv"
pd.DataFrame({'missing_words': list(missing_words)}).sort_values(by='missing_words').to_csv(otput_file, index=False, encoding='utf-8')

print(f"完成！遺漏的詞已儲存至 {otput_file}")
