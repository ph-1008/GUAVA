import os
import ast
import pandas as pd
from transformers import MT5Tokenizer
import re

def compress_con_tree(con_str):
    # 移除多層括號，只保留最外層與詞性資訊
    # (X word) -> X:word
    con_str = re.sub(r'\(([^() ]+)\s+([^() ]+)\)', r'(\1:\2)', con_str)  
    con_str = con_str.replace('(', ' ').replace(')', ' ')  # 括號拿掉
    con_str = re.sub(r'\s+', ' ', con_str)  # 多個空格變成一個空格
    return con_str.strip()

DATASET_PATH = os.path.join("CODE","Other_task", "data", "data.csv") # Adjust as needed
dataset = pd.read_csv(DATASET_PATH)
dataset["pos"] = dataset["pos"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
dataset["dep"] = dataset["dep"].apply(lambda x: ast.literal_eval(x))
dataset['con'] = dataset['con'].apply(lambda x: compress_con_tree(str(x)))
#print(dataset["dep"].head())
# dataset["tok"] = dataset["tok"].apply(lambda x: ast.literal_eval(x))  <POS> {' '.join(row['pos'])}  <DEP> {' '.join([i[1] for i in row['dep']])}   <CON> {row['con']}
dataset["input_text"] = dataset.apply(lambda row: f"translate Chinese to Gloss: {row['input_text']}", axis=1)

# --- START: Token Length Analysis ---
print("\n--- Analyzing Token Lengths ---")
# Initialize tokenizer (without special tokens for this general length analysis,
# or add them if your input_text/target_text already contains them BEFORE tokenization)
# For the current input_text format f"translate Chinese to Gloss: {row['input_text']}",
# the special tokens <POS>, <DEP>, <CON> are NOT part of the string yet.
# So, adding them to the tokenizer here won't affect the length of *these specific strings*.
# If your input_text construction were to include those special tokens as strings,
# then you would add them to the tokenizer here as well.
tokenizer_for_analysis = MT5Tokenizer.from_pretrained("google/mt5-base")

# If you were to use the more complex input format that includes <POS>, <DEP>, <CON> as strings:
special_tokens_dict_analysis = {'additional_special_tokens': ['<POS>', '<DEP>', '<CON>']}
tokenizer_for_analysis.add_special_tokens(special_tokens_dict_analysis)
# And ensure your dataset["input_text"] is constructed with these tokens for this analysis.

input_lengths = []
for text in dataset["input_text"]:
    tokens = tokenizer_for_analysis.encode(text, add_special_tokens=True) # encode gives token IDs
    input_lengths.append(len(tokens))

target_lengths = []
for text in dataset["target_text"]:
    tokens = tokenizer_for_analysis.encode(text, add_special_tokens=True)
    target_lengths.append(len(tokens))

print("\nInput Text Token Lengths:")
input_series = pd.Series(input_lengths)
print(f"Max length: {input_series.max()}")
print(f"Min length: {input_series.min()}")
print(f"Mean length: {input_series.mean():.2f}")
print(f"90th percentile: {input_series.quantile(0.90)}")
print(f"95th percentile: {input_series.quantile(0.95)}")
print(f"99th percentile: {input_series.quantile(0.99)}")
# You can also print the full description:
# print(input_series.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99]))


print("\nTarget Text Token Lengths:")
target_series = pd.Series(target_lengths)
print(f"Max length: {target_series.max()}")
print(f"Min length: {target_series.min()}")
print(f"Mean length: {target_series.mean():.2f}")
print(f"90th percentile: {target_series.quantile(0.90)}")
print(f"95th percentile: {target_series.quantile(0.95)}")
print(f"99th percentile: {target_series.quantile(0.99)}")
# print(target_series.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99]))

print("--- End of Token Length Analysis ---\n")
# Now, looking at these stats, you can decide on your max_length (e.g., 128).
# If your 99th percentile is, say, 110, then max_length=128 is reasonable.
# If your 99th percentile is 200, then max_length=128 will truncate a lot,
# and you might consider increasing it to 256 or something appropriate.
# --- END: Token Length Analysis ---