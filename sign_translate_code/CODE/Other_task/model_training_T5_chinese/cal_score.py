import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

if __name__ == '__main__':
    """
    步驟:
    1. 讀取 Excel 檔案
    2. 讀取手語欄位，分割
    4. 使用 BLEU/ROUGE 評估結果 
    """
    
    file_path = r"CODE\Other_task\model_training_T5_chinese\gloss_output.xlsx"
    df = pd.read_excel(file_path)
    df = df[["target_text",	"gloss_output"]]
    
    df.columns = ["pred", "refs"]  # Rename columns for consistency
    print(df.head())
    df["pred"] = df["pred"].apply(lambda x: str(x))
    df["refs"] = df["refs"].apply(lambda x: str(x))

    df["pred"] = df["pred"].apply(lambda x: ' '.join(x.replace("^^", " ^^ ").replace("//", "/").replace("/", " ").split()))
    df["refs"] = df["refs"].apply(lambda x: ' '.join(x.replace("^^", " ^^ ").replace("//", "/").replace("/", " ").split()))

    rouge = Rouge()
    
    smooth_func = SmoothingFunction().method1
    # BLEU（Before）
    bleu_1_before_scores = []
    bleu_2_before_scores = []
    bleu_3_before_scores = []
    bleu_4_before_scores = []

    for pred, ref in zip(df["pred"], df["refs"]):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(1, 0, 0, 0))
        bleu_1_before_scores.append(bleu_score)
        
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(0.5, 0.5, 0, 0))
        bleu_2_before_scores.append(bleu_score)
        
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(1/3, 1/3, 1/3, 0))
        bleu_3_before_scores.append(bleu_score)
        
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_4_before_scores.append(bleu_score)
    avg_bleu_1_before = sum(bleu_1_before_scores) / len(bleu_1_before_scores)
    avg_bleu_2_before = sum(bleu_2_before_scores) / len(bleu_2_before_scores)
    avg_bleu_3_before = sum(bleu_3_before_scores) / len(bleu_3_before_scores)
    avg_bleu_4_before = sum(bleu_4_before_scores) / len(bleu_4_before_scores)
    print(f"Avg BLEU-1: {avg_bleu_1_before:.4f}")
    print(f"Avg BLEU-2: {avg_bleu_2_before:.4f}")
    print(f"Avg BLEU-3: {avg_bleu_3_before:.4f}")
    print(f"Avg BLEU-4: {avg_bleu_4_before:.4f}")
    # ROUGE（Before）
    before_score = rouge.get_scores(df["pred"], df["refs"], avg=True)
    print(f"ROUGE-L F1: {before_score['rouge-l']['f']:.4f}")
    print(f"ROUGE-1 F1: {before_score['rouge-1']['f']:.4f}")
