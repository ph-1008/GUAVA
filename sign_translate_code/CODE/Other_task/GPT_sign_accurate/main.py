import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


if __name__ == '__main__':
    """
    驗證GPT 性能是否準確
    步驟:
    1. 讀取 Excel 檔案
    2. 讀取手語欄位，分割
    3. ckip 斷詞
    4. 使用 BLEU/ROUGE 評估 GPT 輸出結果 
    """
    
    file_path = r"CODE\Other_task\GPT_sign_accurate\sampled_100_gpt.xlsx"
    df = pd.read_excel(file_path)

    
    rouge = Rouge()
    
    smooth_func = SmoothingFunction().method1
    # BLEU（Before）
    bleu_1_before_scores = []
    bleu_2_before_scores = []
    bleu_3_before_scores = []
    bleu_4_before_scores = []

    result = []
    sign = df['手語'].str.replace(r'[^\u4e00-\u9fff\s1-9]', ' ',regex=True)
    sign = sign.str.replace(r'//', '/', regex=True)  # 去除/
    sign = sign.str.replace(r'/', ' ', regex=True)  # 去除/
    sign = sign.str.replace(r'^^', ' ^^ ', regex=True)  # 去除^^
    sign = sign.str.replace(r'\s+', ' ', regex=True)  # 去除多餘空格
    
    for pred, ref in zip(df['GPT'],sign):
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
    before_score = rouge.get_scores(df['GPT'], sign, avg=True)
    print(f"ROUGE-L F1: {before_score['rouge-l']['f']:.4f}")
    print(f"ROUGE-1 F1: {before_score['rouge-1']['f']:.4f}")

    # for a,b in zip(sign,df['GPT']):
    #     a = ' '.join(a.split(' '))
    #     score = (rouge.get_scores(a,b)[0])
    #             # 計算 ROUGE 分數的平均值（取小數點後 5 位）
    #     rouge_1_f = round(score['rouge-1']['f'], 4)
    #     rouge_1_p = round(score['rouge-1']['p'], 4)
    #     rouge_1_r = round(score['rouge-1']['r'], 4)
    #     rouge_2_f = round(score['rouge-2']['f'], 4)
    #     rouge_2_p = round(score['rouge-2']['p'], 4)
    #     rouge_2_r = round(score['rouge-2']['r'], 4)
    #     rouge_l_f = round(score['rouge-l']['f'], 4)
    #     rouge_l_p = round(score['rouge-l']['p'], 4)
    #     rouge_l_r = round(score['rouge-l']['r'], 4)

    #     result.append([rouge_1_f,rouge_1_p,rouge_1_r,rouge_2_f,rouge_2_p,rouge_2_r,rouge_l_f,rouge_l_p,rouge_l_r])

    # df_result = pd.DataFrame(result,columns=['rouge_1_f','rouge_1_p','rouge_1_r','rouge_2_f','rouge_2_p','rouge_2_r','rouge_l_f','rouge_l_p','rouge_l_r'])
    # 
    # df_result.to_csv(r'CODE\Other_task\GPT_sign_accurate\gpt_rouge.csv',index=False)
    

