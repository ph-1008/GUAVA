
import os
import pandas as pd

from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "predictions.csv"
    file_path = os.path.join(current_dir, file_name)
    # print(file_path)
    predict_data = pd.read_csv(file_path)
    predict_data["predictions"] = predict_data["predictions"].apply(lambda x: ' '.join(x.replace("^^", " ^^ ").split()))
    predict_data["references"] = predict_data["references"].apply(lambda x: ' '.join(x.replace("^^", " ^^ ").split()))
    rouge = Rouge()
    smooth_func = SmoothingFunction().method1

    # BLEU（Before）
    bleu_1_before_scores = []
    bleu_2_before_scores = []
    bleu_3_before_scores = []
    bleu_4_before_scores = []
    for pred, ref in zip(predict_data["predictions"], predict_data["references"]):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func,weights=(1, 0, 0, 0))
        bleu_1_before_scores.append(bleu_score)
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func,weights=(0.5, 0.5, 0, 0))
        bleu_2_before_scores.append(bleu_score)
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func,weights=(1/3, 1/3, 1/3, 0))
        bleu_3_before_scores.append(bleu_score)
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func,weights=(0.25, 0.25, 0.25, 0.25))
        bleu_4_before_scores.append(bleu_score)

    avg_bleu_1_before = sum(bleu_1_before_scores) / len(bleu_1_before_scores)
    avg_bleu_2_before = sum(bleu_2_before_scores) / len(bleu_2_before_scores)
    avg_bleu_3_before = sum(bleu_3_before_scores) / len(bleu_3_before_scores)
    avg_bleu_4_before = sum(bleu_4_before_scores) / len(bleu_4_before_scores)
    
    # ROUGE（Before）
    before_score = rouge.get_scores(predict_data["predictions"], predict_data["references"], avg=True)


    print(f"Avg BLEU-1: {avg_bleu_1_before:.4f}")
    print(f"Avg BLEU-2: {avg_bleu_2_before:.4f}")
    print(f"Avg BLEU-3: {avg_bleu_3_before:.4f}")
    print(f"Avg BLEU-4: {avg_bleu_4_before:.4f}")
    print(f"ROUGE-L F1: {before_score['rouge-l']['f']:.4f}")
    print(f"ROUGE-1 F1: {before_score['rouge-1']['f']:.4f}")
"""

Avg BLEU-1: 
Avg BLEU-2: 
Avg BLEU-3: 
Avg BLEU-4: 
ROUGE-L F1: 
ROUGE-1 F1: 





"""