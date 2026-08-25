import numpy as np
import pandas as pd
import ast
from rouge import Rouge 
import hanlp.pretrained
from preprocess import ckipnlptool, vocabularytool
import hanlp
from hanlp_common.document import Document
import rule  # 匯入 rule.py

if __name__ == '__main__':
    """
    讀取口語資料
    丟入規則庫
    輸出
    對照手語
    製表(依據樹長(句長))
    """
    omit_df = set(pd.read_csv('omit.csv', encoding='utf-8',)['missing_words'])
    dataset = pd.read_excel(r"output2.xlsx")
    dataset = dataset[dataset['是否採用'] == 1].reset_index(drop=True)
    raw_chinese = dataset['口語']
    sign_goal = dataset['手語_分割'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    rule.init()  # 初始化 rule.py 內的全域變數

    con = hanlp.load(hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ERNIE_GRAM)
    nlp = hanlp.pipeline() \
    .append(rule.ckiptool.tagger, input_key='tok', output_key='pos') \
    .append(con, input_key='tok', output_key='con') \
    .append(rule.merge_pos_into_con, input_key='*') \
    .append(rule.get_tree_depth, input_key='con', output_key='tree_depth')

    seg_sentence = rule.ckiptool.seg(raw_chinese)
    output_data = []  # 用來存結果的列表

    for i, s in enumerate(seg_sentence):
        doc = nlp(tok=[s])
        doc['goal'] = rule.filiter_mid(sign_goal[i])  # 使用 rule.py 內的函數
        result = rule.convert_to_sign_language(doc)  # 使用 rule.py 內的函數
        goal = doc['goal']
        scores = []

        for r, g in zip(result, goal):
            score = rule.rouge.get_scores(r, g)[0]
            scores.append(score)

        # 計算 ROUGE 分數的平均值（取小數點後 5 位）
        rouge_1_f = round(np.mean([score['rouge-1']['f'] for score in scores]), 5)
        rouge_1_p = round(np.mean([score['rouge-1']['p'] for score in scores]), 5)
        rouge_1_r = round(np.mean([score['rouge-1']['r'] for score in scores]), 5)

        rouge_2_f = round(np.mean([score['rouge-2']['f'] for score in scores]), 5)
        rouge_2_p = round(np.mean([score['rouge-2']['p'] for score in scores]), 5)
        rouge_2_r = round(np.mean([score['rouge-2']['r'] for score in scores]), 5)

        rouge_l_f = round(np.mean([score['rouge-l']['f'] for score in scores]), 5)
        rouge_l_p = round(np.mean([score['rouge-l']['p'] for score in scores]), 5)
        rouge_l_r = round(np.mean([score['rouge-l']['r'] for score in scores]), 5)

        # 將結果存入列表
        output_data.append({
            "Index": i,
            "原始句子": raw_chinese.iloc[i],
            "TOK": doc['tok'],
            "CON": doc['con'],
            "POS": doc['pos'],
            "Tree_depth": doc['tree_depth'][0],
            "手語轉換結果": result,
            "目標手語": goal,
            "ROUGE-1 F1": rouge_1_f,
            "ROUGE-1 P": rouge_1_p,
            "ROUGE-1 R": rouge_1_r,
            "ROUGE-2 F1": rouge_2_f,
            "ROUGE-2 P": rouge_2_p,
            "ROUGE-2 R": rouge_2_r,
            "ROUGE-L F1": rouge_l_f,
            "ROUGE-L P": rouge_l_p,
            "ROUGE-L R": rouge_l_r
        })

    # 轉成 DataFrame
    df_output = pd.DataFrame(output_data)

    # 存成 CSV
    df_output.to_csv("output_results.csv", index=False, encoding="utf-8-sig")  # `utf-8-sig` 避免中文亂碼
