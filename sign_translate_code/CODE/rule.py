import hanlp.pretrained
import phrasetree.tree
from preprocess import ckipnlptool, vocabularytool
from nlptool import *
import hanlp
from hanlp_common.document import Document
import pandas as pd
from rouge import Rouge
import numpy as np
import re
from phrasetree.tree import Tree
from typing import List
ckiptool = None
voctool = None
rouge = None

def find_object_phrase(tree, depth=0):
    """
    遞迴搜尋 Constituency Tree，尋找受詞 (名詞短語 NP)
    """
    if not isinstance(tree, list) or not tree:  # 若為詞，則返回 None
        return None, -1
    
    label, children = tree[0], tree[1:][0]  # 提取標籤 (e.g., NP, VP, V)
    
    while isinstance(label, list):
        label = label[0]

    if label == "NP" or label == "Na":  # 若當前節點為名詞短語
        return tree, depth

    # 遞迴遍歷所有子節點，尋找最深的 NP
    best_np, max_depth = None, -1
    for child in children:
        np, d = find_object_phrase(child, depth + 1)
        if d > max_depth:
            best_np, max_depth = np, d

    return best_np, max_depth

def init():
    global ckiptool, voctool, rouge
    ckiptool = ckipnlptool()
    voctool = vocabularytool()
    rouge = Rouge()

def filiter_mid(dataset):
    processed_data = []
    for row in dataset:
        new_row = []
        for item in row:
            item = item.replace("(", "").replace(")", "").replace('++', '__TEMP__')
            item = re.sub(r'(?<!\+)\+(?!\+)', ' ', item)  # 只替換單個 '+'
            item = item.replace('__TEMP__', '++')         # 恢復 '++'
            item = re.sub(r'\{.*?\}', '', item)           # 移除所有 {.*?}
            if "[" in item and "]" in item:
                new_item = item[item.find("[")+1:item.find("]")]
            else:
                new_item = item
            new_row.append(new_item)
        
        processed_data.append(' '.join(new_row))
    return processed_data

def get_rouge_score(sample,goal):
    scores = []
    for r,g in zip(sample,goal):
        score = rouge.get_scores(r, g)[0]
        scores.append(score)

    return np.mean([score['rouge-l']['f'] for score in scores])

def convert_to_sign_language(text_pack:Document):
    omit_df = set(pd.read_csv('omit.csv', encoding='utf-8',)['missing_words'])
    """
    sentence 口語
    步驟:
    1. 濾除特定字
        a. replace data 替換字詞
        b. omit 省略字詞
    2. 特定字詞 特定處理
        a. '是'
    評分 75%以上出去，剩下DEP
    """
    

    # print(tok)
    """
    濾除奇奇怪怪有的沒的
    """
    omit_tok = [word for word in text_pack['tok'][0] if word not in omit_df]
    print(omit_tok)
    voc =  voctool.change([omit_tok])
    print(voc)
    """
    替換字詞
    """
    tok = [voctool.replace_data.get(word, word) for word in voc[0]]

    """
    濾除'是'
    這/是
    就/是
    是,在附近
    是啊!
    都/是
    """
    SHI_prefix = ('這', '就', '都')
    SHI_suffix = ('!', ',', '。')
    filtered_tok = []
    skip_next = False  # 用來標記是否要跳過下一個詞

    for i, word in enumerate(tok):
        if skip_next:
            skip_next = False  # 跳過後立即重置
            continue

        if word.startswith(SHI_prefix) and i + 1 < len(tok) :
            if tok[i + 1] == '是':
                skip_next = True  # 跳過 "是"
            elif tok[i + 1] == '人':
                skip_next = True  # 跳過 "人"
                filtered_tok.append('他')
        elif word == '是':
            if i + 1 < len(tok) and tok[i + 1] in SHI_suffix:
                # 如果 "是" 的下一個詞在 SHI_suffix，則保留 "是" 及下一個詞
                filtered_tok.append(word)
                filtered_tok.append(tok[i + 1])
                skip_next = True  # 跳過下一個詞，因為已經加入
            else:
                continue  # 直接跳過 "是"
        
        else:
            filtered_tok.append(word)

    temp = []
    result = []
    for word in filtered_tok:
        if word == '?':
            if temp:
                temp[-1] = temp[-1]+'?'
            else:
                temp.append(word)
        elif word in SHI_suffix:
            if temp:
                result.append(' '.join(temp).replace('++', '__TEMP__'))
                result[-1] = re.sub(r'(?<!\+)\+(?!\+)', ' ', result[-1])  # 只替換單個 '+'
                result[-1] = result[-1].replace('__TEMP__', '++')         # 恢復 '++'
                temp = []
        else:
            temp.append(word)
    if temp:
        result.append(' '.join(temp).replace('++', '__TEMP__'))
        result[-1] = re.sub(r'(?<!\+)\+(?!\+)', ' ', result[-1])
        result[-1] = result[-1].replace('__TEMP__', '++')


    if get_rouge_score( result,text_pack['goal']) > 0.75:
        return result
    else:
        """
        DEP 分析
        """
        
        con_analyse: List[Tree] = text_pack['con']
        tree = con_analyse[0]
        max_depth = -1
        deepest_np = None

        for node in tree:  
            if node.label() in {"NP", "Na"}:  # 確保是名詞短語
                if node.height() > max_depth:  # 更新最深的受詞
                    max_depth = node.height()
                    deepest_np = node
        print("最深的受詞:", deepest_np)
    return result


if __name__ == '__main__':
    """
    規則式生成手語至中文
    輸入資料格式
    [[資料A],[資料B].....]

    1. ckip 斷詞
    2. 詞彙庫篩選
    3. 轉換成hanlp結構(句式分析)
    4. 轉換成中文
    """
    sentence = ['這人表裡如一','下巴男這人怎麼樣?','大概8個人。','你認識他嗎?','那聾人我不認識','兩個兒子,一個女兒。','我對花粉過敏。','哈哈,我忘了!','您從哪來?','節稅方法?','我喜歡學自然手語']
    goals = [[['他', '表面', '裡面', '一樣']],[['下巴男', '人', '如何?']],[['大概', '8個人']],[['他', '(你)', '認識?']],[['那', '聾人'], ['我對他', '不認識']],[['兒子', '兩'], ['女兒', 'ー']],[['花粉', '我', '對⋯過敏[敵對]']],[['哈哈', '我忘了']],[['您', '哪', '來?']],[['稅', '節省', '什麼?']],[['數學', '我', '沒輒[敵對]']]]
    goals = [filiter_mid(goal) for goal in goals]
    # sentence = ['你是聾人嗎?', '警方說這是民事糾紛', '他是誰?','是,在附近','是啊!不管碰到什麼問題,一定要找王組長']
    init()
    con = hanlp.load(hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ERNIE_GRAM)
    dep = hanlp.load(hanlp.pretrained.dep.CTB9_UDC_ELECTRA_SMALL, conll=False)
    nlp = hanlp.pipeline() \
    .append(ckiptool.tagger, input_key='tok', output_key='pos') \
    .append(dep, input_key='tok', output_key='dep') \
    .append(con, input_key='tok', output_key='con') \
    .append(merge_pos_into_con, input_key='*') \
    .append(get_tree_depth, input_key='con', output_key='tree_depth')
    
    seg_sentence = ckiptool.seg(sentence)
    # seg_sign_sentence = voctool.change(seg_sentence)
    output = []
    for i,s in enumerate(seg_sentence):
        #print(s)
        doc = nlp(tok=[s])
        doc['goal'] = goals[i]
        doc.pretty_print()
        print(doc)
        result = convert_to_sign_language(doc)
        print(result)
        output.append(result)
        print('-' * 50)
    for s in output:
        print(s)
    print(seg_sentence )
    # print(seg_sign_sentence)