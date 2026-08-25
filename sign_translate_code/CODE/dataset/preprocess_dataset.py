# conda:ckip
import pandas as pd
import re

# 讀取 Excel 檔案
file_path = r"C:\Users\Dreamy\Documents\work\paper\output.xlsx"
df = pd.read_excel(file_path)


def ckip():
    from ckip_transformers.nlp import CkipPosTagger, CkipWordSegmenter

    def pack_ws_pos_sentece(sentence_ws, sentence_pos):
        assert len(sentence_ws) == len(sentence_pos)
        res = []
        for word_ws, word_pos in zip(sentence_ws, sentence_pos):
            res.append(f"{word_ws}({word_pos})")
        return "\u3000".join(res)
    

    # Initialize drivers
    ws_driver = CkipWordSegmenter(model="bert-base")
    pos_driver = CkipPosTagger(model="bert-base")



    # 取得口語資料並進行斷詞和詞性標注
    str_list = []  

    for string in df['口語']:
        string = string.replace(',','&').replace('。','&').replace('，','&').replace('！','')
        str_list.append(string)
    ws_results = ws_driver(str_list)  # 斷詞
    pos_results = pos_driver(ws_results)  # 詞性標注

    # 新增欄位：ws、pos 和 pack_ws_pos_sentece(ws, pos)
    df['ws'] = ws_results
    df['pos'] = pos_results
    df['ws_pos_sentence'] = [
        pack_ws_pos_sentece(ws, pos) for ws, pos in zip(ws_results, pos_results)
    ]

    # 顯示結果
    print(df.head())




def sign_process():
    sign_list = list(df['手語'])
    result = []
    for item in sign_list:
        item = item.replace(" ", "").replace("(/", "/(").replace("/)", ")/").replace("*", "")  # 去除空格
        segments = []  # 存放每組資料

        # 先以 // 和 ^^ 作為分割，取得句子的段落
        for part in item.replace("^^", "?//").split("//"):
            if part:  # 避免空段落
                # 再以 / 分割該段落
                if part[0] == '/': part = part[1:]
                if part[-1] == '/': part = part[:-1]
                split_parts = part.split("/")
                
                segments.append(split_parts)

        result.append(segments)  # 儲存所有段落
    # for s in result:
    #     print(s)
    df['手語_分割'] = result
    return result

def extract_bracket(sign_split):
    import json
    result = {}
    for data in sign_split:
        # 處理資料
        for row in data:
            for item in row:
                # 找出[]中的部分
                matches = re.findall(r'\[(.*?)\]', item)
                for match in matches:
                    # 找到[]左邊的部分，提取最後一個詞
                    left_part = re.split(r'[](|:]', item.split(f'[{match}]')[0])[-1]
                    if left_part not in result:
                        result[left_part] = match
                    # # 將結果加入字典，確保一個key對應多個值時以列表儲存
                    # if match not in result:
                    #     result[match] = []
                    # if left_part not in result[match]:
                    #     result[match].append(left_part)

    # 輸出結果
    print(result)
    # 將結果存成 JSON 檔案
    output_file = "replace_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    return result

# extract_bracket(sign_split=sign_split)
# 如果需要將結果保存到檔案中
sign_process()
ckip()
df.to_excel("output2.xlsx", index=False)