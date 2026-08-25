from text2vec import SentenceModel, cos_sim
from CwnGraph import CwnImage
import json
import os
import pandas as pd

from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
class vocabularytool():
    def __init__(self):
        # 初始化模型
        self.model = SentenceModel('shibing624/text2vec-base-chinese')
        self.cwn = CwnImage.latest()
        """
        # 讀取中文詞彙
        方式1. 全部詞彙庫
        方式2. 只取資料集
        """
        # 方式1
        # with open('chinese_values2.txt', 'r', encoding='utf-8') as f:
        #     self.vocabulary = [i.strip() for i in f.readlines()]

        # 方式2
        self.vocabulary = set()
        DATASET_PATH = os.path.join("CODE","Other_task", "data", "data.csv")
        for row in pd.read_csv(DATASET_PATH)["target_text"].tolist():
            row = row.replace('^^', ' ').replace('//', '/').replace('/', ' ')
            for word in row.split():
                self.vocabulary.add(word)
        self.vocabulary = list(self.vocabulary)


        with open(r"replace_data.json",'r',encoding='utf-8') as f:
            self.replace_data = json.load(f)
        self.replace_data_keyset = set(self.replace_data.keys())
        self.vocabulary_vectors = self.model.encode(self.vocabulary)
        self.symbols = set("#、,.\n(){}[]!?;:\"'<>@%^&*~`|+-=_「」。")
        self.temp_synonyms = set()
    
    def calculate_similarity(self,word, word_list, word_vectors, top_n=1, threshold=0.8):
        word_vector = self.model.encode(word)
        similarities = [cos_sim(word_vector, vec).item() for vec in word_vectors]
        similar_words = sorted(zip(word_list, similarities), key=lambda x: x[1], reverse=True)
        return [(w, s) for w, s in similar_words[:top_n] if s >= threshold]
    
    def find_synonyms(self,word):
        """
        按數量小的先執行
        cwn 先停用
        """
        if word in self.symbols:
            return [(word, 1.0)]
        if word in self.temp_synonyms:
            return [(word, 1.0)]
        if word in self.replace_data_keyset:
            self.temp_synonyms.add(word)
            return [(word, 1.0)]
        if word in self.vocabulary:
            self.temp_synonyms.add(word)
            return [(word, 1.0)]
        # 使用 text2vec 方法
        similar_words = self.calculate_similarity(word, self.vocabulary, self.vocabulary_vectors)
        if similar_words:
            print(f"word: {word}  similar_words: {similar_words}")
            self.temp_synonyms.add(similar_words[0])
            return similar_words
        # 使用 Cwn 方法
        #print(word)

        # cwn_synonyms = self.cwn.find_lemma(word)
        # if not cwn_synonyms:
        #     return []

        # cwn_synonyms = sorted(list({w.lemma for w in cwn_synonyms}))
        # cwn_vectors = self.model.encode(cwn_synonyms)
        # refined_similar_words = self.calculate_similarity(word, cwn_synonyms, cwn_vectors)
        refined_similar_words = []
        return refined_similar_words
    
    def change(self,sentence, reference):
        """
        輸入為,以及對照組reference
        "他 跑步 快"
        """
        output_list = []
        reference = reference.split()
        sentence = sentence.split()
        for word, comp in zip(sentence, reference):
            if word == comp:
                output_list.append(word)
                continue
            synonyms = self.find_synonyms(word)
            if synonyms:
                w,s = synonyms[0]
                output_list.append(w)
            else:
                output_list.append(word)
        return " ".join(output_list)
    
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "train_set_predictions.csv"
    file_path = os.path.join(current_dir, file_name)
    # print(file_path)
    predict_data = pd.read_csv(file_path)
    predict_data["predictions"] = predict_data["predictions"].apply(lambda x: x.strip().replace('//', '/').replace('/', ' '))
    predict_data["references"] = predict_data["references"].apply(lambda x: x.strip().replace('//', '/').replace('/', ' '))
    predict_data["predictions"] = predict_data["predictions"].apply(lambda x: ' '.join(x.replace("^^", " ^^ ").split()))
    predict_data["references"] = predict_data["references"].apply(lambda x: ' '.join(x.replace("^^", " ^^ ").split()))
    voc = vocabularytool()
    rouge = Rouge()
    smooth_func = SmoothingFunction().method1

    # BLEU（Before）
    bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores = [], [], [], []

    for pred, ref in zip(predict_data["predictions"], predict_data["references"]):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        bleu_1_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(1, 0, 0, 0)))
        bleu_2_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(0.5, 0.5, 0, 0)))
        bleu_3_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(1/3, 1/3, 1/3, 0)))
        bleu_4_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(0.25, 0.25, 0.25, 0.25)))

    avg_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores) if bleu_1_scores else 0
    avg_bleu_2 = sum(bleu_2_scores) / len(bleu_2_scores) if bleu_2_scores else 0
    avg_bleu_3 = sum(bleu_3_scores) / len(bleu_3_scores) if bleu_3_scores else 0
    avg_bleu_4 = sum(bleu_4_scores) / len(bleu_4_scores) if bleu_4_scores else 0
    
    # ROUGE（Before）
    before_score = rouge.get_scores(predict_data["predictions"], predict_data["references"], avg=True)

    for index, row in predict_data.iterrows():
        if row["predictions"] != row["references"]:
            predict_data.loc[index, "predictions"] = voc.change(row["predictions"], row["references"])
    

    print(f"Avg BLEU-1: {avg_bleu_1:.4f}")
    print(f"Avg BLEU-2: {avg_bleu_2:.4f}")
    print(f"Avg BLEU-3: {avg_bleu_3:.4f}")
    print(f"Avg BLEU-4: {avg_bleu_4:.4f}")
    print(f"ROUGE-L F1: {before_score['rouge-l']['f']:.4f}")


    #print(f"ROUGE-1 F1: {rouge_1_f1:.4f}")
    # ROUGE（After）
    after_score = rouge.get_scores(predict_data["predictions"], predict_data["references"], avg=True)
    
    # BLEU（After）
    bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores = [], [], [], []
    for pred, ref in zip(predict_data["predictions"], predict_data["references"]):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        bleu_1_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(1, 0, 0, 0)))
        bleu_2_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(0.5, 0.5, 0, 0)))
        bleu_3_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(1/3, 1/3, 1/3, 0)))
        bleu_4_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_func, weights=(0.25, 0.25, 0.25, 0.25)))

    avg_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores) if bleu_1_scores else 0
    avg_bleu_2 = sum(bleu_2_scores) / len(bleu_2_scores) if bleu_2_scores else 0
    avg_bleu_3 = sum(bleu_3_scores) / len(bleu_3_scores) if bleu_3_scores else 0
    avg_bleu_4 = sum(bleu_4_scores) / len(bleu_4_scores) if bleu_4_scores else 0

    print(f"Avg BLEU-1: {avg_bleu_1:.4f}")
    print(f"Avg BLEU-2: {avg_bleu_2:.4f}")
    print(f"Avg BLEU-3: {avg_bleu_3:.4f}")
    print(f"Avg BLEU-4: {avg_bleu_4:.4f}")
    print(f"After ROUGE-L F1: {after_score['rouge-l']['f']:.7f}")
    print(f"After ROUGE-L F1: {after_score['rouge-1']['f']:.7f}")