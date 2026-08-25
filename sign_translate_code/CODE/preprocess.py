from ckip_transformers.nlp import CkipWordSegmenter,CkipPosTagger
from text2vec import SentenceModel, cos_sim
from CwnGraph import CwnImage
import json
"""
各類工具
1. ckip 斷詞
2. 詞彙庫篩選
"""

class ckipnlptool():
    def __init__(self):
        self.ws_driver = CkipWordSegmenter(model="bert-base")
        self.pos_driver = CkipPosTagger(model="bert-base")
    def seg(self, string):
        """
        記得輸入為
        [string]
        """
        string[0] = string[0].replace('!','').replace(' ','').replace('。',',')
        ws_results = self.ws_driver(string)
        return ws_results
    def tagger(self,string):
        ans = self.pos_driver(string)
        print("finish pos")
        return ans
    
class vocabularytool():
    def __init__(self):
        # 初始化模型
        self.model = SentenceModel('shibing624/text2vec-base-chinese')
        self.cwn = CwnImage.latest()
        with open('chineese_values.txt', 'r', encoding='utf-8') as f:
            self.vocabulary = [i.strip() for i in f.readlines()]
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
    
    def change(self,sentence):
        """
        輸入為
        [['a','b',"c"]]
        """
        change_list = []
        for words in sentence:
            tmp = []
            for word in words:
                synonyms = self.find_synonyms(word)
                if synonyms:
                    w,s = synonyms[0]
                    tmp.append(w)
                else:
                    tmp.append('ERROR')
            change_list.append(tmp)
        return change_list