# from sentence_transformers import SentenceTransformer
# import numpy as np
# from ckip_transformers.nlp import CkipWordSegmenter
# # 載入預訓練的語意模型
# model = SentenceTransformer('all-MiniLM-L6-v2')  # 小型但高效的語意模型

# # 詞彙資料庫
# def get_vocabulary():
#     with open('chineese_values.txt', 'r', encoding='utf-8') as f:
#         vocabulary = [i[:-1] for i in f.readlines()]
#         f.close()
#     return vocabulary


# # 輸入句子
# input_sentence = "我喜歡健康的生活和運動"

# # 將詞彙資料庫轉換為向量
# vocabulary_embeddings = model.encode(get_vocabulary())



# # 計算餘弦相似度
# def cosine_similarity(vec1, vec2):
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# # 找出與單詞最相似的詞
# def find_closest_word(word, vocabulary, vocab_embeddings):
#     if word in vocabulary:
#         return word
#     else:
#         word_embedding = model.encode(word)
#         similarities = [cosine_similarity(word_embedding, vec) for vec in vocab_embeddings]
#         best_index = np.argmax(similarities)
#         return vocabulary[best_index]

# def cut(input_sentence):
#     ws_driver  = CkipWordSegmenter(model="bert-base")
#     ws  = ws_driver([input_sentence])
#     return ws[0]

# # 主邏輯：斷詞並比對最接近的詞
# def reconstruct_sentence(input_sentence, vocabulary, vocab_embeddings):
#     words = cut(input_sentence)  # 使用 ckip 進行中文斷詞
#     reconstructed_words = [find_closest_word(word, vocabulary, vocab_embeddings) for word in words]
#     print(reconstructed_words)
#     return "".join(reconstructed_words)

# # 執行
# result_sentence = reconstruct_sentence(input_sentence, get_vocabulary(), vocabulary_embeddings)
# print("重建的句子：", result_sentence)


def FastText_method():
    """
    Use conda
    conda activate genism
    使用 FastText 模型查詢同義詞。
    :param word: 要查詢的詞
    :param model: FastText 模型
    :param top_n: 返回的同義詞數量
    :return: 同義詞列表
    """

    from gensim.models import KeyedVectors

    # 載入 FastText 中文詞向量
    model_path = r"D:\Fasttext\w2v_CNA_ASBC_300d.vec"  # 替換為實際文件路徑
    model = KeyedVectors.load_word2vec_format(model_path)
    def find_synonyms(word, model, top_n=10):

        try:
            similar_words = model.most_similar(word, topn=top_n)
            return similar_words
        except KeyError:
            return ["詞不在模型中，請嘗試其他詞。"]

    # 測試
    words = ['你', '玩', '過', '橄欖球', '嗎']
    for word in words:
        synonyms = find_synonyms(word, model)
        print(f"{word} 的同義詞：")
        for w,s in synonyms:
            print(round(s,3), w)
        print("-"*50)

def w2v_CNA_ASBC_300d():
    from gensim.models import Word2Vec

    # 載入 FastText 中文詞向量
    model_path = r"D:\Fasttext\w2v_CNA_ASBC_300d.vec"  # 替換為實際文件路徑
    model = Word2Vec.load(model_path)
    def find_synonyms(word, model, top_n=10):

        try:
            similar_words = model.most_similar(word, topn=top_n)
            return similar_words
        except KeyError:
            return ["詞不在模型中，請嘗試其他詞。"]

    # 測試
    words = ['你', '玩', '過', '橄欖球', '嗎']
    for word in words:
        synonyms = find_synonyms(word, model)
        print(f"{word} 的同義詞：")
        for w,s in synonyms:
            print(round(s,3), w)
        print("-"*50)


def text2vec_method():
    from text2vec import SentenceModel, cos_sim

    # 加載預設中文模型
    model = SentenceModel('shibing624/text2vec-base-chinese')

    # 詞彙資料庫
    def get_vocabulary():
        with open('chineese_values.txt', 'r', encoding='utf-8') as f:
            vocabulary = [i.strip() for i in f.readlines()]
        return vocabulary

    word_list = get_vocabulary()
    word_vectors = model.encode(word_list)

    def find_synonyms(word, word_list, model, top_n=5):
        word_vector = model.encode(word)

        # 計算相似度
        similarities = [cos_sim(word_vector, vec).item() for vec in word_vectors]  # 使用 `.item()` 將 Tensor 轉為 float

        # 配對並排序
        similar_words = sorted(zip(word_list, similarities), key=lambda x: x[1], reverse=True)
        return similar_words[:top_n]

    # 測試輸入詞
    words = ['是', '啊', '！', '不管', '碰到', '什麼', '問題', '，', '一定', '要', '找', '王', '組長', '，', '有', '些', '送貨員', '態度', '不', '好', '，', '你', '不要', '跟', '他們', '當面', '起', '衝突', '，', '可以', '把', '情形', '告訴', '王', '組長', '請', '他', '協助', '解決', '。']

    for word in words:
        synonyms = find_synonyms(word, word_list, model)
        print(f"{word} 的同義詞：")
        for w, s in synonyms:
            print(f"{round(s, 3)}: {w}")
        print("-" * 50)



def Cwn_method():
    from CwnGraph import CwnImage

    cwn = CwnImage.latest()


    # 測試輸入詞
    words = ['是', '啊', '！', '不管', '碰到', '什麼', '問題', '，', '一定', '要', '找', '王', '組長', '，', '有', '些', '送貨員', '態度', '不', '好', '，', '你', '不要', '跟', '他們', '當面', '起', '衝突', '，', '可以', '把', '情形', '告訴', '王', '組長', '請', '他', '協助', '解決', '。']

    for word in words:
        synonyms = cwn.find_lemma(word)
        print(f"{word} 的同義詞：")
        if len(synonyms):
            synonyms = sorted(list({w.lemma for w in synonyms}))
            for w in synonyms:
                print(f"{w}")
        print("-" * 50)


def combined_method():
    from text2vec import SentenceModel, cos_sim
    from CwnGraph import CwnImage

    # 初始化模型
    model = SentenceModel('shibing624/text2vec-base-chinese')
    cwn = CwnImage.latest()

    # 加載詞彙庫
    def get_vocabulary():
        with open('chineese_values.txt', 'r', encoding='utf-8') as f:
            vocabulary = [i.strip() for i in f.readlines()]
        return vocabulary

    vocabulary = get_vocabulary()
    vocabulary_vectors = model.encode(vocabulary)

    # 計算相似度
    def calculate_similarity(word, word_list, word_vectors, top_n=5, threshold=0.8):
        word_vector = model.encode(word)
        similarities = [cos_sim(word_vector, vec).item() for vec in word_vectors]
        similar_words = sorted(zip(word_list, similarities), key=lambda x: x[1], reverse=True)
        return [(w, s) for w, s in similar_words[:top_n] if s >= threshold]

    # 查找詞的同義詞
    def find_synonyms(word, vocabulary, vocabulary_vectors):
        if word in vocabulary:
            return [(word, 1.0)]

        # 使用 text2vec 方法
        similar_words = calculate_similarity(word, vocabulary, vocabulary_vectors)
        if similar_words:
            return similar_words

        # 使用 Cwn 方法
        cwn_synonyms = cwn.find_lemma(word)
        if not cwn_synonyms:
            return []

        cwn_synonyms = sorted(list({w.lemma for w in cwn_synonyms}))
        cwn_vectors = model.encode(cwn_synonyms)
        refined_similar_words = calculate_similarity(word, cwn_synonyms, cwn_vectors)
        
        return refined_similar_words

    # 測試輸入詞
    words = ['是', '啊', '！', '不管', '碰到', '什麼', '問題', '，', '一定', '要', '找', '王', '組長', '，', '有', '些', '送貨員', '態度', '不', '好', '，', '你', '不要', '跟', '他們', '當面', '起', '衝突', '，', '可以', '把', '情形', '告訴', '王', '組長', '請', '他', '協助', '解決', '。']

    for word in words:
        synonyms = find_synonyms(word, vocabulary, vocabulary_vectors)
        print(f"{word} 的同義詞：")
        if synonyms:
            for w, s in synonyms:
                print(f"{round(s, 3)}: {w}")
        else:
            print("無相關結果")
        print("-" * 50)

# 調用方法
FastText_method()