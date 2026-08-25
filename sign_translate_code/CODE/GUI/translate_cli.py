import os
import sys
import re
import json
import argparse

import hanlp
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from text2vec import SentenceModel, cos_sim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
sys.path.append(PROJECT_ROOT)

from CODE.preprocess import ckipnlptool
from CODE.nlptool import merge_pos_into_con


def build_translator():
    ckip_tool = ckipnlptool()

    con = hanlp.load(hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ERNIE_GRAM)
    dep = hanlp.load(hanlp.pretrained.dep.CTB9_UDC_ELECTRA_SMALL, conll=False)

    nlp = hanlp.pipeline() \
        .append(ckip_tool.tagger, input_key='tok', output_key='pos') \
        .append(dep, input_key='tok', output_key='dep') \
        .append(con, input_key='tok', output_key='con') \
        .append(merge_pos_into_con, input_key='*')

    model_path = os.path.join(BASE_DIR, "best_model")
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = MT5Tokenizer.from_pretrained(model_path)

    with open(os.path.join(PROJECT_ROOT, 'chinese_values2.txt'), 'r', encoding='utf-8') as f:
        vocabulary = [i.strip() for i in f.readlines()]

    with open(os.path.join(PROJECT_ROOT, 'replace_data.json'), 'r', encoding='utf-8') as f:
        replace_data = json.load(f)

    after_process_model = SentenceModel('shibing624/text2vec-base-chinese')
    vocabulary_vectors = after_process_model.encode(vocabulary)
    replace_data_keyset = set(replace_data.keys())
    symbols = set("#、,.\n(){}[]!?;:\"'<>@%^&*~`|+-=_「」。")

    return {
        "ckip_tool": ckip_tool,
        "nlp": nlp,
        "model": model,
        "tokenizer": tokenizer,
        "after_process_model": after_process_model,
        "vocabulary": vocabulary,
        "vocabulary_vectors": vocabulary_vectors,
        "replace_data_keyset": replace_data_keyset,
        "symbols": symbols,
    }


def calculate_similarity(after_process_model, word, word_list, word_vectors, top_n=1, threshold=0.8):
    word_vector = after_process_model.encode(word)
    similarities = [cos_sim(word_vector, vec).item() for vec in word_vectors]
    similar_words = sorted(zip(word_list, similarities), key=lambda x: x[1], reverse=True)
    return [(w, s) for w, s in similar_words[:top_n] if s >= threshold]


def find_synonyms(state, word):
    if word in state["symbols"]:
        return [(word, 1.0)]
    if word in state["replace_data_keyset"]:
        return [(word, 1.0)]
    if word in state["vocabulary"]:
        return [(word, 1.0)]

    similar_words = calculate_similarity(
        state["after_process_model"],
        word,
        state["vocabulary"],
        state["vocabulary_vectors"],
    )
    return similar_words if similar_words else [(word, 1.0)]


def text_to_gloss(state, text, use_pos=True, use_dep=True, use_con=True):
    segmented_text = state["ckip_tool"].seg([text])
    nlp_result = state["nlp"](tok=segmented_text)
    pos, dep = nlp_result['pos'][0], nlp_result['dep'][0]
    constr = re.sub(r'\s+', ' ', str(nlp_result['con'][0]).replace("\n", ""))

    model_input = f"translate Chinese to Gloss: {text}"
    if use_pos:
        model_input += f" <POS> {' '.join(pos)}"
    if use_dep:
        model_input += f" <DEP> {' '.join([i[1] for i in dep])}"
    if use_con:
        model_input += f" <CON> {constr}"

    inputs = state["tokenizer"](
        model_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(state["model"].device)

    with torch.no_grad():
        translated_tokens = state["model"].generate(**inputs)
        gloss_output_raw = state["tokenizer"].batch_decode(translated_tokens, skip_special_tokens=True)

    raw_glosses = gloss_output_raw[0].strip().replace('//', '/。/').replace('^^', '/^^/').split('/')
    gloss_list = [find_synonyms(state, g)[0][0] for g in raw_glosses if g]
    return {
        "model_input": model_input,
        "gloss_raw": gloss_output_raw[0],
        "gloss_list": gloss_list,
        "gloss_joined": "/".join(gloss_list),
    }


def main():
    parser = argparse.ArgumentParser(description="Translate Chinese text to sign gloss.")
    parser.add_argument("--text", type=str, help="Input Chinese sentence")
    parser.add_argument("--no-pos", action="store_true")
    parser.add_argument("--no-dep", action="store_true")
    parser.add_argument("--no-con", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    text = args.text
    if not text:
        text = input("請輸入自然語言句子：").strip()

    if not text:
        print("錯誤：沒有輸入文字", file=sys.stderr)
        sys.exit(1)

    state = build_translator()
    result = text_to_gloss(
        state,
        text,
        use_pos=not args.no_pos,
        use_dep=not args.no_dep,
        use_con=not args.no_con,
    )

    if args.json:
        print(json.dumps({
            "text": text,
            "model_input": result["model_input"],
            "gloss_raw": result["gloss_raw"],
            "gloss_list": result["gloss_list"],
            "gloss_joined": result["gloss_joined"],
        }, ensure_ascii=False, indent=2))
    else:
        print("輸入句子:", text)
        print("模型輸入:", result["model_input"])
        print("Generated gloss:", result["gloss_list"])
        print("Gloss joined:", result["gloss_joined"])


if __name__ == "__main__":
    main()
