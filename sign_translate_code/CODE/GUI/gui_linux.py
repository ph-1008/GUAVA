import os
import sys
import json
import re
import html as ht
import webbrowser

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPlainTextEdit, QPushButton,
    QTextBrowser, QVBoxLayout, QWidget, QMessageBox,
    QHBoxLayout, QCheckBox, QLabel, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QTextCursor, QFont

from nltk.tree import Tree
import hanlp
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
from text2vec import SentenceModel, cos_sim

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
DATA_ROOT = PROJECT_ROOT
sys.path.append(PROJECT_ROOT)

from CODE.preprocess import ckipnlptool
from CODE.nlptool import merge_pos_into_con


def get_node_depths(tree_str):
    try:
        tree = Tree.fromstring(tree_str)
    except ValueError as e:
        print(f"Error parsing tree string: {tree_str}\nError: {e}")
        return {}
    result = {}
    node_queue = [(tree, 0)]
    node_counts = {}
    while node_queue:
        current_node, current_depth = node_queue.pop(0)
        if not isinstance(current_node, Tree):
            continue
        node_label = current_node.label()
        count = node_counts.get(node_label, 0)
        unique_label = f"{node_label}_{count}" if count > 0 else node_label
        node_counts[node_label] = count + 1
        result[unique_label] = current_depth
        for child in current_node:
            if isinstance(child, Tree):
                node_queue.append((child, current_depth + 1))
    return result


def clean_text(item):
    if not isinstance(item, str):
        return item
    item = item.replace("(", "").replace(")", "")
    item = re.sub(r'\{.*?\}', '', item)
    item = re.sub(r'\[.*?\]', '', item)
    return item.strip()


class Text2signgui:
    """
    MobaXterm-friendly lite GUI:
    - no microphone input
    - no QtMultimedia
    - no embedded YouTube / QWebEngine
    - only text conversion + clickable links / local file info
    """

    def __init__(self, parent=None):
        self.parent = parent
        self.initUI()

        self.ckip_tool = ckipnlptool()
        try:
            con = hanlp.load(hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ERNIE_GRAM)
            dep = hanlp.load(hanlp.pretrained.dep.CTB9_UDC_ELECTRA_SMALL, conll=False)
        except Exception as e:
            print(f"Error loading HanLP models: {e}\nEnsure HanLP models are downloaded/accessible.")
            sys.exit(1)

        self.nlp = hanlp.pipeline() \
            .append(self.ckip_tool.tagger, input_key='tok', output_key='pos') \
            .append(dep, input_key='tok', output_key='dep') \
            .append(con, input_key='tok', output_key='con') \
            .append(merge_pos_into_con, input_key='*')

        model_path = os.path.join(BASE_DIR, "best_model")
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = MT5Tokenizer.from_pretrained(model_path)

        self.after_process_model = SentenceModel('shibing624/text2vec-base-chinese')

        with open(os.path.join(PROJECT_ROOT, 'chinese_values2.txt'), 'r', encoding='utf-8') as f:
            self.vocabulary = [i.strip() for i in f.readlines()]

        with open(os.path.join(PROJECT_ROOT, 'replace_data.json'), 'r', encoding='utf-8') as f:
            self.replace_data = json.load(f)

        self.replace_data_keyset = set(self.replace_data.keys())
        self.vocabulary_vectors = self.after_process_model.encode(self.vocabulary)
        self.symbols = set("#、,.\n(){}[]!?;:\"'<>@%^&*~`|+-=_「」。")

        youtube_json = os.path.join(BASE_DIR, 'youtube_link.json')
        try:
            with open(youtube_json, 'r', encoding='utf-8') as f:
                youtube_links_data = json.load(f)
                self.youtube_dict = {item['chinese']: item['youtube_id'] for item in youtube_links_data}
        except Exception as e:
            print(f"Warning: Could not load {youtube_json}: {e}")
            self.youtube_dict = {}

        merged_json = os.path.join(BASE_DIR, 'merged_labels.json')
        try:
            with open(merged_json, 'r', encoding='utf-8') as f:
                self.twtsl_download_dict = json.load(f)
            print(f"Successfully loaded local TWSL video mapping from {merged_json}")
        except Exception as e:
            print(f"Warning: Could not load {merged_json}: {e}")
            self.twtsl_download_dict = {}

        self.twsl_local_video_base_path = os.environ.get(
            "TWSL_VIDEO_DIR",
            os.path.join(DATA_ROOT, "videos")
        )

    def initUI(self):
        self.app = QApplication.instance() or QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle("Text to Sign Language GUI (Lite)")
        self.window.setGeometry(100, 100, 900, 800)

        self.text_input = QPlainTextEdit()
        self.text_input.setPlaceholderText("請輸入要轉換的文本...")

        self.convert_button = QPushButton("轉換為手語")
        self.convert_button.clicked.connect(self.convert_text_to_sign)

        self.result_display = QTextBrowser()
        self.result_display.setOpenExternalLinks(True)

        self.link_display = QTextBrowser()
        self.link_display.setOpenExternalLinks(True)
        self.link_display.setHtml("<p style='color: gray;'>影片連結或本機檔案資訊將顯示於此處。</p>")

        font = QFont("Microsoft JhengHei", 11)
        self.text_input.setFont(font)
        self.result_display.setFont(font)
        self.link_display.setFont(font)
        
        self.exit_button = QPushButton("退出")
        self.exit_button.clicked.connect(self.close)

        label = QLabel("選擇 Prompt 組合：")
        self.option1, self.option2, self.option3 = QCheckBox("POS"), QCheckBox("DEP"), QCheckBox("CON")
        self.checke_box = QCheckBox("全選")
        self.checke_box.setChecked(True)
        self.checke_box.stateChanged.connect(self.toggle_all_options)
        for cb in [self.option1, self.option2, self.option3]:
            cb.setChecked(True)
            cb.setEnabled(False)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.text_input)

        options_layout = QHBoxLayout()
        options_layout.addWidget(label)
        for cb in [self.option1, self.option2, self.option3, self.checke_box]:
            options_layout.addWidget(cb)
        options_layout.addStretch()
        main_layout.addLayout(options_layout)

        main_layout.addWidget(self.convert_button)
        main_layout.addWidget(QLabel("模型輸出"))
        main_layout.addWidget(self.result_display)
        main_layout.addWidget(QLabel("影片 / 連結資訊"))
        main_layout.addWidget(self.link_display)
        main_layout.addWidget(self.exit_button)

        container = QWidget()
        container.setLayout(main_layout)
        self.window.setCentralWidget(container)
        self.result_display.setHtml("<p style='color: gray;'>模型輸出將會顯示於此處。</p>")

    def toggle_all_options(self):
        is_checked = self.checke_box.isChecked()
        for cb in [self.option1, self.option2, self.option3]:
            cb.setChecked(is_checked)
            cb.setEnabled(not is_checked)

    def calculate_similarity(self, word, word_list, word_vectors, top_n=1, threshold=0.8):
        word_vector = self.after_process_model.encode(word)
        similarities = [cos_sim(word_vector, vec).item() for vec in word_vectors]
        similar_words = sorted(zip(word_list, similarities), key=lambda x: x[1], reverse=True)
        return [(w, s) for w, s in similar_words[:top_n] if s >= threshold]

    def find_synonyms(self, word):
        if word in self.symbols:
            return [(word, 1.0)]
        if word in self.replace_data_keyset:
            return [(word, 1.0)]
        if word in self.vocabulary:
            return [(word, 1.0)]
        similar_words = self.calculate_similarity(word, self.vocabulary, self.vocabulary_vectors)
        return similar_words if similar_words else [(word, 1.0)]

    def convert_text_to_sign(self):
        input_text = self.text_input.toPlainText().strip()
        if not input_text:
            self.result_display.setHtml("<p style='color:red;'>請先輸入文本。</p>")
            self.link_display.setHtml("<p style='color: gray;'>請先輸入文本以轉換。</p>")
            return

        gloss_list = self.text_to_gloss(input_text)
        html_text_output = self.generate_gloss_text_output(gloss_list)
        self.result_display.setHtml(html_text_output)
        self.result_display.moveCursor(QTextCursor.MoveOperation.Start)
        self.display_links(gloss_list)

    def text_to_gloss(self, text, test=False):
        print(text)
        segmented_text = self.ckip_tool.seg([text])
        print(segmented_text)
        nlp_result = self.nlp(tok=segmented_text)
        pos, dep = nlp_result['pos'][0], nlp_result['dep'][0]
        constr = re.sub(r'\s+', ' ', str(nlp_result['con'][0]).replace("\n", ""))
        self.model_input = f"translate Chinese to Gloss: {text}"
        if self.option1.isChecked():
            self.model_input += f" <POS> {' '.join(pos)}"
        if self.option2.isChecked():
            self.model_input += f" <DEP> {' '.join([i[1] for i in dep])}"
        if self.option3.isChecked():
            self.model_input += f" <CON> {constr}"

        if test:
            gloss_output_raw = ["公司/旅行/玩//英國/法國/這兩個/二選一//"]
            raw_glosses = gloss_output_raw[0].strip().replace('//', '/。/').replace('^^', '/^^/').split('/')
            return [self.find_synonyms(g)[0][0] for g in raw_glosses if g]

        print(self.model_input)
        inputs = self.tokenizer(self.model_input, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            translated_tokens = self.model.generate(**inputs)
            gloss_output_raw = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        print(f"Generated gloss: {gloss_output_raw}")
        raw_glosses = gloss_output_raw[0].strip().replace('//', '/。/').replace('^^', '/^^/').split('/')
        return [self.find_synonyms(g)[0][0] for g in raw_glosses if g]

    def generate_gloss_text_output(self, gloss_list):
        parts = re.split(r'(<POS>|<DEP>|<CON>)', self.model_input)
        html = "<p>模型輸入：</p>"
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if part in ['<POS>', '<DEP>', '<CON>']:
                tag = ht.escape(part)
                content = ht.escape(parts[i + 1].strip()) if i + 1 < len(parts) else ""
                html += f"<p style='white-space: nowrap;'><b>{tag}</b> <span style='color: gray;'>{content}</span></p>"
                i += 2
            else:
                html += f"<p style='color: gray; white-space: nowrap;'>{ht.escape(part)}</p>"
                i += 1

        html += "<p>手語轉換結果：</p><p>"
        for gloss in gloss_list:
            style = "color: gray; margin: 0 8px;" if gloss in ['^^', '。'] else "margin: 0 8px;"
            display_text = {'^^': '疑問表情', '。': '停頓'}.get(gloss, gloss)
            html += f"<span style='{style}'>{display_text}&nbsp;</span>"
        html += "</p>"
        return html

    def display_links(self, gloss_list):
        html = ["<p><b>詞彙對應資源：</b></p><ul>"]
        found_any = False

        for gloss in gloss_list:
            if gloss in ['^^', '。'] or gloss in self.symbols or not gloss or not gloss.strip():
                continue

            found_any = True
            item_html = f"<li><b>{ht.escape(gloss)}</b>: "

            local_info = ""
            if gloss in self.twtsl_download_dict:
                video_filename = self.twtsl_download_dict[gloss]
                local_video_path = os.path.join(self.twsl_local_video_base_path, video_filename)
                if os.path.exists(local_video_path):
                    local_info = f"本機影片：<code>{ht.escape(local_video_path)}</code>"
                else:
                    local_info = f"本機影片缺失：<code>{ht.escape(local_video_path)}</code>"

            youtube_info = ""
            if gloss in self.youtube_dict:
                video_id = self.youtube_dict[gloss]
                youtube_url = f"https://www.youtube.com/watch?v={video_id}"
                youtube_info = f"YouTube：<a href='{youtube_url}'>{youtube_url}</a>"
            else:
                search_query = f"{gloss} 手語"
                search_url = f"https://www.youtube.com/results?search_query={ht.escape(search_query)}"
                youtube_info = f"搜尋：<a href='{search_url}'>{ht.escape(search_query)}</a>"

            if local_info and youtube_info:
                item_html += local_info + "<br>" + youtube_info
            elif local_info:
                item_html += local_info
            else:
                item_html += youtube_info

            item_html += "</li>"
            html.append(item_html)

        html.append("</ul>")

        if not found_any:
            self.link_display.setHtml("<p style='color: gray;'>沒有可顯示的影片或連結資訊。</p>")
        else:
            self.link_display.setHtml("".join(html))

    def run(self):
        self.window.show()
        self.app.exec()

    def close(self):
        reply = QMessageBox.question(
            self.window,
            '退出確認',
            '確定要退出程式嗎？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            QCoreApplication.instance().quit()


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    QCoreApplication.setApplicationName("Text2SignGUI_Lite")
    QCoreApplication.setOrganizationName("MyOrg")

    gui = Text2signgui()
    gui.run()
