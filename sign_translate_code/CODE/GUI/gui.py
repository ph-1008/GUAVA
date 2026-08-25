from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPlainTextEdit, QPushButton,
    QTextBrowser, QVBoxLayout, QWidget, QMessageBox,
    QHBoxLayout, QCheckBox, QLabel, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import QCoreApplication, QUrl, Qt, QDir, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput 
from PyQt6.QtMultimediaWidgets import QVideoWidget

import os
import sys
import urllib.parse 
import json

# NOTE: The following libraries need to be installed for the new voice input feature:
# pip install SpeechRecognition PyAudio
try:
    import speech_recognition as sr
except ImportError:
    print("Warning: speech_recognition library not found. Voice input will be disabled.")
    print("Please install it using: pip install SpeechRecognition PyAudio")
    sr = None


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from CODE.preprocess import ckipnlptool
from CODE.nlptool import merge_pos_into_con
import re
from nltk.tree import Tree
import hanlp
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
from text2vec import SentenceModel, cos_sim
import html as ht


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
        if not isinstance(current_node, Tree): continue
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
    if not isinstance(item, str): return item
    item = item.replace("(", "").replace(")", "")
    item = re.sub(r'\{.*?\}', '', item)
    item = re.sub(r'\[.*?\]', '', item)
    return item.strip()

# Thread for handling voice recognition to avoid freezing the GUI
class VoiceRecognitionThread(QThread):
    recognized_text = pyqtSignal(str)
    recognition_error = pyqtSignal(str)
    is_listening = pyqtSignal(bool)

    def run(self):
        if not sr: # Check if the library was imported successfully
            self.recognition_error.emit("語音辨識模組未安裝。")
            return
            
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.is_listening.emit(True)
            print("Listening for voice input...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                # Listen for speech with a timeout
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                self.is_listening.emit(False)
                print("Recognizing...")
                # Recognize speech using Google's Web Speech API
                text = recognizer.recognize_google(audio, language='zh-TW')
                self.recognized_text.emit(text)
            except sr.WaitTimeoutError:
                self.recognition_error.emit("聆聽超時，未偵測到聲音。")
            except sr.UnknownValueError:
                self.recognition_error.emit("無法辨識語音，請再試一次。")
            except sr.RequestError as e:
                self.recognition_error.emit(f"無法連線至語音辨識服務: {e}")
            finally:
                # Ensure the listening status is set to false
                self.is_listening.emit(False)


class Text2signgui:
    """
    展示文本轉手語的GUI界面。
    輸入文本，點擊按鈕後，將文本轉換為手語文字（gloss），
    並顯示為每個字詞帶有超連結的形式。
    """
    VIDEO_WIDTH = 320
    VIDEO_HEIGHT = 180
    SCROLL_AREA_VIDEO_PLAYER_HEIGHT = VIDEO_HEIGHT + 65
    TWSL_LOCAL_VIDEO_BASE_PATH = r"D:\Sign_dataset\tw_sign_language" 

    def __init__(self, parent=None):
        """
        初始化GUI界面和相關工具。
        1. 初始化CKIP NLP工具。
        2. 加載HanLP模型進行詞性標註、依存句法分析和句法樹分析。
        3. 加載mT5模型和tokenizer進行文本到手語的轉換。
        4. 加載詞彙表和替換數據。
        """
        self.parent = parent
        self.media_players_and_outputs = [] # To keep QMediaPlayer and QAudioOutput instances alive
        self.initUI()

        # NEW: Setup for voice input thread and its signals
        self.voice_thread = VoiceRecognitionThread()
        self.voice_thread.recognized_text.connect(self.update_text_from_voice)
        self.voice_thread.recognition_error.connect(self.handle_voice_error)
        self.voice_thread.is_listening.connect(self.update_voice_button_status)


        # 初始化CKIP NLP工具
        self.ckip_tool = ckipnlptool()
        try:
            con = hanlp.load(hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ERNIE_GRAM)
            dep = hanlp.load(hanlp.pretrained.dep.CTB9_UDC_ELECTRA_SMALL, conll=False)
        except Exception as e:
            print(f"Error loading HanLP models: {e}\nEnsure HanLP models are downloaded/accessible.")
            sys.exit(1)

        # 定義HanLP處理管道
        self.nlp = hanlp.pipeline() \
            .append(self.ckip_tool.tagger, input_key='tok', output_key='pos') \
            .append(dep, input_key='tok', output_key='dep') \
            .append(con, input_key='tok', output_key='con') \
            .append(merge_pos_into_con, input_key='*')
        
        # 初始化mT5模型和tokenizer
        model_path = r"CODE\GUI\best_model"
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = MT5Tokenizer.from_pretrained(model_path)

        self.after_process_model = SentenceModel('shibing624/text2vec-base-chinese')
        with open('chinese_values2.txt', 'r', encoding='utf-8') as f:
            self.vocabulary = [i.strip() for i in f.readlines()]
        with open(r"replace_data.json",'r',encoding='utf-8') as f:
            self.replace_data = json.load(f)
        self.replace_data_keyset = set(self.replace_data.keys())
        self.vocabulary_vectors = self.after_process_model.encode(self.vocabulary)
        self.symbols = set("#、,.\n(){}[]!?;:\"'<>@%^&*~`|+-=_「」。")

        # Load YouTube links from JSON file
        try:
            with open(r"CODE\GUI\youtube_link.json", 'r', encoding='utf-8') as f:
                youtube_links_data = json.load(f)
                self.youtube_dict = {item['chinese']: item['youtube_id'] for item in youtube_links_data}
        except Exception as e:
            print(f"Warning: Could not load CODE\GUI\youtube_link.json: {e}")
            self.youtube_dict = {}

        # Load local TWSL video mapping from JSON file
        local_twtsl_json_path = r"CODE\GUI\merged_labels.json"
        try:
            with open(local_twtsl_json_path, 'r', encoding='utf-8') as f:
                self.twtsl_download_dict = json.load(f)
            print(f"Successfully loaded local TWSL video mapping from {local_twtsl_json_path}")
        except Exception as e:
            print(f"Warning: Could not load {local_twtsl_json_path}: {e}")
            self.twtsl_download_dict = {}
        

    def initUI(self):
        self.app = QApplication.instance() or QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle("Text to Sign Language GUI")
        self.window.setGeometry(100, 100, 800, 750)

        self.text_input = QPlainTextEdit()
        self.text_input.setPlaceholderText("請輸入要轉換的文本，或使用語音輸入...")
        
        self.convert_button = QPushButton("轉換為手語")
        self.convert_button.clicked.connect(self.convert_text_to_sign)
        
        # NEW: Voice input button
        self.voice_input_button = QPushButton("🎤 語音輸入")
        self.voice_input_button.clicked.connect(self.start_voice_recognition)
        if sr is None: # Disable button if library is not available
            self.voice_input_button.setEnabled(False)
            self.voice_input_button.setToolTip("語音辨識模組未安裝 (SpeechRecognition)")

        self.result_display = QTextBrowser()
        self.result_display.setOpenExternalLinks(True)
        self.exit_button = QPushButton("退出")
        self.exit_button.clicked.connect(self.close)

        label = QLabel("選擇Prompt 組合：")
        self.option1, self.option2, self.option3 = QCheckBox("POS"), QCheckBox("DEP"), QCheckBox("CON")
        self.checke_box = QCheckBox("全選")
        self.checke_box.setChecked(True)
        self.checke_box.stateChanged.connect(self.toggle_all_options)
        for cb in [self.option1, self.option2, self.option3]: cb.setChecked(True); cb.setEnabled(False)

        
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.voice_input_button)
        main_layout.addWidget(self.text_input)
        options_layout = QHBoxLayout()
        options_layout.addWidget(label)
        for cb in [self.option1, self.option2, self.option3, self.checke_box]: options_layout.addWidget(cb)
        options_layout.addStretch()
        main_layout.addLayout(options_layout)

        main_layout.addWidget(self.convert_button)
        ## MODIFIED: Button layout to include voice input button
        #button_layout = QHBoxLayout()
        #button_layout.addWidget(self.convert_button)
        #button_layout.addWidget(self.voice_input_button)
        #button_layout.addStretch()
        # main_layout.addLayout(button_layout)
        
        main_layout.addWidget(self.result_display)

        self.video_scroll_area = QScrollArea()
        self.video_scroll_area.setWidgetResizable(True)
        self.video_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.video_scroll_area.setFixedHeight(self.SCROLL_AREA_VIDEO_PLAYER_HEIGHT)
        self.video_container_widget = QWidget()
        self.video_layout = QHBoxLayout(self.video_container_widget)
        self.video_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._add_video_placeholder("手語影片將會顯示於此處。")
        self.video_scroll_area.setWidget(self.video_container_widget)
        main_layout.addWidget(self.video_scroll_area)
        main_layout.addWidget(self.exit_button)
        container = QWidget()
        container.setLayout(main_layout)
        self.window.setCentralWidget(container)
        self.result_display.setHtml("<p style='color: gray;'><br><br><br><br><br>"+" "*79+"模型輸出將會顯示於此處。</p>")

    # --- NEW: Methods for Voice Input ---
    def start_voice_recognition(self):
        if not self.voice_thread.isRunning():
            self.voice_thread.start()
    
    def update_voice_button_status(self, is_listening):
        if is_listening:
            self.voice_input_button.setText("...聆聽中...")
            self.voice_input_button.setEnabled(False)
        else:
            self.voice_input_button.setText("🎤 語音輸入")
            self.voice_input_button.setEnabled(True)

    def update_text_from_voice(self, text):
        current_text = self.text_input.toPlainText()
        # Append recognized text, adding a space if needed
        new_text = (current_text + " " + text) if current_text else text
        self.text_input.setPlainText(new_text)
        self.text_input.moveCursor(QTextCursor.MoveOperation.End) # Move cursor to the end

    def handle_voice_error(self, error_message):
        QMessageBox.warning(self.window, "語音辨識錯誤", error_message)
        # Button status is reset by the thread's 'finally' block via the is_listening signal

    def _clear_video_layout(self):
        # Stop and release QMediaPlayer resources
        for player, audio_output in self.media_players_and_outputs:
            player.stop()
            player.setSource(QUrl()) # Release source
            player.setVideoOutput(None)
            player.setAudioOutput(None)
        self.media_players_and_outputs.clear()

        # Clear widgets from layout
        while self.video_layout.count():
            item = self.video_layout.takeAt(0)
            widget = item.widget()
            if widget: 
                widget.deleteLater()

    def _add_video_placeholder(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_layout.addWidget(lbl)

    def toggle_all_options(self): # Removed state argument, use isChecked()
        is_checked = self.checke_box.isChecked()
        for cb in [self.option1, self.option2, self.option3]:
            cb.setChecked(is_checked)
            cb.setEnabled(not is_checked)

    def calculate_similarity(self,word, word_list, word_vectors, top_n=1, threshold=0.8):
        word_vector = self.after_process_model.encode(word)
        similarities = [cos_sim(word_vector, vec).item() for vec in word_vectors]
        similar_words = sorted(zip(word_list, similarities), key=lambda x: x[1], reverse=True)
        return [(w, s) for w, s in similar_words[:top_n] if s >= threshold]

    def find_synonyms(self,word):
        if word in self.symbols: return [(word, 1.0)]
        if word in self.replace_data_keyset: return [(word, 1.0)]
        if word in self.vocabulary: return [(word, 1.0)]
        similar_words = self.calculate_similarity(word, self.vocabulary, self.vocabulary_vectors)
        return similar_words if similar_words else [(word, 1.0)]

    def convert_text_to_sign(self):
        input_text = self.text_input.toPlainText().strip()
        if not input_text:
            self.result_display.setHtml("<p style='color:red;'>請先輸入文本。</p>")
            self._clear_video_layout()
            self._add_video_placeholder("請先輸入文本以轉換。")
            return
        gloss_list = self.text_to_gloss(input_text)
        html_text_output = self.generate_gloss_text_output(gloss_list)
        self.result_display.setHtml(html_text_output)
        self.result_display.moveCursor(QTextCursor.MoveOperation.Start) # PyQt6
        self.display_embedded_videos(gloss_list)

    def text_to_gloss(self, text, test=False):
        print(text)
        segmented_text = self.ckip_tool.seg([text])
        print(segmented_text)
        nlp_result = self.nlp(tok=segmented_text)
        pos, dep = nlp_result['pos'][0], nlp_result['dep'][0] 
        constr = re.sub(r'\s+', ' ', str(nlp_result['con'][0]).replace("\n", ""))
        self.model_input = f"translate Chinese to Gloss: {text}"
        if self.option1.isChecked(): self.model_input += f" <POS> {' '.join(pos)}"
        if self.option2.isChecked(): self.model_input += f" <DEP> {' '.join([i[1] for i in dep])}"
        if self.option3.isChecked(): self.model_input += f" <CON> {constr}"
        if test:
            print(self.model_input)
            gloss_output_raw = ["公司/旅行/玩//英國/法國/這兩個/二選一//"]
            print(f"Generated gloss: {gloss_output_raw}")
            raw_glosses = gloss_output_raw[0].strip().replace('//', '/。/').replace('^^', '/^^/').split('/')
            return [self.find_synonyms(g)[0][0] for g in raw_glosses if g]
        else:
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

        html = html + "<p>手語轉換結果：</p><p>"
        for gloss in gloss_list:
            style = "color: gray; margin: 0 8px;" if gloss in ['^^', '。'] else "margin: 0 8px;"
            display_text = {'^^': "疑問表情", '。': "停頓"}.get(gloss, gloss)
            html += f"<span style='{style}'>{display_text} </span>"
        return html + "</p>"

    def display_embedded_videos(self, gloss_list):
        self._clear_video_layout() # Also clears self.media_players_and_outputs
        web_views_added_count = 0 

        for gloss in gloss_list:
            if gloss in ['^^', '。'] or gloss in self.symbols or not gloss or not gloss.strip():
                continue

            widget_to_add = None
            action_taken = False

            # Priority 1: Local TWSL with QMediaPlayer
            if gloss in self.twtsl_download_dict:
                video_filename = self.twtsl_download_dict[gloss]
                local_video_path = os.path.join(self.TWSL_LOCAL_VIDEO_BASE_PATH, video_filename)

                if os.path.exists(local_video_path):
                    video_widget = QVideoWidget()
                    video_widget.setFixedSize(self.VIDEO_WIDTH, self.VIDEO_HEIGHT)
                    
                    player = QMediaPlayer()
                    audio_output = QAudioOutput()
                    
                    player.setVideoOutput(video_widget)
                    player.setAudioOutput(audio_output)
                    player.setSource(QUrl.fromLocalFile(QDir.toNativeSeparators(local_video_path)))
                    player.errorOccurred.connect(
                        lambda error, p=player: print(f"QMediaPlayer Error for {p.source().fileName()}: {error}, {p.errorString()}")
                    )
                    player.mediaStatusChanged.connect(
                        lambda status, p=player: print(f"QMediaPlayer Status for {p.source().fileName()}: {status}")
                    )
                    audio_output.setMuted(True) 
                    player.setLoops(QMediaPlayer.Loops.Infinite)

                    self.media_players_and_outputs.append((player, audio_output))
                    
                    widget_to_add = video_widget
                    print(f"Setting up QMediaPlayer for: {gloss} from {local_video_path}")
                    action_taken = True
                    player.play()
                else:
                    print(f"Local video file NOT FOUND for {gloss}: {local_video_path}")
            
            # Priority 2 & 3: YouTube (embed or search) with QWebEngineView
            if not action_taken:
                video_view = QWebEngineView()
                video_view.setFixedSize(self.VIDEO_WIDTH, self.VIDEO_HEIGHT)
                settings = video_view.page().settings()
                settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
                settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
                # ... other settings ...
                
                if gloss in self.youtube_dict:
                    video_id = self.youtube_dict[gloss]
                    embed_url = QUrl(f"https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1&loop=1&playlist={video_id}")
                    video_view.setUrl(embed_url)
                    print(f"Embedding YouTube video for: {gloss}")
                    action_taken = True
                else: 
                    search_query_text = f"{gloss} 手語"
                    encoded_query = urllib.parse.quote_plus(search_query_text)
                    search_url = QUrl(f"https://www.youtube.com/results?search_query={encoded_query}")
                    video_view.setUrl(search_url)
                    print(f"Searching YouTube for: \"{search_query_text}\"")
                    action_taken = True
                widget_to_add = video_view
            
            # MODIFIED: Wrap the video widget and a label in a container
            if widget_to_add: 
                # Create a container to hold the video and its label
                video_container = QWidget()
                video_vbox = QVBoxLayout(video_container)
                video_vbox.setContentsMargins(2, 2, 2, 2)
                video_vbox.setSpacing(5) 

                # Add the video widget (QVideoWidget or QWebEngineView)
                video_vbox.addWidget(widget_to_add)

                # Add the gloss label underneath
                gloss_label = QLabel(gloss)
                gloss_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                gloss_label.setWordWrap(True)
                video_vbox.addWidget(gloss_label)

                # Add the entire container to the main horizontal layout
                self.video_layout.addWidget(video_container)
                web_views_added_count += 1
        
        if web_views_added_count == 0:
            searchable_glosses_were_present = any(
                g for g in gloss_list if g and g.strip() and g not in ['^^', '。'] and g not in self.symbols
            )
            if not searchable_glosses_were_present:
                 self._add_video_placeholder("沒有可顯示影片或搜尋的詞彙。")
            else: 
                 self._add_video_placeholder("詞彙無對應影片，或影片檔案遺失。")

    def run(self):
        self.window.show()
        self.app.exec()

    def close(self):
        reply = QMessageBox.question(self.window, '退出確認', '確定要退出程式嗎？',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            QCoreApplication.instance().quit()

if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    QCoreApplication.setApplicationName("Text2SignGUI")
    QCoreApplication.setOrganizationName("MyOrg") 

    gui = Text2signgui()
    gui.run()