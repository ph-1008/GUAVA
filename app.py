import os
import time
import json
import re
import html as ht
import shlex
import unicodedata
import gradio as gr
from pathlib import Path
from functools import partial
import subprocess
import socket
from urllib.parse import quote

# --- Constants ---
OUTPUT_DIR = 'outputs/app'
OUTNAME = 'render'
DEVICES = '0'
EHM_TRACKER_DIR = 'EHM-Tracker' # Define the tracker directory

TRACKED_IMG_DIR = 'assets/example/tracked_image'
TRACKED_VID_DIR = 'assets/example/tracked_video'
TRACKED_SOURCE_IMAGE_DIR = 'outputs/app/tracked_source_image'
OUTPUTS2_DIR = 'outputs_2'
TEXT_PIPELINE_SCRIPT = 'run_translate_to_guava.sh'
DEFAULT_TEXT_SOURCE = 'Gemini_Generated_Image_kzne4skzne4skzne'
WORD_VIDEO_LABELS_JSON = 'EHM-Tracker/assets/merged_labels.json'
WORD_VIDEO_ASSET_ROOT = 'EHM-Tracker/assets'
TRANSLATE_OUTPUT_TXT = 'sign_translate_code/gloss_output.txt'
PUNCTUATION_TOKENS = {"", ".", ",", "!", "?", ";", ":", "。", "，", "、", "！", "？", "；", "：", "^^"}
MAX_WORD_VIDEO_COMPONENTS = 12
APP_CSS = """
.render-video {
    max-width: 860px;
    margin: 0 auto;
}
.render-video video {
    max-height: 360px !important;
    object-fit: contain !important;
}
.word-video video {
    height: 150px !important;
    object-fit: contain !important;
}
.word-caption {
    text-align: center;
    font-weight: 700;
    margin-top: 4px;
}
.speech-input-row {
    align-items: center;
}
"""

SPEECH_TO_TEXT_JS = """
(currentText) => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert("此瀏覽器不支援語音輸入，請使用 Chrome 或 Edge。");
        return currentText || "";
    }

    return new Promise((resolve) => {
        const recognition = new SpeechRecognition();
        recognition.lang = "zh-TW";
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        recognition.onresult = (event) => {
            const transcript = Array.from(event.results)
                .map((result) => result[0]?.transcript || "")
                .join("")
                .trim();
            const previousText = (currentText || "").trim();
            resolve(previousText && transcript ? `${previousText} ${transcript}` : (transcript || previousText));
        };

        recognition.onerror = (event) => {
            const reason = event.error === "not-allowed"
                ? "瀏覽器未允許麥克風權限。"
                : `語音輸入失敗：${event.error}`;
            alert(reason);
            resolve(currentText || "");
        };

        recognition.onnomatch = () => {
            alert("沒有辨識到語音內容，請再試一次。");
            resolve(currentText || "");
        };

        recognition.start();
    });
}
"""

# --- Core Functions ---

def run_cmd(command, current_dir=None):
    """Executes a shell command and streams its output."""
    print(f"▶️  Executing command:\n{command}", flush=True)
    print(f"▶️  Working directory: {current_dir or os.getcwd()}", flush=True)
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=current_dir,
        bufsize=1
    )

    # Stream stdout
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)
        process.stdout.close()

    return_code = process.wait()

    # Handle errors
    if return_code != 0:
        print(f"‼️ Command failed with return code {return_code}", flush=True)
        raise subprocess.CalledProcessError(return_code, command)

    print(f"✅Command executed successfully.", flush=True)


def master_check_status(source_selection, source_upload, driven_selection, driven_upload):
    """
    Checks the processing status based on the combination of gallery and uploaded inputs.
    """
    src_name, dst_name = None, None

    if source_upload:
        src_name = os.path.splitext(os.path.basename(source_upload))[0]
    elif source_selection:
        src_name = source_selection['caption']
    else:
        return "Please provide a source to check.", None

    if driven_upload:
        dst_name = os.path.splitext(os.path.basename(driven_upload))[0]
    elif driven_selection:
        dst_name = driven_selection['caption']
    else:
        return "Please provide a driving video to check.", None

    output_file = os.path.join(OUTPUT_DIR, f'{OUTNAME}_cross_act', src_name, f'{src_name}_{dst_name}', f'{src_name}_{dst_name}_video.mp4')
    print('Try to find => ' + output_file)

    if not os.path.exists(output_file):
        return "Still processing... You can check progress again later. ⏳", None

    return "Processing completed successfully! 🎉", output_file


def run_master_process(source_selection, source_upload, driven_selection, driven_upload, progress=gr.Progress()):
    """
    A master function to handle all combinations of gallery/upload for source and driven inputs.
    """
    print("\n--- run_master_process called ---")
    print(f"source_selection (from gallery): {source_selection}")
    print(f"source_upload (from upload):   {source_upload}")
    print(f"driven_selection (from gallery): {driven_selection}")
    print(f"driven_upload (from upload):   {driven_upload}")
    print("---------------------------------\n")
    
    try:
        # --- 1. Input Validation ---
        has_source = source_selection is not None or source_upload is not None
        has_driven = driven_selection is not None or driven_upload is not None
        if not has_source or not has_driven:
            return "Error: Please provide both a source and a driving input.", None
        
        progress(0.01, desc="Preparing...")
        
        # --- 2. Setup Paths and Commands ---
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        output_dir = os.path.join(current_dir, OUTPUT_DIR)
        tracker_dir = os.path.join(current_dir, EHM_TRACKER_DIR)
        
        cmd_in_basedir = partial(run_cmd, current_dir=current_dir)
        cmd_in_tracker = partial(run_cmd, current_dir=tracker_dir)

        tracked_source_image_dir = os.path.join(output_dir, 'tracked_source_image')
        tracked_driven_video_dir = os.path.join(output_dir, 'tracked_driven_video')
        os.makedirs(tracked_source_image_dir, exist_ok=True)
        os.makedirs(tracked_driven_video_dir, exist_ok=True)

        # --- 3. Resolve Source Input ---
        progress(0.05, desc="☕️ Processing source...")
        if source_upload:
            print("Processing uploaded source image...")
            source_image_fp = os.path.abspath(source_upload)
            src_name = os.path.splitext(os.path.basename(source_image_fp))[0]
            src_img_root = os.path.join(tracked_source_image_dir, src_name)
            
            if os.path.exists(os.path.join(src_img_root, 'optim_tracking_ehm.pkl')):
                print(f'🐶 Uploaded source image "{src_name}" has been processed before, skipping tracking.')
            else:
                cmd_in_tracker(f'python -m src.tracking_single_image -i "{source_image_fp}" -o "{tracked_source_image_dir}"')
        else: # A gallery item was selected
            src_name = source_selection['caption']
            src_img_root = os.path.join(current_dir, TRACKED_IMG_DIR, src_name)
            print(f"Using pre-tracked source from gallery: {src_name}")
            if not os.path.exists(src_img_root):
                return f"Error: Unable to find source character data path {src_img_root}", None
        
        progress(0.2, desc="✅Source processed.")

        # --- 4. Resolve Driven Input ---
        progress(0.25, desc="☕️ Processing driven video (can take a while)...")
        if driven_upload:
            print("Processing uploaded driven video...")
            driven_video_fp = os.path.abspath(driven_upload)
            dst_name = os.path.splitext(os.path.basename(driven_video_fp))[0]
            dcv_vid_root = os.path.join(tracked_driven_video_dir, dst_name)

            if os.path.exists(os.path.join(dcv_vid_root, 'optim_tracking_ehm.pkl')):
                 print(f'🐶 Uploaded driven video "{dst_name}" has been processed before, skipping tracking.')
            else:
                cmd_in_tracker(f'python tracking_video.py -i "{driven_video_fp}" -o "{tracked_driven_video_dir}" --check_hand_score 0.0 -p 0,1 -n 1 -v 0')
        else: # A gallery item was selected
            dst_name = driven_selection['caption']
            dcv_vid_root = os.path.join(current_dir, TRACKED_VID_DIR, dst_name)
            print(f"Using pre-tracked driven video from gallery: {dst_name}")
            if not os.path.exists(dcv_vid_root):
                 return f"Error: Unable to find driven video data path {dcv_vid_root}", None
        
        progress(0.65, desc="✅Driven video processed. Starting final generation...")

        # --- 5. Generate Final Avatar ---
        print('⚡️ Initiating GUAVA generation results, please wait...')
        output_file = os.path.join(output_dir, f'{OUTNAME}_cross_act', src_name, f'{src_name}_{dst_name}', f'{src_name}_{dst_name}_video.mp4')
        
        if os.path.exists(output_file):
            print(f'🐶 The result already exists, skipping generation....')
        else:
            command = (
                f'PYTHONPATH=. python -m main.test -d {DEVICES} -n {OUTNAME} -m assets/GUAVA'
                f' --source_data_path "{src_img_root}"'
                f' --data_path "{dcv_vid_root}"'
                f' --save_path "{output_dir}"'
                f' --skip_self_act --render_cross_act'
            )
            cmd_in_basedir(command)
            print(f'Completion! The results are saved in {output_dir}/{OUTNAME}_cross_act')
            
        progress(1.0, desc="🎉 Complete!")
        return "🎉? Processing complete!", output_file

    except Exception as e:
        return f"An error occurred: {str(e)}", None


# --- History and Gallery Functions ---
def get_history_videos():
    """Get all previously generated videos and format them for a gr.Gallery."""
    results = []
    base_dir = f"{OUTPUT_DIR}/{OUTNAME}_cross_act"
    if not os.path.exists(base_dir):
        return []
        
    for source_dir in sorted(os.listdir(base_dir)):
        source_path = os.path.join(base_dir, source_dir)
        if os.path.isdir(source_path):
            for driven_dir in sorted(os.listdir(source_path)):
                driven_path = os.path.join(source_path, driven_dir)
                if os.path.isdir(driven_path):
                    videos = list(Path(driven_path).glob("*.mp4"))
                    for video in videos:
                        label = f"Source: {source_dir}\nDriven: {driven_dir[len(source_dir)+1:]}"
                        results.append((str(video), label))
    
    results.sort(key=lambda x: os.path.getctime(x[0]), reverse=True)
    return results

def prepare_gallery_data(base_dir):
    """Prepares image-label pairs for the gr.Gallery component."""
    gallery_list = []
    if not os.path.exists(base_dir):
        print(f"Warning: Gallery directory not found at {base_dir}")
        return gallery_list
        
    for item_name in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item_name)
        if os.path.isdir(item_path):
            preview_path = os.path.join(item_path, 'preview.png')
            if os.path.exists(preview_path):
                gallery_list.append((preview_path, item_name))
    return gallery_list


def first_existing_file(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def prepare_text_source_gallery():
    """Source gallery for text-to-GUAVA: prefer currently tracked source photos."""
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    gallery = []
    

    tracked_source_root = os.path.join(current_dir, TRACKED_SOURCE_IMAGE_DIR)
    if os.path.isdir(tracked_source_root):
        for item_name in sorted(os.listdir(tracked_source_root)):
            item_path = os.path.join(tracked_source_root, item_name)
            if not os.path.isdir(item_path):
                continue

            images_dir = os.path.join(item_path, "images")
            if not os.path.isdir(images_dir):
                continue

            png_candidates = sorted(Path(images_dir).glob("*.png"))
            preview_path = str(png_candidates[0]) if png_candidates else None
            if preview_path:
                gallery.append((preview_path, item_name))
                

    gallery.sort(key=lambda item: 0 if item[1] == DEFAULT_TEXT_SOURCE else 1)
    return gallery


def resolve_text_source(selection, upload_path):
    """Return (source_name, tracked_source_dir), processing uploads if needed."""
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    output_dir = os.path.join(current_dir, OUTPUT_DIR)
    tracker_dir = os.path.join(current_dir, EHM_TRACKER_DIR)
    tracked_source_image_dir = os.path.join(output_dir, 'tracked_source_image')
    os.makedirs(tracked_source_image_dir, exist_ok=True)

    if upload_path:
        source_image_fp = os.path.abspath(upload_path)
        source_name = os.path.splitext(os.path.basename(source_image_fp))[0]
        source_root = os.path.join(tracked_source_image_dir, source_name)

        if os.path.exists(os.path.join(source_root, 'optim_tracking_ehm.pkl')):
            print(f'Uploaded source image "{source_name}" already tracked; skipping tracking.')
        else:
            command = (
                f'python -m src.tracking_single_image '
                f'-i {shlex.quote(source_image_fp)} '
                f'-o {shlex.quote(tracked_source_image_dir)}'
            )
            run_cmd(command, current_dir=tracker_dir)
        return source_name, source_root

    if selection and isinstance(selection, dict):
        source_name = selection.get('caption')
    elif selection:
        source_name = str(selection)
    else:
        source_name = DEFAULT_TEXT_SOURCE

    candidates = [
        os.path.join(current_dir, TRACKED_SOURCE_IMAGE_DIR, source_name),
        os.path.join(current_dir, TRACKED_IMG_DIR, source_name),
    ]
    source_root = first_existing_file(candidates)
    if not source_root:
        raise FileNotFoundError(f"找不到 source character tracking data: {source_name}")
    return source_name, source_root


def run_cmd_capture(command, current_dir=None):
    """Run a command and return combined stdout/stderr."""
    print(f"▶️  Executing command:\n{' '.join(command) if isinstance(command, list) else command}", flush=True)
    print(f"▶️  Working directory: {current_dir or os.getcwd()}", flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=current_dir,
        bufsize=1,
    )
    output_lines = []
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)
            output_lines.append(line)
        process.stdout.close()

    return_code = process.wait()
    output_text = ''.join(output_lines)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output=output_text)
    return output_text


def parse_translate_output():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    output_path = os.path.join(current_dir, TRANSLATE_OUTPUT_TXT)
    if not os.path.exists(output_path):
        return "", "", []

    text = Path(output_path).read_text(encoding='utf-8')
    model_input_match = re.findall(r'^模型輸入:\s*(.+)$', text, re.MULTILINE)
    gloss_joined_match = re.findall(r'^Gloss joined:\s*(.+)$', text, re.MULTILINE)
    generated_match = re.findall(r'^Generated gloss:\s*(.+)$', text, re.MULTILINE)

    model_input = model_input_match[-1] if model_input_match else ""
    gloss_joined = gloss_joined_match[-1] if gloss_joined_match else ""
    generated = generated_match[-1] if generated_match else ""
    gloss_tokens = [token.strip() for token in gloss_joined.split('/') if token.strip()]
    return model_input, generated, gloss_tokens


def is_punctuation_or_symbol(token):
    stripped = (token or '').strip()
    if stripped in PUNCTUATION_TOKENS:
        return True
    return bool(stripped) and all(unicodedata.category(char)[0] in {'P', 'S'} for char in stripped)


def load_word_video_labels():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    labels_path = os.path.join(current_dir, WORD_VIDEO_LABELS_JSON)
    with open(labels_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_word_video_path(token, labels):
    if is_punctuation_or_symbol(token):
        return None

    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    asset_root = os.path.join(current_dir, WORD_VIDEO_ASSET_ROOT)
    lookup_tokens = [token]
    normalized = unicodedata.normalize("NFKC", token).replace("臺", "台")
    if normalized not in lookup_tokens:
        lookup_tokens.append(normalized)

    for lookup_token in lookup_tokens:
        rel_path = labels.get(lookup_token)
        if not rel_path:
            continue
        rel_path = rel_path.replace("\\", "/")
        video_path = rel_path if os.path.isabs(rel_path) else os.path.join(asset_root, rel_path)
        if os.path.exists(video_path):
            return video_path
    return None


def gradio_file_url(path):
    if not path:
        return ""
    absolute_path = os.path.abspath(path).replace("\\", "/")
    return "/file=" + quote(absolute_path, safe="/:")


def build_translation_html(input_text, model_input, generated, gloss_tokens):
    if not gloss_tokens:
        return "<p style='color: #777;'>尚未產生手語轉換結果。</p>"

    token_html = []
    for token in gloss_tokens:
        display = {'^^': '疑問表情', '。': '停頓'}.get(token, token)
        color = "#777" if is_punctuation_or_symbol(token) else "#111"
        token_html.append(
            f"<span style='display:inline-block;margin:4px 6px;padding:4px 8px;"
            f"border:1px solid #ddd;border-radius:6px;color:{color};'>{ht.escape(display)}</span>"
        )

    return f"""
    <div style="line-height:1.6;">
      <div><b>輸入句子</b><br>{ht.escape(input_text)}</div>
      <div style="margin-top:10px;"><b>模型輸入</b><br><code style="white-space:normal;">{ht.escape(model_input)}</code></div>
      <div style="margin-top:10px;"><b>模型輸出</b><br><code>{ht.escape(generated)}</code></div>
      <div style="margin-top:10px;"><b>手語轉換結果</b><br>{''.join(token_html)}</div>
    </div>
    """


def find_rendered_video(output_dir):
    output_path = Path(output_dir)
    candidates = list(output_path.rglob("*_video.mp4"))
    if not candidates:
        candidates = list(output_path.rglob("*.mp4"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def build_word_videos_html(gloss_tokens):
    try:
        labels = load_word_video_labels()
    except Exception as exc:
        return f"<p style='color:#b00020;'>無法讀取詞語影片 JSON：{ht.escape(str(exc))}</p>"

    cards = []
    for token in gloss_tokens:
        if is_punctuation_or_symbol(token):
            continue
        video_path = resolve_word_video_path(token, labels)
        if video_path:
            video = (
                f"<video src='{gradio_file_url(video_path)}' controls muted "
                f"style='width:100%;max-height:220px;background:#111;border-radius:6px;'></video>"
                f"<div style='font-size:12px;color:#777;word-break:break-all;margin-top:4px;'>{ht.escape(video_path)}</div>"
            )
        else:
            video = "<div style='height:160px;display:flex;align-items:center;justify-content:center;background:#f3f3f3;border-radius:6px;color:#777;'>找不到對應影片</div>"

        cards.append(
            f"<div style='min-width:220px;max-width:280px;flex:1 1 240px;'>"
            f"{video}"
            f"<div style='font-weight:700;text-align:center;margin-top:8px;'>{ht.escape(token)}</div>"
            f"</div>"
        )

    if not cards:
        return "<p style='color:#777;'>沒有可顯示的單詞影片。</p>"

    return "<div style='display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;'>" + "".join(cards) + "</div>"


def empty_word_video_updates():
    updates = []
    for _ in range(MAX_WORD_VIDEO_COMPONENTS):
        updates.extend([
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
        ])
    return updates


def build_word_video_updates(gloss_tokens):
    try:
        labels = load_word_video_labels()
    except Exception as exc:
        updates = [
            gr.update(value=None, visible=False),
            gr.update(value=f"無法讀取詞語影片 JSON：{exc}", visible=True),
        ]
        updates.extend(empty_word_video_updates()[2:])
        return updates

    slots = []
    for token in gloss_tokens:
        if is_punctuation_or_symbol(token):
            continue
        video_path = resolve_word_video_path(token, labels)
        if video_path:
            slots.append((video_path, token))
        else:
            slots.append((None, f"{token}\n\n找不到對應影片"))
        if len(slots) >= MAX_WORD_VIDEO_COMPONENTS:
            break

    updates = []
    for index in range(MAX_WORD_VIDEO_COMPONENTS):
        if index < len(slots):
            video_path, caption = slots[index]
            updates.extend([
                gr.update(value=video_path, visible=bool(video_path)),
                gr.update(value=f"**{caption}**", visible=True),
            ])
        else:
            updates.extend([
                gr.update(value=None, visible=False),
                gr.update(value="", visible=False),
            ])
    return updates


def build_render_html(video_path):
    if not video_path:
        return "<p style='color:#777;'>尚未產生渲染影片。</p>"
    return f"""
    <div>
      <video src="{gradio_file_url(video_path)}" controls style="width:100%;max-height:520px;background:#111;border-radius:6px;"></video>
      <div style="margin-top:8px;"><b>渲染影片儲存位置：</b><code style="word-break:break-all;">{ht.escape(video_path)}</code></div>
    </div>
    """


def run_text_to_guava(source_selection, source_upload, input_text, smooth_word_segments, progress=gr.Progress()):
    if not input_text or not input_text.strip():
        return (
            "請先輸入自然語言句子。",
            build_translation_html("", "", "", []),
            None,
            "",
            *empty_word_video_updates(),
        )

    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    progress(0.05, desc="Preparing source character...")

    try:
        source_name, source_root = resolve_text_source(source_selection, source_upload)
        safe_text = re.sub(r'[^\w\u4e00-\u9fff]+', '_', input_text.strip())[:40].strip('_')
        output_name = f"ui_{time.strftime('%Y%m%d_%H%M%S')}_{safe_text or 'text'}"

        progress(0.15, desc="Translating and rendering...")
        command = [
            "bash",
            os.path.join(current_dir, TEXT_PIPELINE_SCRIPT),
            input_text.strip(),
            "--source-data-path",
            source_root,
            "--output-name",
            output_name,
        ]
        if smooth_word_segments:
            command.append("--smooth-word-segments")
        run_cmd_capture(command, current_dir=current_dir)

        model_input, generated, gloss_tokens = parse_translate_output()
        output_dir = os.path.join(current_dir, OUTPUTS2_DIR, output_name)
        rendered_video = find_rendered_video(output_dir)
        if not rendered_video:
            raise FileNotFoundError(f"找不到渲染完成影片，輸出資料夾：{output_dir}")

        progress(1.0, desc="Complete")
        smooth_status = "開啟" if smooth_word_segments else "關閉，保留原始 EHM 單詞動作"
        status = f"完成。Source: {source_name}。單詞片段平滑：{smooth_status}"
        return (
            status,
            build_translation_html(input_text.strip(), model_input, generated, gloss_tokens),
            rendered_video,
            rendered_video,
            *build_word_video_updates(gloss_tokens),
        )
    except subprocess.CalledProcessError as exc:
        error_text = exc.output[-4000:] if getattr(exc, "output", None) else str(exc)
        return (
            f"執行失敗：{exc}",
            f"<pre style='white-space:pre-wrap;color:#b00020;'>{ht.escape(error_text)}</pre>",
            None,
            "",
            *empty_word_video_updates(),
        )
    except Exception as exc:
        return (
            f"發生錯誤：{exc}",
            f"<pre style='white-space:pre-wrap;color:#b00020;'>{ht.escape(str(exc))}</pre>",
            None,
            "",
            *empty_word_video_updates(),
        )

# --- Gradio Interface ---

with gr.Blocks(title="虛擬角色手語翻譯系統", theme=gr.themes.Soft(), css=APP_CSS) as demo:
    gr.Markdown("# 虛擬角色手語翻譯系統")
    text_source_gallery_data = prepare_text_source_gallery()
    selected_text_source = gr.State({"caption": DEFAULT_TEXT_SOURCE})

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 選擇人物形象來源")
            gr.Markdown(
                "預設人物形象為圖庫中的第一張圖片。上傳的人物形象須為真實人物照片；"
                "若上傳動漫、插畫或其他非真人形象，將無法進行生成。"
            )
            with gr.Tabs():
                with gr.TabItem("從圖庫選擇"):
                    text_source_gallery = gr.Gallery(
                        value=text_source_gallery_data,
                        label="人物圖庫",
                        columns=2,
                        height="auto",
                        object_fit="contain",
                    )
                with gr.TabItem("上傳圖片"):
                    text_upload_source_image = gr.Image(label="人物圖片", type="filepath", height=360)

        with gr.Column(scale=2):
            gr.Markdown("### 2. 輸入自然語言")
            text_input_sentence = gr.Textbox(
                label="自然語言句子",
                placeholder="例如：你結婚了嗎",
                lines=3,
            )
            with gr.Row(elem_classes=["speech-input-row"]):
                speech_input_btn = gr.Button("語音輸入", variant="secondary")
            smooth_word_segments_checkbox = gr.Checkbox(
                label="啟用單詞片段平滑",
                value=False,
                info="取消勾選時，單詞影片動作參數會最接近原始 EHM-Tracker 追蹤結果。",
            )
            text_generate_btn = gr.Button("產生手語影片", variant="primary", size="lg")
            text_status = gr.Textbox(label="狀態", lines=2)
            text_translation_html = gr.HTML(label="模型輸出與手語轉換結果")

    gr.Markdown("### 3. 影片播放")
    text_render_video = gr.Video(label="渲染完成影片", height=360, elem_classes=["render-video"])
    text_render_path = gr.Textbox(label="渲染影片儲存位置", interactive=False)
    gr.Markdown("### 手語轉換結果單詞影片")
    word_video_outputs = []
    with gr.Row():
        for word_index in range(MAX_WORD_VIDEO_COMPONENTS):
            with gr.Column(min_width=220):
                word_video = gr.Video(
                    label=f"單詞影片 {word_index + 1}",
                    visible=False,
                    height=150,
                    elem_classes=["word-video"],
                )
                word_caption = gr.Markdown(visible=False, elem_classes=["word-caption"])
                word_video_outputs.extend([word_video, word_caption])
    
    # --- Event Handlers ---

    def on_gallery_select(evt: gr.SelectData):
        """Store the selected gallery caption for source-character lookup."""
        if evt.value:
            if isinstance(evt.value, dict):
                caption = evt.value.get("caption")
            elif isinstance(evt.value, (list, tuple)) and len(evt.value) > 1:
                caption = evt.value[1]
            else:
                caption = None
            selection = {"caption": caption} if caption else evt.value
            print(f"已選擇人物：{caption or evt.value}")
            return selection
        return None

    text_source_gallery.select(on_gallery_select, None, selected_text_source, show_progress="hidden")

    speech_input_btn.click(
        fn=None,
        inputs=text_input_sentence,
        outputs=text_input_sentence,
        js=SPEECH_TO_TEXT_JS,
        show_progress="hidden",
    )

    text_generate_btn.click(
        fn=run_text_to_guava,
        inputs=[
            selected_text_source,
            text_upload_source_image,
            text_input_sentence,
            smooth_word_segments_checkbox,
        ],
        outputs=[
            text_status,
            text_translation_html,
            text_render_video,
            text_render_path,
            *word_video_outputs,
        ],
    )

# --- Launch the App ---
if __name__ == "__main__":
    import argparse

    def find_available_port(preferred_port, max_tries=50):
        for port in range(preferred_port, preferred_port + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("0.0.0.0", port))
                except OSError:
                    continue
                return port
        raise OSError(f"Cannot find empty port in range: {preferred_port}-{preferred_port + max_tries - 1}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Whether to share the app.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 8188)))
    parser.add_argument("--strict-port", action="store_true", help="Fail instead of trying the next port when --port is busy.")
    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    os.makedirs(os.path.join(current_dir, OUTPUT_DIR), exist_ok=True)
    
    allowed_paths = [
        current_dir,
        os.path.join(current_dir, OUTPUT_DIR),
        os.path.join(current_dir, OUTPUTS2_DIR),
        os.path.join(current_dir, TRACKED_IMG_DIR),
        os.path.join(current_dir, TRACKED_VID_DIR),
        os.path.join(current_dir, TRACKED_SOURCE_IMAGE_DIR),
        os.path.join(current_dir, WORD_VIDEO_ASSET_ROOT),
        os.path.join(current_dir, EHM_TRACKER_DIR),
    ]

    launch_port = args.port if args.strict_port else find_available_port(args.port)
    if launch_port != args.port:
        print(f"Port {args.port} is busy; launching on {launch_port} instead.")
    
    demo.launch(
        server_name="0.0.0.0", 
        allowed_paths=allowed_paths, 
        share=args.share,
        server_port=launch_port
    )
