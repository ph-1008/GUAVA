import gradio as gr
import os
from pathlib import Path
import subprocess

def search_video_by_keyword(keyword, base_path):
    """搜尋單一關鍵字並返回影片路徑（完全符合）"""
    try:
        # 搜尋包含關鍵字的資料夾
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)
            
            # 檢查是否為資料夾
            if os.path.isdir(folder_path):
                # 從資料夾名稱中提取關鍵字（假設格式為 prefix_關鍵字）
                # 例如: Gemini_Generated_Image_kzne4skzne4skzne_工作
                parts = folder_name.split('_')
                # 取最後一個部分作為關鍵字
                folder_keyword = parts[-1] if parts else ""
                
                # 完全符合檢查
                if folder_keyword == keyword:
                    # 尋找 mp4 檔案
                    video_name = f"{folder_name}_video.mp4"
                    video_path = os.path.join(folder_path, video_name)
                    
                    if os.path.exists(video_path):
                        return video_path
                    else:
                        # 嘗試尋找其他 mp4 檔案
                        mp4_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
                        if mp4_files:
                            video_path = os.path.join(folder_path, mp4_files[0])
                            return video_path
        
        return None
    except Exception as e:
        print(f"搜尋 '{keyword}' 時發生錯誤: {str(e)}")
        return None

def concatenate_videos_ffmpeg(video_paths, output_path):
    """使用 ffmpeg 串接影片"""
    try:
        # 建立暫存的檔案清單
        temp_dir = os.path.dirname(output_path)
        list_file = os.path.join(temp_dir, "concat_list.txt")
        
        # 寫入影片清單
        with open(list_file, 'w') as f:
            for video_path in video_paths:
                f.write(f"file '{video_path}'\n")
        
        # 使用 ffmpeg 串接
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',
            '-y',  # 覆蓋輸出檔案
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 刪除暫存檔案
        if os.path.exists(list_file):
            os.remove(list_file)
        
        if result.returncode == 0:
            return True, "影片串接成功"
        else:
            return False, f"ffmpeg 錯誤: {result.stderr}"
    
    except Exception as e:
        return False, f"串接影片時發生錯誤: {str(e)}"

def search_and_play_video(keyword_input):
    """搜尋包含關鍵字的資料夾並返回影片路徑"""
    base_path = "/home/paohan/GUAVA/outputs/app/render_cross_act/Gemini_Generated_Image_kzne4skzne4skzne/"
    
    if not keyword_input:
        return None, "請輸入關鍵字"
    
    try:
        # 檢查基礎路徑是否存在
        if not os.path.exists(base_path):
            return None, f"路徑不存在: {base_path}"
        
        # 將輸入按空格分割成多個關鍵字
        keywords = keyword_input.strip().split()
        
        if len(keywords) == 1:
            # 單一關鍵字，直接搜尋並返回
            video_path = search_video_by_keyword(keywords[0], base_path)
            if video_path:
                return video_path, f"找到影片: {os.path.basename(video_path)}"
            else:
                return None, f"找不到完全符合 '{keywords[0]}' 的影片"
        
        else:
            # 多個關鍵字，搜尋所有影片並串接
            video_paths = []
            found_keywords = []
            missing_keywords = []
            
            for keyword in keywords:
                video_path = search_video_by_keyword(keyword, base_path)
                if video_path:
                    video_paths.append(video_path)
                    found_keywords.append(keyword)
                else:
                    missing_keywords.append(keyword)
            
            if not video_paths:
                return None, f"找不到任何影片。缺少的關鍵字: {', '.join(missing_keywords)}"
            
            # 建立臨時檔案
            temp_dir = "/home/paohan/GUAVA/outputs/app/temp"
            os.makedirs(temp_dir, exist_ok=True)
            output_path = os.path.join(temp_dir, f"concatenated_{'_'.join(found_keywords)}.mp4")
            
            status_msg = f"找到 {len(video_paths)} 個影片: {', '.join(found_keywords)}"
            if missing_keywords:
                status_msg += f"\n找不到的關鍵字: {', '.join(missing_keywords)}"
            status_msg += "\n正在串接影片..."
            
            # 使用 ffmpeg 串接影片
            success, message = concatenate_videos_ffmpeg(video_paths, output_path)
            
            if success:
                status_msg = f"成功串接 {len(video_paths)} 個影片: {', '.join(found_keywords)}"
                if missing_keywords:
                    status_msg += f"\n找不到的關鍵字: {', '.join(missing_keywords)}"
                return output_path, status_msg
            else:
                return None, f"串接失敗: {message}"
    
    except Exception as e:
        return None, f"錯誤: {str(e)}"

# 建立 Gradio 介面
with gr.Blocks(title="影片搜尋播放器") as demo:
    gr.Markdown("# 影片搜尋播放器")
    gr.Markdown("輸入關鍵字來搜尋並播放對應的影片")
    gr.Markdown("**提示**: 輸入多個關鍵字（用空格分隔）可以串接多個影片，例如: `工作 休息 運動`")
    gr.Markdown("**注意**: 關鍵字必須完全符合資料夾名稱的最後部分")
    
    with gr.Row():
        with gr.Column(scale=1):
            keyword_input = gr.Textbox(
                label="關鍵字",
                placeholder="例如: 工作 或 工作 休息 運動",
                lines=1
            )
            search_btn = gr.Button("搜尋", variant="primary")
            status_text = gr.Textbox(
                label="狀態",
                interactive=False,
                lines=4
            )
        
        with gr.Column(scale=2):
            video_output = gr.Video(
                label="影片播放",
                autoplay=True
            )
    
    # 綁定事件
    search_btn.click(
        fn=search_and_play_video,
        inputs=keyword_input,
        outputs=[video_output, status_text]
    )
    
    # 也支援按 Enter 鍵搜尋
    keyword_input.submit(
        fn=search_and_play_video,
        inputs=keyword_input,
        outputs=[video_output, status_text]
    )

if __name__ == "__main__":
    demo.launch(share=False)