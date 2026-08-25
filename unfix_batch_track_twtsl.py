import os
import subprocess
from pathlib import Path
import argparse
import json
from datetime import datetime

# 設定路徑
PROJECT_ROOT = Path(__file__).parent  # GUAVA 專案根目錄
VIDEO_DIR = PROJECT_ROOT / "assets" / "TWTSL"
SOURCE_IMAGE_DIR = PROJECT_ROOT / "outputs" / "app" / "tracked_source_image" / "Gemini_Generated_Image_kzne4skzne4skzne"
TRACKED_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "TWTSL_tracked"
RENDER_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "TWTSL_rendered"

def find_video_files(directory, extensions=['.mp4', '.avi', '.mov', '.mkv']):
    """尋找目錄中的所有影片檔案"""
    video_files = []
    directory = Path(directory)
    
    for ext in extensions:
        video_files.extend(directory.glob(f'**/*{ext}'))
    
    return sorted(video_files)

def track_video_with_ehm(video_path, output_dir):
    """使用 EHM-Tracker 追蹤影片"""
    video_name = video_path.stem
    output_path = Path(output_dir) / video_name
    
    print(f"\n{'='*60}")
    print(f"正在追蹤影片: {video_path.name}")
    print(f"輸出路徑: {output_path}")
    print(f"{'='*60}\n")
    
    # 使用 EHM-Tracker 追蹤
    cmd = [
        "python", "EHM-Tracker/tracking_video.py",
        str(video_path),
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        
        # 檢查追蹤結果是否存在
        if (output_path / "optim_tracking_ehm.pkl").exists():
            return output_path
        else:
            print(f"警告：追蹤完成但找不到 optim_tracking_ehm.pkl")
            return None
    except subprocess.CalledProcessError as e:
        print(f"錯誤：追蹤失敗 - {e}")
        return None

def render_with_avatar(tracked_data_path, source_image_path, output_dir):
    """使用 GUAVA 將追蹤數據渲染到目標角色"""
    video_name = Path(tracked_data_path).name
    output_path = Path(output_dir) / video_name
    
    print(f"\n{'='*60}")
    print(f"正在渲染: {video_name}")
    print(f"追蹤數據: {tracked_data_path}")
    print(f"目標角色: {source_image_path}")
    print(f"輸出路徑: {output_path}")
    print(f"{'='*60}\n")
    
    cmd = [
        "python", "-m", "main.test",
        "-d", "0",
        "-m", "assets/GUAVA",
        "-s", str(output_path),
        "--data_path", str(tracked_data_path),
        "--source_data_path", str(source_image_path),
        "--skip_self_act",
        "--render_cross_act"
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"錯誤：渲染失敗 - {e}")
        return None

def batch_process_twtsl(
    video_dir=None,
    source_image=None,
    tracking_output_dir=None,
    render_output_dir=None,
    track_only=False,
    render_only=False
):
    """批次處理 TWTSL 資料夾中的所有影片"""
    
    # 使用預設值
    if video_dir is None:
        video_dir = VIDEO_DIR
    if source_image is None:
        source_image = SOURCE_IMAGE_DIR
    if tracking_output_dir is None:
        tracking_output_dir = TRACKED_OUTPUT_DIR
    if render_output_dir is None:
        render_output_dir = RENDER_OUTPUT_DIR
    
    # 確認路徑存在
    video_path = Path(video_dir)
    source_path = Path(source_image)
    
    if not video_path.exists():
        print(f"錯誤：影片目錄不存在 - {video_dir}")
        return
    
    if not render_only and not source_path.exists():
        print(f"警告：來源角色路徑不存在 - {source_image}")
        print("將僅執行追蹤，不進行渲染")
        track_only = True
    
    # 建立輸出目錄
    Path(tracking_output_dir).mkdir(parents=True, exist_ok=True)
    Path(render_output_dir).mkdir(parents=True, exist_ok=True)
    
    # 尋找所有影片
    video_files = find_video_files(video_dir)
    
    if not video_files:
        print(f"錯誤：在 {video_dir} 中找不到影片檔案")
        return
    
    print(f"\n找到 {len(video_files)} 個影片檔案")
    print("影片列表:")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf.name}")
    
    # 處理結果統計
    results = {
        'total': len(video_files),
        'tracked': 0,
        'rendered': 0,
        'failed': [],
        'details': []
    }
    
    # 記錄結果
    log_file = Path(tracking_output_dir) / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"批次處理開始時間: {datetime.now()}\n")
        f.write(f"影片來源: {video_dir}\n")
        f.write(f"角色來源: {source_image}\n")
        f.write(f"追蹤輸出: {tracking_output_dir}\n")
        f.write(f"渲染輸出: {render_output_dir}\n")
        f.write(f"總共 {len(video_files)} 個影片\n")
        f.write("="*80 + "\n\n")
    
    # 逐一處理
    for idx, video_file in enumerate(video_files, 1):
        print(f"\n\n{'='*60}")
        print(f"處理進度: {idx}/{len(video_files)}")
        print(f"{'='*60}")
        
        video_name = video_file.stem
        tracked_path = Path(tracking_output_dir) / video_name
        
        result_info = {
            'index': idx,
            'video': str(video_file),
            'video_name': video_file.name,
            'success': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # 步驟1: 使用 EHM-Tracker 追蹤影片
        if not render_only:
            # 檢查是否已有追蹤數據
            if tracked_path.exists() and (tracked_path / "optim_tracking_ehm.pkl").exists():
                print(f"✓ 跳過追蹤（已存在）: {video_name}")
                result_info['tracking_status'] = 'skipped'
            else:
                tracked_result = track_video_with_ehm(video_file, tracking_output_dir)
                if tracked_result:
                    results['tracked'] += 1
                    result_info['tracking_status'] = 'success'
                    print(f"✓ 追蹤成功: {video_file.name}")
                else:
                    results['failed'].append(f"{video_name} (tracking)")
                    result_info['success'] = False
                    result_info['tracking_status'] = 'failed'
                    result_info['error'] = 'tracking failed'
                    print(f"✗ 追蹤失敗: {video_file.name}")
                    
                    # 記錄失敗並繼續下一個
                    results['details'].append(result_info)
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{idx}/{len(video_files)}] {video_file.name}\n")
                        f.write(f"  狀態: 失敗 (追蹤)\n")
                        f.write(f"  時間: {result_info['timestamp']}\n\n")
                    continue
        
        # 步驟2: 使用 GUAVA 渲染到目標角色
        if not track_only:
            if not tracked_path.exists() or not (tracked_path / "optim_tracking_ehm.pkl").exists():
                print(f"✗ 警告：找不到追蹤數據 - {tracked_path}")
                results['failed'].append(f"{video_name} (no tracking data)")
                result_info['success'] = False
                result_info['rendering_status'] = 'no_data'
                result_info['error'] = 'tracking data not found'
            else:
                rendered_result = render_with_avatar(
                    tracked_path,
                    source_path,
                    render_output_dir
                )
                
                if rendered_result:
                    results['rendered'] += 1
                    result_info['rendering_status'] = 'success'
                    print(f"✓ 渲染成功: {video_file.name}")
                else:
                    results['failed'].append(f"{video_name} (rendering)")
                    result_info['success'] = False
                    result_info['rendering_status'] = 'failed'
                    result_info['error'] = 'rendering failed'
                    print(f"✗ 渲染失敗: {video_file.name}")
        
        results['details'].append(result_info)
        
        # 寫入日誌
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{idx}/{len(video_files)}] {video_file.name}\n")
            f.write(f"  狀態: {'成功' if result_info['success'] else '失敗'}\n")
            if 'tracking_status' in result_info:
                f.write(f"  追蹤: {result_info['tracking_status']}\n")
            if 'rendering_status' in result_info:
                f.write(f"  渲染: {result_info['rendering_status']}\n")
            if not result_info['success'] and 'error' in result_info:
                f.write(f"  錯誤: {result_info['error']}\n")
            f.write(f"  時間: {result_info['timestamp']}\n\n")
    
    # 顯示最終統計
    print(f"\n\n{'='*60}")
    print("處理完成！統計結果：")
    print(f"{'='*60}")
    print(f"總影片數: {results['total']}")
    if not render_only:
        print(f"成功追蹤: {results['tracked']}")
    if not track_only:
        print(f"成功渲染: {results['rendered']}")
    if results['failed']:
        print(f"\n失敗項目 ({len(results['failed'])}):")
        for failed in results['failed']:
            print(f"  - {failed}")
    print(f"\n日誌檔案: {log_file}")
    print(f"輸出位置:")
    print(f"  追蹤數據: {tracking_output_dir}")
    print(f"  渲染影片: {render_output_dir}")
    print(f"{'='*60}\n")
    
    # 儲存 JSON 結果
    json_file = Path(tracking_output_dir) / f"processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': results['total'],
                'tracked': results['tracked'],
                'rendered': results['rendered'],
                'failed': len(results['failed']),
                'failed_list': results['failed']
            },
            'details': results['details']
        }, f, indent=2, ensure_ascii=False)
    
    print(f"結果摘要: {json_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批次處理 TWTSL 手語影片')
    parser.add_argument('--video_dir', type=str, 
                        default=None,
                        help='手語影片目錄')
    parser.add_argument('--source_image', type=str,
                        default=None,
                        help='目標角色的追蹤數據路徑')
    parser.add_argument('--tracking_output', type=str,
                        default=None,
                        help='追蹤結果輸出目錄')
    parser.add_argument('--render_output', type=str,
                        default=None,
                        help='渲染結果輸出目錄')
    parser.add_argument('--track_only', action='store_true',
                        help='僅執行追蹤，不渲染')
    parser.add_argument('--render_only', action='store_true',
                        help='僅執行渲染（假設已有追蹤數據）')
    
    args = parser.parse_args()
    
    batch_process_twtsl(
        video_dir=args.video_dir,
        source_image=args.source_image,
        tracking_output_dir=args.tracking_output,
        render_output_dir=args.render_output,
        track_only=args.track_only,
        render_only=args.render_only
    )