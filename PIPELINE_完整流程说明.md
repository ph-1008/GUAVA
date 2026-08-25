# GUAVA 中文至手語化身影片完整流程

> 最後核對日期：2026-08-20  
> 專案路徑：`/home/paohan/GUAVA`  
> 本文件說明目前實際使用的「中文自然語言 → 手語 gloss → 動作串接 → GUAVA 渲染」流程。  
> 專案備份、Conda 環境與交接步驟請另見 `README_HANDOVER.md`。

---

## 1. 系統用途

本系統將中文自然語言轉換成手語 gloss，根據 gloss 尋找已完成 EHM tracking 的手語片段，再將各片段的 SMPL-X、FLAME 與 MANO 參數裁切、串接並加入詞間過渡，最後以 GUAVA 將動作渲染到指定的人物形象。

主要使用情境是直接從既有 tracking 資料庫取用手語片段。只有在上傳新的人物圖片，或需要增加新的手語詞彙時，才需另外執行 EHM-Tracker。

---

## 2. 現行完整資料流

```text
使用者選擇或上傳人物形象
  ↓
輸入中文句子（文字或瀏覽器語音輸入）
  ↓ app.py / run_translate_to_guava.sh
sign_translate_env：translate_cli.py
  ↓
手語 gloss（例如：今天/哪裡/上課）
  ↓ gloss_to_guava_pipeline.py
依 merged_labels_tracked.json 解析每個 gloss
  ↓
既有 EHM tracking 片段
  ↓ tracking_concatenation_final_5.py
動作區間擷取 + 詞間過渡 + LMDB 合併
  ↓
outputs_2/<output-name>/
  ↓ main/test.py --render_cross_act
來源人物外觀 + 串接後的手語動作
  ↓
最終 MP4 影片
```

---

## 3. 正式入口與主要程式

| 用途 | 現行入口 |
| --- | --- |
| Gradio 圖形介面 | `app.py` |
| 命令列完整流程 | `run_translate_to_guava.sh` |
| gloss 解析、tracking 查找、串接與渲染調度 | `gloss_to_guava_pipeline.py` |
| 中文至 gloss 翻譯 | `sign_translate_code/CODE/GUI/translate_cli.py` |
| 動作片段裁切與串接 | `tracking_concatenation_final_5.py` |
| GUAVA 推論與渲染 | `main/test.py` |
| 新圖片或新影片的 EHM tracking | `EHM-Tracker` 內的 tracking 程式 |

以下檔案屬舊版、歷史實驗或特定案例用途，不是目前中文至影片主流程的入口：

- `app_GUAVA.py`
- `tracking_params_concatenation_fixed.py`
- `render_concatenated_with_avatar.sh`
- `outputs/concatenated_tracking_1224/`
- `outputs/concatenated_render_cross/`

---

## 4. 執行環境

主流程會使用兩個 Conda 環境：

| Conda 環境 | 用途 |
| --- | --- |
| `sign_translate_env` | 中文自然語言轉手語 gloss |
| `GUAVA` | tracking 資料處理、串接、Gradio 與 GPU 渲染 |

`run_translate_to_guava.sh` 會先進入 `sign_translate_env`，完成翻譯後再切換到 `GUAVA`。目前腳本預設 Conda 位於：

```text
/home/paohan/anaconda3
```

腳本與 pipeline 內仍有 `/home/paohan/GUAVA` 絕對路徑。若專案還原到其他使用者或其他目錄，必須同步調整這些路徑。

---

## 5. 使用 Gradio 介面

### 5.1 啟動

```bash
cd /home/paohan/GUAVA
source /home/paohan/anaconda3/etc/profile.d/conda.sh
conda activate GUAVA
python app.py --port 8188 --strict-port
```

介面預設監聽 `0.0.0.0:8188`。若不使用 `--strict-port`，當 8188 已被佔用時，程式會嘗試後續的可用 port。

### 5.2 操作流程

1. 從人物圖庫選擇已 tracking 的人物，或上傳新的真人圖片。
2. 輸入中文句子，也可使用瀏覽器的語音輸入功能。
3. 視需要勾選「啟用單詞片段平滑」。預設為關閉，以保留最接近 EHM-Tracker 原始追蹤的動作參數。
4. 按下「產生手語影片」。
5. 系統會顯示翻譯結果、對應單詞影片、最終渲染影片與存檔路徑。

上傳圖片時，若 `outputs/app/tracked_source_image/<圖片名>/` 尚沒有 `optim_tracking_ehm.pkl`，`app.py` 會先呼叫 EHM-Tracker 的 `src.tracking_single_image`。目前流程預期輸入為真人照片，動漫或插畫不在支援範圍內。

---

## 6. 使用命令列

### 6.1 完整翻譯與渲染

```bash
cd /home/paohan/GUAVA
bash run_translate_to_guava.sh "今天在哪裡上課" --output-name demo -d 0
```

預設人物來源：

```text
/home/paohan/GUAVA/outputs/app/tracked_source_image/Gemini_Generated_Image_kzne4skzne4skzne
```

指定其他已 tracking 的人物：

```bash
bash run_translate_to_guava.sh "今天在哪裡上課" \
  --source-data-path /path/to/tracked_source \
  --output-name demo \
  -d 0
```

### 6.2 只檢查 gloss 映射

```bash
bash run_translate_to_guava.sh "哈囉你好" --dry-run
```

`--dry-run` 會執行中文至 gloss 翻譯並檢查每個 gloss 是否能對應到 tracking 資料，但不會建立串接輸出或執行渲染。

### 6.3 只生成串接資料

```bash
bash run_translate_to_guava.sh "哈囉你好" \
  --no-render \
  --output-name concat_only
```

### 6.4 開啟單詞片段平滑

```bash
bash run_translate_to_guava.sh "今天在哪裡上課" \
  --smooth-word-segments \
  --output-name smooth_demo
```

### 6.5 主要 pipeline 參數

| 參數 | 功能 | 預設 |
| --- | --- | --- |
| `--gloss-output` | 指定包含 `Gloss joined:` 的檔案 | `sign_translate_code/gloss_output.txt` |
| `--labels-json` | gloss 至 tracking 路徑的映射檔 | `merged_labels_tracked.json` |
| `--output-root` | 輸出根目錄 | `outputs_2` |
| `--output-name` | 指定本次輸出目錄名稱 | 時間與 gloss 動態生成 |
| `--model-path` | GUAVA config 與 checkpoint 目錄 | `assets/GUAVA` |
| `--source-data-path` | 人物形象 tracking 目錄 | 預設 Gemini 人物 |
| `-d`, `--devices` | GPU 編號 | `0` |
| `--dry-run` | 只翻譯與解析 tracking 路徑 | 關閉 |
| `--no-render` | 完成串接後停止 | 關閉 |
| `--no-visualize` | 不產生 motion envelope PNG | 關閉 |
| `--smooth-word-segments` | 對每個單詞片段做輕度平滑 | 關閉 |

---

## 7. 中文至 gloss

`run_translate_to_guava.sh` 在 `sign_translate_env` 內執行：

```bash
cd /home/paohan/GUAVA/sign_translate_code/CODE/GUI
python translate_cli.py \
  --text "<中文句子>"
```

結果寫入：

```text
sign_translate_code/gloss_output.txt
```

Pipeline 優先讀取最後一行 `Gloss joined:`，並以 `/` 分割為 gloss tokens。若沒有 `Gloss joined:`，才嘗試解析 `Generated gloss:`。

例如：

```text
Gloss joined: 今天/哪裡/上課/^^
```

會解析為：

```text
今天
哪裡
上課
^^（符號，略過）
```

---

## 8. gloss 至 tracking 資料

### 8.1 主要映射檔

```text
EHM-Tracker/Sign_dataset/tw_sign_dataset/merged_labels_tracked.json
```

Pipeline 會先查詢映射檔，再對路徑進行標準化和 fallback 查找。目前會嘗試的主要位置包括：

```text
EHM-Tracker/Sign_dataset/tw_sign_dataset/
EHM-Tracker/results/dad_drive_1hr_step700/
EHM-Tracker/results/TWTSL_tracked/
EHM-Tracker/results/
outputs/TWTSL_tracked/
outputs/app/tracked_driven_video/
```

### 8.2 有效 tracking 目錄

每個可用的 tracking 目錄至少必須包含：

```text
<tracked_dir>/
├── optim_tracking_ehm.pkl
├── id_share_params.pkl
├── videos_info.json
└── img_lmdb/
    ├── data.mdb
    └── lock.mdb
```

`optim_tracking_flame.pkl`、`base_tracking.pkl`、`images/`、`masks/` 與 `visual_results/` 可存在於 EHM tracking 原始輸出中，但 gloss resolver 判斷該路徑是否可用時，主要檢查上述四項。

### 8.3 特殊解析規則

- 會略過標點符號和類似 `^^` 的符號 token。
- 支援部分同義或字形正規化，例如 `臺北` 對應 `台北`。
- 支援少數複合 token 展開，例如將 `二個小時` 展開為 `二` 與 `小時`。
- 只要有一個必要 gloss 無法解析，pipeline 就會中止，並要求更新 label 映射或補入 tracking 資料。

---

## 9. Tracking 資料結構

### 9.1 `optim_tracking_ehm.pkl`

該檔案以 frame key 為索引：

```python
{
    "frame_000000": {
        "smplx_coeffs": {
            "shape": ...,
            "joints_offset": ...,
            "head_scale": ...,
            "hand_scale": ...,
            "body_pose": ...,
            "global_pose": ...,
            "transl": ...,
            "left_hand_pose": ...,
            "right_hand_pose": ...,
            "expression": ...,
        },
        "flame_coeffs": {
            "shape_params": ...,
            "expression_params": ...,
            "neck_pose": ...,
            "jaw_pose": ...,
            "eyes_pose": ...,
        },
        "left_mano_coeffs": {...},
        "right_mano_coeffs": {...},
        "dwpose_rlt": {...},
        "body_crop": ...,
        "head_crop": ...,
        "left_hand_crop": ...,
        "right_hand_crop": ...,
        "head_lmk_203": ...,
        "head_lmk_70": ...,
        "head_lmk_mp": ...,
    },
    "frame_000001": {...},
}
```

實際 key 和維度應以當前 EHM-Tracker 輸出與 `dataset/data_loader.py` 為準，不應在交接文件中假設所有版本都有完全相同的維度。

### 9.2 `id_share_params.pkl`

包含身份共用參數，主要為 SMPL-X 體型與縮放、關節偏移，以及 FLAME 臉部形狀等資訊。串接結果會保留第一個片段的 `id_share_params.pkl`；真正 cross-reenactment 時還會以來源人物的身份參數取代驅動片段的身份參數。

### 9.3 `videos_info.json`

串接後的結構為：

```json
{
  "concatenated_video": {
    "frames_num": 218,
    "frames_keys": [
      "frame_000000",
      "frame_000001"
    ]
  }
}
```

`218` 只是格式範例。實際幀數由 gloss 數量、每個 tracking 片段的動作區間和詞間過渡數量決定。

### 9.4 `img_lmdb/`

LMDB 的 key 格式例如：

```text
frame_000000/ori_image
frame_000000/body_image
frame_000000/body_mask
frame_000000/body_matting
frame_000000/head_image
frame_000000/left_hand_image
frame_000000/right_hand_image
```

---

## 10. 動作裁切與串接

現行串接由 `tracking_concatenation_final_5.concatenate_tracking_data_with_peaks()` 處理。

### 10.1 動作區間偵測

對每個片段：

1. 從 DWPose 追蹤結果擷取左右手與身體位置。
2. 以左右手相鄰幀位移量的總和作為速度包絡。
3. 使用 Gaussian filter 平滑速度。
4. 使用 median、MAD、P35、P65 與 P90 建立自適應 high/low threshold。
5. 合併接近的動作區間，並保護過短片段不被過度裁切。
6. 第一個片段保留開頭，最後一個片段保留結尾；中間片段根據 motion envelope 裁切前後靜止區間。

目前 pipeline 傳入的主要設定：

```text
window_size = 5
min_peak_distance = 10
offset_frames = 5
num_transition_frames = 15
```

舊文件中的固定 `motion_threshold=0.01` 不再是目前主 pipeline 的裁切方法。

### 10.2 詞間過渡

相鄰兩個手語片段之間預設插入 15 幀。一般數值參數使用線性插值：

```python
alpha = i / (num_transition_frames + 1)
value = value_a * (1 - alpha) + value_b * alpha
```

SMPL-X 的 `global_pose`、`body_pose`、`left_hand_pose` 與 `right_hand_pose` 屬 axis-angle 旋轉參數，現行版本會先選擇較短旋轉路徑再插值，以減少跨越 `±π` 時繞長路徑旋轉的問題。

過渡幀在合併後 LMDB 中使用下一個片段的第一幀圖像資料；動作主要由插值後的 tracking 參數驅動。

### 10.3 可選的單詞平滑

預設狀態：

```text
smooth_word_segments = false
fix_hand_jitter = false
```

啟用 `--smooth-word-segments` 後，pipeline 會對 body、hand 與 face 參數套用輕度平滑。這個選項不影響詞間 15 幀過渡；詞間過渡仍會產生。

### 10.4 串接後幀數

對 `N` 個成功解析的 gloss 片段：

```text
總幀數 = 各片段裁切後幀數總和 + 15 × (N - 1)
```

因此總幀數不是固定 437，也不應用某一次實驗的幀數當成系統規格。

---

## 11. 串接輸出結構

每次非 `--dry-run` 執行會在 `outputs_2` 內建立獨立目錄。`peak_analysis_*.png` 只在未使用 `--no-visualize` 時產生；`render.log` 只在實際進入渲染階段後產生：

```text
outputs_2/<output-name>/
├── optim_tracking_ehm.pkl
├── id_share_params.pkl
├── videos_info.json
├── img_lmdb/
│   ├── data.mdb
│   └── lock.mdb
├── peak_analysis_1_<tracked-name>.png
├── peak_analysis_2_<tracked-name>.png
├── ...
├── gloss_pipeline_manifest.json
└── render.log
```

`gloss_pipeline_manifest.json` 會記錄：

- 原始 gloss tokens。
- 每個 token 的解析狀態與 tracking 路徑。
- 本次實際使用的 render command。
- 是否啟用單詞片段平滑與 hand-jitter 修復。

執行渲染時，`render.log` 會保留 `main/test.py` 的完整渲染輸出，排除渲染失敗時應先檢查此檔。

---

## 12. GUAVA Cross-Reenactment

Pipeline 產生的渲染命令等同於：

```bash
PYTHONPATH=/home/paohan/GUAVA:$PYTHONPATH \
/home/paohan/anaconda3/envs/GUAVA/bin/python \
  /home/paohan/GUAVA/main/test.py \
  -d 0 \
  -m /home/paohan/GUAVA/assets/GUAVA \
  -s /home/paohan/GUAVA/outputs_2/<output-name> \
  --data_path /home/paohan/GUAVA/outputs_2/<output-name> \
  --source_data_path /path/to/tracked_source \
  --skip_self_act \
  --render_cross_act
```

### 12.1 來源人物

`main/test.py` 會從 `source_data_path` 讀取來源人物的圖像、mask 與 tracking 參數，再以 `Ubody_Gaussian_inferer` 建立綁定到來源形象的 3D Gaussian 化身。

### 12.2 身份與動作分離

Cross-reenactment 的核心為：

```text
來源人物身份 + 串接後動作 = 最終化身影片
```

`change_id_info()` 會將目標幀中的以下身份參數替換為來源人物的值：

```text
smplx_coeffs.shape
smplx_coeffs.joints_offset
smplx_coeffs.head_scale
smplx_coeffs.hand_scale
flame_coeffs.shape_params
```

目標幀的身體、手部、臉部動作及位置參數則繼續保留，用來驅動來源人物。

### 12.3 逐幀渲染

對 `concatenated_video` 的每個 frame：

1. 載入串接後的 target tracking 參數。
2. 以 `change_id_info()` 替換身份參數。
3. 以 `Ubody_Gaussian` 根據當前姿態變形 Gaussian 化身。
4. 以 `GaussianRenderer` 在黑色背景上渲染。
5. 保存 `render/00000.png`、`render/00001.png` 等單幀圖片。
6. 以 30 FPS、quality 8 編碼為 MP4。

---

## 13. 最終輸出

當 pipeline 使用預設 saving name `render` 時，完整結構為：

```text
outputs_2/<output-name>/
├── optim_tracking_ehm.pkl
├── id_share_params.pkl
├── videos_info.json
├── img_lmdb/
├── gloss_pipeline_manifest.json
├── render.log
└── render_cross_act/
    └── <source-id>/
        └── <source-id>_concatenated_video/
            ├── render/
            │   ├── 00000.png
            │   ├── 00001.png
            │   └── ...
            ├── source_image.png
            └── <source-id>_concatenated_video_video.mp4
```

例如：

```text
outputs_2/ui_20260713_155614_哪裡/
└── render_cross_act/
    └── Gemini_Generated_Image_kzne4skzne4skzne/
        └── Gemini_Generated_Image_kzne4skzne4skzne_concatenated_video/
            └── Gemini_Generated_Image_kzne4skzne4skzne_concatenated_video_video.mp4
```

該案例的幀數為 116，只代表當次「哪裡」的實際輸出，不是系統固定值。

---

## 14. 增加新詞彙或新 tracking 資料

一般使用者輸入句子時，不會重新 tracking 所有影片。若某個 gloss 無法解析，才需進行資料準備：

1. 準備對應手語影片。
2. 使用 EHM-Tracker 產生 tracking 資料。
3. 確認目錄含 `optim_tracking_ehm.pkl`、`id_share_params.pkl`、`videos_info.json` 與 `img_lmdb/`。
4. 將 gloss 與 tracking 路徑寫入 `merged_labels_tracked.json`。
5. 使用 `--dry-run` 檢查映射。
6. 確認映射成功後再執行完整渲染。

目前 EHM-Tracker 所需的預訓練模型與資產數量很大，完整備份與還原方式請依 `README_HANDOVER.md`。

---

## 15. 常見問題

### 15.1 找不到 gloss 對應 tracking

檢查：

```text
EHM-Tracker/Sign_dataset/tw_sign_dataset/merged_labels_tracked.json
```

並確認映射目錄包含必要 tracking payload。可先執行：

```bash
bash run_translate_to_guava.sh "<測試句子>" --dry-run
```

### 15.2 串接後動作過度不理想

- 檢查各單詞 tracking 品質。
- 查看 `peak_analysis_*.png` 的裁切區間。
- 比較啟用與關閉 `--smooth-word-segments` 的結果。
- 修改過渡幀、motion envelope 參數前，應保留 manifest 和測試輸出以便比較。

### 15.3 渲染失敗

優先檢查：

- `outputs_2/<output-name>/render.log`
- `assets/GUAVA/config.yaml`
- `assets/GUAVA/checkpoints/`
- `source_data_path` 是否為完整 tracking 目錄。
- `nvidia-smi` 和 `torch.cuda.is_available()`。
- GUAVA 環境中的 CUDA rasterizer 是否可匯入。

### 15.4 介面沒有找到最終影片

`app.py` 會在本次 `outputs_2/<output-name>/` 內遞迴尋找 `*_video.mp4`。若影片不存在，請先查看同目錄的 `render.log` 和 `gloss_pipeline_manifest.json`。

---

## 16. 時間與容量

本流程不設固定處理時間或檔案大小，因為實際結果受以下因素影響：

- gloss 數量與每個 tracking 片段的幀數。
- motion envelope 裁切後的保留範圍。
- 過渡幀數。
- 圖像解析度與 LMDB 大小。
- GPU、CUDA、PyTorch 與 rasterizer 版本。
- 是否需要先對新圖片或影片進行 EHM tracking。

實際交接或性能驗證時，應以當次 `videos_info.json`、`render.log` 與輸出目錄容量為準，不應使用舊案例的固定 437 幀、固定秒數或固定容量作為系統規格。

---

## 17. 交接前建議測試

### 17.1 環境匯入測試

```bash
cd /home/paohan/GUAVA
source /home/paohan/anaconda3/etc/profile.d/conda.sh

conda activate GUAVA
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import lmdb, gradio, smplx, onnxruntime; print("GUAVA imports OK")'

conda activate sign_translate_env
python -c 'import torch, transformers; print("translation imports OK")'
```

### 17.2 gloss 解析測試

```bash
bash run_translate_to_guava.sh "哈囉你好" --dry-run
```

### 17.3 串接測試

```bash
bash run_translate_to_guava.sh "哈囉你好" \
  --no-render \
  --output-name handover_concat_test
```

### 17.4 完整 GPU 渲染測試

```bash
bash run_translate_to_guava.sh "哈囉你好" \
  --output-name handover_full_test \
  -d 0
```

測試完成後應同時確認：

- `gloss_pipeline_manifest.json` 的 tokens 與 tracking 解析路徑正確。
- `videos_info.json` 的 `frames_num` 與 `frames_keys` 一致。
- `img_lmdb/` 可讀取。
- `render.log` 沒有未處理錯誤。
- `render_cross_act/` 下已產生連續 PNG 與可播放 MP4。

---

## 18. 總結

目前系統的正式 pipeline 可簡化為：

1. **人物準備**：選擇已 tracking 的人物，或對新上傳圖片進行 tracking。
2. **中文翻譯**：以 `translate_cli.py` 將中文自然語言轉成 gloss。
3. **詞彙解析**：依 `merged_labels_tracked.json` 尋找既有 EHM tracking 片段。
4. **動作串接**：以 `tracking_concatenation_final_5.py` 進行 motion-envelope 裁切、詞間過渡與 LMDB 合併。
5. **GUAVA 渲染**：以來源人物的身份與串接後動作執行 cross-reenactment。
6. **輸出與紀錄**：將 tracking、manifest、render log、單幀 PNG 與 MP4 保存在 `outputs_2/<output-name>/`。

若本文件與程式行為不一致，應以 `app.py`、`run_translate_to_guava.sh`、`gloss_to_guava_pipeline.py`、`tracking_concatenation_final_5.py` 和 `main/test.py` 的實際實作為準，並同步更新本文件的最後核對日期。
