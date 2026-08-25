# GUAVA 專案備份、還原與交接說明

> 最後核對日期：2026-08-20
> 專案來源：`/home/paohan/GUAVA`
> 平台：Linux x86_64、Ubuntu 22.04
> 原始 GUAVA 論文與安裝說明請見 [README.md](README.md)。

## 1. 文件目的

本文件供 GUAVA 專案的完整備份、災難還原與人員交接使用。這個工作目錄除了上游 GUAVA 程式碼，也包含手語翻譯、EHM-Tracker、模型、資料集、實驗結果，以及尚未提交到 Git 的修改。因此：

- 完整 working tree 備份才是本專案的交接主體。
- 只備份 GitHub repository 並不足以還原目前成果。
- 不可用 `git clean`、重新 clone 或重新初始化 submodule 取代原始工作目錄。
- Conda 環境位於專案目錄外，必須另外封裝。

2026-08-10 完整盤點時，專案約為 **1.4 TiB、2,929,483 個檔案**。其中約 1.3 TiB 位於 `EHM-Tracker/Sign_dataset/tw_sign_dataset`。正式備份前請在目的地預留至少 **2 TiB** 可用空間。

## 2. 系統功能與資料流程

本專案將中文自然語言轉換成手語 gloss，尋找對應的已追蹤手語片段，裁切、串接並加入詞間過渡，最後交由 GUAVA 產生高斯化身影片。單詞片段內平滑預設關閉。

```text
中文句子
  ↓ sign_translate_env
自然語言 → 手語 gloss
  ↓ 依標籤查找已追蹤片段
EHM tracking 資料
  ↓ 動作裁切、串接、詞間過渡
合併後的 SMPL-X / FLAME / MANO 參數
  ↓ GUAVA environment + GPU rendering
最終化身影片
```

詳細的 tracking 資料結構與 cross-reenactment 說明另見 [PIPELINE_完整流程说明.md](PIPELINE_%E5%AE%8C%E6%95%B4%E6%B5%81%E7%A8%8B%E8%AF%B4%E6%98%8E.md)。

## 3. 重要入口

| 用途 | 入口 |
| --- | --- |
| 完整「中文 → gloss → 串接 → 渲染」流程 | `run_translate_to_guava.sh` |
| 流程核心 | `gloss_to_guava_pipeline.py` |
| Gradio 操作介面 | `app.py` |
| GUAVA 渲染 | `main/test.py` |
| GUAVA 訓練 | `main/train.py` |
| EHM 影片追蹤 | `EHM-Tracker/tracking_video.py` |
| 動作串接主要版本 | `tracking_concatenation_final_5.py` |
| 翻譯 CLI | `sign_translate_code/CODE/GUI/translate_cli.py` |

## 4. 目錄說明

大型資料集與 tracking 目錄容量沿用 2026-08-10 的完整盤點；其餘主要目錄已於 2026-08-20 快速複核。容量可能因實驗輸出或檔案整理而改變。

| 路徑 | 約略容量 | 用途 |
| --- | ---: | --- |
| `EHM-Tracker/Sign_dataset` | 1.3 TiB | 主要台灣手語資料集，完整交接時不可遺漏 |
| `EHM-Tracker/results` | 82 GiB | EHM tracking 與 LMDB 結果 |
| `outputs_2` | 6.1 GiB | 中文到手語流程的新輸出與測試結果 |
| `outputs` | 9.5 GiB | tracking、串接、render、training 等歷史輸出 |
| `EHM-Tracker/pretrained` | 6.6 GiB | EHM 所需預訓練模型 |
| `sign_translate_code` | 2.2 GiB | 中文至手語 gloss 模型、程式與資料 |
| `experiment` | 3.5 GiB | 平滑度、motion envelope 等實驗 |
| `assets` | 2.9 GiB | GUAVA checkpoint、SMPL-X、FLAME 與範例資產 |
| `main`、`models`、`submodules` | 約 35 MiB | GUAVA 訓練、推論與 CUDA 擴充模組 |
| `latex` | 86 MiB | 論文、圖表與研究文件 |
| `檔案前身`、`unused` | 小量 | 舊版或暫未使用程式；完整交接仍保留 |

### 4.1 必要模型與資產

至少需要保留：

- `assets/GUAVA/`：GUAVA checkpoint，約 1.3 GiB。
- `assets/SMPLX/`：SMPL-X 模型與 symbolic link。
- `assets/FLAME/`：FLAME 模型。
- `EHM-Tracker/pretrained/`：DWPose、HaMeR、PIXIE、landmark、matting 等模型。
- `EHM-Tracker/assets/`：SMPL-X、FLAME、MANO、StyleMatte 及手語索引資產。
- `sign_translate_code/CODE/GUI/best_model/`：中文至 gloss 模型。

SMPL-X、FLAME 及部分資料可能有下載或再散布限制。交接者應沿用原授權範圍，不應直接公開完整備份內容。

## 5. Git 狀態注意事項

本目錄不是乾淨的上游 repository：

- 根目錄為 `Pixel-Talk/GUAVA` 的 Git repository。
- `EHM-Tracker` 是另一個 Git 工作區／submodule。
- `sign_translate_code` 屬於根目錄 Git repository，不是獨立 Git 工作區。
- 根目錄與 `EHM-Tracker` 都有尚未提交的變更；根目錄另有大量未追蹤的模型、程式、輸出與研究資料。

因此備份必須包含每一層 `.git` 以及整個 working tree。接手前不要執行會清除未追蹤檔案的 Git 指令。

正式備份時，將以下資訊存入備份目錄的 `manifests/git/`：

```bash
git -C /home/paohan/GUAVA remote -v
git -C /home/paohan/GUAVA rev-parse HEAD
git -C /home/paohan/GUAVA status --short --branch

git -C /home/paohan/GUAVA/EHM-Tracker remote -v
git -C /home/paohan/GUAVA/EHM-Tracker rev-parse HEAD
git -C /home/paohan/GUAVA/EHM-Tracker status --short --branch

```

## 6. 執行環境

主流程實際使用兩個 Conda 環境：

| 環境 | 目前容量 | 用途 |
| --- | ---: | --- |
| `GUAVA` | 約 9.1 GiB | tracking 串接、GUAVA 推論、render、Gradio |
| `sign_translate_env` | 約 6.7 GiB | 中文自然語言轉成手語 gloss |

`run_translate_to_guava.sh` 會先進入 `sign_translate_env`，再切換到 `GUAVA`。

### 6.1 環境檔的重要限制

- `GUAVA_backup.yml` 是既有的 GUAVA 環境快照，可作參考，但交接時仍應重新匯出並使用 `conda-pack`。
- `sign_translate_code/environment.yml` 內含 Windows 套件，且環境名稱為 `sign_model`；它不是目前 Linux 主流程所使用的 `sign_translate_env`，不可當成唯一還原來源。
- `requirements.txt` 與目前環境的 PyTorch/CUDA 套件未必完全一致。
- NVIDIA driver 不包含在 Conda 環境裡，接手機器仍需安裝相容的 driver。

### 6.2 正式備份時封裝環境

以下使用 `/path/to/backups` 作為範例備份根目錄。實際執行前務必替換成已確認容量、權限與檔案系統功能的真實路徑。

```bash
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
readonly GUAVA_HANDOVER_ROOT=/path/to/backups/GUAVA_handover_20260820

mkdir -p "${GUAVA_HANDOVER_ROOT}/environments"

conda env export -n GUAVA --no-builds \
  | sed '/^prefix:/d' \
  > "${GUAVA_HANDOVER_ROOT}/environments/GUAVA-full.yml"
conda list -n GUAVA --explicit \
  > "${GUAVA_HANDOVER_ROOT}/environments/GUAVA-explicit.txt"
conda run -n GUAVA python -m pip freeze \
  > "${GUAVA_HANDOVER_ROOT}/environments/GUAVA-pip-freeze.txt"
"${HOME}/anaconda3/bin/conda-pack" -n GUAVA \
  -o "${GUAVA_HANDOVER_ROOT}/environments/GUAVA-linux-x86_64.tar.gz"

conda env export -n sign_translate_env --no-builds \
  | sed '/^prefix:/d' \
  > "${GUAVA_HANDOVER_ROOT}/environments/sign_translate_env-full.yml"
conda list -n sign_translate_env --explicit \
  > "${GUAVA_HANDOVER_ROOT}/environments/sign_translate_env-explicit.txt"
conda run -n sign_translate_env python -m pip freeze \
  > "${GUAVA_HANDOVER_ROOT}/environments/sign_translate_env-pip-freeze.txt"
"${HOME}/anaconda3/bin/conda-pack" -n sign_translate_env \
  -o "${GUAVA_HANDOVER_ROOT}/environments/sign_translate_env-linux-x86_64.tar.gz"
```

## 7. 完整備份

### 7.1 建議的備份結構

```text
GUAVA_handover_20260820/
├── project/
│   └── GUAVA/                 # 專案完整原樣副本
├── environments/             # Conda 封裝與套件清單
├── manifests/
│   ├── git/
│   ├── DIRECTORY_SIZES.txt
│   ├── FILE_COUNT.txt
│   ├── SYSTEM_INFO.txt
│   └── SHA256SUMS
└── README_HANDOVER.md
```

每次交接建立新的日期目錄，不覆蓋前一次備份。

### 7.2 備份前檢查

目前已確認沒有正在執行的訓練、追蹤、渲染或 UI 工作。正式開始前仍建議快速確認一次：

```bash
pgrep -af 'app.py|main/train.py|main/test.py|tracking_video.py|gloss_to_guava_pipeline.py'
```

若沒有輸出，即可執行單次完整同步。若未來備份時有上述程序在寫入 LMDB 或 output，應等工作結束後再同步。

### 7.3 使用 rsync 複製完整工作目錄

專案含近 300 萬個檔案，建議使用 `rsync` 以便續傳與驗證。目前系統尚未安裝 `rsync`，正式備份前應先依系統管理規範完成安裝，再執行以下檢查：

```bash
command -v rsync
rsync --version
```

```bash
readonly GUAVA_SOURCE=/home/paohan/GUAVA/
readonly GUAVA_HANDOVER_ROOT=/path/to/backups/GUAVA_handover_20260820

mkdir -p "${GUAVA_HANDOVER_ROOT}/project/GUAVA"

rsync -aHAX \
  --partial \
  --human-readable \
  --info=progress2 \
  "${GUAVA_SOURCE}" \
  "${GUAVA_HANDOVER_ROOT}/project/GUAVA/"
```

注意：

- 來源路徑結尾的 `/` 代表複製 GUAVA 目錄內容。
- 目的地必須是新建立且專屬於這次交接的目錄。
- 不使用排除規則；`.git`、模型、資料集、outputs、快取與中文檔名全部保留。
- 目的檔案系統若不支援 ACL 或 extended attributes，應先確認影響，再調整 `-A`、`-X`。
- 不建議製作單一 1.4 TiB ZIP，因為不利於續傳、增量同步與局部還原。

### 7.4 記錄系統與容量

```bash
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
readonly GUAVA_HANDOVER_ROOT=/path/to/backups/GUAVA_handover_20260820

mkdir -p "${GUAVA_HANDOVER_ROOT}/manifests/git"

du -h --max-depth=2 /home/paohan/GUAVA \
  > "${GUAVA_HANDOVER_ROOT}/manifests/DIRECTORY_SIZES.txt"
find /home/paohan/GUAVA -xdev -type f -printf . \
  | wc -c \
  > "${GUAVA_HANDOVER_ROOT}/manifests/FILE_COUNT.txt"

{
  uname -a
  sed -n '1,20p' /etc/os-release
  gcc --version
  ffmpeg -version
  conda --version
  nvidia-smi
} > "${GUAVA_HANDOVER_ROOT}/manifests/SYSTEM_INFO.txt" 2>&1
```

## 8. 備份驗證

### 8.1 比對來源與備份

`rsync` 完成後執行 checksum dry-run。若備份完全一致，不應列出待複製檔案：

```bash
readonly GUAVA_SOURCE=/home/paohan/GUAVA/
readonly GUAVA_BACKUP=/path/to/backups/GUAVA_handover_20260820/project/GUAVA/

rsync -aHAXnrc \
  --itemize-changes \
  "${GUAVA_SOURCE}" \
  "${GUAVA_BACKUP}"
```

這會完整讀取來源與備份的約 1.4 TiB 資料，可能需要數小時，但可驗證檔案內容，而不只比較時間與容量。

### 8.2 建立長期校驗碼

在備份根目錄執行：

```bash
cd /path/to/backups/GUAVA_handover_20260820

find project/GUAVA environments -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > manifests/SHA256SUMS

sha256sum --check manifests/SHA256SUMS
```

完成後保存 `rsync` log、檔案數、容量與 checksum 驗證結果。

## 9. 從備份還原

### 9.1 還原專案

目前腳本預期專案位於接手者的 `${HOME}/GUAVA`，Conda 安裝位於 `${HOME}/anaconda3`。最簡單的方式是維持這個配置。

```bash
rsync -aHAX \
  --human-readable \
  --info=progress2 \
  /path/to/backups/GUAVA_handover_20260820/project/GUAVA/ \
  /home/接手帳號/GUAVA/
```

若使用不同路徑，需同步修改 `run_translate_to_guava.sh` 及其他個人化腳本中的路徑設定。

### 9.2 還原封裝環境

先安裝 Miniconda/Anaconda，並確認 `${HOME}/anaconda3` 存在。目標環境目錄必須是新的空目錄。

```bash
mkdir -p "${HOME}/anaconda3/envs/GUAVA"
tar -xzf /path/to/backups/GUAVA_handover_20260820/environments/GUAVA-linux-x86_64.tar.gz \
  -C "${HOME}/anaconda3/envs/GUAVA"
source "${HOME}/anaconda3/envs/GUAVA/bin/activate"
conda-unpack
deactivate

mkdir -p "${HOME}/anaconda3/envs/sign_translate_env"
tar -xzf /path/to/backups/GUAVA_handover_20260820/environments/sign_translate_env-linux-x86_64.tar.gz \
  -C "${HOME}/anaconda3/envs/sign_translate_env"
source "${HOME}/anaconda3/envs/sign_translate_env/bin/activate"
conda-unpack
deactivate
```

`conda-pack` 適用於相同 Linux x86_64 類型的機器。若跨作業系統或架構，應改用 YAML/explicit/pip 清單重建，並重新編譯 CUDA submodules。

## 10. 啟動與交接測試

### 10.1 最小環境檢查

```bash
cd "${HOME}/GUAVA"
source "${HOME}/anaconda3/etc/profile.d/conda.sh"

conda activate GUAVA
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import lmdb, gradio, smplx, onnxruntime; print("GUAVA imports OK")'

conda activate sign_translate_env
python -c 'import torch, transformers; print("translation imports OK")'
```

### 10.2 只解析 gloss，不產生串接或影片

```bash
cd "${HOME}/GUAVA"
bash run_translate_to_guava.sh "哈囉你好" --dry-run
```

此測試會執行中文至 gloss 翻譯，再檢查 gloss 是否能對應到 tracking 資料，但不建立最終輸出。

### 10.3 產生串接資料但不渲染

```bash
cd "${HOME}/GUAVA"
bash run_translate_to_guava.sh "哈囉你好" --no-render --output-name handover_smoke_test
```

### 10.4 完整 GPU 渲染

```bash
cd "${HOME}/GUAVA"
bash run_translate_to_guava.sh "哈囉你好" --output-name handover_full_test -d 0
```

### 10.5 啟動 Gradio UI

```bash
cd "${HOME}/GUAVA"
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate GUAVA
python app.py --port 8188 --strict-port
```

預設監聽 `0.0.0.0:8188`。若伺服器有防火牆或對外網路，請先限制可存取來源；不要直接把研究資料與介面暴露到公網。

## 11. 常見問題

### 找不到 Conda 或環境

確認 Anaconda/Miniconda 安裝路徑，以及下列檔案是否存在：

```text
${HOME}/anaconda3/etc/profile.d/conda.sh
${HOME}/anaconda3/envs/GUAVA/
${HOME}/anaconda3/envs/sign_translate_env/
```

### CUDA、PyTorch 或自訂 rasterizer 載入失敗

檢查 NVIDIA driver、`torch.version.cuda`、`torch.cuda.is_available()`，以及下列 CUDA 擴充是否能 import：

- `diff-gaussian-rasterization-32`
- `simple-knn`
- `fused-ssim`

跨機器還原後若 ABI、CUDA 或 GPU 架構不同，可能需要在 `submodules/` 重新安裝。

### 找不到模型或 tracking 資料

先確認 `assets/GUAVA`、`assets/SMPLX`、`assets/FLAME`、`EHM-Tracker/pretrained`、`EHM-Tracker/Sign_dataset` 與翻譯 `best_model` 均已完整還原，且 symbolic link 沒有在備份或還原過程中轉成一般文字檔。

### Git 顯示大量修改或未追蹤檔案

這是交接快照的已知狀態，不代表還原失敗。先比對備份中保存的 Git status 與 checksum，不要直接 reset 或 clean。

## 12. 資安、隱私與授權

- 手語資料及 outputs 可能包含可識別人物，備份目錄與共用資料夾應採最小權限原則。
- 不要把備份系統帳密、SSH private key、API token 寫入本文件或 Git。
- `.gradio/certificate.pem` 是專案內現有憑證檔；若另有 private key，應經核准的秘密管理管道交接。
- SMPL-X、FLAME、預訓練模型及第三方資料集須遵循各自授權。
- 單一儲存裝置不能取代多份備份；重要交接資料仍建議保留第二份離線或異地副本。

## 13. 交接完成條件

- [ ] 備份目的地有日期化、未覆蓋舊版的完整 GUAVA 目錄。
- [ ] `GUAVA` 與 `sign_translate_env` 均有 conda-pack、YAML、explicit、pip freeze。
- [ ] 根目錄與 `EHM-Tracker` 兩個 Git 工作區的 remote、commit、branch、status 已保存。
- [ ] 檔案數與各主要目錄容量已比對。
- [ ] checksum dry-run 沒有差異。
- [ ] SHA-256 清單已建立並驗證。
- [ ] 新機器能通過 import 測試。
- [ ] 中文至 gloss 的 `--dry-run` 測試成功。
- [ ] 串接 `--no-render` 測試成功。
- [ ] GPU 完整渲染測試成功。
- [ ] 模型授權、資料權限、聯絡窗口及已知問題已向接手者說明。
