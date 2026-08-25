# Motion envelope threshold sweep

這個實驗只分析下列兩句中的 7 個詞，不修改正式 GUAVA pipeline：

- `弟弟 / 吃 / 番茄 / 不喜歡`
- `今天 / 哪裡 / 上課`

資料來源依照既有 `gloss_pipeline_manifest.json` 所記錄的原始
`sign_net_tracked/<video_id>/optim_tracking_ehm.pkl`。Motion envelope 與正式
`tracking_concatenation_final_5.py` 相同，使用左右手腕每幀位移相加，再以
Gaussian `sigma=2` 平滑。

## 掃描範圍

```text
high = median + high_coefficient * MAD
low  = median + low_coefficient  * MAD

high_coefficient = 0.8, 1.0, 1.2, 1.5, 2.0
low_coefficient  = -0.5, -0.35, 0.0, 0.2, 0.35, 0.5, 0.8
```

`high_coefficient=0.8, low_coefficient=0.8` 因為 `low >= high` 而標記為無效，
其餘共 34 組有效設定。正式設定 `1.2 / 0.35` 會標記為 `is_current_setting`。

## 執行

```bash
/home/paohan/anaconda3/envs/GUAVA/bin/python \
  /home/paohan/GUAVA/experiment/motion_envelope_threshold_sweep/run_threshold_sweep.py
```

若 GUAVA conda 環境路徑不同，也可以使用已安裝 `numpy`、`scipy`、
`matplotlib` 的 Python 執行。

## 輸出

結果預設寫入 `results/`：

- `threshold_sweep_all_words.csv`：每詞、每組係數的完整數據。
- `aggregate_by_threshold.csv`：跨 7 詞彙的平均擷取比例、邊界變化與退化率。
- `aggregate_heatmaps.png`：所有詞的平均結果矩陣。
- `experiment_summary.json`：輸入、參數與主要統計。
- `report.md`：自動產生的文字摘要。
- `per_word/<詞>/threshold_heatmaps.png`：單詞的起點、終點與長度矩陣。
- `per_word/<詞>/selected_range_comparison.png`：代表性 high/low 設定的擷取範圍。

CSV 主要比較兩種範圍：

1. `raw_*`：threshold 直接形成的 motion region。
2. `final_*`：套用 offset 與最短片段限制後的最終範圍。本實驗忽略詞彙原本的句首或句尾位置，七個詞全部依句中詞規則計算起點與終點。

`pipeline_*` 保留為程式處理階段的相容欄位；在本實驗中其數值與 `final_*` 相同。

本實驗沒有人工起訖標註，因此不能計算語意正確率或宣稱某組閾值是絕對最佳。
它回答的是各設定會切在哪裡、保留多少，以及結果對係數是否敏感。
