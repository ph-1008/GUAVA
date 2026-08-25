import json
import os

def filter_scraped_data(input_file='twtsl.json', output_file='twtsl_filtered.json'):
    """
    讀取爬蟲產生的 JSON 檔案，過濾並整理資料。

    規則：
    1. 去除 key 中的 '_A', '_B' 等後綴。
    2. 合併相同 key 的資料時，優先保留有效的 URL。
    3. 如果一個 key 的所有版本都是錯誤訊息，則保留錯誤訊息。
    """
    # 檢查輸入檔案是否存在
    if not os.path.exists(input_file):
        print(f"錯誤：找不到輸入檔案 '{input_file}'。請確保它與此腳本在同一個目錄下。")
        return

    print(f"正在讀取來源檔案: {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except json.JSONDecodeError:
        print(f"錯誤：檔案 '{input_file}' 不是一個有效的 JSON 格式。")
        return
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return

    filtered_data = {}
    print("開始過濾與合併資料...")

    # 遍歷原始資料中的每一個項目
    for key, value in original_data.items():
        # 1. 取得基礎 key (例如 "一小時_A" -> "一小時")
        base_key = key.split('_')[0]

        # 檢查當前的值是否為一個錯誤訊息
        # 假設錯誤訊息都是以 "ERROR" 開頭的字串
        is_current_value_error = isinstance(value, str) and value.startswith("ERROR")

        # 2. 應用合併規則
        if base_key not in filtered_data:
            # 如果是第一次遇到這個 base_key，直接加入
            filtered_data[base_key] = value
        else:
            # 如果 base_key 已經存在，則需要判斷是否要更新
            existing_value = filtered_data[base_key]
            is_existing_value_error = isinstance(existing_value, str) and existing_value.startswith("ERROR")

            # 只有在「現有的是錯誤，而新來的是有效網址」時，才進行更新
            if is_existing_value_error and not is_current_value_error:
                print(f"  - 更新 '{base_key}': 發現有效網址，取代舊的錯誤訊息。")
                filtered_data[base_key] = value
            # 其他情況（現有的是網址，新來的也是網址；或兩者都是錯誤）則不變，保留第一個遇到的版本

    # 寫入新的 JSON 檔案
    print(f"\n資料處理完成，共整理出 {len(filtered_data)} 筆獨立詞彙。")
    print(f"正在將結果寫入新檔案: {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 讓中文字正常顯示，indent=4 美化格式
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"成功！已產生過濾後的檔案 '{output_file}'。")


if __name__ == "__main__":
    filter_scraped_data()