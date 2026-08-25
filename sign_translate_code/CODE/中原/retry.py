import json
import os
from playwright.sync_api import sync_playwright, TimeoutError

def retry_failed_searches(filename='twtsl_filtered.json'):
    """
    讀取包含錯誤標記的 JSON 檔案，並使用網站的搜尋功能來嘗試修復它們。
    """
    # 步驟 1: 讀取檔案並找出需要重試的項目
    if not os.path.exists(filename):
        print(f"錯誤：找不到目標檔案 '{filename}'。")
        return

    print(f"正在讀取檔案: {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 找出所有值包含 "ERROR" 的 key
    error_keys = [key for key, value in data.items() if isinstance(value, str) and "ERROR" in value]

    if not error_keys:
        print("恭喜！檔案中沒有需要修復的錯誤項目。")
        return

    print(f"發現 {len(error_keys)} 個需要修復的項目: {error_keys}")
    print("即將啟動瀏覽器進行自動修復...")

    # 步驟 2: 啟動 Playwright 進行修復
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False) # 建議設為 False 以便觀察過程
        page = browser.new_page()
        
        # 只需要訪問一次首頁
        page.goto("https://twtsl.ccu.edu.tw/TSL/index.php")
        print("已載入網站首頁。")

        # 遍歷所有需要修復的詞彙
        for search_term in error_keys:
            try:
                print(f"\n--- 正在修復: '{search_term}' ---")

                # 找到搜尋框，清空並輸入詞彙
                page.fill("input#searchBar", search_term)
                
                # 點擊搜尋按鈕
                page.click("button#send")
                print(f"  - 已送出搜尋: '{search_term}'")

                # 等待搜尋結果列表出現 (等待第一個結果連結)
                page.wait_for_selector("a[onclick^='querySearch']", timeout=10000)
                print("  - 搜尋結果已載入。")

                # 點擊搜尋結果中的第一個連結
                first_result_link = page.query_selector("a[onclick^='querySearch']")
                if not first_result_link:
                    print("  ❌ 錯誤：搜尋成功但找不到結果連結。")
                    continue
                
                first_result_link.click()
                print("  - 已點擊第一個搜尋結果。")

                # 使用最可靠的方式等待影片 src 更新
                page.wait_for_function(
                    """() => {
                        const source = document.querySelector('#wordMovie source');
                        // 等待 source 存在，且 src 有值，且看起來是個影片檔
                        return source && source.src && source.src.includes('.mp4');
                    }""",
                    timeout=5000
                )

                # 抓取更新後的影片網址
                video_url = page.query_selector("#wordMovie source").get_attribute("src")
                
                # 在記憶體中更新資料
                data[search_term] = video_url
                print(f"  ✔ 成功！已將 '{search_term}' 更新為: {video_url}")

            except TimeoutError:
                print(f"  ❌ [超時錯誤] 修復 '{search_term}' 失敗。可能是查無結果或頁面載入太久。")
            except Exception as e:
                print(f"  ❌ [未知錯誤] 修復 '{search_term}' 時發生錯誤: {e}")

        # 所有錯誤都處理完畢，關閉瀏覽器
        print("\n所有錯誤項目處理完畢，關閉瀏覽器。")
        browser.close()

    # 步驟 3: 將更新後的完整資料寫回原檔案
    print(f"正在將修復後的資料寫回檔案 '{filename}'...")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print("修復完成！")


if __name__ == "__main__":
    retry_failed_searches()