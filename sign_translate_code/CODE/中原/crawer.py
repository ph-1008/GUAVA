import json
from playwright.sync_api import sync_playwright, TimeoutError

# 建立一個空字典來儲存所有抓取到的資料
scraped_data = {}

with sync_playwright() as p:
    # 啟動瀏覽器，headless=False 可以看到過程，設為 True 則在背景執行
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://twtsl.ccu.edu.tw/TSL/index.php")

    # --- 步驟 1: 預先獲取所有筆畫分類的名稱 ---
    print("正在獲取所有筆畫分類...")
    page.click('button#pinOpener')
    print("等待筆畫視窗顯示...")
    page.wait_for_selector('#pinModal.show')
    stroke_links_in_modal = page.query_selector_all("#pinModal a[onclick^='pinSearch']")
    stroke_categories = [link.inner_text() for link in stroke_links_in_modal]
    print(f"成功獲取分類: {stroke_categories}")
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)

    # --- 步驟 2: 遍歷每一個筆畫分類進行抓取 ---
    for category in stroke_categories:
        print(f"\n{'='*10} 開始處理分類: {category} {'='*10}")
        page.click('button#pinOpener')
        page.wait_for_selector('#pinModal.show')
        print(f"點擊分類 '{category}'...")
        page.click(f"#pinModal a:has-text('{category}')")

        # --- 步驟 3: 在分類內循環翻頁並抓取詞彙 ---
        page_num = 1
        while True:
            print(f"\n--- {category} - 第 {page_num} 頁 ---")
            page.wait_for_selector("a[onclick^='querySearch']")
            word_links = page.query_selector_all("a[onclick^='querySearch']")
            word_texts_on_page = [link.inner_text() for link in word_links]
            print(f"本頁共有 {len(word_texts_on_page)} 個詞彙。")

            # 遍歷當前頁面的每一個詞彙
            for word_text in word_texts_on_page:
                if word_text in scraped_data:
                    print(f"  - 詞彙 '{word_text}' 已抓取過，跳過。")
                    continue

                try:
                    # ★★★ 關鍵修改點開始 ★★★

                    # 1. 點擊前，先獲取當前影片的 src
                    #    這可能是空的、上一個影片的、或是預設影片的
                    old_src = ""
                    source_element = page.query_selector("#wordMovie source")
                    if source_element:
                        old_src = source_element.get_attribute("src")

                    # 2. 點擊詞彙連結
                    page.click(f"a[onclick^='querySearch']:has-text('{word_text}')")

                    # 3. 使用 wait_for_function 等待影片的 src 發生變化
                    #    這是最可靠的方法，我們在等待一個明確的 DOM 變化
                    page.wait_for_function(
                        """(old_src_to_check) => {
                            const source = document.querySelector('#wordMovie source');
                            // 必須滿足三個條件：
                            // 1. <source> 元素存在
                            // 2. <source> 元素有 src 屬性
                            // 3. src 屬性的值和舊的不一樣
                            return source && source.src && source.src !== old_src_to_check;
                        }""",
                        arg=old_src,  # 將 Python 中的 old_src 變數傳入 JS 函數
                        timeout=10000 # 設定 10 秒超時
                    )

                    # 4. 等待結束後，DOM 已更新，現在可以安全地抓取新網址
                    new_source_element = page.query_selector("#wordMovie source")
                    video_url = new_source_element.get_attribute("src")
                    
                    # ★★★ 關鍵修改點結束 ★★★

                    scraped_data[word_text] = video_url
                    print(f"  ✔ 已抓取: {word_text} -> {video_url}")

                except TimeoutError:
                    print(f"  ❌ [超時錯誤] 抓取 '{word_text}' 時，影片 src 未在指定時間內更新，跳過。")
                    scraped_data[word_text] = "ERROR: Timeout waiting for src change"
                except Exception as e:
                    print(f"  ❌ [未知錯誤] 抓取 '{word_text}' 時發生錯誤: {e}")
                    scraped_data[word_text] = f"ERROR: {e}"

            # 檢查並點擊下一頁
            next_page_li = page.query_selector("li.page-item:has(a.page-link:has-text('下一頁'))")
            if next_page_li and "disabled" not in next_page_li.get_attribute("class"):
                print("\n  -> 前往下一頁...")
                page.click("a.page-link:has-text('下一頁')")
                page_num += 1
                page.wait_for_load_state('domcontentloaded')
            else:
                print(f"\n分類 '{category}' 已到達最後一頁。")
                break

    print("\n所有分類抓取完成，正在關閉瀏覽器...")
    browser.close()

# --- 步驟 4: 將所有抓取到的資料寫入 JSON 檔案 ---
print(f"總共抓取了 {len(scraped_data)} 筆資料，正在寫入 twtsl.json...")
with open("twtsl.json", 'w', encoding='utf-8') as f:
    json.dump(scraped_data, f, ensure_ascii=False, indent=4)

print("寫入完成！程式執行結束。")