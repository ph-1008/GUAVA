citation

衛生福利部. “身心障礙者人數按等級及類別分.” Accessed: May 12, 2025. \[Online].

Available:https://www.mohw.gov.tw/dl-69422-eed90854-f90f-4cf9-8b9c-dd90f6b2163b.html



天氣手語 工研院

AI手語-虛擬氣象主播 | 電腦與通訊 https://share.google/OIJWm52XE4JO6mZRf

\-------------------------------------------------



我現在需要寫一個 py檔是

可以接收 run\_translate\_to\_guava.sh 產出的 gloss\_output.txt 中的 詞語拆解結果

在 "/home/paohan/GUAVA/EHM-Tracker/Sign\_dataset/tw\_sign\_dataset/merged\_labels\_tracked.json"中查找 詞語檔案所在位置



接這是 tracking\_concatenation\_final\_4.py 的功能

按照gloss\_output.txt 中的 詞語順序作為輸入

書出到 ../GUAVA/outputs\_2/... 中 (一個合適檔名的資料夾)



接者 這個指令完成最終渲染 cd /home/paohan/GUAVA \&\& PYTHONPATH=/home/paohan/GUAVA:$PYTHONPATH python main/test.py   -d '0' -m assets/GUAVA -s outputs\_2/...  --data\_path /home/paohan/GUAVA/outputs\_2/...   --source\_data\_path /home/paohan/GUAVA/outputs/app/tracked\_source\_image/Gemini\_Generated\_Image\_kzne4skzne4skzne   --skip\_self\_act --render\_cross\_act



\--------------------------------------------

**gloss\_to\_guava\_pipeline.py**



它會做這幾件事：

讀 /home/paohan/GUAVA/sign\_translate\_code/gloss\_output.txt 的 Gloss joined:

用 merged\_labels\_tracked.json 查詞對應的 tracked 資料夾

自動處理 臺北 -> 台北、略過 。

二個小時 會展開成 二/小時

呼叫 tracking\_concatenation\_final\_4.py 輸出到 /home/paohan/GUAVA/outputs\_2/<自動命名資料夾>

預設接著執行 main/test.py 做最終 render



bash /home/paohan/GUAVA/run\_translate\_to\_guava.sh "哈囉，你好"

bash /home/paohan/GUAVA/run\_translate\_to\_guava.sh "小朋友在草地散步"

bash /home/paohan/GUAVA/run\_translate\_to\_guava.sh "我喜歡去兒童樂園玩旋轉木馬"

bash /home/paohan/GUAVA/run\_translate\_to\_guava.sh "我們去草地上一起玩"

bash /home/paohan/GUAVA/run\_translate\_to\_guava.sh "你結婚了嗎"



