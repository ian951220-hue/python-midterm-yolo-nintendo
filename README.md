# 基礎程式設計(二) 期中專案：復古掌機 AI 辨識器 🎮

**作者**：庭逸 林 (Ting-Yi Lin)

## 專案簡介
本專案為「基礎程式設計(二)」之期中作業 (Python Side Project)。
利用 Python 結合電腦視覺套件 (OpenCV) 與 Roboflow 進階影像分割 (Segmentation) API，實作一個能夠精準辨識並描繪出「Nintendo DS」復古掌機輪廓的 AI 辨識程式。

相較於傳統的方框標註，本專案採用更進階的 Segmentation 模型，能透過解析回傳的 JSON 座標資料，自動以多邊形 (Polygons) 完美貼合並描繪出遊戲機的真實形狀。

## 開發環境與依賴套件
* **程式語言**：Python 3.11.5
* **核心套件**：
  * `opencv-python` (用於影像讀取、視窗顯示與輪廓繪製)
  * `inference-sdk` (用於介接 Roboflow Serverless API)
  * `numpy` (用於處理 JSON 座標陣列轉換)

## 執行方式 (How to Run)

**1. 下載專案**
請將本專案的 `python_midterm.py` 與測試用的圖片 `test.jpg` 下載並放置於同一個資料夾中。

**2. 安裝必要套件**
請開啟終端機 (Terminal) 或命令提示字元，確認環境為 Python 3.11 後，輸入以下指令安裝所需套件：
```bash
pip install opencv-python inference-sdk numpy
