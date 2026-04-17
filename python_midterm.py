import cv2
import numpy as np  # 這次我們需要 numpy 來處理數學座標陣列
from inference_sdk import InferenceHTTPClient

# 1. 載入圖片
image_path = 'test.jpg'
img = cv2.imread(image_path)

if img is None:
    print(f"找不到圖片 '{image_path}'！")
else:
    print("圖片讀取成功，正在連線至 Roboflow 執行進階工作流...")
    
    # 2. 連線到 Roboflow
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="q6WxrQB2MjRTTz3gH5gY"
    )

    try:
        # 3. 取得 AI 辨識結果
        result = client.run_workflow(
            workspace_name="ian-yakcn",
            workflow_id="general-segmentation-api",
            images={"image": image_path},
            parameters={"classes": "nintendo, toy"},
            use_cache=True
        )
        
        print("🎉 成功取得 AI 辨識座標！正在將輪廓繪製到圖片上...")

        # 4. 畫圖小幫手 (尋找 JSON 裡的座標並畫出多邊形)
        def draw_predictions(image, predictions_data):
            if isinstance(predictions_data, dict):
                # 如果找到目標名稱 (class)
                if 'class' in predictions_data:
                    class_name = predictions_data['class']
                    # 尋找裡面的 x, y 座標陣列
                    for key, value in predictions_data.items():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict) and 'x' in value[0]:
                            # 成功挖出座標！把它們變成 OpenCV 看得懂的陣列
                            points = np.array([[int(p['x']), int(p['y'])] for p in value], np.int32)
                            points = points.reshape((-1, 1, 2))
                            
                            # 用帥氣的綠色畫出精準的多邊形輪廓 (厚度 3)
                            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=3)
                            
                            # 在輪廓的左上角寫上 AI 辨識出的名字 (例如: nintendo)
                            x_min, y_min = np.min(points[:, 0, 0]), np.min(points[:, 0, 1])
                            cv2.putText(image, class_name, (x_min, y_min - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            return 
                # 繼續往下挖
                for key, value in predictions_data.items():
                    draw_predictions(image, value)
            elif isinstance(predictions_data, list):
                for item in predictions_data:
                    draw_predictions(image, item)

        # 5. 呼叫小幫手幫我們的圖片畫上輪廓
        draw_predictions(img, result)
        
        # 6. 顯示最終成果！
        cv2.namedWindow("My Python Midterm Project", cv2.WINDOW_NORMAL)
        cv2.imshow("My Python Midterm Project", img)
        print("\n✅ 繪製完成！請在跳出的圖片視窗上「按一下鍵盤任意鍵」即可關閉程式。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")