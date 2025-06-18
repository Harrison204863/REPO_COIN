import cv2
import numpy as np
import tensorflow as tf #2.19
from PIL import Image
import json
import time
#
MODEL_PATH = r"C:\Users\joe\Downloads\coin\coin\model\coins_model_NTD_0610.tflite"
LABEL_PATH = r"C:\Users\joe\Downloads\coin\coin\model\coins_label_NTD.json"
# MODEL_PATH = r"E:\Study\PyCharm\coin_classification\model\training_results_061123\coins_model_lite_061123.tflite"
# LABEL_PATH = r"E:\Study\PyCharm\coin_classification\model\coins_label_lite.json"

def init_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def preprocess_roi(roi, target_size=(224, 224)):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # 轉換為 RGB
    roi_pil = Image.fromarray(roi)
    roi_pil = roi_pil.resize(target_size, Image.BILINEAR)
    roi = np.array(roi_pil, dtype=np.float32) / 255.0  # 歸一化
    return np.expand_dims(roi, axis=0)  # (1, 224, 224, 3)

def detect_coins(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (3, 3), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=30,
        minRadius=30,
        maxRadius=150
    )

    coin_rois = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for x, y, r in circles:
            # 創建圓形遮罩
            mask = np.zeros_like(frame)
            cv2.circle(mask, (x, y), r, (255, 255, 255), -1)  # 白色圓形遮罩

            # 應用遮罩，提取硬幣區域
            masked_coin = cv2.bitwise_and(frame, mask)

            # 提取 ROI（可選：裁切硬幣區域）
            y1, y2 = max(0, y - r), min(frame.shape[0], y + r)
            x1, x2 = max(0, x - r), min(frame.shape[1], x + r)
            roi = masked_coin[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            coin_rois.append((roi, (x, y, r)))

    return coin_rois

def main():
    interpreter, input_details, output_details = init_model(MODEL_PATH)

    with open(LABEL_PATH, 'r') as f:
        labels = json.load(f)

    cap = cv2.VideoCapture(0)
    print("按下 's' 辨識，按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Coin Detector", frame)
        key = cv2.waitKey(1)
        key = ord('s')
        #print(" 辨識，")

        if key == ord('s'):
            # 保存當前彩色畫面
            original_frame = frame.copy()

            # 偵測硬幣並提取遮罩後的 ROI
            coin_rois = detect_coins(original_frame)
            if not coin_rois:
                print("未偵測到硬幣！")
                continue
            
            for roi, (x, y, r) in coin_rois:
                # 預處理並輸入模型
                
                input_data = preprocess_roi(roi)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # 取得預測結果
                output_data = interpreter.get_tensor(output_details[0]['index'])
                print("output_data:", output_data)
                predicted_class = np.argmax(output_data)
                print("Prediction index:", predicted_class)
                confidence = float(output_data[0][predicted_class]) * 100
                coin_name = labels.get(str(predicted_class), "Unknown")

                # 在原始畫面上標註
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.putText(frame, f"{coin_name} {confidence:.1f}%",
                            (x - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

            # 顯示結果
            cv2.imshow("Detection Result", frame)
            #cv2.imshow("Masked Coins", np.hstack([roi for roi, _ in coin_rois]))  # 可選：顯示所有遮罩後的硬幣
            #cv2.imwrite('Masked_Coin.jpg', np.hstack([roi for roi, _ in coin_rois]))

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()