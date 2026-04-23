import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

def get_contour(image_base64):
    try:
        img_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        return max(contours, key=cv2.contourArea)
    except:
        return None

@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    target_b64 = data.get('target_image')
    
    target_cnt = get_contour(target_b64)
    if target_cnt is None:
        return jsonify({"error": "Không tìm thấy vật thể trong ảnh hoặc ảnh không hợp lệ"}), 200 # Trả về 200 để không báo lỗi đỏ

    # Tạm thời trả về thông báo đã thấy ảnh để test luồng
    return jsonify([{"to_id": "TEST_SUCCESS", "score": 0.0}])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
