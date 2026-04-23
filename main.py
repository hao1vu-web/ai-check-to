import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

def get_contour(image_base64):
    # Giải mã ảnh từ Base64
    img_data = base64.b64decode(image_base64)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Tiền xử lý: Chuyển xám -> Làm mờ -> Tách biên
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # Tìm đường viền
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Lấy đường viền lớn nhất (giả định là cái TO)
    return max(contours, key=cv2.contourArea)

@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    target_b64 = data.get('target_image')
    masters = data.get('masters') # Danh sách các hình Master gửi từ Sheets
    
    target_cnt = get_contour(target_b64)
    if target_cnt is None:
        return jsonify({"error": "Không tìm thấy vật thể trong ảnh"}), 400

    results = []
    for m in masters:
        master_cnt = get_contour(m['image_b64'])
        if master_cnt is not None:
            # Thuật toán so sánh hình dạng (Hu Moments) - Càng thấp càng giống
            score = cv2.matchShapes(target_cnt, master_cnt, 1, 0.0)
            results.append({"to_id": m['to_id'], "score": score})

    # Sắp xếp kết quả: Điểm thấp nhất (giống nhất) lên đầu
    results.sort(key=lambda x: x['score'])
    return jsonify(results[:3]) # Trả về Top 3

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)