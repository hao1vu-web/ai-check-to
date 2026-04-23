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
        # Khử nhiễu và bóc biên dạng
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
    masters_data = data.get('masters', []) # Danh sách ảnh mẫu từ Sheets gửi sang
    
    target_cnt = get_contour(target_b64)
    if target_cnt is None:
        return jsonify({"error": "Không bóc tách được biên dạng ảnh chụp"}), 200

    results = []
    for m in masters_data:
        m_cnt = get_contour(m['image_b64'])
        if m_cnt is not None:
            # Thuật toán MatchShapes: 0.0 là giống hệt nhau, càng cao càng khác
            score = cv2.matchShapes(target_cnt, m_cnt, 1, 0.0)
            results.append({"to_id": m['to_id'], "score": score})

    # Sắp xếp theo điểm số (thấp nhất lên đầu)
    results.sort(key=lambda x: x['score'])
    
    # Trả về Top 3 mã giống nhất
    return jsonify(results[:3])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
