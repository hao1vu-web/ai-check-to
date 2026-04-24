import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

# Biến toàn cục để lưu trữ "ký ức" biên dạng của 1000 mã TO
trained_masters = []

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

# ĐIỂM MỚI: Endpoint để huấn luyện AI
@app.route('/train', methods=['POST'])
def train():
    global trained_masters
    data = request.json
    masters = data.get('masters', [])
    
    new_knowledge = []
    for m in masters:
        cnt = get_contour(m['image_b64'])
        if cnt is not None:
            new_knowledge.append({"to_id": m['to_id'], "contour": cnt.tolist()}) # Chuyển contour sang list để lưu
    
    trained_masters = new_knowledge
    return jsonify({"status": "success", "message": f"Đã học thuộc {len(trained_masters)} mã TO"})

@app.route('/compare', methods=['POST'])
def compare():
    global trained_masters
    data = request.json
    target_b64 = data.get('target_image')
    
    target_cnt = get_contour(target_b64)
    if target_cnt is None:
        return jsonify({"error": "Không bóc tách được biên dạng ảnh chụp"}), 200

    if not trained_masters:
        return jsonify({"error": "AI chưa được huấn luyện. Hãy chạy lệnh Train trước"}), 200

    results = []
    for m in trained_masters:
        # Chuyển list ngược lại thành numpy array để so sánh
        m_cnt = np.array(m['contour'], dtype=np.int32)
        score = cv2.matchShapes(target_cnt, m_cnt, 1, 0.0)
        results.append({"to_id": m['to_id'], "score": score})

    results.sort(key=lambda x: x['score'])
    return jsonify(results[:3])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
