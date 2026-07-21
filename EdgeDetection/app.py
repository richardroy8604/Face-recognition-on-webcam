import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

# Global parameters for edge detection
params = {
    'blur_kernel': 5,
    'canny_min': 50,
    'canny_max': 150,
    'invert': False
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # 1. Get the image from the request
    file = request.files.get('image')
    if not file:
        return jsonify({"status": "error", "message": "No image sent"}), 400

    # 2. Read parameters sent with the request or fallback to global ones
    blur_kernel = int(request.form.get('blur_kernel', params['blur_kernel']))
    canny_min = int(request.form.get('canny_min', params['canny_min']))
    canny_max = int(request.form.get('canny_max', params['canny_max']))
    invert = request.form.get('invert', 'false') == 'true'

    # 3. Convert image bytes to opencv image
    filestr = file.read()
    np_img = np.frombuffer(filestr, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"status": "error", "message": "Invalid image"}), 400

    # 4. Apply image processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if blur_kernel < 1:
        blur_kernel = 1
    if blur_kernel % 2 == 0:
        blur_kernel += 1
        
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blur, canny_min, canny_max)

    if invert:
        edges = cv2.bitwise_not(edges)

    # 5. Encode back to jpeg
    ret, buffer = cv2.imencode('.jpg', edges)
    if not ret:
        return jsonify({"status": "error", "message": "Failed to encode image"}), 500

    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/update_params', methods=['POST'])
def update_params():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400
        
    if 'blur_kernel' in data:
        params['blur_kernel'] = int(data['blur_kernel'])
    if 'canny_min' in data:
        params['canny_min'] = int(data['canny_min'])
    if 'canny_max' in data:
        params['canny_max'] = int(data['canny_max'])
    if 'invert' in data:
        params['invert'] = bool(data['invert'])
        
    return jsonify({"status": "success", "params": params})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
