import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

# Global parameters for face detection
params = {
    'scale_factor': 1.1,
    'min_neighbors': 5,
    'color': 'green',  # green, cyan, pink, amber
    'thickness': 2
}

# Load the Haar Cascade face classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Colors dictionary (BGR format for OpenCV)
COLORS = {
    'green': (0, 255, 0),
    'cyan': (255, 255, 0),    # OpenCV BGR: Blue + Green = Cyan
    'pink': (203, 192, 255),  # light pink / hot pink BGR
    'amber': (0, 191, 255)    # Orange/Amber BGR
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # 1. Get image from request
    file = request.files.get('image')
    if not file:
        return jsonify({"status": "error", "message": "No image sent"}), 400

    # 2. Get parameters from POST request or fallback
    sf = float(request.form.get('scale_factor', params['scale_factor']))
    mn = int(request.form.get('min_neighbors', params['min_neighbors']))
    col_name = request.form.get('color', params['color'])
    thick = int(request.form.get('thickness', params['thickness']))

    # 3. Decode frame
    filestr = file.read()
    np_img = np.frombuffer(filestr, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"status": "error", "message": "Invalid image"}), 400

    # 4. Perform face detection
    color = COLORS.get(col_name, (0, 255, 0))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=sf,
        minNeighbors=mn,
        minSize=(30, 30)
    )
    
    # Draw boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thick)

    # 5. Encode back to jpeg
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return jsonify({"status": "error", "message": "Failed to encode image"}), 500

    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/update_params', methods=['POST'])
def update_params():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400
        
    if 'scale_factor' in data:
        params['scale_factor'] = float(data['scale_factor'])
    if 'min_neighbors' in data:
        params['min_neighbors'] = int(data['min_neighbors'])
    if 'color' in data:
        params['color'] = str(data['color'])
    if 'thickness' in data:
        params['thickness'] = int(data['thickness'])
        
    return jsonify({"status": "success", "params": params})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
