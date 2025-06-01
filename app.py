from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
import cv2
import easyocr

app = Flask(__name__)

model = YOLO('./models/number_plate.pt')
reader = easyocr.Reader(['en'], gpu=False)
UPLOAD_FOLDER = './static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    results = model.predict(source=filepath, save=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    img = cv2.imread(filepath)
    plate_text = "No text detected"
    first_box = None

    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0])
        first_box = [x1, y1, x2, y2]
        cropped = img[y1:y2, x1:x2]

        # EasyOCR works with RGB
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb)
        if results:
            plate_text = results[0][1].strip()

    return jsonify({
        'plate_text': plate_text,
        'box': first_box
    })

if __name__ == '__main__':
    app.run(debug=True)
