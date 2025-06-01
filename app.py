from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2
import os
import pytesseract
import re

app = Flask(__name__)

# Set path to Tesseract if needed (Windows example)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLO model
model = YOLO('./models/number_plate.pt')  # Update this path as needed
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
        return jsonify({'error': 'Empty filename'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    results = model.predict(source=filepath, save=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    img = cv2.imread(filepath)
    plate_text = "No text detected"
    box_coords = None

    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0])
        box_coords = [x1, y1, x2, y2]
        plate_crop = img[y1:y2, x1:x2]

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        plate_text = pytesseract.image_to_string(resized, config=config).strip()
        plate_text = re.sub(r'[^A-Z0-9 ]', '', plate_text)

    return jsonify({
        'plate_text': plate_text,
        'box': box_coords
    })

if __name__ == '__main__':
    app.run(debug=True)
