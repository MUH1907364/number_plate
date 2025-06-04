from flask import Flask, request, render_template
from ultralytics import YOLO
import os
import cv2
import easyocr
import requests

app = Flask(__name__)

# Load local YOLO model and OCR reader
model = YOLO('./models/number_plate.pt')
reader = easyocr.Reader(['en'], gpu=False)

# Folder setup
UPLOAD_FOLDER = './static/uploads'
RESULT_FOLDER = './static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Your Plate Recognizer API key
API_KEY = '8aa098a3f7ff9dcd8205440b25713179db7d0c6f'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # === YOLO + EasyOCR ===
    results = model.predict(source=filepath, save=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    img = cv2.imread(filepath)
    yolo_plate_text = "No text detected"
    yolo_confidence = None
    result_filename = f"result_{filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)

    if len(boxes) > 0:
        x1, y1, x2, y2 = map(int, boxes[0])
        cropped = img[y1:y2, x1:x2]

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        ocr_results = reader.readtext(rgb)

        if ocr_results:
            yolo_plate_text = ocr_results[0][1].strip()
            yolo_confidence = round(ocr_results[0][2] * 100, 2)

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, yolo_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imwrite(result_path, img)

    # === API Detection ===
    with open(filepath, 'rb') as img_file:
        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            files={'upload': img_file},
            headers={'Authorization': f'Token {API_KEY}'}
        )

    api_plate_text = None
    api_confidence = None

    if response.status_code == 200:
        data = response.json()
        if data['results']:
            api_plate_text = data['results'][0]['plate']
            api_confidence = round(data['results'][0]['score'] * 100, 2)
    else:
        api_plate_text = "API call failed"

    return render_template('index.html',
                           uploaded_file=filename,
                           result_file=result_filename,
                           yolo_text=yolo_plate_text,
                           yolo_confidence=yolo_confidence,
                           api_text=api_plate_text,
                           api_confidence=api_confidence)

if __name__ == '__main__':
    app.run(debug=True)
