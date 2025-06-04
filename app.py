from flask import Flask, request, render_template
from ultralytics import YOLO
import requests
import os
import cv2
import easyocr

app = Flask(__name__)

# Model & OCR
model = YOLO('./models/number_plate.pt')
reader = easyocr.Reader(['en'], gpu=False)

# Folders
UPLOAD_FOLDER = './static/uploads'
RESULT_FOLDER = './static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# API Config
USE_API = True  # Toggle this to True to use Plate Recognizer API
PLATE_RECOGNIZER_API_KEY = "8aa098a3f7ff9dcd8205440b25713179db7d0c6f"
PLATE_RECOGNIZER_URL = "https://api.platerecognizer.com/v1/plate-reader/"

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

    plate_text = "No text detected"
    confidence = "N/A"
    result_filename = f"result_{filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)

    if USE_API:
        try:
            with open(filepath, 'rb') as f:
                response = requests.post(
                    PLATE_RECOGNIZER_URL,
                    files={'upload': f},
                    headers={'Authorization': f'Token {PLATE_RECOGNIZER_API_KEY}'}
                )

            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    result = data['results'][0]
                    plate_text = result['plate'].upper()
                    confidence = f"{round(result['score'] * 100, 2)}%"

                    box = result['box']
                    img = cv2.imread(filepath)
                    cv2.rectangle(img, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0, 255, 0), 2)
                    cv2.putText(img, plate_text, (box['xmin'], box['ymin'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imwrite(result_path, img)
                else:
                    plate_text = "API Error: No result found"
            else:
                plate_text = f"API Error: {response.status_code}"
        except Exception as e:
            plate_text = f"API Exception: {str(e)}"

    else:
        results = model.predict(source=filepath, save=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        img = cv2.imread(filepath)
        if len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0])
            cropped = img[y1:y2, x1:x2]
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            ocr_results = reader.readtext(rgb)
            if ocr_results:
                plate_text = " ".join([res[1].strip() for res in ocr_results])
                confidence = f"{round(ocr_results[0][2] * 100, 2)}%"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imwrite(result_path, img)

    return render_template('index.html', plate_text=plate_text, confidence=confidence, uploaded_file=filename, result_file=result_filename)

if __name__ == '__main__':
    app.run(debug=True)
