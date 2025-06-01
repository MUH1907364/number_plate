from flask import Flask, request, render_template
from ultralytics import YOLO
import os
import cv2
import easyocr
import requests

app = Flask(__name__)

# Load YOLO model and OCR
model = YOLO('./models/number_plate.pt')
reader = easyocr.Reader(['en'], gpu=False)

# Directories
UPLOAD_FOLDER = './static/uploads'
RESULT_FOLDER = './static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# API setup
PLATE_RECOGNIZER_API_URL = "https://api.platerecognizer.com/v1/plate-reader/"
PLATE_RECOGNIZER_API_KEY = "8aa098a3f7ff9dcd8205440b25713179db7d0c6f"  # 🔁 Replace with your actual API key

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

    detection_method = request.form.get('detection_method', 'yolo')
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    result_filename = f"result_{filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    plate_text = "No text detected"

    if detection_method == "api":
        # Use Plate Recognizer API
        try:
            with open(filepath, 'rb') as img_file:
                response = requests.post(
                    PLATE_RECOGNIZER_API_URL,
                    files={'upload': img_file},
                    headers={'Authorization': PLATE_RECOGNIZER_API_KEY}
                )

            if response.status_code in [200, 201]:
                result = response.json()
                if result['results']:
                    plate_text = result['results'][0]['plate']
                    box = result['results'][0]['box']
                    x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                    img = cv2.imread(filepath)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imwrite(result_path, img)
                else:
                    plate_text = "No plate detected"
            else:
                plate_text = f"API error: {response.status_code}"
        except Exception as e:
            plate_text = f"API failed: {str(e)}"

    else:
        # Local YOLO detection
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

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imwrite(result_path, img)

    return render_template(
        'index.html',
        plate_text=plate_text,
        uploaded_file=filename,
        result_file=result_filename
    )

if __name__ == '__main__':
    app.run(debug=True)
