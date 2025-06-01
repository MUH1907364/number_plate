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

# Folders
UPLOAD_FOLDER = './static/uploads'
RESULT_FOLDER = './static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# External API info (replace with your actual API details)
API_URL = 'https://app.platerecognizer.com/service/snapshot-cloud/'
API_KEY = '8aa098a3f7ff9dcd8205440b25713179db7d0c6f'  # Replace with your key


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['file']
    mode = request.form.get('mode', 'local')  # "local" or "api"

    if file.filename == '':
        return render_template('index.html', error="No file selected")

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    plate_text = "No text detected"
    result_filename = f"result_{filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)

    # -----------------------------
    # Mode 1: LOCAL YOLO + EasyOCR
    # -----------------------------
    if mode == 'local':
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

            # Draw box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imwrite(result_path, img)

    # --------------------------
    # Mode 2: EXTERNAL API CALL
    # --------------------------
    elif mode == 'api':
        with open(filepath, 'rb') as img_file:
            response = requests.post(
                API_URL,
                headers={'Authorization': f'Bearer {API_KEY}'},
                files={'image': img_file}
            )

        if response.status_code == 200:
            data = response.json()
            plate_text = data.get('plate', 'No text detected')
            # Optionally save API image result if provided
        else:
            plate_text = f"API error: {response.text}"

    else:
        return render_template('index.html', error="Invalid detection mode")

    return render_template(
        'index.html',
        plate_text=plate_text,
        uploaded_file=filename,
        result_file=result_filename
    )


if __name__ == '__main__':
    app.run(debug=True)
