from flask import Flask, request, render_template, url_for
from ultralytics import YOLO
import os
import cv2
import easyocr

app = Flask(__name__)

# Load model and OCR
model = YOLO('./models/number_plate.pt')
reader = easyocr.Reader(['en'], gpu=False)

# Define folders
UPLOAD_FOLDER = './static/uploads'
RESULT_FOLDER = './static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

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

    # Save uploaded file
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Run YOLOv8 model
    results = model.predict(source=filepath, save=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    img = cv2.imread(filepath)
    plate_text = "No text detected"
    result_filename = f"result_{filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)

    if len(boxes) > 0:
        # Get first detected box
        x1, y1, x2, y2 = map(int, boxes[0])
        cropped = img[y1:y2, x1:x2]

        # Convert to RGB and run OCR
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        ocr_results = reader.readtext(rgb)

        if ocr_results:
            # Combine all lines of text
            plate_text = " ".join([res[1].strip() for res in ocr_results])

        # Draw box and label on image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, plate_text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Save the result image
    cv2.imwrite(result_path, img)

    # Debug info
    print("✔️ OCR Output:", plate_text)
    print("✔️ Saved processed image to:", result_path)

    return render_template(
        'index.html',
        plate_text=plate_text,
        uploaded_file=filename,
        result_file=result_filename
    )

if __name__ == '__main__':
    app.run(debug=True)
