import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
from ultralytics import YOLO
import json
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# създаване на папките, ако не съществуват
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


model = YOLO('yolov8m.pt')  # YOLOv8 medium model


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # генериране на име за файла
            unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(upload_path)

            # обработване на снимката
            result_filename = process_image(upload_path, unique_filename)

            # Return the result page with the processed image
            return render_template('index.html',
                                   result_image=url_for('static', filename=f'results/{result_filename}'),
                                   has_result=True)

    return render_template('index.html', has_result=False)


def process_image(image_path, filename):
    img = cv2.imread(image_path)

    # увереност на модела - ако е над 50% уверен разпознава обекта
    results = model(img, conf=0.5)

    detections = []

    for result in results:
        boxes = result.boxes.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            # Запази само открития с достатъчна точност
            if conf >= 0.5:
                detections.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': round(conf, 4),
                    'bbox': [x1, y1, x2, y2]
                })

                # Изчертаване на bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name}: {conf:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 255, 0), -1)
                cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


    result_filename = f"result_{filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, img)

    json_filename = result_filename.rsplit('.', 1)[0] + '.json'
    json_path = os.path.join(app.config['RESULT_FOLDER'], json_filename)
    with open(json_path, 'w') as f:
        json.dump({
            'image': filename,
            'detections': detections
        }, f, indent=4)

    os.remove(image_path)

    return result_filename

if __name__ == '__main__':
    app.run(debug=True)