from flask import Flask, render_template, Response, jsonify
import cv2
import easyocr
import os
import time

app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# EasyOCR Reader
reader = easyocr.Reader(['en'])

# Path to Haar Cascade for License Plate Detection
harcascade = "model/haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(harcascade)

# Ensure the 'plates' folder exists to save detected plates
if not os.path.exists("plates"):
    os.makedirs("plates")

# Flask Route to Stream Webcam Feed
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect license plates
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        for (x, y, w, h) in plates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_plate', methods=['POST'])
def detect_plate():
    # Capture frame from the webcam
    success, img = cap.read()
    if not success:
        return jsonify({"error": "Failed to capture image"})

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # If plates are detected, save the ROI and run OCR
    for (x, y, w, h) in plates:
        img_roi = img[y:y + h, x:x + w]

        # Save the detected plate image
        plate_filename = f"plates/plate_{int(time.time())}.jpg"  # Using timestamp for unique names
        cv2.imwrite(plate_filename, img_roi)

        # Use EasyOCR to extract text from the license plate
        result = reader.readtext(plate_filename)
        plate_text = " ".join([detection[1] for detection in result])

        return jsonify({"plate_text": plate_text.strip(), "image_url": plate_filename})

    return jsonify({"error": "No plate detected"})

if __name__ == '__main__':
    app.run(debug=True)
