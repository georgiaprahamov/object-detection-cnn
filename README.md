# 🧐 Object Detection Web App using YOLOv5 & Flask

This is a lightweight web application that allows users to upload an image (`.jpg`, `.jpeg`, `.png`, etc.) and uses a pre-trained deep learning model (YOLOv5) to detect and identify objects within the image. Detected objects are visualized with bounding boxes and class labels.

---

## 🚀 Features

- Upload an image via a simple web interface  
- Object detection using a **CNN-based YOLOv5** model  
- Visual feedback: bounding boxes + labels on the image  
- Lightweight, fast, and easy to run locally  

---

## 🧰 Tech Stack

- **Backend:** Python, Flask  
- **Deep Learning Model:** YOLOv5 (pre-trained on COCO dataset)  
- **Image Processing:** OpenCV, Pillow  
- **Frontend:** HTML5, Bootstrap (Flask templates)

---

## ⚙️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/georgiaprahamov/object-detector-app.git
   cd object-detector-app
   ```

2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

5. Open your browser and go to `http://localhost:5000`

---

## 🧠 About the AI Model

This app uses **YOLOv5s**, a fast and efficient convolutional neural network (CNN) trained on the **COCO dataset**. It is capable of detecting 80+ common object classes like:

- Person  
- Car  
- Truck  
- Bicycle  
- Stop sign  
- Dog  
- etc.

---

## 📢 Contact

Developed by **Georgi Aprahamov**  
- 📸 Instagram: [@g.aprahamov](https://www.instagram.com/g.aprahamov/)  
- 🧑‍💻 GitHub: [github.com/georgiaprahamov](https://github.com/georgiaprahamov)

---

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify it!
