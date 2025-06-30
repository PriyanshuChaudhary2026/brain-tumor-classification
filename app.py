import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize app
app = Flask(__name__)

# Load your trained model
model = load_model("brain_tumor_model.keras")

# Class labels in order
CLASS_LABELS = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Upload directory setup
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            # Always overwrite with fixed name
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded.jpg")
            file.save(filepath)
            image_url = filepath

            # Preprocess image
            img = image.load_img(filepath, target_size=(64, 64))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array)
            class_index = np.argmax(pred)
            prediction = f"{CLASS_LABELS[class_index]} ({pred[0][class_index]*100:.2f}% confidence)"

    return render_template("index.html", prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
