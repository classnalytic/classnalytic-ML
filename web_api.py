import cv2
import os
import numpy
from flask import Flask, request, Response
import prediction

app = Flask(__name__)
IMG_PATH = os.path.abspath("./temps")

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files["image"]
    file_loc = os.path.join(IMG_PATH, image_file.filename)
    image_file.save(file_loc)

    img = cv2.imread(file_loc)
    results = prediction.predict_all(img)

    return results
    # results = prediction.predict_all([img])

    # return results

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
