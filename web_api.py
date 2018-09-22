import cv2
import os
import numpy
import uuid
from flask import Flask, request, Response, jsonify, url_for, send_file, abort
import prediction

app = Flask(__name__)
IMG_PATH = os.path.abspath("./temps")
FACES_IMG_ROOT_PATH = os.path.abspath("./faces")

def face_url(student_id, filename):
    temp_link = "/api/temp/faces/{student_id}/{filename}"


@app.route('/api/predict', methods=['POST'])
def predict():
    image_file = request.files["image"]
    file_loc = os.path.join(IMG_PATH, uuid.uuid4())
    image_file.save(file_loc)

    img = cv2.imread(file_loc)
    os.remove(file_loc)
    results = jsonify(prediction.predict_all(img))

    return results

@app.route('/api/regis_face', methods=['POST'])
def regis_face():
    student_id = request.form['student_id']
    student_image = request.files['image']
    student_faces_path = os.path.join(IMG_PATH, "faces", student_id)
    if(not os.path.isdir(student_faces_path)):
        os.makedirs(student_faces_path)
    filename = "{}.{}".format(uuid.uuid4(), student_image.filename.split(".")[-1])
    save_loc = os.path.join(IMG_PATH, filename)
    student_image.save(save_loc)

    img = cv2.imread(save_loc)
    face_locations = prediction.face_location(img)

    temp_face_urls = []
    for face_location in face_locations:
        face_location = face_location["face_location"]
        face_img = prediction.crop_and_resize(img, face_location, (160, 160))

        image_uuid = uuid.uuid4()
        filename = "{}.png".format(image_uuid)
        save_path = os.path.join(IMG_PATH, "faces", student_id, filename)
        cv2.imwrite(save_path, face_img)

        temp_face_urls.append({
            "id": image_uuid,
            "url": url_for("temp_face", student_id=student_id, filename=filename)
        })

    return jsonify(temp_face_urls)

@app.route('/api/temp/faces/<student_id>/<filename>')
def temp_face(student_id, filename):
    image_path = os.path.join(IMG_PATH, "faces", student_id, filename)
    print(image_path)
    print(os.path.isfile(image_path))
    if(os.path.isfile(image_path)):
        return send_file(image_path)
    else:
        abort(404)

@app.route('/api/regis_face/select/')

@app.route('/')
def main():
    return "Hello Classnalytic!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
