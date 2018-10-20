import cv2
import os
import shutil
import numpy
import uuid
from flask import Flask, request, Response, jsonify, url_for, send_file, abort
import prediction
from web import app, model_train
import redis


IMG_PATH = os.path.abspath("./temps")
FACES_IMG_ROOT_PATH = os.path.abspath("./faces")

def face_url(student_id, filename):
    temp_link = "/api/predict/faces/temp/{student_id}/{filename}"

@app.route('/api/predict/train', methods=['POST'])
def train_task():
    r = redis.StrictRedis(host='localhost', port=6379, db=0)

    task_id = r.get('current_job')

    task = model_train.AsyncResult(task_id)

    if task.state == 'PENDING' or task.state == 'RUNNING':
        response = {
            'ready' : False,
            'task': task_id.decode('utf-8'),
            'state': task.state

        }
    else:
        result = model_train.apply_async()

        response = {
            'ready' : True,
            'task': result.task_id,
            'state': result.state
        }

        r.set('current_job', result.task_id)

    return jsonify(response)


@app.route('/api/predict/train/reload', methods=['POST'])
def model_reload():
    prediction.load_facenet_model()
    return jsonify({'success': True})


@app.route('/api/predict/train/status', methods=['POST'])
def train_status():
    r = redis.StrictRedis(host='localhost', port=6379, db=0)

    task_id = r.get('current_job')

    task = model_train.AsyncResult(task_id)

    if task.state == 'PENDING' or task.state == 'RUNNING':
        response = {
            'ready' : False,
            'task' : task_id.decode('utf-8'),
            'state': task.state
        }
    else:
        response = {
            'ready' : True,
            'task' : task_id.decode('utf-8'),
            'state': task.state
        }
        prediction.load_facenet_model()

    return jsonify(response)


@app.route("/api/predict/train/<task_id>", methods=['POST'])
def show_result(task_id):
    task = model_train.AsyncResult(task_id)

    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
        }
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@app.route('/api/predict', methods=['POST'])
def predict():
    image_file = request.files["image"]
    file_loc = os.path.join(IMG_PATH, str(uuid.uuid4()))
    image_file.save(file_loc)

    img = cv2.imread(file_loc)
    results = jsonify(prediction.predict_all(img))
    os.remove(file_loc)

    return results

@app.route('/api/predict/faces/register', methods=['POST'])
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

@app.route('/api/predict/faces/temp/<student_id>/<filename>')
def temp_face(student_id, filename):
    image_path = os.path.join(IMG_PATH, "faces", student_id, filename)
    print(image_path)
    print(os.path.isfile(image_path))
    if(os.path.isfile(image_path)):
        return send_file(image_path)
    else:
        abort(404)

@app.route('/api/predict/faces/select', methods=['POST'])
def select_face():
    student_id = request.form["student_id"]
    image_id = request.form["image_id"]

    filename = "{}.png".format(image_id)
    temp_img_path = os.path.join(IMG_PATH, "faces", student_id)
    if(not os.path.isdir(temp_img_path)):
        response = jsonify(success=False)
        response.status_code = 400
        return response

    student_faces_path = os.path.join(FACES_IMG_ROOT_PATH, student_id)
    source_path = os.path.join(temp_img_path, filename)
    dest_path = os.path.join(FACES_IMG_ROOT_PATH, student_id, filename)
    if(os.path.isfile(source_path)):
        if(not os.path.isdir(student_faces_path)):
            os.makedirs(student_faces_path)
        os.rename(source_path, dest_path)
        shutil.rmtree(temp_img_path)
        response = jsonify(success=True)
    else:
        response = jsonify(success=False)
        response.status_code = 400
    return response

@app.route('/api/predict')
def main():
    return "Hello Classnalytic!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
