worker:
	celery -A web.celery worker
web:
	uwsgi --socket 0.0.0.0:5000 --protocol=http --http-websockets --gevent 1000 -w wsgi:app
train:
	python3 ./facenet/classifier.py TRAIN ~/classalytic/faces ~/classalytic/models/facenet/20180402-114759/20180402-114759.pb ~/classalytic/models/facenet/20180402-114759/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35
classify:
	 python3 ./facenet/classifier.py CLASSIFY ~/classalytic/faces ~/classalytic/models/facenet/20180402-114759/20180402-114759.pb ~/classalytic/models/facenet/20180402-114759/lfw_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 40 --nrof_train_images_per_class 35
