from flask import Flask
from celery import Celery
from facenet import classifier

app = Flask("classnalytic")

app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)

@celery.task(name='classnalytic.model_train', bind=True)
def model_train(self):
    args = classifier.parse_arguments(['TRAIN', '/home/wiput/classalytic/faces', '/home/wiput/classalytic/models/facenet/20180402-114759/20180402-114759.pb', '/home/wiput/classalytic/models/facenet/20180402-114759/lfw_classifier.pkl', '--batch_size', '1000', '--min_nrof_images_per_class', '40', '--nrof_train_images_per_class', '35'])
    classifier.main(args, self)
    return "Completed!"