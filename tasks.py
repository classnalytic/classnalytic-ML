from web import celery
import facenet.train_softmax

@celery.task(name='classnalytic.model_train')
def model_train():

    return "Oooo"