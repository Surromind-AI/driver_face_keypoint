from mtcnn import MTCNN
import cv2


class BBoxNotFound(Exception):
    pass


def detect_face_mtcnn(image_path):
    detector = MTCNN()
    try:
        img = cv2.imread(image_path)
        box = detector.detect_faces(img)
        bbox = box[0]["box"]
    except IndexError:
        raise BBoxNotFound
    
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
