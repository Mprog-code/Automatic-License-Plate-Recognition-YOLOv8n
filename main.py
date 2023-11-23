from ultralytics import YOLO  # importing YOLOv8 Model
import cv2

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# load models (2 models: to detect & read lisense plate + car detection)
coco_model = YOLO('yolov8n.pt')  # pretrain model on coco dataset (car, truck, motorcycle,....)
license_plate_detector = YOLO('./license_plate_detector.pt')  # DIRECTORY PATH
# "C:\Users\malku\PycharmProjects\CV_Project\license_plate_detector.pt"

# load video
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7]  # car, motorbike, bus, truck

# read frames from video
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        # if frame_nmr > 10:
        #     break
        results[frame_nmr] = {}
        # detect vehicles
        # pass
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            # print(detections) ############
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)  ## Threshold

                # cv2.imshow('original crop', license_plate_crop)
                # cv2.imshow('threshold', license_plate_crop_thresh)
                # cv2.waitKey(0)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results
write_csv(results, './test2.csv')

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print('PyCharm')

