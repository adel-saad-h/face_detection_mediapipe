import cv2
import mediapipe as mp

cap = cv2.VideoCapture("man face.mp4")
mp_face_detect = mp.solutions.face_detection
face_detect = mp_face_detect.FaceDetection(min_detection_confidence=.9)
mp_draw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detect.process(img_rgb)
    # print(results.detections)
    if results.detections:
        for id, detect in enumerate(results.detections):
            # print(detect)
            # mp_draw.draw_detection(img, detect)
            b_box_class = detect.location_data.relative_bounding_box
            # print(b_box_class)
            h, w, c = img.shape
            cx, cy, cw, ch = int(b_box_class.xmin * w), int(b_box_class.ymin * h), int(b_box_class.width * w), int(
                b_box_class.height * h)
            cv2.rectangle(img, (cx, cy, cw, ch), (0, 0, 255), 2)
            sub_img = img[cy:cy + ch, cx:cx + cw]
            new_w, new_h = int(cw / 50), int(ch / 50)
            img_temp = cv2.resize(sub_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR_EXACT)
            img_out = cv2.resize(img_temp, (cw, ch), interpolation=cv2.INTER_LINEAR_EXACT)
            img[cy:cy + ch, cx:cx + cw] = img_out

    cv2.namedWindow("IMAGE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("IMAGE", 640, 480)
    cv2.imshow("IMAGE", img)
    cv2.waitKey(1)
