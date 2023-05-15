import mediapipe as mp
import pandas as pd
import numpy as np
import cv2


def blinking():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh


    # A: location of the eye-landamarks in the facemesh collection
    RIGHT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
    LEFT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces = 1,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
        ) as face_mesh:

        blinkcounter = 0
        count = 0


        while True:
            ret,img = cap.read()
            if not ret:
                break
            img = cv2.flip(img,1)
            rgb_frame = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_h,img_w = img.shape[:2]
            # print(img_h,img_w)

            result = face_mesh.process(rgb_frame)

            if result.multi_face_landmarks:
                all_landmarks = np.array([np.multiply([p.x,p.y],[img_w,img_h]).astype(int) for p in result.multi_face_landmarks[0].landmark])

                right_eye = all_landmarks[RIGHT_EYE]

                right_159 = all_landmarks[159]
                right_145 = all_landmarks[145]

                d_right = abs(right_145[1]-right_159[1])
                # cv2.line(img,right_159,right_145,(255,255,0),2)
                # print('distance_right ',d_right)

                left_eye = all_landmarks[LEFT_EYE]

                left_386 = all_landmarks[386]
                left_374 = all_landmarks[374]

                d_left = abs(left_386[1] - left_374[1])
                # cv2.line(img,left_386,left_374,(255,255,0),2)
                # print('distance_left ',d_left)


                if (d_right <= 8 and d_left <= 8) and count == 0:
                    blinkcounter += 1
                    count = 1
                if count != 0:
                    count += 1
                    if count > 10:
                        count=0

                cv2.putText(img,f"Blinks: {blinkcounter}",(50,50),1,2,(255,0,0),2)

                # cv2.polylines(img,[left_eye],True,(0,255,0),1,cv2.LINE_8)
                # cv2.polylines(img,[right_eye],True,(0,255,0),1,cv2.LINE_8)

                # cv2.circle(img,center=d_left//2)

                if blinkcounter > 3:
                    cv2.putText(img,"Real Person",(50,100),1,2,(255,0,0),2)
                else:
                    cv2.putText(img,"Spoofing",(50,100),1,2,(255,0,0),2)
                


            cv2.imshow('img',img)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return f"total blinks: {blinkcounter}"


blinking()

# blinks = blinking()
# print(blinks)
