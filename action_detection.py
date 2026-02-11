import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            
            left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
            right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
            
            if left_wrist_y < nose_y and right_wrist_y < nose_y:
                action_stage = "HANDS UP!"
                color = (0, 255, 0)
            else:
                action_stage = "Neutral"
                color = (0, 0, 255)

            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            cv2.putText(image, 'ACTION:', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, action_stage, 
                        (10,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()