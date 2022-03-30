import pickle
import cv2
import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


# For webcam input:
def extract_face_landmarks(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    device_index=10,
    duration=120,
    pickle_file="keypoints.pickle",
):

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    # drawing_spec = None
    cap = cv2.VideoCapture('/dev/video2')
    with mp_face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh:
        keypoints = []
        start = time.time()
        while cap.isOpened():
            if time.time() - start > duration:
                break
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            # break

            # Draw the face mesh annotations on the image.
            # image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # add facelandmarks to the keypoints list
                    keypoint = protobuf_to_dict(face_landmarks)["landmark"]
                    keypoints.append(keypoint)
                    print(len(keypoint))

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow("MediaPipe Face Mesh", cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    

    # save the keypoints/features
    with open(pickle_file, "wb") as fp:
        pickle.dump(keypoints, fp)


extract_face_landmarks(device_index=0, pickle_file='experiment_data/not_facing_speaking_2.pickle')
