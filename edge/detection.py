import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np

from config import FACE_MODEL_PATH, POSE_MODEL_PATH, YOLO_MODEL_PATH, YOLO_FRAME_SKIP

class VisionModels:
    """
    STRICT RESPONSIBILITY: Model Initialization and Inference only.
    No decisions. No alerts. Only outputs raw landmarks and bounding boxes.
    """
    def __init__(self):
        print("[VISION] Initializing YOLOv8 Object Detection...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.frame_counter = 0
        
        # Cache the YOLO results so we don't return None on skipped frames
        self.last_yolo_boxes = []

        print("[VISION] Initializing MediaPipe Face Landmarker...")
        base_options_face = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        options_face = vision.FaceLandmarkerOptions(
            base_options=base_options_face,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.face_detector = vision.FaceLandmarker.create_from_options(options_face)

        print("[VISION] Initializing MediaPipe Pose Landmarker...")
        base_options_pose = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
        options_pose = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            num_poses=1)
        self.pose_detector = vision.PoseLandmarker.create_from_options(options_pose)
        
        print("[VISION] Models loaded successfully.")

    def process_frame(self, frame):
        """
        Runs inference on the provided frame.
        :param frame: BGR Image from OpenCV
        :return: (face_landmarks, facial_matrix, wrist_landmarks, ear_landmarks, phone_boxes)
        """
        self.frame_counter += 1

        # 1. YOLOv8 (with Frame Skipping)
        # Class 67 is 'cell phone' in COCO dataset
        if self.frame_counter % YOLO_FRAME_SKIP == 0 or self.frame_counter == 1:
            yolo_results = self.yolo_model(frame, classes=[67], verbose=False)[0]
            phones = []
            for box in yolo_results.boxes:
                b = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                phones.append(b)
            self.last_yolo_boxes = phones

        # Prepare for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 2. Face Landmarker
        face_result = self.face_detector.detect(mp_image)
        face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else None
        facial_matrix = face_result.facial_transformation_matrixes[0] if face_result.facial_transformation_matrixes else None

        # 3. Pose Landmarker
        pose_result = self.pose_detector.detect(mp_image)
        hand_landmarks = []
        ear_landmarks = []
        
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            pose_lms = pose_result.pose_landmarks[0]
            # Some MediaPipe versions wrap the landmark list in an object with a .landmark property
            lms_list = pose_lms.landmark if hasattr(pose_lms, "landmark") else pose_lms
            
            if len(lms_list) >= 23: # Make sure we have enough landmarks
                hand_landmarks = [
                    lms_list[15], lms_list[16], # Wrists
                    lms_list[17], lms_list[18], # Pinkies
                    lms_list[19], lms_list[20], # Indexes
                    lms_list[21], lms_list[22]  # Thumbs
                ]
                ear_landmarks = [lms_list[7], lms_list[8]]

        return face_landmarks, facial_matrix, hand_landmarks, ear_landmarks, self.last_yolo_boxes

    def close(self):
        """Cleanup models"""
        self.face_detector.close()
        self.pose_detector.close()
