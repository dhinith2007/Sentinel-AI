import math

def calculate_ear(landmarks):
    """
    Calculates Eye Aspect Ratio (EAR) for both eyes.
    :param landmarks: Normalized face landmarks from MediaPipe
    :return: Float average EAR
    """
    if not landmarks:
        return 0.0

    # MediaPipe Face Mesh Indices for Eyes
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def eye_aspect_ratio(eye_indices, lms):
        p = [lms[i] for i in eye_indices]
        # Vertical distances
        v1 = math.dist((p[1].x, p[1].y), (p[5].x, p[5].y))
        v2 = math.dist((p[2].x, p[2].y), (p[4].x, p[4].y))
        # Horizontal distance
        h = math.dist((p[0].x, p[0].y), (p[3].x, p[3].y))
        
        # Avoid division by zero
        if h == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
    right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
    return (left_ear + right_ear) / 2.0


def calculate_head_pose(transformation_matrix):
    """
    Extracts the head pitch (up/down) and yaw (left/right) from the MediaPipe 4x4 facial transformation matrix.
    :return: (pitch_rad, yaw_rad)
    """
    if transformation_matrix is None:
        return 0.0, 0.0
    
    try:
        # R02 is sin(yaw)
        # R12 is -sin(pitch)*cos(yaw)
        # R22 is cos(pitch)*cos(yaw)
        yaw = math.asin(max(-1.0, min(1.0, transformation_matrix[0, 2])))
        pitch = math.atan2(-transformation_matrix[1, 2], transformation_matrix[2, 2])
        return pitch, yaw
    except Exception:
        return 0.0, 0.0


def check_phone_intersection(hand_landmarks, phone_boxes, frame_width, frame_height, padding=30):
    """
    Checks if any hand landmark (wrist/fingers) intersects with any phone bounding box.
    """
    if not hand_landmarks or not phone_boxes:
        return False

    for phone_box in phone_boxes:
        x1, y1, x2, y2 = phone_box
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding

        for pt in hand_landmarks:
            px = int(pt.x * frame_width)
            py = int(pt.y * frame_height)
            
            if (x1 <= px <= x2) and (y1 <= py <= y2):
                return True
                
    return False


def check_wrist_near_ear(wrist_landmarks, ear_landmarks, distance_threshold):
    """
    Checks if the distance between a wrist and an ear is below a threshold (calling).
    Note: Distances are normalized [0.0, 1.0] if using raw MP outputs.
    """
    if not wrist_landmarks or not ear_landmarks:
        return False

    for wrist in wrist_landmarks:
        for ear_node in ear_landmarks:
            d = math.dist((wrist.x, wrist.y), (ear_node.x, ear_node.y))
            if d < distance_threshold:
                return True
                
    return False
