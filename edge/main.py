import cv2
import time
import json

from config import (
    CAMERA_INDEX, RESOLUTION_W, RESOLUTION_H, 
    EAR_THRESHOLD, PITCH_THRESHOLD, YAW_THRESHOLD, PHONE_PADDING
)
from detection import VisionModels
import features
from event_engine import EventEngine
from event_logger import EventLogger
from alert import dispatch_alert, stop_alert

def send_event(event_json):
    """Stub function for future backend API integration."""
    # Example: requests.post("https://api.backend-server.com/event", json=event_json)
    # print(f"[BACKEND STUB] Event pushed to cloud: {event_json.get('event')}")
    pass

def main():
    print("========================================")
    print(" Sentinel-AI Production Edge Node       ")
    print("========================================")

    # 1. Initialize pipelines
    vision = VisionModels()
    engine = EventEngine()
    logger = EventLogger()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_H)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[SYSTEM] Pipeline Ready. Press 'q' to quit.")

    # FPS monitoring vars
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time > prev_time else 0
        prev_time = current_time

        # Optimize resolution as strictly requested
        frame = cv2.resize(frame, (RESOLUTION_W, RESOLUTION_H))

        # --- 2. DETECTION LAYER ---
        face_lms, face_matrix, hand_lms, ear_lms, phone_boxes = vision.process_frame(frame)

        # --- 3. FEATURE EXTRACTION LAYER ---
        ear = features.calculate_ear(face_lms)
        pitch, yaw = features.calculate_head_pose(face_matrix)
        holding_phone = features.check_phone_intersection(hand_lms, phone_boxes, frame.shape[1], frame.shape[0], PHONE_PADDING)

        # Draw phone boxes for debugging locally
        for b in phone_boxes:
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)

        # Build feature boolean vector
        features_dict = {
            "DROWSINESS": False,
            "PHONE_USAGE": False,
            "DISTRACTION": False
        }

        # Drowsiness Logic (Is EAR below threshold?)
        if face_lms and 0.0 < ear < EAR_THRESHOLD:
            features_dict["DROWSINESS"] = True
            
        # Phone Usage Logic (Is wrist/finger intersecting phone box?)
        if holding_phone:
            features_dict["PHONE_USAGE"] = True
            
        # Distraction / Pitch Logic (Looking down/up or left/right for a long time)
        # We need a baseline calibration. For many webcams, looking straight ahead doesn't yield 0,0 pitch/yaw perfectly.
        # But for now, we assume large absolute deviations indicate distraction.
        if abs(pitch) > PITCH_THRESHOLD or abs(yaw) > YAW_THRESHOLD:
            features_dict["DISTRACTION"] = True

        # --- 4. EVENT ENGINE LAYER ---
        # The engine handles prioritization and per-event cooldowns securely.
        event_json = engine.evaluate_features(features_dict)

        # --- 5. LOGGING & ALERT ACTIONS ---
        if event_json:
            # Save to JSON/CSV trails
            logger.log_event(event_json)

            # Trigger Hardware/Sound Outputs
            dispatch_alert(event_json)

            # Stub for backend
            send_event(event_json)
            
        # Handle cleanup: If state went from ALERT back to IDLE, stop persistent alarms
        if engine.state == "IDLE":
            stop_alert()

        # --- UI DISPLAY ---
        color = (0, 255, 0)
        status_text = "STATUS: MONITORING"
        if engine.state == "ALERT":
            color = (0, 0, 255)
            status_text = f"ALERT: {engine.active_alert}"
        elif engine.state == "COOLDOWN":
            color = (0, 165, 255) # Orange
            status_text = "STATUS: COOLDOWN"

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Display raw feature values and boolean triggers for transparency
        cv2.putText(frame, f"EAR: {ear:.2f} (Drowsy: {features_dict['DROWSINESS']})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Pitch: {pitch:.2f} Yaw: {yaw:.2f} (Distracted: {features_dict['DISTRACTION']})", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Phone Detect: {len(phone_boxes)} (Holding: {features_dict['PHONE_USAGE']})", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Sentinel-AI: Production Edge Node", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    vision.close()
    stop_alert()
    print("[SYSTEM] Safe Shutdown Complete.")

if __name__ == "__main__":
    main()
