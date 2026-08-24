import time
from config import EVENT_PRIORITIES, TIME_THRESHOLDS, COOLDOWNS

class EventEngine:
    def __init__(self):
        # Track when a condition started to become true { "EVENT": timestamp }
        self.condition_starts = {}
        
        # Track when the last event was fired { "EVENT": timestamp }
        self.last_fired = {}
        
        # Current state: IDLE, DETECTED, ALERT, COOLDOWN
        self.state = "IDLE"
        self.active_alert = None

    def evaluate_features(self, features_dict):
        """
        Evaluates boolean features over time and returns the highest priority event to alert.
        :param features_dict: Dictionary of booleans e.g. {"DROWSINESS": True, "PHONE_USAGE": False}
        :return: JSON Event object or None
        """
        detected_events = []
        current_time = time.time()

        # 1. Update condition trackers
        for event, is_active in features_dict.items():
            if is_active:
                # If it wasn't tracked, start tracking
                if event not in self.condition_starts:
                    self.condition_starts[event] = current_time
                    
                # Check if it has crossed the time threshold to become a real event
                duration = current_time - self.condition_starts[event]
                required_duration = TIME_THRESHOLDS.get(event, 0.0) # default 0s if not specified
                
                if duration >= required_duration:
                    detected_events.append((event, duration))
            else:
                # Condition stopped, remove from tracking
                if event in self.condition_starts:
                    del self.condition_starts[event]

        # 2. If nothing is actively breaching thresholds, handle state
        if not detected_events:
            if self.state in ["ALERT", "COOLDOWN"]:
                # We do not immediately go IDLE if in COOLDOWN, wait for cooldown to expire
                # However, for simplicity here, we clear active_alert if no conditions exist.
                # The cooldown check will happen when a NEW event tries to fire.
                pass
            self.state = "IDLE"
            self.active_alert = None
            return None

        # 3. Filter out events that are currently on Cooldown
        valid_events = []
        for event, duration in detected_events:
            last_time = self.last_fired.get(event, 0)
            cooldown_period = COOLDOWNS.get(event, 5) # Default 5 if missing
            
            if (current_time - last_time) >= cooldown_period:
                valid_events.append((event, duration))

        if not valid_events:
            self.state = "COOLDOWN"
            return None

        # 4. Priority Sorting
        # Sort by priority value (Lower is better).
        # We use a default high number if not in dict so it falls to bottom.
        valid_events.sort(key=lambda x: EVENT_PRIORITIES.get(x[0], 99))

        # 5. Select Highest Priority Event
        best_event, best_duration = valid_events[0]

        # 6. Fire the event
        self.last_fired[best_event] = current_time
        self.state = "ALERT"
        self.active_alert = best_event

        return self._build_event_json(best_event, best_duration)


    def _build_event_json(self, event_name, duration):
        """Constructs the standard backend-ready JSON payload"""
        
        # Simple severity mapping based on our priority queue
        severity_map = {
            "MEDICAL_EMERGENCY": "CRITICAL",
            "DROWSINESS": "HIGH",
            "PHONE_USAGE": "MEDIUM",
            "DISTRACTION": "MEDIUM",
            "TEXTING": "HIGH"
        }

        message_map = {
            "MEDICAL_EMERGENCY": "Driver unreachable! Possible emergency.",
            "DROWSINESS": "Driver is drowsy. Please take rest.",
            "PHONE_USAGE": "Avoid using phone while driving.",
            "DISTRACTION": "Focus on the road ahead.",
            "TEXTING": "Texting while driving is dangerous. Keep your eyes up."
        }

        # Format ISO timestamp
        timestamp = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())

        return {
            "event": event_name,
            "severity": severity_map.get(event_name, "LOW"),
            "duration": round(duration, 2),
            "timestamp": timestamp,
            "message": message_map.get(event_name, "Alert triggered.")
        }
