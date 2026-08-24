import winsound
import threading
import subprocess
import time

# Global event to control the continuous beep
_stop_event = threading.Event()
_stop_event.set() # Initially stopped

def _play_continuous_beep():
    """Internal loop that plays the beep until the stop event is set."""
    try:
        while not _stop_event.is_set():
            winsound.Beep(2500, 400)  # Sharp, annoying frequency
            time.sleep(0.1)
    except Exception:
        pass

def _speak_message(message):
    """Uses Windows native PowerShell to read out the message."""
    cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{message}')"
    try:
        subprocess.run(["powershell", "-Command", cmd], capture_output=True)
    except Exception:
        pass

def dispatch_alert(event_json):
    """
    Accepts a structured JSON event payload and triggers outputs.
    Strictly Output Actions. DO NOT add decision logic here!
    """
    if not event_json:
        return

    event_type = event_json.get("event", "UNKNOWN")
    severity = event_json.get("severity", "LOW")
    message = event_json.get("message", "Attention required.")
    timestamp = event_json.get("timestamp")

    print(f"\n=========================================================")
    print(f"[{timestamp}] ALERT DISPATCHED: {event_type}")
    print(f"SEVERITY: {severity}")
    print(f"MESSAGE: {message}")
    print(f"=========================================================\n")

    # 1. Start continuous beep for HIGH or CRITICAL severity
    if severity in ["HIGH", "CRITICAL"]:
        if _stop_event.is_set() or not any(t.name == "BeepThread" for t in threading.enumerate()):
            _stop_event.clear()
            beep_thread = threading.Thread(target=_play_continuous_beep, name="BeepThread", daemon=True)
            beep_thread.start()
    
    # 2. Trigger the voice message based on the JSON payload
    threading.Thread(target=_speak_message, args=(message,), name="VoiceThread", daemon=True).start()


def stop_alert():
    """Stops the continuous alert beep."""
    if not _stop_event.is_set():
        _stop_event.set()
        print("[INFO] Alert cleared. Buzzers deactivated.")
