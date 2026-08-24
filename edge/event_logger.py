import json
import csv
import os
from config import CSV_LOG_FILE, JSON_LOG_FILE

class EventLogger:
    def __init__(self):
        self.csv_file = CSV_LOG_FILE
        self.json_file = JSON_LOG_FILE
        self._initialize_files()

    def _initialize_files(self):
        """Creates the log files and writes CSV headers if they don't exist."""
        # Check CSV
        file_exists = os.path.isfile(self.csv_file)
        if not file_exists:
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "event", "severity", "duration", "message"])
        
        # Check JSON
        if not os.path.isfile(self.json_file):
            with open(self.json_file, mode='w') as file:
                # Initialize an empty list
                json.dump([], file)

    def log_event(self, event_data):
        """
        Logs the structured JSON event to both CSV and JSON local stores.
        :param event_data: Dictionary representing the event payload
        """
        try:
            # 1. Write to CSV
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    event_data.get("timestamp"),
                    event_data.get("event"),
                    event_data.get("severity"),
                    event_data.get("duration"),
                    event_data.get("message")
                ])
            
            # 2. Append to JSON
            # Note: For production with huge files, appending to JSON array like this is inefficient. 
            # Better to use JSONLines (.jsonl). For now, we do standard reading/writing.
            with open(self.json_file, mode='r+') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    data = []
                data.append(event_data)
                file.seek(0)
                json.dump(data, file, indent=4)
                
            print(f"[LOGGER] Event successfully written to local storage: {event_data.get('event')}")
        except Exception as e:
            print(f"[LOGGER ERROR] Failed to write logs: {e}")
