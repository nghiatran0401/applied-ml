"""
Attendance Logging System
Stores and manages attendance records
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AttendanceLogger:
    """Manages attendance records"""
    
    def __init__(self, log_file: str = "attendance_logs.json"):
        self.log_file = log_file
        self.logs = []
        self.load_logs()
    
    def load_logs(self):
        """Load attendance logs from file"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.logs = json.load(f)
            except Exception as e:
                logger.error(f"Error loading logs: {e}")
                self.logs = []
        else:
            self.logs = []
    
    def save_logs(self):
        """Save attendance logs to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving logs: {e}")
    
    def log_attendance(
        self,
        person_id: int,
        person_name: str,
        attendance_type: str = "check_in",
        confidence: float = 0.0,
        emotion: Optional[str] = None,
        liveness_score: float = 0.0,
        location: str = "main_entrance"
    ) -> Dict:
        """
        Log an attendance record
        
        Args:
            person_id: Employee ID
            person_name: Employee name
            attendance_type: "check_in" or "check_out"
            confidence: Recognition confidence score
            emotion: Detected emotion
            liveness_score: Anti-spoofing confidence
            location: Entry point location
            
        Returns:
            Dictionary with attendance record
        """
        record = {
            "id": len(self.logs) + 1,
            "person_id": person_id,
            "person_name": person_name,
            "attendance_type": attendance_type,
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "confidence": float(confidence),
            "emotion": emotion,
            "liveness_score": float(liveness_score),
            "location": location
        }
        
        self.logs.append(record)
        self.save_logs()
        
        return record
    
    def get_recent_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent attendance logs"""
        return self.logs[-limit:][::-1]  # Return most recent first
    
    def get_logs_by_date(self, date: str) -> List[Dict]:
        """Get logs for a specific date (YYYY-MM-DD)"""
        return [log for log in self.logs if log.get("date") == date]
    
    def get_logs_by_person(self, person_id: int, limit: int = 100) -> List[Dict]:
        """Get logs for a specific person"""
        person_logs = [log for log in self.logs if log.get("person_id") == person_id]
        return person_logs[-limit:][::-1]
    
    def get_today_logs(self) -> List[Dict]:
        """Get today's attendance logs"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.get_logs_by_date(today)
    
    def get_person_today_status(self, person_id: int) -> Optional[Dict]:
        """Get today's attendance status for a person"""
        today_logs = self.get_today_logs()
        person_today = [log for log in today_logs if log.get("person_id") == person_id]
        
        if not person_today:
            return None
        
        # Get latest check-in and check-out
        check_ins = [log for log in person_today if log.get("attendance_type") == "check_in"]
        check_outs = [log for log in person_today if log.get("attendance_type") == "check_out"]
        
        latest_check_in = max(check_ins, key=lambda x: x["timestamp"]) if check_ins else None
        latest_check_out = max(check_outs, key=lambda x: x["timestamp"]) if check_outs else None
        
        return {
            "person_id": person_id,
            "latest_check_in": latest_check_in,
            "latest_check_out": latest_check_out,
            "is_checked_in": latest_check_in is not None and (
                latest_check_out is None or 
                latest_check_in["timestamp"] > latest_check_out["timestamp"]
            )
        }
    
    def should_log_check_in(self, person_id: int, cooldown_minutes: int = 5) -> bool:
        """
        Check if we should log a check-in (prevent duplicates)
        
        Args:
            person_id: Employee ID
            cooldown_minutes: Minutes to wait before allowing another check-in
            
        Returns:
            True if check-in should be logged, False otherwise
        """
        today_logs = self.get_today_logs()
        person_today = [log for log in today_logs if log.get("person_id") == person_id]
        
        if not person_today:
            return True
        
        # Get most recent check-in
        check_ins = [log for log in person_today if log.get("attendance_type") == "check_in"]
        if not check_ins:
            return True
        
        latest_check_in = max(check_ins, key=lambda x: x["timestamp"])
        latest_time = datetime.fromisoformat(latest_check_in["timestamp"])
        time_diff = (datetime.now() - latest_time).total_seconds() / 60
        
        return time_diff >= cooldown_minutes
    
    def delete_all_logs(self) -> bool:
        """
        Delete all attendance logs
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logs = []
            self.save_logs()
            return True
        except Exception as e:
            logger.error(f"Error deleting all logs: {e}")
            return False

