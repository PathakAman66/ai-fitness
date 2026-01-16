import os
import json
import time
from datetime import datetime

class WorkoutReportManager:
    @staticmethod
    def save(exercise_analyzer):
        exercise = exercise_analyzer.exercise_type

        report = {
            "exercise": exercise,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if exercise in ["bicep_curl", "pushups", "lunges"]:
            report["total_reps"] = exercise_analyzer.rep_count

        elif exercise == "plank":
            report["duration_seconds"] = exercise_analyzer.get_duration()

        os.makedirs("reports", exist_ok=True)
        filename = f"reports/workout_{exercise}_{int(time.time())}.json"

        with open(filename, "w") as file:
            json.dump(report, file, indent=4)

        print(f"📄 Workout report saved → {filename}")
