"""
MQTT-style telemetry simulator for the Industrial Oven Digital Twin.
Publishes synthetic readings to the FastAPI /predict endpoint at 1 Hz.
"""

import random
import time

import requests
import numpy as np
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"


def generate_reading(t: float):
    """Generate one synthetic telemetry reading."""
    return {
        "Zone1_Temp_C": round(np.random.normal(160.0, 1.2), 2),
        "Zone2_Temp_C": round(210.0 - 38.0 * max(0, min(1, (t % 3000 - 1500) / 300)) + np.random.normal(0, 1.5), 2),
        "Zone3_Temp_C": round(np.random.normal(180.0, 1.0), 2),
        "Humidity_Pct": round(np.random.normal(40.0, 1.5), 2),
        "Belt_Speed_m_min": round(np.random.normal(12.0, 0.1), 2),
        "Gas_Flow_m3_h": round(np.random.normal(32.6, 0.4), 2),
        "Dough_Inlet_Moisture_Pct": round(np.random.normal(42.0, 2.0), 2),
        "Product_Load_kg_min": round(np.random.normal(30.0, 1.0), 2),
        "Exhaust_Damper_Pct": round(np.clip(np.random.normal(55.0, 5.0), 20, 90), 2),
        "Heat_Recovery_Eff_Pct": round(np.clip(np.random.normal(70.0, 3.0), 50, 95), 2),
    }


def main():
    print("Starting telemetry simulator...")
    print(f"Posting to {API_URL}")
    print("Press Ctrl+C to stop.\n")

    t = 0
    while True:
        try:
            reading = generate_reading(t)
            resp = requests.post(API_URL, json=reading, timeout=5)
            result = resp.json()
            status = "🚨 ALARM" if result["failure_flag"] else "✅ OK"
            print(
                f"{status} | failure={result['failure_probability']:.2%} | "
                f"RUL={result['rul_burner_hours']:.0f}h | Z2={reading['Zone2_Temp_C']:.1f}°C"
            )
            t += 1
            time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nSimulator stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2.0)


if __name__ == "__main__":
    main()
