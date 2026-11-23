import asyncio
import math
from collections import deque
from pathlib import Path
import threading
import time
import pandas as pd
from bleak import BleakClient, BleakScanner

# Standard Bluetooth GATT UUIDs for Heart Rate Service & Measurement
HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


class WearableBridge:
    def __init__(self, mode: str = "Simulated Replay", csv_path: str = "assets/sample_telemetry.csv"):
        self.mode = mode
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._start_time = time.time()
        self._rr_intervals_ms = deque(maxlen=20)
        self._thread = None

        self.latest_data = {
            "hr_bpm": 72,
            "hrv_rmssd": 48,
            "spo2_pct": 98,
            "accel_g": 1.0,
            "connected": False,
            "device_name": "Simulated Smartwatch",
        }

        if mode == "Simulated Replay":
            resolved_csv = Path(csv_path)
            if not resolved_csv.is_file():
                resolved_csv = Path(__file__).resolve().parent / csv_path
            self.df = pd.read_csv(resolved_csv)
            if self.df.empty:
                raise ValueError(f"Telemetry CSV is empty: {resolved_csv}")
            self.latest_data["connected"] = True
        elif mode == "Live BLE Smartwatch":
            self.latest_data["device_name"] = "Scanning BLE (0x180D)..."
            self._thread = threading.Thread(target=self._run_ble_loop, daemon=True)
            self._thread.start()
        else:
            raise ValueError(f"Unsupported wearable mode: {mode}")

    def get_telemetry(self) -> dict:
        if self.mode == "Simulated Replay":
            elapsed = int(time.time() - self._start_time) % len(self.df)
            row = self.df.iloc[elapsed]
            with self._lock:
                self.latest_data.update({
                    "hr_bpm": int(row["hr_bpm"]),
                    "hrv_rmssd": int(row["hrv_rmssd"]),
                    "spo2_pct": int(row["spo2_pct"]),
                    "accel_g": float(row["accel_g"]),
                })
                return self.latest_data.copy()

        with self._lock:
            return self.latest_data.copy()

    def compute_biometric_risk(self, telemetry: dict) -> float:
        """
        Converts HR spikes, low HRV, SpO2 drops, and G-force impact (>2.5g)
        into a normalized physiological anomaly score (0.0 to 1.0).
        """
        hr = float(telemetry["hr_bpm"])
        hrv = float(telemetry["hrv_rmssd"])
        spo2 = float(telemetry["spo2_pct"])
        accel = float(telemetry["accel_g"])

        if hr >= 95.0:
            hr_score = min(1.0, max(0.0, (hr - 95.0) / 50.0))
        elif hr < 50.0:
            hr_score = min(1.0, max(0.0, (50.0 - hr) / 20.0))
        else:
            hr_score = 0.0

        hrv_score = min(1.0, max(0.0, (35.0 - hrv) / 25.0))
        spo2_score = min(1.0, max(0.0, (95.0 - spo2) / 6.0))
        impact_score = 1.0 if accel > 2.5 else 0.0

        p_bio = 0.35 * hr_score + 0.25 * hrv_score + 0.20 * spo2_score + 0.20 * impact_score
        return round(min(1.0, max(0.0, p_bio)), 3)

    def _run_ble_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ble_worker())
        except Exception as exc:
            with self._lock:
                self.latest_data["connected"] = False
                self.latest_data["device_name"] = f"BLE Error: {type(exc).__name__}"
        finally:
            loop.close()

    async def _ble_worker(self) -> None:
        devices_and_adv = await BleakScanner.discover(timeout=5.0, return_adv=True)
        target = None
        for device, adv in devices_and_adv.values():
            uuids = [u.lower() for u in (adv.service_uuids or [])]
            name = (device.name or adv.local_name or "").lower()
            if HR_SERVICE_UUID in uuids or any(
                k in name for k in ("polar", "garmin", "wahoo", "band", "watch", "hr")
            ):
                target = device
                break

        if not target:
            with self._lock:
                self.latest_data["connected"] = False
                self.latest_data["device_name"] = "No BLE HR Device Found"
            return

        def handle_hr(_, data: bytearray) -> None:
            if len(data) < 2:
                return
            flags = data[0]
            hr_format_16bit = bool(flags & 0x01)
            energy_expended_present = bool(flags & 0x08)
            rr_present = bool(flags & 0x10)

            offset = 1
            if hr_format_16bit:
                if len(data) < 3:
                    return
                hr_bpm = int.from_bytes(data[offset:offset + 2], "little")
                offset += 2
            else:
                hr_bpm = data[offset]
                offset += 1

            if energy_expended_present:
                offset += 2

            hrv_update = None
            if rr_present:
                while offset + 1 < len(data):
                    rr_raw = int.from_bytes(data[offset:offset + 2], "little")
                    self._rr_intervals_ms.append((rr_raw / 1024.0) * 1000.0)
                    offset += 2
                if len(self._rr_intervals_ms) >= 3:
                    diffs = [
                        self._rr_intervals_ms[i] - self._rr_intervals_ms[i - 1]
                        for i in range(1, len(self._rr_intervals_ms))
                    ]
                    rmssd = math.sqrt(sum(d * d for d in diffs) / len(diffs))
                    hrv_update = int(round(rmssd))

            with self._lock:
                self.latest_data["hr_bpm"] = hr_bpm
                if hrv_update is not None:
                    self.latest_data["hrv_rmssd"] = hrv_update
                self.latest_data["connected"] = True

        async with BleakClient(target.address) as client:
            with self._lock:
                self.latest_data["device_name"] = target.name or target.address
                self.latest_data["connected"] = True
            await client.start_notify(HR_CHAR_UUID, handle_hr)
            try:
                while not self._stop_event.is_set() and client.is_connected:
                    await asyncio.sleep(0.5)
            finally:
                if client.is_connected:
                    await client.stop_notify(HR_CHAR_UUID)
                with self._lock:
                    self.latest_data["connected"] = False

    def stop(self) -> None:
        self._stop_event.set()
