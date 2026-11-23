import math
import os
from pathlib import Path
import tempfile
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit as st

from geometry import FallGeometryEngine, POSE_CONNECTIONS
from wearable_bridge import WearableBridge

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pose_landmarker_full.task"

st.set_page_config(page_title="EdgeGuard Multimodal AI", layout="wide")
st.title("EdgeGuard: 100% Offline Multimodal Fall & Acute Health Risk MVP")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def fuse_risk_and_fsm(p_vision: float, immobile_sec: float, p_bio: float) -> tuple[float, str, str]:
    """
    Combines Vision, Immobility, and Biometric anomaly probabilities into P_alert
    via a sigmoid function and assigns the Finite State Machine (FSM) state and Alert Tier.
    """
    w_immobile = min(1.0, max(0.0, immobile_sec / 6.0))
    # Weighted logistic fusion centered around alert threshold
    logit = (3.2 * p_vision) + (2.5 * w_immobile) + (3.0 * p_bio) - 3.6
    p_alert = round(sigmoid(logit), 3)

    if p_vision < 0.55 and p_alert < 0.40:
        return p_alert, "MONITORING", "Normal"
    elif p_alert >= 0.85:
        return p_alert, "ALARMING_HEALTH_ALERT", "Tier 3: Critical Medical Risk"
    elif p_alert >= 0.70 or immobile_sec >= 4.0:
        return p_alert, "POST_FALL_ASSESSMENT", "Tier 2: Suspected Injury / Immobility"
    else:
        return p_alert, "IMPACT_DETECTED", "Tier 1: Mechanical Fall (Vitals Stable)"


def _banner_color_bgr(fsm_state: str) -> tuple[int, int, int]:
    if fsm_state == "ALARMING_HEALTH_ALERT":
        return (0, 0, 220)      # Deep Red
    if fsm_state == "POST_FALL_ASSESSMENT":
        return (0, 100, 230)    # Dark Orange
    if fsm_state == "IMPACT_DETECTED":
        return (0, 165, 230)    # Amber
    return (30, 140, 30)        # Green


# --- Sidebar Controls ---
st.sidebar.header("Pipeline Configuration")
wearable_mode = st.sidebar.selectbox(
    "Wearable Telemetry Source",
    ["Simulated Replay", "Live BLE Smartwatch"],
)
video_source = st.sidebar.radio(
    "Video Input",
    ["Live MacBook Webcam", "Upload Test Video (.mp4)"],
)
uploaded_video = None
if video_source == "Upload Test Video (.mp4)":
    uploaded_video = st.sidebar.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

run_pipeline = st.sidebar.toggle("Start EdgeGuard Engine", value=False)

# --- Dashboard Layout ---
col_video, col_metrics = st.columns([1.3, 1.0])
frame_placeholder = col_video.empty()
metrics_placeholder = col_metrics.empty()

if run_pipeline:
    if not MODEL_PATH.is_file():
        st.error(f"MediaPipe model weights not found at `{MODEL_PATH}`.")
        st.stop()

    if video_source == "Upload Test Video (.mp4)" and uploaded_video is None:
        st.warning("Please upload a test video file in the sidebar to start the pipeline.")
        st.stop()

    # Initialize MediaPipe Pose Landmarker in VIDEO mode
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    geo_engine = FallGeometryEngine(fps=25.0)
    wearable = WearableBridge(mode=wearable_mode)

    temp_video_path = None
    if video_source == "Upload Test Video (.mp4)":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            temp_video_path = tmp.name
        cap = cv2.VideoCapture(temp_video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        detector.close()
        wearable.stop()
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        st.error("Unable to open video source. Check macOS Camera permissions or video file.")
        st.stop()

    timestamp_ms = 0
    try:
        while cap.isOpened() and run_pipeline:
            ret, frame = cap.read()
            if not ret:
                if video_source == "Upload Test Video (.mp4)":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop demo video
                    continue
                break

            if video_source == "Live MacBook Webcam":
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms += 40
            result = detector.detect_for_video(mp_image, timestamp_ms)

            # Default values if no person is in frame
            geo_metrics = {
                "l_torso": 0.0,
                "torso_angle": 0.0,
                "drop_velocity": 0.0,
                "is_horizontal": False,
                "immobile_seconds": 0.0,
                "p_vision": 0.0,
            }
            h, w, _ = frame.shape

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                geo_metrics = geo_engine.extract_metrics(landmarks)
                skel_color = (0, 0, 255) if geo_metrics["is_horizontal"] else (0, 255, 0)

                # Draw skeleton bones and key joints
                for start_idx, end_idx in POSE_CONNECTIONS:
                    p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                    p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                    cv2.line(frame, p1, p2, skel_color, 2)
                for idx in [0, 11, 12, 23, 24, 25, 26, 27, 28]:
                    cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                    cv2.circle(frame, (cx, cy), 5, skel_color, -1)

            # Poll wearable vitals & run multimodal fusion
            telemetry = wearable.get_telemetry()
            p_bio = wearable.compute_biometric_risk(telemetry)
            p_alert, fsm_state, tier_label = fuse_risk_and_fsm(
                geo_metrics["p_vision"], geo_metrics["immobile_seconds"], p_bio
            )

            # Overlay top status banner on video feed
            banner_color = _banner_color_bgr(fsm_state)
            cv2.rectangle(frame, (10, 10), (w - 10, 55), banner_color, -1)
            cv2.putText(
                frame,
                f"STATE: {fsm_state} | P_alert: {p_alert:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                width="stretch",
            )

            # Render live telemetry & risk metrics on the right column
            with metrics_placeholder.container():
                if fsm_state == "ALARMING_HEALTH_ALERT":
                    st.error(f"🚨 **{tier_label}** — Immediate Assistance Required!")
                elif fsm_state in ("POST_FALL_ASSESSMENT", "IMPACT_DETECTED"):
                    st.warning(f"⚠️ **{tier_label}** ({fsm_state})")
                else:
                    st.success(f"✅ **{tier_label}** ({fsm_state})")

                st.progress(min(1.0, max(0.0, p_alert)))

                m1, m2, m3 = st.columns(3)
                m1.metric("Alert Probability (P_alert)", f"{p_alert * 100:.1f}%")
                m2.metric("Torso Angle (θ)", f"{geo_metrics['torso_angle']}°")
                m3.metric("Immobility Timer", f"{geo_metrics['immobile_seconds']}s")

                w1, w2, w3 = st.columns(3)
                w1.metric("Heart Rate", f"{telemetry['hr_bpm']} BPM")
                w2.metric("HRV (RMSSD)", f"{telemetry['hrv_rmssd']} ms")
                w3.metric("Wrist Impact", f"{telemetry['accel_g']} g")

                st.caption(
                    f"Wearable: **{telemetry['device_name']}** | SpO2: `{telemetry['spo2_pct']}%` | "
                    f"Vision Risk: `{geo_metrics['p_vision']}` | Biometric Risk: `{p_bio}`"
                )

            time.sleep(0.01)
    finally:
        detector.close()
        wearable.stop()
        cap.release()
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
else:
    st.info("Toggle **Start EdgeGuard Engine** in the left sidebar to launch the webcam and wearable fusion dashboard.")
