import math
from collections import deque
import numpy as np

# Skeleton connections for drawing on the video feed
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
    (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
]


class FallGeometryEngine:
    def __init__(self, fps: float = 30.0, buffer_seconds: float = 0.5):
        self.fps = max(1.0, float(fps))
        maxlen = max(5, int(round(self.fps * buffer_seconds)))
        self.hip_y_history = deque(maxlen=maxlen)
        self.keypoint_history = deque(maxlen=maxlen)
        self.immobile_frames = 0

    def reset(self) -> None:
        """Clears rolling motion buffers and resets the immobility counter."""
        self.hip_y_history.clear()
        self.keypoint_history.clear()
        self.immobile_frames = 0

    def extract_metrics(self, landmarks) -> dict:
        """
        Takes MediaPipe's 33 normalized pose landmarks and returns
        scale-invariant fall and immobility metrics.
        """
        # Core landmarks: Nose (0), Shoulders (11, 12), Hips (23, 24)
        nose = np.array([landmarks[0].x, landmarks[0].y], dtype=np.float64)
        l_sh = np.array([landmarks[11].x, landmarks[11].y], dtype=np.float64)
        r_sh = np.array([landmarks[12].x, landmarks[12].y], dtype=np.float64)
        l_hip = np.array([landmarks[23].x, landmarks[23].y], dtype=np.float64)
        r_hip = np.array([landmarks[24].x, landmarks[24].y], dtype=np.float64)

        shoulder_mid = (l_sh + r_sh) / 2.0
        hip_mid = (l_hip + r_hip) / 2.0

        # 1. Scale-invariant torso length (L_torso)
        torso_vec = shoulder_mid - hip_mid
        l_torso = float(np.linalg.norm(torso_vec)) + 1e-6

        # 2. Torso inclination angle (theta) relative to vertical Y-axis
        # In image coordinates, Y increases downward, so upright vector is (0, -1)
        dx = float(torso_vec[0])
        dy = float(-torso_vec[1])  # Positive when shoulders are above hips
        angle_rad = math.atan2(abs(dx), dy)
        torso_angle_deg = math.degrees(angle_rad)

        # 3. Normalized vertical hip drop velocity (V_y) over rolling 0.5s deque buffer
        self.hip_y_history.append(float(hip_mid[1]))
        if len(self.hip_y_history) >= 2:
            dt = (len(self.hip_y_history) - 1) / self.fps
            dy_hip = self.hip_y_history[-1] - self.hip_y_history[0]  # Positive = downward
            drop_velocity = (dy_hip / l_torso) / max(dt, 1e-3)
        else:
            drop_velocity = 0.0

        # 4. Post-fall immobility tracking (seconds)
        core_points = np.vstack([nose, shoulder_mid, hip_mid])
        self.keypoint_history.append(core_points)

        if len(self.keypoint_history) >= 2:
            frame_delta = float(
                np.mean(
                    np.linalg.norm(
                        self.keypoint_history[-1] - self.keypoint_history[-2], axis=1
                    )
                )
            ) / l_torso
        else:
            frame_delta = 1.0

        is_horizontal = torso_angle_deg > 65.0
        if is_horizontal and frame_delta < 0.025:
            self.immobile_frames += 1
        elif not is_horizontal:
            self.immobile_frames = 0

        immobile_seconds = self.immobile_frames / self.fps

        # 5. Vision-only fall probability (0.0 to 1.0)
        angle_score = min(1.0, max(0.0, (torso_angle_deg - 45.0) / 45.0))
        velocity_score = min(1.0, max(0.0, drop_velocity / 2.5))
        p_vision = 0.65 * angle_score + 0.35 * velocity_score
        if is_horizontal:
            p_vision = max(p_vision, 0.72)
        p_vision = min(1.0, max(0.0, float(p_vision)))

        return {
            "l_torso": round(l_torso, 4),
            "torso_angle": round(torso_angle_deg, 1),
            "drop_velocity": round(drop_velocity, 2),
            "is_horizontal": is_horizontal,
            "immobile_seconds": round(immobile_seconds, 1),
            "p_vision": round(p_vision, 3),
        }
