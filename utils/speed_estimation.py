import math
from typing import Optional, Tuple, Dict, List
import cv2

def calculate_speed(
    prev_pos: Optional[Tuple[float, float]],
    curr_pos: Optional[Tuple[float, float]],
    fps: float,
    pixels_per_meter: float,
    max_reasonable_kmph: float = 250.0
) -> float:
    """
    Calculate the real-world speed of the cricket ball between two frames.
    Args:
        prev_pos: (x, y) centroid in previous frame or None
        curr_pos: (x, y) centroid in current frame or None
        fps: frames per second of the video (must be > 0)
        pixels_per_meter: pixels per meter scale factor
        max_reasonable_kmph: cap to remove unrealistic spikes (default 250 km/h)

    Returns:
        speed_kmph (float): speed in km/h rounded to 1 decimal place. Returns 0.0 if invalid.
    """
    if prev_pos is None or curr_pos is None:
        return 0.0
    if fps is None or fps <= 0:
        return 0.0
    if pixels_per_meter is None or pixels_per_meter <= 0:
        return 0.0

    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    distance_pixels = math.hypot(dx, dy)

    distance_meters = distance_pixels / pixels_per_meter
    time_seconds    = 1.0 / fps
    speed_m_per_s   = distance_meters / time_seconds
    speed_kmph      = speed_m_per_s * 3.6

    if math.isnan(speed_kmph) or speed_kmph <= 0 or speed_kmph > max_reasonable_kmph:
        return 0.0

    return round(speed_kmph, 1)

class SpeedAggregator:
    """
    Aggregates per-object instantaneous speeds with smoothing, spike rejection,
    max tracking, and hit freeze logic.
    """
    def __init__(self, window: int = 7, spike_kmph_cap: float = 250.0, min_kmph_threshold: float = 5.0):
        self.window = max(1, int(window))
        self.spike_cap = float(spike_kmph_cap)
        self.min_threshold = float(min_kmph_threshold)
        self.history: Dict[int, List[float]] = {}        # object_id -> rolling speeds (km/h)
        self.prev_centroid: Dict[int, Tuple[int, int]] = {}  # object_id -> last centroid
        self._max_kmph = 0.0
        self._hit_kmph: Optional[float] = None
        self._frozen = False

    def get_prev_centroid(self, object_id: int) -> Optional[Tuple[int, int]]:
        return self.prev_centroid.get(object_id, None)

    def update(self, object_id: int, centroid: Tuple[int, int], inst_kmph: float) -> None:
        # Always store centroid
        self.prev_centroid[object_id] = centroid

        # If frozen after hit, do not update speed history/max
        if self._frozen:
            return

        # Spike rejection and minimum floor
        if inst_kmph <= 0 or inst_kmph < self.min_threshold or inst_kmph > self.spike_cap:
            return

        # Rolling window update
        buf = self.history.setdefault(object_id, [])
        buf.append(inst_kmph)
        if len(buf) > self.window:
            buf.pop(0)

        # Smoothed speed (simple mean)
        smoothed = round(sum(buf) / len(buf), 1)

        # Update global max using smoothed value
        if smoothed > self._max_kmph:
            self._max_kmph = smoothed

    def max_speed(self) -> float:
        return round(self._max_kmph, 1)

    def freeze_on_hit(self) -> None:
        """
        Freeze updates and lock 'speed at hit' to the best available number so far.
        """
        self._frozen = True
        self._hit_kmph = self.max_speed()

    def hit_speed(self) -> Optional[float]:
        return self._hit_kmph

def speed_category(kmph: float) -> str:
    """
    Categorize speed:
    - Fast: 140+
    - Medium: 100+
    - Slow: <100
    """
    if kmph >= 140.0:
        return "Fast"
    if kmph >= 100.0:
        return "Medium"
    return "Slow"

def format_top_right_text(
    frame,
    lines: List[str],
    frame_size: Tuple[int, int],
    pad_right: int = 20,
    top_y: int = 36,
    font_scale: float = 0.9,
    thickness: int = 2,
    color: Tuple[int, int, int] = (0, 255, 255)
) -> None:
    """
    Draw multiple lines aligned to the top-right corner.
    """
    width, _ = frame_size
    y = top_y
    for text in lines:
        (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x = width - text_width - pad_right
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y += int(28 * font_scale)




