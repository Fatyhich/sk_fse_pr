from typing import Optional, Any, Tuple


class FPSBuffer:

    def __init__(self, target_fps: Optional[float] = None):
        self._target_dt = (1. / target_fps) * 1000 if target_fps is not None else None
        self._last_data = None
        self._reference_timestamp = None

    def filter(self, timestamp: float, data: Any) -> Optional[Tuple[int, Any]]:
        if self._target_dt is None:
            return None

        # No previous timestamps, so return current data
        if self._reference_timestamp is None:
            self._reference_timestamp = timestamp
            return timestamp, data
        
        # Error: new timestamp is before the last timestamp
        if self._reference_timestamp > timestamp:
            raise ValueError("Timestmaps must strictly increase")

        # If timestamp is not enough to count for target FPS
        if (timestamp - self._reference_timestamp) < self._target_dt:
            self._last_data = (timestamp, data)
            return None, None
        
        # If we are less than target FPS, select the closest data for target FPS
        if (timestamp - self._reference_timestamp) >= self._target_dt:

            # No buffered data
            if self._last_data is None:
                self._reference_timestamp = timestamp
                self._last_data = None
                return (timestamp, data)

            # Previous data is closer to the target FPS than the new one
            delta_last = abs(self._target_dt - (self._last_data[0] - self._reference_timestamp))
            delta_current = abs(self._target_dt - (timestamp - self._reference_timestamp))
            if delta_current <= delta_last:
                self._reference_timestamp = timestamp
                result = (timestamp, data)
                self._last_data = None
                return result
            else:
                self._reference_timestamp = self._last_data[0]
                result = self._last_data
                self._last_data = None
                return result
