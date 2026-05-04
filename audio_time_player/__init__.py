import os
import streamlit.components.v1 as components
from typing import Optional

_audio_time_player = components.declare_component(
    "audio_time_player",
    path=os.path.dirname(__file__),
)

def audio_time_player(
    audio_b64: str = "",
    mime: str = "audio/wav",
    poll_ms: int = 250,
    height: int = 90,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    autoplay: bool = False,
    segment_token: str = "",
    report_time: bool = True,
    emit_events: bool = True,
    key: Optional[str] = None,
) -> dict:
    """
    audio_b64: base64 payload WITHOUT data: prefix. If empty, keep existing audio.
    Returns: {"t": <currentTimeSeconds>}
    """
    val = _audio_time_player(
        audio_b64=audio_b64,
        mime=mime,
        poll_ms=int(poll_ms),
        height=int(height),
        start_sec=float(start_sec or 0.0),
        end_sec=None if end_sec is None else float(end_sec),
        autoplay=bool(autoplay),
        segment_token=str(segment_token or ""),
        report_time=bool(report_time),
        emit_events=bool(emit_events),
        key=key,
        default={"t": 0.0},
    )
    if isinstance(val, dict) and "t" in val:
        return val
    return {"t": 0.0}
