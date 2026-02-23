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
        key=key,
        default={"t": 0.0},
    )
    if isinstance(val, dict) and "t" in val:
        return val
    return {"t": 0.0}
