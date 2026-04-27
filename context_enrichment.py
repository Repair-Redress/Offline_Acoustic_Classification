import json
import math
import ssl
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional


APP_USER_AGENT = "OfflineAudioSetClassification/1.0 (research context enrichment)"


def build_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

WEATHER_CODE_LABELS = {
    0: ("clear", "clear sky"),
    1: ("mainly clear", "mainly clear"),
    2: ("partly cloudy", "partly cloudy"),
    3: ("overcast", "overcast"),
    45: ("fog", "fog"),
    48: ("depositing rime fog", "fog"),
    51: ("light drizzle", "drizzle"),
    53: ("moderate drizzle", "drizzle"),
    55: ("dense drizzle", "drizzle"),
    56: ("light freezing drizzle", "freezing drizzle"),
    57: ("dense freezing drizzle", "freezing drizzle"),
    61: ("slight rain", "rain"),
    63: ("moderate rain", "rain"),
    65: ("heavy rain", "rain"),
    66: ("light freezing rain", "freezing rain"),
    67: ("heavy freezing rain", "freezing rain"),
    71: ("slight snow", "snow"),
    73: ("moderate snow", "snow"),
    75: ("heavy snow", "snow"),
    77: ("snow grains", "snow"),
    80: ("slight rain showers", "rain showers"),
    81: ("moderate rain showers", "rain showers"),
    82: ("violent rain showers", "rain showers"),
    85: ("slight snow showers", "snow showers"),
    86: ("heavy snow showers", "snow showers"),
    95: ("thunderstorm", "thunderstorm"),
    96: ("thunderstorm with slight hail", "thunderstorm"),
    99: ("thunderstorm with heavy hail", "thunderstorm"),
}


def parse_recording_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None

    if parsed is None:
        parsed = parse_android_capture_timestamp(text)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_android_capture_timestamp(text: str) -> Optional[datetime]:
    cleaned = text.replace("_", " ").strip()
    parts = cleaned.split()
    if len(parts) != 2:
        return None

    date_part, time_part = parts
    parsed_date = None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"):
        try:
            parsed_date = datetime.strptime(date_part, fmt).date()
            break
        except ValueError:
            continue
    if parsed_date is None:
        return None

    time_parts = time_part.split(":")
    if len(time_parts) < 2:
        return None
    try:
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2]) if len(time_parts) >= 3 else 0
        millisecond = int(time_parts[3]) if len(time_parts) >= 4 else 0
    except ValueError:
        return None

    if millisecond > 999:
        millisecond = int(str(millisecond)[:3])
    return datetime(
        parsed_date.year,
        parsed_date.month,
        parsed_date.day,
        hour,
        minute,
        second,
        max(0, millisecond) * 1000,
        tzinfo=timezone.utc,
    )


def fetch_json(url: str, timeout: float = 12.0) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": APP_USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout, context=build_ssl_context()) as response:
        return json.loads(response.read().decode("utf-8"))


def build_url(base_url: str, params: Dict[str, Any]) -> str:
    clean = {k: v for k, v in params.items() if v is not None and v != ""}
    return f"{base_url}?{urllib.parse.urlencode(clean, doseq=True)}"


def nearest_hour_index(times: list[str], timestamp: datetime) -> int:
    if not times:
        return 0
    target = timestamp.replace(minute=0, second=0, microsecond=0)
    if timestamp.minute >= 30:
        target = target + timedelta(hours=1)
    target_text = target.strftime("%Y-%m-%dT%H:00")
    if target_text in times:
        return times.index(target_text)

    def distance(idx: int) -> float:
        candidate = datetime.fromisoformat(times[idx])
        return abs((candidate - target.replace(tzinfo=None)).total_seconds())

    return min(range(len(times)), key=distance)


def weather_code_text(code: Any) -> tuple[str, str]:
    try:
        return WEATHER_CODE_LABELS.get(int(code), ("unknown", "unknown"))
    except (TypeError, ValueError):
        return "unknown", "unknown"


def summarize_weather(hourly: Dict[str, Any], idx: int) -> Dict[str, Any]:
    def at(key: str) -> Any:
        values = hourly.get(key) or []
        if idx >= len(values):
            return None
        return values[idx]

    label, condition = weather_code_text(at("weather_code"))
    windspeed = at("wind_speed_10m")
    windgust = at("wind_gusts_10m")
    precipitation = at("precipitation")
    cloudcover = at("cloud_cover")

    descriptors = [label]
    if isinstance(precipitation, (int, float)) and precipitation > 0:
        descriptors.append("wet")
    if isinstance(windspeed, (int, float)) and windspeed >= 8:
        descriptors.append("windy")
    if isinstance(cloudcover, (int, float)) and cloudcover >= 80 and "overcast" not in descriptors:
        descriptors.append("overcast")

    weather = {
        "summary": " and ".join(dict.fromkeys(descriptors)),
        "description": label,
        "condition": condition,
        "weathercode_label": label,
        "temperature_c": at("temperature_2m"),
        "relativehumidity_2m": at("relative_humidity_2m"),
        "precipitation_mm": precipitation,
        "rain_mm": at("rain"),
        "cloudcover": cloudcover,
        "windspeed_10m": windspeed,
        "windgusts_10m": windgust,
        "visibility_m": at("visibility"),
    }
    return {k: v for k, v in weather.items() if v is not None}


def fetch_weather_context(latitude: float, longitude: float, timestamp: datetime) -> Dict[str, Any]:
    today = date.today()
    endpoint = "https://api.open-meteo.com/v1/forecast"
    if timestamp.date() < today:
        endpoint = "https://archive-api.open-meteo.com/v1/archive"

    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "weather_code",
        "cloud_cover",
        "wind_speed_10m",
        "wind_gusts_10m",
        "visibility",
    ]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": timestamp.date().isoformat(),
        "end_date": timestamp.date().isoformat(),
        "hourly": ",".join(hourly_vars),
        "timezone": "auto",
    }
    data = fetch_json(build_url(endpoint, params))
    hourly = data.get("hourly") or {}
    idx = nearest_hour_index(hourly.get("time") or [], timestamp)
    weather = summarize_weather(hourly, idx)
    weather["source"] = "Open-Meteo"
    weather["matched_hour"] = (hourly.get("time") or [""])[idx] if hourly.get("time") else ""
    return weather


def fetch_location_context(latitude: float, longitude: float) -> Dict[str, Any]:
    params = {
        "lat": latitude,
        "lon": longitude,
        "format": "jsonv2",
        "addressdetails": 1,
        "zoom": 18,
    }
    data = fetch_json(build_url("https://nominatim.openstreetmap.org/reverse", params))
    address = data.get("address") or {}
    location = {
        "name": data.get("name") or address.get("road") or address.get("neighbourhood"),
        "display_name": data.get("display_name"),
        "category": data.get("category"),
        "type": data.get("type"),
        "road": address.get("road"),
        "neighbourhood": address.get("neighbourhood") or address.get("quarter"),
        "suburb": address.get("suburb"),
        "city": address.get("city") or address.get("town") or address.get("village"),
        "state": address.get("state"),
        "country": address.get("country"),
        "latitude": latitude,
        "longitude": longitude,
        "source": "OpenStreetMap Nominatim",
    }
    return {k: v for k, v in location.items() if v not in ("", None)}


def infer_scene_from_location(location: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join(str(location.get(k, "")) for k in ("category", "type", "display_name", "road")).lower()
    scene: Dict[str, Any] = {}
    tags = []

    if any(token in text for token in ("road", "street", "highway", "primary", "secondary", "residential")):
        scene.update(
            {
                "environment": "urban",
                "setting": "roadside",
                "place_type": "road corridor",
                "indoor_outdoor": "outdoor",
                "land_use": "transportation",
            }
        )
        tags.extend(["outdoor", "urban", "roadside"])
    elif any(token in text for token in ("park", "garden", "forest", "trail")):
        scene.update({"environment": "natural", "setting": "park", "indoor_outdoor": "outdoor"})
        tags.extend(["outdoor", "nature"])
    elif location:
        scene.update({"environment": "outdoor", "indoor_outdoor": "outdoor"})
        tags.append("outdoor")

    return {"scene": scene, "tags": tags}


def build_recording_context(timestamp: Optional[datetime], metadata: Dict[str, Any]) -> Dict[str, Any]:
    recording = {k: v for k, v in metadata.get("recording", {}).items() if v not in ("", None)}
    if timestamp:
        recording["timestamp"] = timestamp.isoformat()
    for key in ("device", "source", "notes"):
        if metadata.get(key):
            recording[key] = metadata[key]
    return recording


def merge_context(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if not value:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_context(merged[key], value)
        elif isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = sorted({*merged[key], *value})
        else:
            merged[key] = value
    return merged


def enrich_context(
    latitude: Optional[float],
    longitude: Optional[float],
    timestamp_text: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
    include_geocoding: bool = True,
    include_weather: bool = True,
) -> Dict[str, Any]:
    metadata = metadata or {}
    timestamp = parse_recording_timestamp(timestamp_text or metadata.get("timestamp") or metadata.get("capture_time"))
    context: Dict[str, Any] = {}

    if latitude is None or longitude is None or not math.isfinite(latitude) or not math.isfinite(longitude):
        recording = build_recording_context(timestamp, metadata)
        return {"recording": recording} if recording else {}

    location: Dict[str, Any] = {"latitude": latitude, "longitude": longitude}
    if include_geocoding:
        location = merge_context(location, fetch_location_context(latitude, longitude))
    context["location"] = location

    inferred = infer_scene_from_location(location)
    context = merge_context(context, inferred)

    if timestamp and include_weather:
        context["weather"] = fetch_weather_context(latitude, longitude, timestamp)

    recording = build_recording_context(timestamp, metadata)
    if recording:
        context["recording"] = recording

    manual_context = metadata.get("context")
    if isinstance(manual_context, dict):
        context = merge_context(context, manual_context)

    return context
