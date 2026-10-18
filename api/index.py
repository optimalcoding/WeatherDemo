import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(
    title="Atmosphere Weather Backend",
    description="FastAPI weather aggregation proxy with surface & upper-air isobaric slices",
    version="1.1.0"
)

# Enable CORS for local testing and mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NWS_HEADERS = {
    "User-Agent": "(AtmosphereWeatherBFF, contact@atmosphereapp.internal)",
    "Accept": "application/geo+json"
}


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Atmosphere Weather BFF",
        "endpoints": [
            "/api/weather",
            "/api/atmosphere/levels",
            "/docs"
        ]
    }


@app.get("/api/weather")
async def get_weather(
    lat: float = Query(..., description="Latitude coordinate", ge=-90, le=90),
    lon: float = Query(..., description="Longitude coordinate", ge=-180, le=180)
):
    """
    Primary aggregation endpoint for surface conditions, 48h meteogram,
    active watches/warnings, and RainViewer radar loop timestamps.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        try:
            points_resp = await client.get(points_url, headers=NWS_HEADERS)
            if points_resp.status_code != 200:
                raise HTTPException(
                    status_code=points_resp.status_code,
                    detail=f"NWS Points lookup failed: {points_resp.text}"
                )
            points_data = points_resp.json()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to contact NWS API: {str(exc)}")

        props = points_data.get("properties", {})
        wfo = props.get("cwa")
        grid_x = props.get("gridX")
        grid_y = props.get("gridY")
        rel_loc = props.get("relativeLocation", {}).get("properties", {})
        city = rel_loc.get("city", "Unknown")
        state = rel_loc.get("state", "")
        location_display = f"{city}, {state}" if state else city

        grid_url = f"https://api.weather.gov/gridpoints/{wfo}/{grid_x},{grid_y}"
        alerts_url = f"https://api.weather.gov/alerts/active?point={lat},{lon}"
        radar_url = "https://api.rainviewer.com/public/weather-maps.json"

        grid_task = client.get(grid_url, headers=NWS_HEADERS)
        alerts_task = client.get(alerts_url, headers=NWS_HEADERS)
        radar_task = client.get(radar_url)

        responses = await asyncio.gather(grid_task, alerts_task, radar_task, return_exceptions=True)
        grid_res, alerts_res, radar_res = responses

        if isinstance(grid_res, Exception) or grid_res.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to fetch raw NWS grid telemetry.")

        grid_json = grid_res.json().get("properties", {})

        active_alert = None
        if not isinstance(alerts_res, Exception) and alerts_res.status_code == 200:
            features = alerts_res.json().get("features", [])
            if features:
                active_alert = features[0].get("properties", {}).get("event")

        latest_radar_path = None
        if not isinstance(radar_res, Exception) and radar_res.status_code == 200:
            past_frames = radar_res.json().get("radar", {}).get("past", [])
            if past_frames:
                latest_radar_path = past_frames[-1].get("path")

        meteogram_points = parse_nws_grid_series(grid_json)
        current = meteogram_points[0] if meteogram_points else None
        current_temp = current["tempF"] if current else 68.0
        current_dew = current["dewPointF"] if current else 55.0
        lcl_meters = calculate_lcl(current_temp, current_dew)

        return {
            "locationName": location_display,
            "cwa": wfo,
            "gridX": grid_x,
            "gridY": grid_y,
            "current": {
                "temperatureF": current_temp,
                "dewPointF": current_dew,
                "relativeHumidity": current["relativeHumidity"] if current else 60,
                "windSpeedMph": current["windSpeedMph"] if current else 5.0,
                "cloudBaseMeters": round(lcl_meters, 1)
            },
            "activeAlertHeadline": active_alert,
            "latestRadarPath": latest_radar_path,
            "hourlyMeteogram": meteogram_points
        }


@app.get("/api/atmosphere/levels")
async def get_atmospheric_levels(
    lat: float = Query(..., description="Latitude coordinate", ge=-90, le=90),
    lon: float = Query(..., description="Longitude coordinate", ge=-180, le=180)
):
    """
    Fetches multi-level atmospheric slices across mandatory millibar pressure altitudes
    (1000, 925, 850, 700, 500, 300, 250 hPa) for vertical soundings and wind shear calculation.
    """
    # Parameters for Open-Meteo GFS pressure level variables
    variables = [
        "temperature_1000hPa", "temperature_925hPa", "temperature_850hPa", "temperature_700hPa", "temperature_500hPa", "temperature_300hPa", "temperature_250hPa",
        "dewpoint_1000hPa", "dewpoint_925hPa", "dewpoint_850hPa", "dewpoint_700hPa", "dewpoint_500hPa", "dewpoint_300hPa", "dewpoint_250hPa",
        "windspeed_1000hPa", "windspeed_850hPa", "windspeed_700hPa", "windspeed_500hPa", "windspeed_250hPa",
        "winddirection_1000hPa", "winddirection_850hPa", "winddirection_700hPa", "winddirection_500hPa", "winddirection_250hPa",
        "geopotential_height_850hPa", "geopotential_height_500hPa", "geopotential_height_250hPa"
    ]
    
    query_str = ",".join(variables)
    open_meteo_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&hourly={query_str}&forecast_days=2&timezone=auto"
    )

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(open_meteo_url)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail="Open-Meteo isobaric query failed.")
            data = resp.json().get("hourly", {})
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to contact isobaric model provider: {str(exc)}")

    # Extract current sounding slice (index 0 corresponds to the current hour)
    levels = [1000, 925, 850, 700, 500, 300, 250]
    profile = []

    for lvl in levels:
        t_key = f"temperature_{lvl}hPa"
        td_key = f"dewpoint_{lvl}hPa"
        ws_key = f"windspeed_{lvl}hPa"
        wd_key = f"winddirection_{lvl}hPa"
        gh_key = f"geopotential_height_{lvl}hPa"

        temp_c = data.get(t_key, [None])[0]
        dew_c = data.get(td_key, [None])[0]
        wind_kmh = data.get(ws_key, [None])[0] if ws_key in data else None
        wind_dir = data.get(wd_key, [None])[0] if wd_key in data else None
        geo_m = data.get(gh_key, [None])[0] if gh_key in data else None

        profile.append({
            "pressureMbar": lvl,
            "tempC": temp_c,
            "tempF": round((temp_c * 9.0 / 5.0) + 32.0, 1) if temp_c is not None else None,
            "dewPointC": dew_c,
            "dewPointF": round((dew_c * 9.0 / 5.0) + 32.0, 1) if dew_c is not None else None,
            "windSpeedMph": round(wind_kmh * 0.621371, 1) if wind_kmh is not None else None,
            "windDirectionDeg": wind_dir,
            "geopotentialHeightMeters": geo_m
        })

    # Calculate 0-6 km Bulk Wind Shear indicator (Surface to 500 hPa delta)
    sfc_wind = profile[0].get("windSpeedMph") or 0.0
    mid_wind = next((p.get("windSpeedMph") or 0.0 for p in profile if p["pressureMbar"] == 500), 0.0)
    bulk_shear_estimate = abs(mid_wind - sfc_wind)

    return {
        "timestamp": data.get("time", [""])[0],
        "bulkShearMph0to6km": round(bulk_shear_estimate, 1),
        "verticalSounding": profile
    }


def parse_nws_grid_series(props: Dict[str, Any]) -> List[Dict[str, Any]]:
    temp_vals = props.get("temperature", {}).get("values", [])
    dew_vals = props.get("dewpoint", {}).get("values", [])
    rh_vals = props.get("relativeHumidity", {}).get("values", [])
    pop_vals = props.get("probabilityOfPrecipitation", {}).get("values", [])
    wind_vals = props.get("windSpeed", {}).get("values", [])
    gust_vals = props.get("windGust", {}).get("values", [])
    weather_vals = props.get("weather", {}).get("values", [])

    results = []
    total_hours = min(len(temp_vals), 48)

    for i in range(total_hours):
        raw_valid_time = temp_vals[i].get("validTime", "")
        hour_label = f"{i}h"
        try:
            iso_str = raw_valid_time.split("/")[0][:19]
            dt = datetime.fromisoformat(iso_str)
            hour_label = dt.strftime("%I%p").lstrip("0").lower()
        except Exception:
            pass

        temp_c = temp_vals[i].get("value") or 15.0
        dew_c = (dew_vals[i].get("value") if i < len(dew_vals) else None) or (temp_c - 5.0)
        rh = int(rh_vals[i].get("value") or 60) if i < len(rh_vals) else 60
        pop = int(pop_vals[i].get("value") or 0) if i < len(pop_vals) else 0
        wind_kmh = (wind_vals[i].get("value") if i < len(wind_vals) else None) or 10.0
        gust_kmh = (gust_vals[i].get("value") if i < len(gust_vals) else None)

        has_thunder = False
        if i < len(weather_vals):
            phenomena = weather_vals[i].get("value", [])
            for item in phenomena:
                w_str = str(item.get("weather", "")).lower()
                if "thunderstorm" in w_str:
                    has_thunder = True
                    break

        results.append({
            "hourLabel": hour_label,
            "tempF": round((temp_c * 9.0 / 5.0) + 32.0, 1),
            "dewPointF": round((dew_c * 9.0 / 5.0) + 32.0, 1),
            "relativeHumidity": rh,
            "popPercent": pop,
            "windSpeedMph": round(wind_kmh * 0.621371, 1),
            "windGustMph": round(gust_kmh * 0.621371, 1) if gust_kmh else None,
            "hasThunder": has_thunder
        })

    return results


def calculate_lcl(temp_f: float, dew_point_f: float) -> float:
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    dew_c = (dew_point_f - 32.0) * 5.0 / 9.0
    spread = temp_c - dew_c
    return max(0.0, spread * 125.0)
