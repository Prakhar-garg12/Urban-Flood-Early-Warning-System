# map_generator.py
import folium
import os
from datetime import datetime


def generate_flood_map(risk_label, confidence, live_values,
                       lat=23.2599, lon=77.4126,
                       output_path="static/flood_map.html"):
    """
    risk_label  : "Low" | "Medium" | "High"
    confidence  : float 0-1
    live_values : dict from Flask /predict response
    lat, lon    : user location (default Bhopal)
    output_path : save location
    """

    CITY_CENTER = [lat, lon]

    # ── Flood zones relative to city center ──
    FLOOD_ZONES = [
        {
            "name"       : "Low-lying river basin",
            "description": "Area near main river — floods during heavy monsoon",
            "risk_weight": "high",
            "coordinates": [
                [lat + 0.008, lon - 0.015],
                [lat + 0.012, lon - 0.005],
                [lat + 0.006, lon + 0.010],
                [lat + 0.001, lon + 0.002],
                [lat + 0.004, lon - 0.018],
            ]
        },
        {
            "name"       : "Urban lake zone",
            "description": "Residential area around city lake — overflow risk",
            "risk_weight": "high",
            "coordinates": [
                [lat - 0.050, lon + 0.017],
                [lat - 0.044, lon + 0.027],
                [lat - 0.051, lon + 0.032],
                [lat - 0.057, lon + 0.025],
                [lat - 0.054, lon + 0.015],
            ]
        },
        {
            "name"       : "Northern river basin",
            "description": "Northern river floodplain — drainage overflow zone",
            "risk_weight": "medium",
            "coordinates": [
                [lat + 0.050, lon + 0.037],
                [lat + 0.060, lon + 0.052],
                [lat + 0.055, lon + 0.062],
                [lat + 0.045, lon + 0.057],
                [lat + 0.040, lon + 0.042],
            ]
        },
        {
            "name"       : "Dense urban drain zone",
            "description": "Dense urban area with poor drainage infrastructure",
            "risk_weight": "medium",
            "coordinates": [
                [lat + 0.015, lon + 0.007],
                [lat + 0.020, lon + 0.017],
                [lat + 0.016, lon + 0.024],
                [lat + 0.010, lon + 0.019],
                [lat + 0.009, lon + 0.010],
            ]
        },
        {
            "name"       : "Southern river zone",
            "description": "Southern river floodplain — seasonal flooding",
            "risk_weight": "low",
            "coordinates": [
                [lat - 0.080, lon + 0.047],
                [lat - 0.070, lon + 0.062],
                [lat - 0.075, lon + 0.072],
                [lat - 0.085, lon + 0.067],
                [lat - 0.090, lon + 0.052],
            ]
        },
    ]

    # ── Colors per risk level ──
    RISK_COLORS = {
        "Low"   : {"zone": "#22c55e", "fill": "#22c55e", "opacity": 0.25,
                   "tile": "CartoDB positron"},
        "Medium": {"zone": "#f59e0b", "fill": "#f59e0b", "opacity": 0.35,
                   "tile": "CartoDB positron"},
        "High"  : {"zone": "#ef4444", "fill": "#ef4444", "opacity": 0.50,
                   "tile": "CartoDB positron"},
    }

    # ── Which zones to show per risk level ──
    ZONE_FILTER = {
        "Low"   : ["low"],
        "Medium": ["low", "medium"],
        "High"  : ["low", "medium", "high"],
    }

    colors  = RISK_COLORS[risk_label]
    weights = ZONE_FILTER[risk_label]

    # ── Create map ──
    m = folium.Map(
        location   = CITY_CENTER,
        zoom_start = 12,
        tiles      = colors["tile"],
    )

    # ── City boundary circle ──
    folium.Circle(
        location = CITY_CENTER,
        radius   = 8000,
        color    = colors["zone"],
        weight   = 1.5,
        fill     = False,
        tooltip  = "City monitoring boundary",
    ).add_to(m)

    # ── Flood-prone zone polygons ──
    zones_shown = 0
    for zone in FLOOD_ZONES:
        if zone["risk_weight"] not in weights:
            continue

        folium.Polygon(
            locations    = zone["coordinates"],
            color        = colors["zone"],
            weight       = 2,
            fill         = True,
            fill_color   = colors["fill"],
            fill_opacity = colors["opacity"],
            tooltip      = f"<b>{zone['name']}</b><br>{zone['description']}",
            popup        = folium.Popup(
                f"""
                <div style='font-family:Arial;width:220px;padding:4px'>
                  <b style='color:{colors["zone"]};font-size:13px'>{zone['name']}</b><br>
                  <small style='color:#555'>{zone['description']}</small>
                  <hr style='margin:6px 0;border-color:#ddd'>
                  <b>Risk weight:</b> {zone['risk_weight'].upper()}<br>
                  <b>Current risk:</b>
                  <span style='color:{colors["zone"]};font-weight:bold'>
                    {risk_label.upper()}
                  </span>
                </div>
                """,
                max_width=240
            ),
        ).add_to(m)
        zones_shown += 1

    # ── Center marker ──
    folium.Marker(
        location = CITY_CENTER,
        tooltip  = "Monitoring station",
        icon     = folium.Icon(
            color = "red"    if risk_label == "High"   else
                    "orange" if risk_label == "Medium" else "green",
            icon  = "info-sign"
        ),
    ).add_to(m)

    # ── Risk level colors for info box ──
    risk_bg = {
        "Low"   : "#dcfce7",
        "Medium": "#fef9c3",
        "High"  : "#fee2e2",
    }
    risk_text_col = {
        "Low"   : "#166534",
        "Medium": "#854d0e",
        "High"  : "#991b1b",
    }

    # ── Info overlay box (top-right corner of map) ──
    now_str = datetime.now().strftime('%d-%b-%Y %H:%M')

    info_html = f"""
    <div style="
        position : fixed;
        top       : 12px;
        right     : 12px;
        background: {risk_bg[risk_label]};
        border    : 2px solid {risk_text_col[risk_label]};
        border-radius: 10px;
        padding   : 14px 18px;
        font-family: Arial, sans-serif;
        font-size : 13px;
        z-index   : 9999;
        min-width : 210px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    ">
        <div style="font-size:17px;font-weight:bold;color:{risk_text_col[risk_label]}">
            Flood Risk: {risk_label.upper()}
        </div>
        <div style="color:{risk_text_col[risk_label]};margin-top:4px;font-size:12px">
            Confidence: {confidence * 100:.1f}%
        </div>
        <hr style="border-color:{risk_text_col[risk_label]};opacity:0.25;margin:8px 0">
        <table style="font-size:11px;color:#444;width:100%;border-collapse:collapse">
          <tr>
            <td style="padding:2px 0">Rain today</td>
            <td style="text-align:right;font-weight:bold">
              {live_values.get('precipitation_sum', 0)} mm
            </td>
          </tr>
          <tr>
            <td style="padding:2px 0">River discharge</td>
            <td style="text-align:right;font-weight:bold">
              {live_values.get('river_discharge', 0)} m³/s
            </td>
          </tr>
          <tr>
            <td style="padding:2px 0">3-day rainfall</td>
            <td style="text-align:right;font-weight:bold">
              {live_values.get('rainfall_3d', 0)} mm
            </td>
          </tr>
          <tr>
            <td style="padding:2px 0">7-day rainfall</td>
            <td style="text-align:right;font-weight:bold">
              {live_values.get('rainfall_7d', 0)} mm
            </td>
          </tr>
          <tr>
            <td style="padding:2px 0">Humidity</td>
            <td style="text-align:right;font-weight:bold">
              {live_values.get('humidity', 0)}%
            </td>
          </tr>
          <tr>
            <td style="padding:2px 0">Soil moisture</td>
            <td style="text-align:right;font-weight:bold">
              {live_values.get('soil_moisture_0_to_7cm', 0)} m³/m³
            </td>
          </tr>
          <tr>
            <td style="padding:2px 0">Monsoon season</td>
            <td style="text-align:right;font-weight:bold">
              {'Yes' if live_values.get('is_monsoon') else 'No'}
            </td>
          </tr>
        </table>
        <hr style="border-color:{risk_text_col[risk_label]};opacity:0.25;margin:8px 0">
        <div style="color:#888;font-size:10px;text-align:center">
            {zones_shown} flood zone(s) highlighted
            &nbsp;·&nbsp;
            {now_str}
        </div>
    </div>
    """

    m.get_root().html.add_child(folium.Element(info_html))

    # ── Save map ──
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m.save(output_path)
    print(f"Map saved → {output_path}  ({zones_shown} zones shown)")
    return output_path