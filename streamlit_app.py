import streamlit as st
import ee
import json
import geemap.foliumap as geemap
import xml.etree.ElementTree as ET
import re
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="SpecTralNi30 - RWH Analytics", 
    page_icon="🌧️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS STYLING (Cyber-Glass UI) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&family=Inter:wght@300;400;600&display=swap');
    
    :root {
        --bg-color: #050509;
        --card-bg: rgba(20, 24, 35, 0.7);
        --glass-border: 1px solid rgba(255, 255, 255, 0.08);
        --accent-primary: #00f2ff;
        --accent-secondary: #7000ff;
        --text-primary: #e2e8f0;
    }

    .stApp { 
        background-image: radial-gradient(circle at 50% 0%, #1a1f35 0%, #050509 100%);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, .title-font { font-family: 'Rajdhani', sans-serif !important; text-transform: uppercase; letter-spacing: 1px; }
    p, label, .stMarkdown, div { color: var(--text-primary) !important; }

    section[data-testid="stSidebar"] {
        background-color: rgba(10, 12, 16, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div, .stDateInput > div > div {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 4px;
        color: #fff !important;
    }
    
    div.stButton > button:first-child {
        background: linear-gradient(90deg, var(--accent-secondary) 0%, #4c1d95 100%);
        border: none;
        color: white;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 0.6rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(112, 0, 255, 0.4);
    }

    .hud-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.02);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding: 15px 25px;
        border-radius: 0 0 15px 15px;
        margin-bottom: 25px;
    }
    .hud-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        background: -webkit-linear-gradient(0deg, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hud-badge {
        background: rgba(0, 242, 255, 0.1);
        border: 1px solid rgba(0, 242, 255, 0.3);
        color: var(--accent-primary);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }

    .glass-card {
        background: var(--card-bg);
        border: var(--glass-border);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .card-label {
        font-family: 'Rajdhani', sans-serif;
        color: var(--accent-primary);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 5px;
    }
    
    iframe {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    
    .metric-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AUTHENTICATION ---
try:
    service_account = st.secrets["gcp_service_account"]["client_email"]
    secret_dict = dict(st.secrets["gcp_service_account"])
    key_data = json.dumps(secret_dict) 
    credentials = ee.ServiceAccountCredentials(service_account, key_data=key_data)
    ee.Initialize(credentials)
except Exception:
    try:
        ee.Initialize()
    except Exception as e:
        st.error(f"⚠️ Authentication Error: {e}")
        st.stop()

# --- SESSION STATE INITIALIZATION ---
if 'calculated' not in st.session_state: st.session_state['calculated'] = False
if 'roi' not in st.session_state: st.session_state['roi'] = None

# --- 4. HELPER FUNCTIONS ---
def parse_kml(content):
    try:
        if isinstance(content, bytes): content = content.decode('utf-8')
        match = re.search(r'<coordinates>(.*?)</coordinates>', content, re.DOTALL | re.IGNORECASE)
        if match: return process_coords(match.group(1))
        root = ET.fromstring(content)
        for elem in root.iter():
            if elem.tag.lower().endswith('coordinates') and elem.text:
                return process_coords(elem.text)
    except: pass
    return None

def process_coords(text):
    raw = text.strip().split()
    coords = [[float(x.split(',')[0]), float(x.split(',')[1])] for x in raw if len(x.split(',')) >= 2]
    return ee.Geometry.Polygon([coords]) if len(coords) > 2 else None

def calculate_area_by_class(image, region, scale):
    area_image = ee.Image.pixelArea().addBands(image)
    stats = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName='class_index'),
        geometry=region,
        scale=scale,
        maxPixels=1e10, 
        bestEffort=True
    )
    
    groups = stats.get('groups').getInfo()
    data = []
    total_area = 0
    
    if not groups: return pd.DataFrame()

    for item in groups:
        c_idx = int(item['class_index'])
        area_sqm = item['sum']
        area_ha = area_sqm / 10000.0
        total_area += area_ha
        data.append({"Class": f"Class {c_idx}", "Area (ha)": area_ha})
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by="Area (ha)", ascending=False)
        df["%"] = ((df["Area (ha)"] / total_area) * 100).round(1)
        df["Area (ha)"] = df["Area (ha)"].round(2)
        
    return df

def generate_static_map_display(image, roi, vis_params, title, cmap_colors=None, is_categorical=False, class_names=None):
    try:
        roi_bounds = roi.bounds().getInfo()['coordinates'][0]
        lons = [p[0] for p in roi_bounds]
        lats = [p[1] for p in roi_bounds]
        
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        mid_lat = (min_lat + max_lat) / 2
        width_deg = max_lon - min_lon
        height_deg = max_lat - min_lat
        aspect_ratio = (width_deg * np.cos(np.radians(mid_lat))) / height_deg
        fig_width = 12 
        fig_height = fig_width / aspect_ratio
        if fig_height > 20: fig_height = 20
        if fig_height < 4: fig_height = 4

        if 'palette' in vis_params or 'min' in vis_params:
            ready_img = image.visualize(**vis_params)
        else:
            ready_img = image 
            
        thumb_url = ready_img.getThumbURL({
            'region': roi,
            'dimensions': 1500, 
            'format': 'png',
            'crs': 'EPSG:4326' 
        })
        response = requests.get(thumb_url, timeout=120)
        
        if response.status_code != 200: return None
        img_pil = Image.open(BytesIO(response.content))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300, facecolor='#000000')
        ax.set_facecolor('#000000')
        im = ax.imshow(img_pil, extent=[min_lon, max_lon, min_lat, max_lat], aspect='auto')
        
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#00f2ff')
        ax.tick_params(colors='white', labelcolor='white', labelsize=10)
        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_alpha(0.3)
        
        # North Arrow & Scale Bar logic same as before...
        ax.annotate('N', xy=(0.97, 0.95), xytext=(0.97, 0.88),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(facecolor='white', edgecolor='white', width=4, headwidth=12, headlength=10),
                    ha='center', va='center', fontsize=16, fontweight='bold', color='white',
                    path_effects=[PathEffects.withStroke(linewidth=2, foreground="black")])

        # Legend logic
        if is_categorical and class_names and 'palette' in vis_params:
            patches = []
            for name, color in zip(class_names, vis_params['palette']):
                patches.append(mpatches.Patch(color=color, label=name))
            legend = ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.08), 
                               frameon=False, title="Classes", ncol=min(len(class_names), 4))
            plt.setp(legend.get_title(), color='white', fontweight='bold', fontsize=12)
            for text in legend.get_texts():
                text.set_color("white")
                
        elif cmap_colors and 'min' in vis_params:
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", cmap_colors)
            norm = mcolors.Normalize(vmin=vis_params['min'], vmax=vis_params['max'])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
            cbar = plt.colorbar(sm, cax=cax)
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.set_label('Value', color='white', fontsize=12)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', fontsize=10)
        
        buf = BytesIO()
        plt.savefig(buf, format='jpg', bbox_inches='tight', facecolor='#000000')
        buf.seek(0)
        plt.close(fig)
        return buf
    except: return None

# --- 5. SIDEBAR (CONTROL PANEL) ---
with st.sidebar:
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="font-family: 'Rajdhani'; color: #fff; margin:0;">SpecTralNi30</h2>
            <p style="font-size: 0.8rem; color: #00f2ff; letter-spacing: 2px; margin:0;">RAINWATER HARVESTING</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("### 1. Target Acquisition (ROI)")
        roi_method = st.radio("Selection Mode", ["Upload KML", "Point & Buffer", "Manual Coordinates"], label_visibility="collapsed")
        
        new_roi = None
        if roi_method == "Upload KML":
            kml = st.file_uploader("Drop KML File", type=['kml'])
            if kml:
                kml.seek(0)
                new_roi = parse_kml(kml.read())
        elif roi_method == "Point & Buffer":
            c1, c2 = st.columns([1, 1])
            lat = c1.number_input("Lat", value=20.59, min_value=-90.0, max_value=90.0, format="%.6f")
            lon = c2.number_input("Lon", value=78.96, min_value=-180.0, max_value=180.0, format="%.6f")
            rad = st.number_input("Radius (meters)", value=5000, min_value=10, step=10)
            if lat and lon: 
                new_roi = ee.Geometry.Point([lon, lat]).buffer(rad).bounds()
        elif roi_method == "Manual Coordinates":
            c1, c2 = st.columns(2)
            min_lon = c1.number_input("Min Lon", value=78.0, min_value=-180.0, max_value=180.0, format="%.6f")
            min_lat = c2.number_input("Min Lat", value=20.0, min_value=-90.0, max_value=90.0, format="%.6f")
            max_lon = c1.number_input("Max Lon", value=79.0, min_value=-180.0, max_value=180.0, format="%.6f")
            max_lat = c2.number_input("Max Lat", value=21.0, min_value=-90.0, max_value=90.0, format="%.6f")
            if min_lon < max_lon and min_lat < max_lat: 
                new_roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

        if new_roi:
            if st.session_state['roi'] is None or new_roi.getInfo() != st.session_state['roi'].getInfo():
                st.session_state['roi'] = new_roi
                st.session_state['calculated'] = False
                st.toast("Target Locked: ROI Updated", icon="🎯")

    st.markdown("---")
    
    st.markdown("### 2. MCDA Weights")
    st.caption("Multi-Criteria Decision Analysis")
    w_rain = st.slider("Rainfall Weight (%)", 0, 100, 30)
    w_slope = st.slider("Slope Weight (%)", 0, 100, 20)
    w_lulc = st.slider("Land Use Weight (%)", 0, 100, 30)
    w_soil = st.slider("Soil Weight (%)", 0, 100, 20)
    
    total = w_rain + w_slope + w_lulc + w_soil
    if total != 100:
        st.warning(f"⚠️ Total weight: {total}%. It should ideally be 100%.")
    
    st.markdown("---")
    st.markdown("### 3. Temporal Window")
    st.caption("For average rainfall calculation")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", datetime.now()-timedelta(365*5)) # 5 Years default
    end = c2.date_input("End", datetime.now())

    st.markdown("###")
    if st.button("INITIALIZE RWH SCAN 🚀"):
        if st.session_state['roi']:
            st.session_state.update({
                'calculated': True,
                'start': start.strftime("%Y-%m-%d"),
                'end': end.strftime("%Y-%m-%d"),
                'w_rain': w_rain/100.0,
                'w_slope': w_slope/100.0,
                'w_lulc': w_lulc/100.0,
                'w_soil': w_soil/100.0
            })
        else:
            st.error("❌ Error: ROI not defined.")

# --- 6. MAIN CONTENT ---
st.markdown("""
<div class="hud-header">
    <div>
        <div class="hud-title">SpecTralNi30 ANALYTICS</div>
        <div style="color:#94a3b8; font-size:0.9rem;">RWH SITE SUITABILITY MODE</div>
    </div>
    <div style="text-align:right;">
        <span class="hud-badge">SYSTEM ONLINE</span>
        <div style="font-family:'Rajdhani'; font-size:1.2rem; margin-top:5px;">""" + datetime.now().strftime("%H:%M UTC") + """</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state['calculated']:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:40px;">
        <h2 style="color:#fff;">📡 WAITING FOR INPUT</h2>
        <p style="color:#94a3b8; margin-bottom:20px;">Configure the target ROI and weights in the sidebar to begin Rainwater Harvesting analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    m = geemap.Map(height=500, basemap="HYBRID")
    if st.session_state['roi']:
        m.centerObject(st.session_state['roi'], 12)
        m.addLayer(ee.Image().paint(st.session_state['roi'], 2, 3), {'palette': '#00f2ff'}, 'Target ROI')
    m.to_streamlit()

else:
    roi = st.session_state['roi']
    p = st.session_state
    
    col_map, col_res = st.columns([3, 1])
    m = geemap.Map(height=700, basemap="HYBRID")
    m.centerObject(roi, 13)

    with st.spinner("🌧️ Performing MCDA for Rainwater Harvesting..."):
        # 1. RAINFALL (CHIRPS)
        rain_dataset = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD") \
            .filterDate(p['start'], p['end']) \
            .select('precipitation')
        
        if rain_dataset.size().getInfo() > 0:
            rain_mean = rain_dataset.mean().clip(roi)
            min_rain, max_rain = 50, 800
            rain_norm = rain_mean.clamp(min_rain, max_rain).unitScale(min_rain, max_rain)
        else:
            st.warning("Rainfall data unavailable for range. Using placeholder.")
            rain_norm = ee.Image(0.5).clip(roi)
            rain_mean = rain_norm

        # 2. SLOPE (NASA DEM)
        dem = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
        slope = ee.Terrain.slope(dem).clip(roi)
        # 0-5 deg is best. Invert: 0->1, 30->0
        slope_norm = slope.clamp(0, 30).unitScale(0, 30)
        slope_score = ee.Image(1).subtract(slope_norm) 

        # 3. LULC (ESA WorldCover)
        lulc = ee.Image("ESA/WorldCover/v100/2020").select('Map').clip(roi)
        # High score: Bare(60), Grass(30), Shrub(20). Low score: Built(50), Water(80)
        from_list = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]
        to_list   = [0.6, 0.8, 0.8, 0.7, 0.0, 1.0, 0.0, 0.1, 0.1, 0.1]
        lulc_score = lulc.remap(from_list, to_list).rename('lulc_score')

        # 4. SOIL (OpenLandMap)
        try:
            soil_clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0').mean().clip(roi)
            soil_score = soil_clay.clamp(0, 50).unitScale(0, 50)
        except:
            soil_score = ee.Image(0.5).clip(roi)

        # 5. WEIGHTED OVERLAY
        suitability = (rain_norm.multiply(p['w_rain'])) \
            .add(slope_score.multiply(p['w_slope'])) \
            .add(lulc_score.multiply(p['w_lulc'])) \
            .add(soil_score.multiply(p['w_soil']))

        # VISUALIZATION
        vis_params = {'min': 0, 'max': 0.8, 'palette': ['red', 'orange', 'yellow', 'lightgreen', 'darkgreen']}
        
        m.addLayer(rain_mean, {'min': 0, 'max': 200, 'palette': ['blue', 'purple']}, 'Rainfall (Raw)', False)
        m.addLayer(slope, {'min': 0, 'max': 30, 'palette': ['white', 'black']}, 'Slope (Raw)', False)
        m.addLayer(suitability, vis_params, 'RWH Suitability Index')

        legend_dict = {
            "Very High Suitability": "006400", 
            "High Suitability": "90EE90",      
            "Moderate Suitability": "FFFF00",  
            "Low Suitability": "FFA500",       
            "Unsuitable": "FF0000"             
        }
        m.add_legend(title="RWH Suitability", legend_dict=legend_dict)

        with col_res:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">🌧️ RWH ANALYTICS</div>', unsafe_allow_html=True)
            
            # Classification for stats
            suit_class = ee.Image(0).where(suitability.lt(0.2), 1) \
                .where(suitability.gte(0.2).And(suitability.lt(0.4)), 2) \
                .where(suitability.gte(0.4).And(suitability.lt(0.6)), 3) \
                .where(suitability.gte(0.6).And(suitability.lt(0.8)), 4) \
                .where(suitability.gte(0.8), 5).clip(roi)
            
            st.markdown('<div class="card-label">📊 SUITABLE AREA</div>', unsafe_allow_html=True)
            with st.spinner("Calculating potential area..."):
                df_area = calculate_area_by_class(suit_class, roi, 30)
                if not df_area.empty:
                    name_map = {"Class 1": "Unsuitable", "Class 2": "Low", "Class 3": "Moderate", "Class 4": "High", "Class 5": "Very High"}
                    df_area['Class'] = df_area['Class'].map(name_map).fillna(df_area['Class'])
                    st.dataframe(df_area, hide_index=True, use_container_width=True)

            st.markdown("---")
            if st.button("☁️ Export Suitability Map"):
                    ee.batch.Export.image.toDrive(
                    image=suitability, description=f"RWH_Suitability_{datetime.now().strftime('%Y%m%d')}", 
                    scale=30, region=roi, folder='GEE_Exports'
                ).start()
                    st.toast("Export started")
            
            st.markdown("---")
            map_title = st.text_input("Map Title", "RWH Site Suitability")
            if st.button("📷 Render Map (JPG)"):
                    with st.spinner("Generating Map..."):
                        buf = generate_static_map_display(
                            suitability, roi, vis_params, map_title, 
                            cmap_colors=vis_params['palette']
                        )
                        if buf:
                            st.download_button("⬇️ Save Image", buf, "Ni30_RWH.jpg", "image/jpeg", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with col_map:
        m.to_streamlit()
