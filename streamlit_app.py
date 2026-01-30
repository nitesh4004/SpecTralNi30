import streamlit as st
import ee
import json
import os
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

# --- ML/DL IMPORTS ---
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="SpecTralNi30 Analytics", 
    page_icon="🛰️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS STYLING (Scientific Light UI) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;600&display=swap');
    
    /* GLOBAL VARIABLES */
    :root {
        --bg-color: #ffffff;
        --card-bg: #f8f9fa; /* Light Gray for cards */
        --glass-border: 1px solid #e0e0e0;
        --accent-primary: #002D62; /* Oxford Blue */
        --accent-secondary: #0066b2; /* Lighter Blue */
        --text-primary: #1a202c; /* Near Black */
        --text-header: #002D62; /* Dark Blue for Headers */
    }

    /* BACKGROUND */
    .stApp { 
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
    }

    /* TYPOGRAPHY */
    h1, h2, h3, .title-font { 
        font-family: 'Rajdhani', sans-serif !important; 
        text-transform: uppercase; 
        letter-spacing: 0.5px;
        color: var(--text-header) !important;
    }
    
    p, label, .stMarkdown, div, span { 
        color: var(--text-primary) !important; 
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6; /* Very light gray */
        border-right: 1px solid #d1d5db;
    }
    
    /* WIDGET STYLING */
    .stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div, .stDateInput > div > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 4px;
        color: #000000 !important;
    }
    
    /* Widget Labels */
    .stSelectbox label, .stTextInput label, .stNumberInput label, .stDateInput label, .stSlider label, .stRadio label {
        color: #002D62 !important;
        font-weight: 600;
    }
    
    /* BUTTONS */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
        border: none;
        color: white !important;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 0.6rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 45, 98, 0.3);
    }
    div.stButton > button:first-child p {
        color: white !important;
    }

    /* CUSTOM HUD HEADER */
    .hud-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border-bottom: 2px solid #002D62;
        padding: 15px 25px;
        margin-bottom: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .hud-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.0rem;
        font-weight: 700;
        color: var(--accent-primary);
    }
    .hud-badge {
        background: rgba(0, 45, 98, 0.1);
        border: 1px solid var(--accent-primary);
        color: var(--accent-primary) !important;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }

    /* GLASS CARDS (Now clean cards) */
    .glass-card {
        background: var(--card-bg);
        border: var(--glass-border);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .card-label {
        font-family: 'Rajdhani', sans-serif;
        color: var(--accent-primary) !important;
        font-size: 1.0rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
        border-bottom: 2px solid rgba(0, 45, 98, 0.1);
        padding-bottom: 5px;
    }
    
    /* MAP CONTAINER */
    iframe {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* METRIC VALUE STYLING */
    .metric-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #000000 !important;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #4a5568 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AUTHENTICATION ---
def init_ee():
    # Prefer Streamlit secrets for deployment (guard if secrets file missing)
    secret_dict = None
    try:
        if "gcp_service_account" in st.secrets:
            secret_dict = dict(st.secrets["gcp_service_account"])
        elif st.secrets.get("type") == "service_account":
            # Allow secrets stored as a raw service account JSON
            secret_dict = dict(st.secrets)
    except Exception:
        secret_dict = None

    if secret_dict:
        try:
            service_account = secret_dict.get("client_email", "")
            project_id = (
                secret_dict.get("project_id", "")
                or st.secrets.get("ee_project", "")
                or os.getenv("EE_PROJECT", "")
            )
            credentials = ee.ServiceAccountCredentials(
                service_account, key_data=json.dumps(secret_dict)
            )
            ee.Initialize(credentials, project=project_id or None)
            return
        except Exception as e:
            raise RuntimeError(
                "Service account authentication failed. Verify your Streamlit Secrets "
                "format and that the service account is added in Earth Engine Permissions."
            ) from e

    # Local development fallback
    project_id = (
        st.session_state.get("ee_project")
        or os.getenv("EE_PROJECT", "")
        or "ee-niteshswansat"
    )
    if not project_id:
        st.warning("Google Earth Engine now requires a Cloud Project ID.")
        project_id = st.text_input(
            "GEE Project ID",
            value="",
            help="Use your Google Cloud project ID (not name).",
        )
        if project_id:
            st.session_state["ee_project"] = project_id.strip()
            st.rerun()
        st.stop()
    ee.Initialize(project=project_id)

try:
    init_ee()
except Exception as e:
    st.error(
        "⚠️ Authentication Error. Run `earthengine authenticate` in your terminal, "
        "then restart the app. For Streamlit Cloud, add secrets.\n\n"
        f"Details: {e}"
    )
    st.stop()

# --- SESSION STATE INITIALIZATION ---
if 'calculated' not in st.session_state: st.session_state['calculated'] = False
if 'dates' not in st.session_state: st.session_state['dates'] = []
if 'roi' not in st.session_state: st.session_state['roi'] = None
if 'mode' not in st.session_state: st.session_state['mode'] = 'Spectral Monitor'
if 'basemap' not in st.session_state: st.session_state['basemap'] = None

# Visualization States - Defaults
if 'vis_min' not in st.session_state: st.session_state['vis_min'] = 0.0
if 'vis_max' not in st.session_state: st.session_state['vis_max'] = 1.0
if 'last_calc_key' not in st.session_state: st.session_state['last_calc_key'] = None

# --- 4. EXTENDED COLOR PALETTES ---
def get_palettes():
    return {
        "Red-Yellow-Green (Vegetation)": ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'],
        "Blue-White-Green (Water/Veg)": ['#0000ff', '#ffffff', '#008000'],
        "Blue-Yellow-Red (Thermal)": ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'],
        "Viridis (Sequential)": ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
        "Magma (Sequential)": ['#000004', '#140e36', '#3b0f70', '#641a80', '#8c2981', '#b73779', '#de4968', '#f7705c', '#fe9f6d', '#fcfdbf'],
        "Inferno (Sequential)": ['#000004', '#160b39', '#420a68', '#6a176e', '#932667', '#bc3754', '#dd513a', '#f37819', '#fca50a', '#f6d746'],
        "Plasma (Sequential)": ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'],
        "Turbo (Rainbow Enhanced)": ['#30123b', '#466be3', '#28bbec', '#32f197', '#a2fc3c', '#f2f221', '#fc8961', '#cf2547', '#7a0403'],
        "Ocean (Water Depth)": ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58'],
        "Terrain (Elevation)": ['#006400', '#32CD32', '#FFFF00', '#DAA520', '#8B4513', '#A0522D', '#D2691E', '#CD853F', '#F4A460', '#DEB887', '#D3D3D3', '#FFFFFF'],
        "Greyscale": ['#000000', '#FFFFFF']
    }

# --- 5. HELPER FUNCTIONS ---
def get_basemap_options():
    basemaps = getattr(geemap, "basemaps", {}) or {}
    candidates = ["HYBRID", "SATELLITE", "TERRAIN", "ROADMAP", "Esri.WorldImagery", "OpenStreetMap"]
    options = [name for name in candidates if name in basemaps]
    return options if options else ["OpenStreetMap"]

def resolve_basemap(preferred="HYBRID"):
    basemaps = getattr(geemap, "basemaps", {}) or {}
    if preferred in basemaps:
        return preferred
    for candidate in ("Esri.WorldImagery", "OpenStreetMap", "SATELLITE", "TERRAIN", "ROADMAP"):
        if candidate in basemaps:
            return candidate
    return "OpenStreetMap"

def create_map(height):
    options = get_basemap_options()
    current = st.session_state.get('basemap') or resolve_basemap()
    if current not in options:
        current = resolve_basemap()
    return geemap.Map(height=height, basemap=current)

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

def preprocess_landsat(img):
    opticalBands = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = img.select('ST_B.*').multiply(0.00341802).add(149.0)
    return img.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

def rename_landsat_bands(img):
    # Select optical and rename
    optical = img.select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    )
    # KEEP THERMAL BAND (ST_B10)
    thermal = img.select(['ST_B10'])
    return optical.addBands(thermal)

def compute_index(img, platform, index, formula=None):
    if platform == "Sentinel-2 (Optical)":
        if index == '🛠️ Custom (Band Math)':
            map_b = {
                'B1':img.select('B1'), 'B2':img.select('B2'), 'B3':img.select('B3'), 'B4':img.select('B4'), 
                'B5':img.select('B5'), 'B6':img.select('B6'), 'B7':img.select('B7'),
                'B8':img.select('B8'), 'B8A':img.select('B8A'), 
                'B11':img.select('B11'), 'B12':img.select('B12')
            }
            return img.expression(formula, map_b).rename('Custom')
        map_i = {'NDVI': ['B8','B4'], 'GNDVI': ['B8','B3'], 'NDWI (Water)': ['B3','B8'], 'NDMI': ['B8','B11']}
        if index in map_i: return img.normalizedDifference(map_i[index]).rename(index.split()[0])

    elif "Landsat" in platform:
        if index == 'LST (Thermal)':
            # Convert Kelvin to Celsius: K - 273.15
            return img.select('ST_B10').subtract(273.15).rename('LST')

        if index == '🛠️ Custom (Band Math)':
            map_b = {'B1':img.select('B1'), 'B2':img.select('B2'), 'B3':img.select('B3'), 'B4':img.select('B4'), 'B5':img.select('B5'), 'B6':img.select('B6'), 'B7':img.select('B7')}
            return img.expression(formula, map_b).rename('Custom')
        
        map_i = {'NDVI': ['B5','B4'], 'GNDVI': ['B5','B3'], 'NDWI (Water)': ['B3','B5'], 'NDMI': ['B5','B6']}
        if index in map_i: return img.normalizedDifference(map_i[index]).rename(index.split()[0])

    elif platform == "Sentinel-1 (Radar)":
        if index == '🛠️ Custom (Band Math)':
            map_b = {'VV': img.select('VV'), 'VH': img.select('VH')}
            return img.expression(formula, map_b).rename('Custom')
        if index == 'VV': return img.select('VV')
        if index == 'VH': return img.select('VH')
        if index == 'VH/VV Ratio': return img.select('VH').subtract(img.select('VV')).rename('Ratio')
    return img.select(0)

# --- STATS CALCULATION (Dynamic Stretch) ---
def calculate_dynamic_stretch(image, roi, scale=30):
    """Calculates 2nd and 98th percentile for optimal stretching"""
    try:
        stats = image.reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=roi,
            scale=scale,
            maxPixels=1e9,
            bestEffort=True
        ).getInfo()
        
        vals = list(stats.values())
        if len(vals) >= 2:
            return min(vals), max(vals)
        return 0.0, 1.0 # Fallback
    except Exception as e:
        print(f"Stats Error: {e}")
        return 0.0, 1.0

# --- ROI STATISTICS FUNCTION (For Display) ---
def calculate_roi_stats_display(image, roi, scale=30):
    reducer = ee.Reducer.minMax() \
        .combine(reducer2=ee.Reducer.mean(), sharedInputs=True) \
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
    try:
        stats = image.reduceRegion(
            reducer=reducer, geometry=roi, scale=scale, maxPixels=1e9, bestEffort=True
        ).getInfo()
        res = {}
        keys = list(stats.keys())
        for k in keys:
            if 'mean' in k: res['mean'] = stats[k]
            if 'min' in k: res['min'] = stats[k]
            if 'max' in k: res['max'] = stats[k]
            if 'stdDev' in k: res['std'] = stats[k]
        return res
    except: return None

# --- LULC SPECIFIC FUNCTIONS ---
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0) \
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

def add_lulc_indices(image):
    nir = image.select("B8")
    red = image.select("B4")
    green = image.select("B3")
    blue = image.select("B2")
    swir1 = image.select("B11")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    gndvi = nir.subtract(green).divide(nir.add(green)).rename("GNDVI")
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {"NIR": nir, "RED": red, "BLUE": blue}
    ).rename("EVI")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")
    
    return image.addBands([ndvi, evi, gndvi, ndwi, ndmi])

def calculate_area_by_class(image, region, scale, class_names=None):
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
        
        name = f"Class {c_idx}"
        if class_names:
            if 0 <= c_idx < len(class_names): name = class_names[c_idx]
            else: name = f"Class {c_idx}"
        
        data.append({"Class": name, "Area (ha)": area_ha})
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values(by="Area (ha)", ascending=False)
        df["%"] = ((df["Area (ha)"] / total_area) * 100).round(1)
        df["Area (ha)"] = df["Area (ha)"].round(2)
        
    return df

def generate_static_map_display(image, roi, vis_params, title, cmap_colors=None, is_categorical=False, class_names=None):
    try:
        # 1. CALCULATE GEOMETRY & ASPECT RATIO
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

        # 2. PREPARE IMAGE FROM GEE
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
        
        if response.status_code != 200:
            st.error(f"GEE Server Error (Status {response.status_code})")
            return None
            
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            st.error(f"GEE Error: {response.text}")
            return None

        img_pil = Image.open(BytesIO(response.content))
        
        # 3. PLOT WITH WHITE BACKGROUND AND BLACK TEXT
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300, facecolor='#ffffff')
        ax.set_facecolor('#ffffff')
        
        extent = [min_lon, max_lon, min_lat, max_lat]
        
        im = ax.imshow(img_pil, extent=extent, aspect='auto')
        
        # Dark Blue Title
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#002D62')
        
        # Styling ticks for White background (Black text)
        ax.tick_params(colors='black', labelcolor='black', labelsize=10)
        ax.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.1)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        
        # 4. NORTH ARROW / DIRECTION MARKER
        ax.annotate('N', xy=(0.97, 0.95), xytext=(0.97, 0.88),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', edgecolor='black', width=4, headwidth=12, headlength=10),
                    ha='center', va='center', fontsize=16, fontweight='bold', color='black')

        # 5. SCALE BAR LOGIC
        try:
            center_lat = (min_lat + max_lat) / 2
            met_per_deg_lon = 111320 * np.cos(np.radians(center_lat))
            
            width_met = width_deg * met_per_deg_lon
            target_len_met = width_met / 5
            
            order = 10 ** np.floor(np.log10(target_len_met))
            nice_len_met = round(target_len_met / order) * order
            nice_len_deg = nice_len_met / met_per_deg_lon
            
            pad_x = width_deg * 0.05
            pad_y = height_deg * 0.05
            
            start_x = max_lon - pad_x - nice_len_deg
            start_y = min_lat + pad_y
            
            bar_height = height_deg * 0.015
            
            # Black scale bar rectangle
            rect = mpatches.Rectangle((start_x, start_y), nice_len_deg, bar_height, 
                                    linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)
            
            label = f"{int(nice_len_met/1000)} km" if nice_len_met >= 1000 else f"{int(nice_len_met)} m"
            
            # Text label above scale bar (Black)
            ax.text(start_x + nice_len_deg/2, start_y + bar_height + (height_deg*0.01), label, 
                    color='black', ha='center', va='bottom', fontsize=12, fontweight='bold')
        except:
            pass
        
        # 6. LEGEND LOGIC
        if is_categorical and class_names and 'palette' in vis_params:
            patches = []
            for name, color in zip(class_names, vis_params['palette']):
                patches.append(mpatches.Patch(color=color, label=name))
            legend = ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.08), 
                               frameon=False, title="Classes", ncol=min(len(class_names), 4))
            plt.setp(legend.get_title(), color='#002D62', fontweight='bold', fontsize=12)
            for text in legend.get_texts():
                text.set_color("black")
                text.set_fontsize(10)
                
        elif cmap_colors and 'min' in vis_params:
            cmap = mcolors.LinearSegmentedColormap.from_list("custom", cmap_colors)
            norm = mcolors.Normalize(vmin=vis_params['min'], vmax=vis_params['max'])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
            cbar = plt.colorbar(sm, cax=cax)
            cbar.ax.yaxis.set_tick_params(color='black')
            cbar.set_label('Value', color='black', fontsize=12)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black', fontsize=10)
        
        buf = BytesIO()
        # Save with white facecolor
        plt.savefig(buf, format='jpg', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        plt.close(fig)
        return buf
        
    except UnidentifiedImageError:
        st.error("Error: GEE returned a text error instead of an image. The computation region might be too large or complex.")
        return None
    except Exception as e:
        st.error(f"Map Generation Error: {e}")
        return None

# --- 6. SIDEBAR (CONTROL PANEL) ---
with st.sidebar:
    # --- LOGO INSERTION ---
    st.image("logo.png", use_container_width=True)
    
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="font-family: 'Rajdhani'; color: #002D62; margin:0;">SpecTralNi30</h2>
            <p style="font-size: 0.8rem; color: #0066b2; letter-spacing: 2px; margin:0; font-weight:600;">GEOSPATIAL CORE</p>
        </div>
    """, unsafe_allow_html=True)
    
    # MODE SELECTOR
    mode = st.radio("System Mode", ["Spectral Monitor", "LULC Classifier", "Geospatial-embeddings-use-cases", "Landslide Detection (SAR)"], index=0)
    st.session_state['mode'] = mode

    st.markdown("---")
    
    st.markdown("### 1. Basemap")
    basemap_options = get_basemap_options()
    default_basemap = st.session_state.get('basemap') or resolve_basemap()
    if default_basemap not in basemap_options:
        basemap_options = [default_basemap] + basemap_options
    basemap_choice = st.selectbox("Basemap", basemap_options, index=basemap_options.index(default_basemap))
    st.session_state['basemap'] = basemap_choice

    st.markdown("---")

    with st.container():
        st.markdown("### 2. Target Acquisition (ROI)")
        roi_method = st.radio("Selection Mode", ["Upload KML", "Point & Buffer", "Manual Coordinates"], label_visibility="collapsed")
        
        new_roi = None
        if roi_method == "Upload KML":
            kml = st.file_uploader("Drop KML File", type=['kml'])
            if kml:
                kml.seek(0)
                new_roi = parse_kml(kml.read())
        elif roi_method == "Point & Buffer":
            c1, c2 = st.columns([1, 1])
            # High precision coordinates, wide range
            lat = c1.number_input("Lat", value=20.59, min_value=-90.0, max_value=90.0, format="%.6f")
            lon = c2.number_input("Lon", value=78.96, min_value=-180.0, max_value=180.0, format="%.6f")
            # Radius in METERS now, no max limit
            rad = st.number_input("Radius (meters)", value=5000, min_value=10, step=10)
            if lat and lon: 
                new_roi = ee.Geometry.Point([lon, lat]).buffer(rad).bounds()
        elif roi_method == "Manual Coordinates":
            c1, c2 = st.columns(2)
            # High precision coordinates
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
    
    # --- MODE SPECIFIC SETTINGS ---
    # Init vars
    rf_trees, svm_kernel, svm_gamma, gtb_trees = 100, 'RBF', 0.5, 100
    ann_layers, ann_iter, ann_alpha = (100, 100), 500, 0.0001
    model_choice = "Random Forest"
    embedding_year = 2023
    embedding_task = "LULC (ESA Labels)"
    pre_start, pre_end, post_start, post_end = None, None, None, None
    slide_thresh, slope_thresh = 2.0, 15

    if mode == "Spectral Monitor":
        st.markdown("### 2. Sensor Config")
        platform = st.selectbox("Satellite Network", [
            "Sentinel-2 (Optical)", "Landsat 9 (Optical)", "Landsat 8 (Optical)", "Sentinel-1 (Radar)"
        ])
        
        is_optical = "Optical" in platform
        formula, orbit = "", "BOTH"
        
        # --- SENSOR CONFIG ---
        if is_optical:
            # ADDED LST OPTION FOR LANDSAT
            if "Landsat" in platform:
                idx = st.selectbox("Spectral Product", ['NDVI', 'GNDVI', 'NDWI (Water)', 'NDMI', 'LST (Thermal)', '🛠️ Custom (Band Math)'])
                if 'Custom' in idx:
                    def_form = "(B5-B4)/(B5+B4)" 
                    formula = st.text_input("Math Expression", def_form)
            else:
                idx = st.selectbox("Spectral Product", ['NDVI', 'GNDVI', 'NDWI (Water)', 'NDMI', '🛠️ Custom (Band Math)'])
                if 'Custom' in idx:
                    def_form = "(B8-B4)/(B8+B4)"
                    formula = st.text_input("Math Expression", def_form)
            
            cloud = st.slider("Cloud Tolerance %", 0, 30, 10)
        else:
            idx = st.selectbox("Polarization", ['VV', 'VH', 'VH/VV Ratio', '🛠️ Custom (Band Math)'])
            if 'Custom' in idx:
                formula = st.text_input("Expression", "VH/VV")
            orbit = st.radio("Pass Direction", ["DESCENDING", "ASCENDING", "BOTH"])
            cloud = 0

        # --- VISUALIZATION CONTROL (Dynamic) ---
        st.markdown("### 3. Visualization Settings")
        
        palettes = get_palettes()
        pal_name = st.selectbox("Color Palette", list(palettes.keys()), index=0)
        cur_palette = palettes[pal_name]
        
        st.caption("Value Range (Stretch)")
        
        # Manual Inputs linked to Session State
        # We read from session state, defaulting to 0.0/1.0 if not yet set
        c1, c2 = st.columns(2)
        
        # The key logic here allows us to update these widgets from the main script
        vmin = c1.number_input("Min", value=float(st.session_state['vis_min']), key='vis_min_input')
        vmax = c2.number_input("Max", value=float(st.session_state['vis_max']), key='vis_max_input')
        
        # Sync: If user manually changes these widgets, update session state
        st.session_state['vis_min'] = vmin
        st.session_state['vis_max'] = vmax
        
        st.markdown("---")

    elif mode == "LULC Classifier": # LULC MODE
        st.markdown("### 2. ML Architecture")
        
        # 1. Model Selector
        model_choice = st.selectbox(
            "Select Classifier", 
            [
                "Google Dynamic World (Pre-trained Deep Learning)",
                "Artificial Neural Network (MLP)", 
                "Random Forest", 
                "Support Vector Machine (SVM)", 
                "Gradient Tree Boost", 
                "CART (Decision Tree)", 
                "Naive Bayes"
            ]
        )

        # 2. Dynamic Hyperparameters
        if model_choice == "Google Dynamic World (Pre-trained Deep Learning)":
            st.info("🌍 Uses Google's pre-trained deep learning model (FCN) on Sentinel-2 data. 10m global resolution.")
            cloud = st.slider("Cloud Masking % (For S2 composite)", 0, 30, 20)

        elif model_choice == "Artificial Neural Network (MLP)":
            st.info("🧠 Hybrid Execution: Training runs locally on Streamlit using Scikit-Learn. Map visualization uses a Random Forest proxy.")
            hidden_layers = st.text_input("Hidden Layers (e.g. 100,50)", "100,100")
            ann_layers = tuple(map(int, hidden_layers.split(',')))
            ann_iter = st.slider("Max Iterations", 200, 1000, 500)
            ann_alpha = st.number_input("Alpha (L2)", value=0.0001, format="%.4f")
            cloud = st.slider("Cloud Masking %", 0, 30, 20)
            split_ratio = st.slider("Train/Validation Split", 0.5, 0.9, 0.8)

        elif model_choice == "Random Forest":
            rf_trees = st.slider("Number of Trees", 10, 500, 150)
            cloud = st.slider("Cloud Masking %", 0, 30, 20)
            split_ratio = st.slider("Train/Validation Split", 0.5, 0.9, 0.8)
        
        elif model_choice == "Support Vector Machine (SVM)":
            svm_kernel = st.selectbox("Kernel Type", ["RBF", "LINEAR", "POLY"])
            svm_gamma = st.number_input("Gamma (RBF)", value=0.5)
            cloud = st.slider("Cloud Masking %", 0, 30, 20)
            split_ratio = st.slider("Train/Validation Split", 0.5, 0.9, 0.8)

        else: # Other models
            st.caption("Standard GEE classifiers.")
            cloud = st.slider("Cloud Masking %", 0, 30, 20)
            split_ratio = st.slider("Train/Validation Split", 0.5, 0.9, 0.8)
            
        st.markdown("---")

    elif mode == "Geospatial-embeddings-use-cases":
        st.markdown("### 2. AI Embeddings Task")
        embedding_task = st.selectbox("Select Task", [
            "LULC (Supervised with ESA Labels)", 
            "Alpha Earth: Transfer Learning (Dynamic World)",
            "Water/Change Detection (Unsupervised)"
        ])
        embedding_year = st.slider("Target Year", 2017, 2024, 2023)
        st.caption(f"Using Google Satellite Embeddings (V1) for {embedding_year}")
        cloud = 0 # Embeddings don't use this directly in the same way
        
        st.markdown("---")

    elif mode == "Landslide Detection (SAR)":
        st.markdown("### 2. Event Configuration")
        st.caption("Using Sentinel-1 (Radar) Change Detection")
        
        st.markdown("#### Pre-Event Baseline")
        c1, c2 = st.columns(2)
        pre_start = c1.date_input("From", datetime.now()-timedelta(90), key="pre_s")
        pre_end = c2.date_input("To", datetime.now()-timedelta(30), key="pre_e")
        
        st.markdown("#### Post-Event Analysis")
        c3, c4 = st.columns(2)
        post_start = c3.date_input("From", datetime.now()-timedelta(29), key="post_s")
        post_end = c4.date_input("To", datetime.now(), key="post_e")

        st.markdown("#### Detection sensitivity")
        slide_thresh = st.slider("Backscatter Change (dB)", 1.0, 5.0, 2.5, 0.1, help="Higher value = Stricter detection (less noise)")
        slope_thresh = st.slider("Min Slope (Degrees)", 0, 30, 15, help="Exclude flat areas (agri fields)")
        cloud = 0
        
        st.markdown("---")

    # Common Temporal Window (Only for non-SAR/Embeddings modes)
    if mode not in ["Geospatial-embeddings-use-cases", "Landslide Detection (SAR)"]:
        st.markdown("---")
        st.markdown("### 3. Temporal Window")
        c1, c2 = st.columns(2)
        start = c1.date_input("Start", datetime.now()-timedelta(60))
        end = c2.date_input("End", datetime.now())
    else:
        # Dummy vars for consistency
        start, end = datetime.now(), datetime.now()

    st.markdown("###")
    if st.button("INITIALIZE SCAN 🚀"):
        if st.session_state['roi']:
            params = {
                'calculated': True, 
                'cloud': cloud,
                'model_choice': model_choice,
                'rf_trees': rf_trees,
                'svm_kernel': svm_kernel,
                'svm_gamma': svm_gamma,
                'gtb_trees': gtb_trees,
                'split_ratio': split_ratio if 'split_ratio' in locals() else 0.8
            }
            
            if mode == "Landslide Detection (SAR)":
                params.update({
                    'pre_start': pre_start.strftime("%Y-%m-%d"),
                    'pre_end': pre_end.strftime("%Y-%m-%d"),
                    'post_start': post_start.strftime("%Y-%m-%d"),
                    'post_end': post_end.strftime("%Y-%m-%d"),
                    'slide_thresh': slide_thresh,
                    'slope_thresh': slope_thresh
                })
            elif mode != "Geospatial-embeddings-use-cases":
                params.update({
                    'start': start.strftime("%Y-%m-%d"), 
                    'end': end.strftime("%Y-%m-%d")
                })
            else:
                params.update({
                    'embedding_year': embedding_year,
                    'embedding_task': embedding_task
                })

            if model_choice == "Artificial Neural Network (MLP)":
                params.update({'ann_layers': ann_layers, 'ann_iter': ann_iter, 'ann_alpha': ann_alpha})
            
            if mode == "Spectral Monitor":
                params.update({
                    'platform': platform, 'idx': idx, 'formula': formula, 
                    'orbit': orbit, 'palette': cur_palette
                    # Note: vmin/vmax are pulled from session state dynamically in the main loop
                })
                
            st.session_state.update(params)
            st.session_state['dates'] = [] 
        else:
            st.error("❌ Error: ROI not defined.")

# --- 7. MAIN CONTENT ---
st.markdown("""
<div class="hud-header">
    <div>
        <div class="hud-title">SpecTralNi30 ANALYTICS</div>
        <div style="color:#002D62; font-size:0.9rem; font-weight:600;">""" + st.session_state['mode'].upper() + """</div>
    </div>
    <div style="text-align:right;">
        <span class="hud-badge">SYSTEM ONLINE</span>
        <div style="font-family:'Rajdhani'; font-size:1.2rem; margin-top:5px; color:#002D62; font-weight:700;">""" + datetime.now().strftime("%H:%M UTC") + """</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state['calculated']:
    # Welcome View
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:40px;">
        <h2 style="color:#002D62;">📡 WAITING FOR INPUT</h2>
        <p style="color:#1a202c; margin-bottom:20px;">Configure the sensor parameters and region of interest in the sidebar panel.</p>
    </div>
    """, unsafe_allow_html=True)
    # FORCED GOOGLE HYBRID
    m = create_map(height=500)
    if st.session_state['roi']:
        m.centerObject(st.session_state['roi'], 12)
        m.addLayer(ee.Image().paint(st.session_state['roi'], 2, 3), {'palette': '#002D62'}, 'Target ROI')
    m.to_streamlit()

else:
    roi = st.session_state['roi']
    p = st.session_state
    
    # ==========================================
    # MODE 1: SPECTRAL MONITOR
    # ==========================================
    if p['mode'] == "Spectral Monitor":
        with st.spinner("🛰️ Establishing Uplink... Processing Earth Engine Data..."):
            if p['platform'] == "Sentinel-2 (Optical)":
                col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                       .filterBounds(roi).filterDate(p['start'], p['end'])
                       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', p['cloud'])))
                processed = col 
            elif "Landsat" in p['platform']:
                col_raw = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") if "Landsat 9" in p['platform'] else ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                col = (col_raw.filterBounds(roi).filterDate(p['start'], p['end'])
                       .filter(ee.Filter.lt('CLOUD_COVER', p['cloud'])))
                processed = col.map(preprocess_landsat).map(rename_landsat_bands)
            else:
                col = (ee.ImageCollection('COPERNICUS/S1_GRD')
                       .filterBounds(roi).filterDate(p['start'], p['end'])
                       .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')))
                if p['orbit'] != "BOTH": col = col.filter(ee.Filter.eq('orbitProperties_pass', p['orbit']))
                processed = col
            
            if not st.session_state['dates']:
                cnt = processed.size().getInfo()
                if cnt > 0:
                    dates_list = processed.aggregate_array('system:time_start').map(
                        lambda t: ee.Date(t).format('YYYY-MM-dd')).distinct().sort()
                    st.session_state['dates'] = dates_list.slice(0, 50).getInfo()
                else:
                    st.error(f"⚠️ Signal Lost: No images found.")
                    st.stop()

        if st.session_state['dates']:
            dates = st.session_state['dates']
            col_map, col_data = st.columns([3, 1])
            
            with col_data:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">📅 ACQUISITION DATE</div>', unsafe_allow_html=True)
                sel_date = st.selectbox("Select Timestamp", dates, index=len(dates)-1, label_visibility="collapsed")
                st.caption(f"{len(dates)} Scenes Available")
                st.markdown('</div>', unsafe_allow_html=True)

                d_s = sel_date
                d_e = (datetime.strptime(sel_date, "%Y-%m-%d") + timedelta(1)).strftime("%Y-%m-%d")
                
                # Fetch Base Image (Mosaic for the day)
                base_img = processed.filterDate(d_s, d_e).mosaic().clip(roi)
                
                # Calculate Index
                index_img = compute_index(base_img, p['platform'], p['idx'], p['formula'])
                
                # --- AUTO-CALCULATION LOGIC ---
                # We construct a unique key for the current configuration (Index + Date + ROI)
                # If this changes, we calculate new Stats (p2, p98) and update the session state
                
                roi_hash = str(roi.getInfo()) # ROI might change slightly
                current_calc_key = f"{p['idx']}_{sel_date}_{roi_hash}"
                
                if st.session_state.get('last_calc_key') != current_calc_key:
                    with st.spinner("📏 Calculating dynamic stretch based on ROI statistics..."):
                        p2, p98 = calculate_dynamic_stretch(index_img, roi)
                        
                        # Update the SESSION STATE directly
                        st.session_state['vis_min'] = p2
                        st.session_state['vis_max'] = p98
                        st.session_state['last_calc_key'] = current_calc_key
                        
                        # RERUN to update the sidebar widgets with these new values
                        st.rerun()

                # --- STATISTICS CARD ---
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-label">📊 STATISTICAL ANALYSIS ({p["idx"]})</div>', unsafe_allow_html=True)
                with st.spinner("Calculating display stats..."):
                    # This calculates Mean/Std/Min/Max for display (different from the p2/p98 stretch)
                    stats = calculate_roi_stats_display(index_img, roi)
                    if stats:
                        c_s1, c_s2 = st.columns(2)
                        c_s1.markdown(f"**Mean:** {stats['mean']:.3f}")
                        c_s1.markdown(f"**Std:** {stats['std']:.3f}")
                        c_s2.markdown(f"**Min:** {stats['min']:.3f}")
                        c_s2.markdown(f"**Max:** {stats['max']:.3f}")
                    else:
                        st.warning("Stats unavailable")
                st.markdown('</div>', unsafe_allow_html=True)

                # --- LAYER PREPARATION ---
                vis_params = {'min': st.session_state['vis_min'], 'max': st.session_state['vis_max'], 'palette': p['palette']}
                
                # --- DOWNLOAD & MAP ---
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">💾 DATA EXPORT</div>', unsafe_allow_html=True)
                try:
                    url = index_img.getDownloadURL({'scale': 30 if "Landsat" in p['platform'] else 10, 'region': roi, 'name': f"{p['idx']}_{sel_date}"})
                    st.markdown(f"<a href='{url}' style='color:#002D62; font-weight:bold; text-decoration:none;'>🔗 Download GeoTIFF ({p['idx']})</a>", unsafe_allow_html=True)
                except: st.caption("Region too large for instant link.")
                
                st.markdown("---")
                if st.button("📷 Render Map (JPG)", use_container_width=True):
                    with st.spinner("Rendering..."):
                        buf = generate_static_map_display(
                            index_img, roi, vis_params, 
                            f"{p['idx']} | {sel_date}", 
                            cmap_colors=p['palette'] if 'palette' in vis_params else None
                        )
                        if buf:
                            st.download_button("⬇️ Save Image", buf, f"Ni30_Map_{sel_date}.jpg", "image/jpeg", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_map:
                # FORCED GOOGLE HYBRID
                m = create_map(height=700)
                m.centerObject(roi, 13)
                m.addLayer(index_img, vis_params, p['idx'])
                m.add_colorbar(vis_params, label=p['idx'], layer_name="Legend")
                m.to_streamlit()

    # ==========================================
    # MODE 2: MULTI-MODEL LULC CLASSIFIER
    # ==========================================
    elif p['mode'] == "LULC Classifier":
        
        # 1. SETUP MAP
        col_map, col_res = st.columns([3, 1])
        # FORCED GOOGLE HYBRID
        m = create_map(height=700)
        m.centerObject(roi, 13)
        
        # S2 Background for all modes
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate(p['start'], p['end'])
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", p['cloud']))
            .map(mask_s2_clouds) 
        )
        if s2_collection.size().getInfo() > 0:
            s2_median = s2_collection.median().clip(roi)
            rgb_vis = {'min': 0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2']}
            m.addLayer(s2_median, rgb_vis, 'RGB Composite')
        else:
            st.warning("No clear Sentinel-2 background available.")

        # --- BRANCH A: PRE-TRAINED DEEP LEARNING (DYNAMIC WORLD) ---
        if p['model_choice'] == "Google Dynamic World (Pre-trained Deep Learning)":
            with st.spinner("🧠 Querying Google Dynamic World V1 (Deep Learning)..."):
                
                # Filter DW Collection
                dw_col = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
                    .filterBounds(roi) \
                    .filterDate(p['start'], p['end'])
                
                if dw_col.size().getInfo() == 0:
                    st.error("No Dynamic World data found for this date/region.")
                    st.stop()
                    
                # Create Composite (Mode of labels)
                # The 'label' band contains the class index with highest probability
                dw_image = dw_col.select('label').mode().clip(roi)
                
                # DW Specific Visualization
                dw_vis = {
                    "min": 0, "max": 8,
                    "palette": [
                        '#419bdf', '#397d49', '#88b053', '#7a87c6', '#e49635',
                        '#dfc35a', '#c4281b', '#a59b8f', '#b39fe1'
                    ]
                }
                
                dw_names = ['Water', 'Trees', 'Grass', 'Flooded Veg', 'Crops', 
                            'Shrub/Scrub', 'Built', 'Bare', 'Snow/Ice']
                
                m.addLayer(dw_image, dw_vis, "Dynamic World LULC")
                m.add_legend(title="Dynamic World Classes", legend_dict=dict(zip(dw_names, dw_vis['palette'])))
                
                # Metrics Display (N/A for Pre-trained)
                with col_res:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-label">🧠 MODEL METRICS</div>', unsafe_allow_html=True)
                    st.info("Pre-trained Global Model")
                    
                    st.markdown('<div class="card-label">📊 AREA STATS</div>', unsafe_allow_html=True)
                    with st.spinner("Calculating areas..."):
                          df_area = calculate_area_by_class(dw_image, roi, 10, dw_names)
                          if not df_area.empty:
                              st.dataframe(df_area, hide_index=True, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown('<div class="card-label">💾 EXPORT RESULT</div>', unsafe_allow_html=True)
                    if st.button("☁️ Save to Drive"):
                        ee.batch.Export.image.toDrive(
                            image=dw_image, description=f"DW_LULC_{datetime.now().strftime('%Y%m%d')}", 
                            scale=10, region=roi, folder='GEE_Exports'
                        ).start()
                        st.toast("Export Started to GDrive")
                    
                    st.markdown("---")
                    
                    # USER INPUT FOR MAP TITLE
                    map_title = st.text_input("Map Title", "Dynamic World LULC Analysis")
                    
                    if st.button("📷 Render Map (JPG)"):
                        with st.spinner("Generating Map..."):
                            buf = generate_static_map_display(
                                dw_image, roi, dw_vis, map_title, 
                                is_categorical=True, class_names=dw_names
                            )
                            if buf:
                                st.download_button("⬇️ Save Image", buf, "Ni30_DW_LULC.jpg", "image/jpeg", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        # --- BRANCH B: CUSTOM TRAINING (ANN, RF, SVM) ---
        else:
            TRAIN_URL = "https://raw.githubusercontent.com/nitesh4004/Geospatial-Ni30/main/sentinel2_lulc_synthetic.csv"
            
            with st.spinner(f"🧠 Training {p['model_choice']}..."):
                try:
                    df = pd.read_csv(TRAIN_URL)
                    
                    # Feature Engineering
                    df['B2'] = df['B2'] / 10000.0
                    df['B3'] = df['B3'] / 10000.0
                    df['B4'] = df['B4'] / 10000.0
                    df['B8'] = df['B8'] / 10000.0
                    df['B11'] = df['B11'] / 10000.0
                    
                    df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'])
                    df['GNDVI'] = (df['B8'] - df['B3']) / (df['B8'] + df['B3'])
                    df['EVI'] = 2.5 * ((df['B8'] - df['B4']) / (df['B8'] + 6 * df['B4'] - 7.5 * df['B2'] + 1))
                    df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'])
                    df['NDMI'] = (df['B8'] - df['B11']) / (df['B8'] + df['B11'])
                    
                    class_names = ['Water', 'Forest', 'Cropland', 'Built-up', 'Barren', 'Rock/Exposed']
                    class_lut = {name: i for i, name in enumerate(class_names)}
                    
                    if "Class" in df.columns:
                        df["class_val"] = df["Class"].map(class_lut)
                    
                    df = df.dropna(subset=["class_val"])
                    input_bands = ["NDVI", "EVI", "GNDVI", "NDWI", "NDMI"]
                    
                    # --- MODEL TRAINING ---
                    if p['model_choice'] == "Artificial Neural Network (MLP)":
                        # LOCAL ANN TRAINING
                        X = df[input_bands].values
                        y = df['class_val'].values.astype(int)
                        
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-p['split_ratio']), random_state=42)
                        
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        mlp = MLPClassifier(
                            hidden_layer_sizes=p['ann_layers'], max_iter=p['ann_iter'], 
                            alpha=p['ann_alpha'], activation='relu', solver='adam', random_state=42
                        )
                        mlp.fit(X_train_scaled, y_train)
                        
                        y_pred = mlp.predict(X_test_scaled)
                        overall_accuracy = accuracy_score(y_test, y_pred)
                        kappa = cohen_kappa_score(y_test, y_pred)
                        
                        proxy_name = "Visual Proxy (Random Forest)"
                        # Train proxy for map
                        features = []
                        for i, row in df.iterrows():
                             features.append(ee.Feature(None, {
                                'NDVI': row['NDVI'], 'EVI': row['EVI'], 'GNDVI': row['GNDVI'], 
                                'NDWI': row['NDWI'], 'NDMI': row['NDMI'], 'class': int(row['class_val'])
                            }))
                        fc_raw = ee.FeatureCollection(features)
                        trained_classifier = ee.Classifier.smileRandomForest(100).train(fc_raw, "class", input_bands)

                    else:
                        # GEE NATIVE MODELS
                        features = []
                        for i, row in df.iterrows():
                            features.append(ee.Feature(None, {
                                'NDVI': row['NDVI'], 'EVI': row['EVI'], 'GNDVI': row['GNDVI'], 
                                'NDWI': row['NDWI'], 'NDMI': row['NDMI'], 'class': int(row['class_val'])
                            }))
                        
                        fc_raw = ee.FeatureCollection(features)
                        fc_with_random = fc_raw.randomColumn()
                        training_fc = fc_with_random.filter(ee.Filter.lt('random', p['split_ratio']))
                        validation_fc = fc_with_random.filter(ee.Filter.gte('random', p['split_ratio']))

                        if p['model_choice'] == "Random Forest":
                            classifier_inst = ee.Classifier.smileRandomForest(numberOfTrees=p['rf_trees'], seed=42)
                        elif p['model_choice'] == "Support Vector Machine (SVM)":
                            classifier_inst = ee.Classifier.libsvm(kernelType=p['svm_kernel'], gamma=p['svm_gamma'], cost=10)
                        elif p['model_choice'] == "Gradient Tree Boost":
                            classifier_inst = ee.Classifier.smileGradientTreeBoost(numberOfTrees=p['gtb_trees'], shrinkage=0.005, samplingRate=0.7, seed=42)
                        elif p['model_choice'] == "CART (Decision Tree)":
                            classifier_inst = ee.Classifier.smileCart()
                        elif p['model_choice'] == "Naive Bayes":
                            classifier_inst = ee.Classifier.smileNaiveBayes()

                        trained_classifier = classifier_inst.train(training_fc, "class", input_bands)
                        
                        validated = validation_fc.classify(trained_classifier)
                        error_matrix = validated.errorMatrix('class', 'classification')
                        overall_accuracy = error_matrix.accuracy().getInfo()
                        kappa = error_matrix.kappa().getInfo()
                        proxy_name = p['model_choice']

                except Exception as e:
                    st.error(f"❌ Processing Error: {e}")
                    st.stop()
                
                # Classify Map
                if s2_collection.size().getInfo() > 0:
                    indices_img = add_lulc_indices(s2_median)
                    lulc_class = indices_img.select(input_bands).classify(trained_classifier)
                    
                    # Vis
                    lulc_palette = ['#0000FF', '#006400', '#b2df8a', '#FF0000', '#8B4513', '#808080']
                    vis_params = {"min": 0, "max": 5, "palette": lulc_palette}
                    
                    m.addLayer(lulc_class, vis_params, f"LULC: {proxy_name}")
                    m.add_legend(title="LULC Classes", legend_dict=dict(zip(class_names, lulc_palette)))
                    
                    # Metrics Display
                    with col_res:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown('<div class="card-label">🧠 MODEL METRICS</div>', unsafe_allow_html=True)
                        st.success(f"Arch: {p['model_choice']}")
                        
                        c_a, c_b = st.columns(2)
                        c_a.markdown(f"""<div class="metric-value">{overall_accuracy:.2%}</div><div class="metric-sub">Accuracy</div>""", unsafe_allow_html=True)
                        c_b.markdown(f"""<div class="metric-value">{kappa:.3f}</div><div class="metric-sub">Kappa</div>""", unsafe_allow_html=True)
                        
                        st.markdown('<div class="card-label">📊 AREA STATS</div>', unsafe_allow_html=True)
                        with st.spinner("Calculating areas..."):
                             df_area = calculate_area_by_class(lulc_class, roi, 10, class_names)
                             if not df_area.empty:
                                 st.dataframe(df_area, hide_index=True, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown('<div class="card-label">💾 EXPORT</div>', unsafe_allow_html=True)
                        if st.button("☁️ Save to Drive"):
                            ee.batch.Export.image.toDrive(
                                image=lulc_class, description=f"LULC_Custom_{datetime.now().strftime('%Y%m%d')}", 
                                scale=10, region=roi, folder='GEE_Exports'
                            ).start()
                            st.toast("Export Started")

                        st.markdown("---")
                        
                        # USER INPUT FOR MAP TITLE
                        map_title = st.text_input("Map Title", "LULC Classification Analysis")
                        
                        if st.button("📷 Render Map (JPG)"):
                            with st.spinner("Generating Map..."):
                                buf = generate_static_map_display(
                                    lulc_class, roi, vis_params, map_title, 
                                    is_categorical=True, class_names=class_names
                                )
                                if buf:
                                    st.download_button("⬇️ Save Image", buf, "Ni30_LULC.jpg", "image/jpeg", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("No imagery to classify.")

        with col_map:
            m.to_streamlit()

    # ==========================================
    # MODE 3: GEOSPATIAL EMBEDDINGS USE CASES
    # ==========================================
    elif p['mode'] == "Geospatial-embeddings-use-cases":
        col_map, col_res = st.columns([3, 1])
        # FORCED GOOGLE HYBRID
        m = create_map(height=700)
        m.centerObject(roi, 13)
        
        target_year = int(p['embedding_year'])
        
        # Load Google Satellite Embeddings
        # V1/ANNUAL has embeddings for each year
        embeddings_col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
            .filterDate(f'{target_year}-01-01', f'{target_year+1}-01-01') \
            .filterBounds(roi)
        
        if embeddings_col.size().getInfo() == 0:
             st.error(f"No embeddings found for year {target_year} in this region.")
             st.stop()
             
        embeddings = embeddings_col.mosaic().clip(roi)
        
        # Helper to visualize embeddings (RGB using PCA-like bands)
        # Bands are A00 to A63. We can just visualize A00, A01, A02
        emb_vis = {'min': -0.1, 'max': 0.1, 'bands': ['A00', 'A01', 'A02']}
        m.addLayer(embeddings, emb_vis, f'Alpha Earth RGB {target_year}')

        if p['embedding_task'] == "LULC (Supervised with ESA Labels)":
             with st.spinner("Generating LULC from Embeddings (ESA Ground Truth)..."):
                # 1. Get Ground Truth (ESA WorldCover 2021 - closest available)
                # ESA WorldCover 2021 is used as reference for training the embeddings
                esa = ee.Image('ESA/WorldCover/v200/2021').clip(roi)
                
                # 2. Prepare Training Data
                # We use the embeddings + ESA label to train a classifier
                # Then classify the current year's embeddings
                
                # Sample pixels
                sample_scale = 20 # Embeddings are coarser than S2
                num_points = 1000
                
                # Use the 2021 embeddings for training (to match ESA 2021 labels)
                emb_2021_col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
                    .filterDate('2021-01-01', '2022-01-01') \
                    .filterBounds(roi)
                
                if emb_2021_col.size().getInfo() > 0:
                    emb_2021 = emb_2021_col.mosaic().clip(roi)
                    
                    # Stack embeddings and label
                    train_image = emb_2021.addBands(esa.rename('label'))
                    
                    # Stratified Sample
                    points = train_image.stratifiedSample(
                        numPoints=num_points,
                        classBand='label',
                        region=roi,
                        scale=sample_scale,
                        geometries=True
                    )
                    
                    # Train RF
                    # Embeddings bands are A00..A63
                    band_names = embeddings.bandNames()
                    classifier = ee.Classifier.smileRandomForest(100).train(
                        features=points,
                        classProperty='label',
                        inputProperties=band_names
                    )
                    
                    # Classify Target Year Embeddings
                    classified = embeddings.classify(classifier)
                    
                    # Visualization (ESA Palette)
                    # ESA classes: 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100
                    
                    # Map specific ESA classes to colors
                    class_values = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]
                    class_names_esa = ['Trees', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare/Sparse', 'Water', 'Herbaceous Wetland', 'Mangroves', 'Moss/Lichen']
                    class_colors = ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000', 
                                    '#b4b4b4', '#0064c8', '#0096a0', '#00cf75', '#fae6a0']
                    
                    # Filter classified image to only valid classes and visualize
                    remapped = classified.remap(class_values, list(range(len(class_values))))
                    vis_remap = {'min': 0, 'max': len(class_values)-1, 'palette': class_colors}
                    
                    m.addLayer(remapped, vis_remap, f"LULC {target_year} (Embeddings)")
                    
                    with col_res:
                          st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                          st.markdown('<div class="card-label">🔍 ANALYSIS</div>', unsafe_allow_html=True)
                          st.success("Model Trained on 2021 Data")
                          st.info(f"Inference on {target_year}")
                          
                          st.markdown('<div class="card-label">📊 AREA STATS</div>', unsafe_allow_html=True)
                          with st.spinner("Calculating areas..."):
                               # Note: we use 'remapped' here so the indices 0-9 match 'class_names_esa'
                               df_area = calculate_area_by_class(remapped, roi, 20, class_names_esa)
                               if not df_area.empty:
                                   st.dataframe(df_area, hide_index=True, use_container_width=True)
                          
                          if st.button("☁️ Export Map"):
                            ee.batch.Export.image.toDrive(
                                image=classified, description=f"Emb_LULC_{target_year}", 
                                scale=20, region=roi, folder='GEE_Exports'
                            ).start()
                            st.toast("Export task started")
                          
                          st.markdown("---")
                          
                          # USER INPUT FOR MAP TITLE
                          map_title = st.text_input("Map Title", f"LULC Embeddings {target_year}")

                          if st.button("📷 Render Map (JPG)"):
                             with st.spinner("Generating Map..."):
                                 buf = generate_static_map_display(
                                     remapped, roi, vis_remap, map_title, 
                                     is_categorical=True, class_names=class_names_esa
                                 )
                                 if buf:
                                     st.download_button("⬇️ Save Image", buf, "Ni30_Emb_LULC.jpg", "image/jpeg", use_container_width=True)
                          st.markdown('</div>', unsafe_allow_html=True)

                else:
                    st.error("Training data (2021 Embeddings) missing.")

        elif p['embedding_task'] == "Alpha Earth: Transfer Learning (Dynamic World)":
            # NEW MODE
            with st.spinner("Training Alpha Earth Embeddings on Dynamic World Labels..."):
                # 1. Load Dynamic World Labels for the SAME YEAR as embeddings
                # This makes it better than ESA for multi-year analysis
                dw_col = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
                    .filterDate(f'{target_year}-01-01', f'{target_year+1}-01-01') \
                    .filterBounds(roi)

                if dw_col.size().getInfo() > 0:
                    # Get the most common class for the year (label band)
                    dw_label = dw_col.select('label').mode().clip(roi)
                    
                    # 2. Combine Embeddings (Features) with DW (Labels)
                    training_image = embeddings.addBands(dw_label.rename('label'))
                    
                    # 3. Sample points
                    # Stratified sampling ensures we get examples of all classes present
                    points = training_image.stratifiedSample(
                        numPoints=1000,
                        classBand='label',
                        region=roi,
                        scale=30,
                        geometries=True
                    )
                    
                    # 4. Train Classifier
                    band_names = embeddings.bandNames()
                    classifier = ee.Classifier.smileRandomForest(50).train(
                        features=points,
                        classProperty='label',
                        inputProperties=band_names
                    )
                    
                    # 5. Classify
                    classified_dw = embeddings.classify(classifier)
                    
                    # 6. Visualization (Dynamic World Palette)
                    dw_names = ['Water', 'Trees', 'Grass', 'Flooded Veg', 'Crops', 
                                'Shrub/Scrub', 'Built', 'Bare', 'Snow/Ice']
                    dw_vis = {
                        "min": 0, "max": 8,
                        "palette": [
                            '#419bdf', '#397d49', '#88b053', '#7a87c6', '#e49635',
                            '#dfc35a', '#c4281b', '#a59b8f', '#b39fe1'
                        ]
                    }
                    
                    m.addLayer(dw_label, dw_vis, "Dynamic World (Raw)", False) # Hidden by default
                    m.addLayer(classified_dw, dw_vis, f"Alpha Earth LULC {target_year}")
                    m.add_legend(title="Alpha Earth Classes", legend_dict=dict(zip(dw_names, dw_vis['palette'])))

                    with col_res:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown('<div class="card-label">🧠 SEMANTIC TRANSFER</div>', unsafe_allow_html=True)
                        st.success(f"Year: {target_year}")
                        st.info("Method: Semantic Embeddings trained on Dynamic World labels.")
                        st.caption("Embeddings often smooth out noise found in raw optical classification.")
                        
                        st.markdown('<div class="card-label">📊 AREA STATS</div>', unsafe_allow_html=True)
                        with st.spinner("Calculating areas..."):
                             df_area = calculate_area_by_class(classified_dw, roi, 20, dw_names)
                             if not df_area.empty:
                                 st.dataframe(df_area, hide_index=True, use_container_width=True)
                        
                        st.markdown("---")
                        if st.button("☁️ Export Map"):
                            ee.batch.Export.image.toDrive(
                                image=classified_dw, description=f"AlphaEarth_DW_{target_year}", 
                                scale=20, region=roi, folder='GEE_Exports'
                            ).start()
                            st.toast("Export task started")
                        
                        st.markdown("---")
                        map_title = st.text_input("Map Title", f"Alpha Earth LULC {target_year}")
                        if st.button("📷 Render Map (JPG)"):
                             with st.spinner("Generating Map..."):
                                 buf = generate_static_map_display(
                                     classified_dw, roi, dw_vis, map_title, 
                                     is_categorical=True, class_names=dw_names
                                 )
                                 if buf:
                                     st.download_button("⬇️ Save Image", buf, "Ni30_Alpha_LULC.jpg", "image/jpeg", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                else:
                    st.error(f"No Dynamic World labels found for {target_year} to train the embeddings.")


        elif p['embedding_task'] == "Water/Change Detection (Unsupervised)":
            with st.spinner("Running Unsupervised Clustering on Embeddings..."):
                # Use KMeans clustering on the embeddings
                # This groups pixels with similar spectral/textural properties
                
                # 1. Sample for training clusterer
                sample_points = embeddings.sample(
                    region=roi, scale=50, numPixels=2000
                )
                
                # 2. Clusterer (KMeans)
                n_clusters = 5
                clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(sample_points)
                
                # 3. Cluster Result
                result = embeddings.cluster(clusterer)
                
                # Vis
                cluster_vis = {'min': 0, 'max': n_clusters-1, 'palette': ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff']}
                m.addLayer(result.randomVisualizer(), {}, f"Clusters {target_year}")
                
                with col_res:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-label">💧 CLUSTERING</div>', unsafe_allow_html=True)
                    st.info("Unsupervised grouping of terrain features.")
                    st.caption("Useful for detecting water bodies or major land changes without labels.")
                    
                    st.markdown('<div class="card-label">📊 CLUSTER AREA</div>', unsafe_allow_html=True)
                    with st.spinner("Calculating areas..."):
                          df_area = calculate_area_by_class(result, roi, 20, [f"Cluster {i}" for i in range(n_clusters)])
                          if not df_area.empty:
                              st.dataframe(df_area, hide_index=True, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # USER INPUT FOR MAP TITLE
                    map_title = st.text_input("Map Title", f"Clustering Analysis {target_year}")

                    if st.button("📷 Render Map (JPG)"):
                             with st.spinner("Generating Map..."):
                                 buf = generate_static_map_display(
                                     result, roi, cluster_vis, map_title, 
                                     is_categorical=False # Clusters are cat, but random colors make legend hard
                                 )
                                 if buf:
                                     st.download_button("⬇️ Save Image", buf, "Ni30_Clusters.jpg", "image/jpeg", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        with col_map:
            m.to_streamlit()
    
    # ==========================================
    # MODE 4: LANDSLIDE DETECTION (SAR)
    # ==========================================
    elif p['mode'] == "Landslide Detection (SAR)":
        col_map, col_res = st.columns([3, 1])
        # FORCED GOOGLE HYBRID
        m = create_map(height=700)
        m.centerObject(roi, 13)
        
        with st.spinner("🛰️ Calculating SAR Backscatter Changes & DEM Analysis..."):
            # 1. Get S1 Collections for Pre and Post
            def get_s1_processed(start_d, end_d, roi_geom):
                col = ee.ImageCollection('COPERNICUS/S1_GRD') \
                    .filterDate(start_d, end_d) \
                    .filterBounds(roi_geom) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                
                # Mosaic and Apply Speckle Filter (Boxcar)
                # Using VV polarization as it's often more sensitive to surface roughness changes in landslides
                img = col.select('VV').mosaic().clip(roi_geom)
                return img.focal_median(50, 'circle', 'meters')

            pre_img = get_s1_processed(p['pre_start'], p['pre_end'], roi)
            post_img = get_s1_processed(p['post_start'], p['post_end'], roi)
            
            # Check if images exist
            try:
                # Force a check
                info_check = pre_img.getInfo()
            except:
                st.error("No Sentinel-1 Imagery found for the selected Pre-Event dates.")
                st.stop()

            # 2. Calculate Difference (Log Ratio / dB Difference)
            # Sentinel-1 GRD is often provided in linear scale by default in some GEE contexts, 
            # but usually usually pre-processed to dB. If linear, log ratio. If dB, subtraction.
            # Assuming standard GEE S1 GRD which is sigma0.
            
            # Convert to dB for consistent math: 10*log10(x)
            pre_db = ee.Image(10).multiply(pre_img.log10())
            post_db = ee.Image(10).multiply(post_img.log10())
            
            diff = post_db.subtract(pre_db).abs()
            
            # 3. Terrain Masking (Slope)
            # Landslides typically occur on slopes > 10-15 degrees.
            # Use NASA DEM (Global) or SRTM
            dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
            slope = ee.Terrain.slope(dem).clip(roi)
            
            # Mask: Significant Change AND Significant Slope
            # Landslides usually increase roughness (brighter) or remove veg (darker/brighter depending on moisture)
            # We look for high absolute change.
            slide_mask = diff.gt(p['slide_thresh']).And(slope.gt(p['slope_thresh']))
            
            detected_slides = diff.updateMask(slide_mask)
            
            # 4. Visualization
            m.addLayer(pre_db, {'min': -25, 'max': 0}, 'Pre-Event (dB)', False)
            m.addLayer(post_db, {'min': -25, 'max': 0}, 'Post-Event (dB)', False)
            m.addLayer(slope, {'min': 0, 'max': 60, 'palette': ['white', 'black']}, 'Slope Map', False)
            
            slide_vis = {'palette': ['red']}
            m.addLayer(detected_slides, slide_vis, '⚠️ Potential Landslides')
            
            # 5. Calculate Affected Area
            area_img = ee.Image.pixelArea().updateMask(slide_mask)
            stats = area_img.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=roi,
                scale=10,
                maxPixels=1e9
            )
            area_sq_m = stats.get('area')
            area_ha = ee.Number(area_sq_m).divide(10000).getInfo()

            with col_res:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-label">⚠️ DETECTION REPORT</div>', unsafe_allow_html=True)
                st.warning("Sentinel-1 Change Detection")
                
                if area_ha is not None:
                    st.markdown(f"""
                        <div style="margin: 15px 0;">
                            <div class="metric-sub">Affected Area</div>
                            <div class="metric-value" style="color:#ff4b4b;">{area_ha:.2f} ha</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No significant changes detected above threshold.")

                st.markdown(f"**Sensitivity:** {p['slide_thresh']} dB")
                st.markdown(f"**Slope Mask:** > {p['slope_thresh']}°")
                
                st.markdown("---")
                if st.button("☁️ Export Detection"):
                     ee.batch.Export.image.toDrive(
                        image=detected_slides, description=f"Landslide_Mask_{p['post_end']}", 
                        scale=10, region=roi, folder='GEE_Exports'
                    ).start()
                     st.toast("Export started")
                
                st.markdown("---")
                
                # USER INPUT FOR MAP TITLE
                map_title = st.text_input("Map Title", "Landslide Event Analysis")

                if st.button("📷 Save Map Report"):
                     buf = generate_static_map_display(
                         detected_slides, roi, {'min':0, 'max':1, 'palette':['red']}, 
                         map_title, is_categorical=True, class_names=["Landslide"]
                     )
                     if buf:
                         st.download_button("⬇️ Save Image", buf, "Ni30_Landslide.jpg", "image/jpeg", use_container_width=True)

                st.markdown('</div>', unsafe_allow_html=True)

        with col_map:
            m.to_streamlit()
