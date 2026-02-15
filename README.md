# 🛰️ **SpecTralNi30** – Geospatial Remote Sensing Analytics Engine

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-red?logo=streamlit)](https://spectralni30.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Geospatial](https://img.shields.io/badge/Geospatial-Remote%20Sensing-brightgreen)](#)

---

## 📋 **Overview**

**SpecTralNi30** is a production-grade geospatial web application built with **Streamlit** and **Google Earth Engine** for real-time satellite analytics. It empowers geospatial scientists and environmental professionals to process, analyze, and export multi-sensor Earth observation data without writing code.

### 🎯 **Core Capabilities**

- **Multi-sensor workflows**: Sentinel-1 (SAR), Sentinel-2 (Optical), Landsat (Thermal + Optical)
- **Advanced spectral analysis**: NDVI, GNDVI, NDWI, NDMI, EVI, LST, custom band math
- **LULC classification**: 7 pre-trained ML architectures + Google Dynamic World integration
- **Geospatial embeddings**: AI-powered semantic feature extraction
- **SAR change detection**: Automated landslide & flood mapping
- **Flexible export**: GeoTIFF, Google Drive, publication-ready JPG maps with legends

---

## ✨ **Key Features**

### **1. 🔍 Spectral Monitor** (Vegetation & Thermal Analysis)

Analyze vegetation health, water bodies, and thermal signatures in real-time.

**Supported Satellites:**
- Sentinel-2 (10m/20m optical, 13 bands)
- Sentinel-1 (Radar SAR, VV/VH polarization)
- Landsat 8/9 (30m optical + 100m thermal)

**Spectral Products:**
| Index | Formula | Use Case |
|-------|---------|----------|
| NDVI | (NIR - Red) / (NIR + Red) | Vegetation health, biomass |
| GNDVI | (NIR - Green) / (NIR + Green) | Crop stress detection |
| NDWI | (Green - NIR) / (Green + NIR) | Water body mapping |
| NDMI | (NIR - SWIR) / (NIR + SWIR) | Soil moisture content |
| EVI | 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1) | Improved vegetation monitoring |
| LST | Brightness Temp (Kelvin) → Celsius | Land surface temperature |
| Custom | User-defined expressions | Flexible band combinations |

**Capabilities:**
- Real-time satellite scene discovery with cloud filtering (0-100%)
- Automatic dynamic stretching (P2-P98 percentile)
- Multi-date median compositing
- Statistical analysis (mean, std, min, max)
- Time-series visualization

---

### **2. 🏘️ LULC Classifier** (Land Use/Land Cover)

Automatic land cover mapping using pre-trained and custom ML models.

**Pre-trained Models:**
- **Google Dynamic World** (FCN deep learning, 10m global coverage)
- **Custom ML Classifiers:**
  - Random Forest (10-500 trees, Gini/entropy splitting)
  - Support Vector Machine (RBF, LINEAR, POLY kernels)
  - Artificial Neural Network (MLP with tunable hidden layers)
  - Gradient Tree Boost (XGBoost-style)
  - CART Decision Tree
  - Naive Bayes

**9-Class LULC Schema:**
Water | Trees | Grassland | Crops | Built-up | Bare/Rock | Shrub | Flooded Vegetation | Snow/Ice

**Metrics Computed:**
- Overall Accuracy & Kappa coefficient
- Per-class area statistics (hectares)
- Confusion matrix from validation set
- Train/validation split (configurable 50-90%)

---

### **3. 🌐 Geospatial Embeddings** (AI Foundation Model)

Leverages Google Satellite Embeddings V1 (64-band feature vectors) for semantic understanding.

**Use Cases:**
- LULC with ESA WorldCover ground truth
- Transfer learning on Dynamic World labels
- Unsupervised clustering (KMeans)
- Water body & change detection (unsupervised)

**Technical Approach:**
- 64 learned semantic bands from optical/textural data
- Transfer learning via Random Forest on embeddings
- Year-to-year trend analysis

---

### **4. 🏔️ Landslide Detection** (Sentinel-1 SAR)

Automated hazard mapping using radar backscatter change analysis.

**Method:**
- Pre/post-event SAR backscatter comparison (dB scale)
- Terrain masking (DEM-based slope filtering)
- Multi-parametric threshold detection

**Configuration:**
- Backscatter sensitivity: 1.0-5.0 dB adjustment
- Slope threshold: 0-30° (filters flat agricultural areas)
- Customizable temporal windows (pre/post event)

**Outputs:**
- Change magnitude raster
- Affected area statistics (hectares)
- Confidence-based event polygons

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.9+
- Google Earth Engine account (sign up free at [earthengine.google.com](https://earthengine.google.com))
- Internet connection

### **Installation**

```bash
# Clone repository
git clone https://github.com/nitesh4004/SpecTralNi30.git
cd SpecTralNi30

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize Google Earth Engine
earthengine authenticate

# Run Streamlit app
streamlit run streamlit_app.py
```

### **Access Application**

Open browser to `http://localhost:8501`

---

## 📂 **Project Structure**

```
SpecTralNi30/
├── streamlit_app.py              # Main Streamlit app (UI + GEE workflows)
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
├── logo.png                      # Application branding
├── .devcontainer/                # Docker development environment
└── data/                         # Sample datasets
    ├── LULC_Sentinel2_Indian_Region_15000rows.csv
    ├── lulc_spectral_indices_10000.csv
    ├── lulc_spectral_indices_30000.csv
    └── sentinel2_lulc_synthetic.csv
```

---

## 📍 **ROI Selection Methods**

Flexible region-of-interest definition:

| Method | Input | Limit | Use Case |
|--------|-------|-------|----------|
| **Upload KML** | KML/KMZ file | 200 MB | Complex boundaries, admin zones |
| **Point + Buffer** | Lat/Lon + radius (m) | Unlimited | Quick spot analysis |
| **Manual Bounding Box** | Min/Max coordinates | Unlimited | Rectangular AOI |

---

## 💾 **Export Formats**

| Format | Characteristics | Best For |
|--------|-----------------|----------|
| **GeoTIFF** | Cloud-optimized, georeferenced | GIS analysis, archival |
| **Google Drive** | Batch export, cloud storage | Long-term collaboration |
| **JPG Map** | Publication-ready cartography | Reports, presentations |

---

## 🎨 **Visualization**

**Color Palettes:**
- Red-Yellow-Green (vegetation emphasis)
- Blue-White-Green (water + vegetation)
- Blue-Yellow-Red (thermal gradient)
- Viridis, Magma, Plasma, Turbo (scientific)
- Ocean, Terrain (topographic)

**Interactive Mapping:**
- Leaflet.js basemap (satellite, terrain, street)
- Polygon/polyline drawing tools
- Layer toggling and opacity control
- Full-screen mode

---

## 🧪 **Example Use Cases**

1. **Agricultural Monitoring**
   - NDVI/GNDVI time-series for crop health assessment
   - Yield prediction model training
   - Precision irrigation planning

2. **Flood Mapping**
   - NDWI-based water extent mapping
   - SAR backscatter change detection
   - Inundation area quantification

3. **Urban Expansion**
   - Multi-year LULC classification
   - Built-up area change analysis
   - Infrastructure planning

4. **Forest Monitoring**
   - Deforestation detection via change analysis
   - Biomass estimation (NDVI correlation)
   - Seasonal phenology tracking

5. **Environmental Risk Assessment**
   - Landslide susceptibility mapping
   - Coastal erosion monitoring
   - Mine impact assessment

---

## 📊 **Technical Stack**

| Layer | Technology | Purpose |
|-------|-----------|----------|
| **Backend** | Google Earth Engine | Planetary-scale data processing |
| **Frontend** | Streamlit | Interactive web UI (Python-native) |
| **Mapping** | Geemap/Leaflet.js | Interactive map visualization |
| **ML/DL** | Scikit-learn, TensorFlow | Local model training |
| **Data Handling** | GeoPandas, Rasterio, NumPy | Geospatial operations |
| **Deployment** | Streamlit Cloud | Production hosting |

---

## 🤝 **Contributing**

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push to branch: `git push origin feature/my-feature`
5. Open a Pull Request with a clear description

### **Development Guidelines**
- Follow PEP 8 style guide
- Add docstrings to all functions
- Test locally before submitting PR
- Document new features in README

---

## 📜 **License**

MIT License – See LICENSE file for details.

---

## 📬 **Contact & Support**

**Author:** Nitesh Kumar  
**Role:** Geospatial Data Scientist  
**Email:** nitesh.gulzar@gmail.com  
**GitHub:** [@nitesh4004](https://github.com/nitesh4004)  
**LinkedIn:** [in/nitesh4004](https://linkedin.com/in/nitesh4004/)  
**Portfolio:** [nitesh4004.github.io](https://nitesh4004.github.io/)  

### **Support Channels**
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/nitesh4004/SpecTralNi30/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/nitesh4004/SpecTralNi30/discussions)
- 📧 **Email**: Direct inquiry for commercial use or custom development

---

## 🎯 **Roadmap**

- [ ] Hyperspectral data integration (PRISMA, EnMAP)
- [ ] Real-time alerting system (flood/fire detection)
- [ ] Multi-temporal change detection workflows
- [ ] Crop yield ML model library
- [ ] REST API for programmatic access
- [ ] Mobile app (React Native)

---

## 📚 **References & Resources**

- [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
- [Streamlit Framework](https://streamlit.io)
- [Geemap Project](https://geemap.org)
- [ESA Sentinel Documentation](https://sentinel.esa.int)
- [USGS Landsat](https://www.usgs.gov/landsat)

---

## ⭐ **Acknowledgments**

- Google for Earth Engine & Satellite Embeddings
- Streamlit team for the excellent framework
- Geemap community for mapping utilities
- ESA & USGS for satellite data

---

**Made with ❤️ by Nitesh Kumar | GIS Engineer @ SWANSAT OPC Pvt. Ltd**


---

**Last Updated**: February 15, 2026

---

**Last Updated**: February 15, 2026
