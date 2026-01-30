# SpecTralNi30 Setup & Deployment Guide

## 🚀 Quick Start

This guide will help you get SpecTralNi30 running on both local development and Streamlit Cloud.

---

## 📋 Prerequisites

- Python 3.9 or higher
- Pip package manager
- Git (for version control)
- Google Account (for Earth Engine access)
- Google Cloud Project (for Streamlit Cloud deployment)

---

## ✅ OPTION 1: Local Development (Recommended First Step)

### Step 1: Clone the Repository

```bash
git clone https://github.com/nitesh4004/SpecTralNi30.git
cd SpecTralNi30
```

### Step 2: Create Virtual Environment

```bash
# On Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Authenticate with Google Earth Engine

```bash
earthengine authenticate
```

This will:
1. Open your browser for OAuth authentication
2. Ask for Google account permissions
3. Generate credentials in `~/.config/earthengine/`

### Step 5: Set Your Earth Engine Project

```bash
earthengine config set project ee-niteshswansat
```

### Step 6: Run the App

```bash
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

---

## ☁️ OPTION 2: Streamlit Cloud Deployment (Production)

### Step 1: Create GCP Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/iam-admin/serviceaccounts)
2. Select your project (or create a new one)
3. Click "Create Service Account"
4. Fill in details:
   - Service Account Name: `SpecTralNi30-SA`
   - Service Account ID: `spectralni30-sa`
5. Click "Create and Continue"
6. Grant roles: `Editor` (for testing), then `Viewer` (for production)
7. Click "Continue" → "Done"

### Step 2: Create & Download JSON Key

1. Click on the newly created service account
2. Go to "Keys" tab
3. Click "Add Key" → "Create new key"
4. Choose "JSON" format
5. Click "Create" (JSON file auto-downloads)
6. **Important**: Keep this file secure and never commit to Git

### Step 3: Authorize Service Account in Earth Engine

1. Go to [Earth Engine Registration Page](https://earthengine.google.com/signup/)
2. Sign in with your Google account
3. Go to Settings → "Users and Permissions"
4. Click "+ Add User"
5. Paste your service account email (from JSON key: `client_email`)
6. Select "Viewer" role
7. Click "Add user"

### Step 4: Add Secrets to Streamlit Cloud

1. Deploy your app to Streamlit Cloud
2. In the Streamlit Cloud dashboard, click "Advanced settings"
3. Select "Secrets"
4. Open the JSON key file you downloaded
5. Copy the entire contents
6. In Streamlit Secrets, paste it in TOML format:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-gcp-project-id"
private_key_id = "xxxxxxxxxxxxxxxx"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-sa@your-project.iam.gserviceaccount.com"
client_id = "123456789"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-sa%40your-project.iam.gserviceaccount.com"

# Optional: Set project ID
ee_project = "your-gcp-project-id"
```

7. Click "Save"
8. Redeploy your app

### Step 5: Verify Deployment

1. Go to your Streamlit Cloud app URL
2. App should load without authentication errors
3. Use the sidepanel to configure sensors and ROI
4. Click "INITIALIZE SCAN 🚀" to test

---

## 🔧 Configuration Files

### `.streamlit/secrets.example.toml`

Template file showing required secrets structure. Copy to `.streamlit/secrets.toml` for local development.

### `.gitignore` (Important!)

Make sure `.streamlit/secrets.toml` is in `.gitignore`:

```
.streamlit/secrets.toml
__pycache__/
*.pyc
.DS_Store
.env
venv/
.venv/
```

---

## 📦 Updated Dependencies

Key libraries:
- `streamlit>=1.28.0` - Web framework
- `earthengine-api>=0.1.348` - GEE Python API
- `geemap>=0.37.0` - Geospatial analysis
- `pandas>=1.5.0` - Data manipulation
- `scikit-learn>=1.3.0` - ML models
- `matplotlib>=3.7.0` - Visualization

---

## 🐛 Troubleshooting

### Authentication Error on Streamlit Cloud

**Problem**: "Authentication Error. Run `earthengine authenticate`"

**Solution**:
1. Verify secrets are added in Streamlit Cloud dashboard
2. Check service account email is authorized in Earth Engine
3. Redeploy the app after adding secrets

### Local Authentication Issues

**Problem**: "EEException: User not authorized"

**Solution**:
```bash
# Re-authenticate
earthengine authenticate

# Verify credentials
earthengine info
```

### Import Errors

**Problem**: "ModuleNotFoundError: No module named 'ee'"

**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Map Not Loading

**Problem**: Blank map or "GEE Server Error"

**Solution**:
1. Check ROI coordinates are valid
2. Reduce date range
3. Check cloud cover threshold
4. Verify GEE collection availability for your location

---

## 🔐 Security Best Practices

1. **Never commit secrets**: Always add `.streamlit/secrets.toml` to `.gitignore`
2. **Use environment variables**: For sensitive data in CI/CD
3. **Service account permissions**: Use "Viewer" role in production, not "Editor"
4. **Rotate keys periodically**: Update GCP service account keys every 90 days
5. **Limit service account scope**: Only grant necessary permissions

---

## 📚 Documentation Links

- [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Geemap Project](https://geemap.org/)
- [Earth Engine Python API](https://earthengine.google.com/)

---

## 🤝 Support

For issues or questions:
1. Check GitHub Issues: [SpecTralNi30 Issues](https://github.com/nitesh4004/SpecTralNi30/issues)
2. Email: nitesh.gulzar@gmail.com
3. Review the README.md for more details

---

## ✨ Next Steps

1. ✅ Set up local development environment
2. ✅ Test with sample data
3. ✅ Configure your ROI and sensors
4. ✅ Deploy to Streamlit Cloud
5. ✅ Share with team/stakeholders
