"""Enhanced Earth Engine Authentication Handler for Streamlit

This module provides robust authentication for Google Earth Engine on Streamlit,
with support for both local development and cloud deployment scenarios.
"""

import streamlit as st
import ee
import json
import os
from typing import Optional, Dict, Any


class EarthEngineAuthHandler:
    """Handles Earth Engine authentication for Streamlit applications."""
    
    @staticmethod
    def validate_secrets() -> bool:
        """Validate that secrets are properly configured."""
        try:
            if "gcp_service_account" in st.secrets:
                return True
            elif st.secrets.get("type") == "service_account":
                return True
        except (AttributeError, KeyError):
            pass
        return False
    
    @staticmethod
    def get_service_account_dict() -> Optional[Dict[str, Any]]:
        """Extract service account dictionary from Streamlit secrets."""
        try:
            if "gcp_service_account" in st.secrets:
                return dict(st.secrets["gcp_service_account"])
            elif st.secrets.get("type") == "service_account":
                return dict(st.secrets)
        except Exception:
            pass
        return None
    
    @staticmethod
    def initialize_ee() -> bool:
        """Initialize Earth Engine with multi-method authentication.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Method 1: Try Streamlit Secrets (Cloud Deployment)
            secret_dict = EarthEngineAuthHandler.get_service_account_dict()
            
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
                    st.session_state['ee_initialized'] = True
                    st.session_state['ee_project'] = project_id
                    return True
                except Exception as e:
                    st.error(f"Service account authentication failed: {str(e)}")
                    return False
            
            # Method 2: Try Environment Variable
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                try:
                    project_id = os.getenv("EE_PROJECT", "")
                    ee.Initialize(project=project_id or None)
                    st.session_state['ee_initialized'] = True
                    st.session_state['ee_project'] = project_id
                    return True
                except Exception as e:
                    st.warning(f"Environment credential initialization failed: {str(e)}")
            
            # Method 3: Try Local Authentication (Development)
            try:
                project_id = (
                    st.session_state.get("ee_project")
                    or os.getenv("EE_PROJECT", "")
                    or "ee-niteshswansat"
                )
                ee.Initialize(project=project_id)
                st.session_state['ee_initialized'] = True
                st.session_state['ee_project'] = project_id
                return True
            except ee.EEException:
                pass
            except Exception:
                pass
            
            return False
            
        except Exception as e:
            st.error(f"Unexpected authentication error: {str(e)}")
            return False
    
    @staticmethod
    def show_setup_guide():
        """Display setup guide for Earth Engine authentication."""
        st.error("🔐 Earth Engine Authentication Required")
        
        with st.expander("📚 Setup Instructions", expanded=True):
            st.markdown("""
            ### Option 1: Streamlit Cloud Deployment (Recommended)
            
            1. **Create GCP Service Account**:
               - Go to [Google Cloud Console](https://console.cloud.google.com)
               - Create a new service account
               - Download the JSON key file
            
            2. **Add to Streamlit Secrets**:
               - Go to your Streamlit app settings
               - Navigate to "Secrets"
               - Copy the entire JSON and paste it as TOML:
               ```toml
               [gcp_service_account]
               type = "service_account"
               project_id = "your-project"
               private_key_id = "..."
               private_key = "..."
               client_email = "..."
               client_id = "..."
               auth_uri = "https://accounts.google.com/o/oauth2/auth"
               token_uri = "https://oauth2.googleapis.com/token"
               auth_provider_x509_cert_url = "..."
               client_x509_cert_url = "..."
               ```
            
            3. **Authorize in Earth Engine**:
               - Add service account email to [Earth Engine account permissions](https://earthengine.google.com/)
            
            ### Option 2: Local Development
            
            ```bash
            # Authenticate locally
            earthengine authenticate
            
            # Set your project
            earthengine config set project ee-niteshswansat
            
            # Run the app
            streamlit run streamlit_app.py
            ```
            """)
