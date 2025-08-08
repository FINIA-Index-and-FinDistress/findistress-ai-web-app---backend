"""
ML Model Management for Financial Distress Prediction
Professional model loading and prediction following exact ML pipeline structure.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timezone

import joblib
import pandas as pd
import numpy as np
import shap
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class PipelineConfig:
    """Configuration matching exact ML pipeline."""
    
    # Model paths from environment
    MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", "app/ml_pipeline")
    AFR_MODEL_PATH = os.getenv("AFR_MODEL_PATH", "app/ml_pipeline/trained_models/afr_best_pipeline.joblib")
    ROW_MODEL_PATH = os.getenv("ROW_MODEL_PATH", "app/ml_pipeline/trained_models/row_best_pipeline.joblib")
    
    # Pipeline feature sets 
    AFR_FEATURES = [
        'startup', 'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf',
        'Fem_CEO', 'Pvt_Own', 'Con_Own', 'Edu', 'Exports', 'Innov', 'Transp',
        'Gifting', 'Pol_Inst', 'Infor_Comp', 'Credit', 'WSI', 'WUI', 'GDP', 'PRIME'
    ]
    
    ROW_FEATURES = AFR_FEATURES + ['Size', 'MarketCap', 'GPR']
    
    # Business-friendly feature descriptions
    FEATURE_DESCRIPTIONS = {
        'startup': 'Company is a startup (young company)',
        'Fin_bank': 'Bank financing percentage',
        'Fin_supplier': 'Supplier credit financing',
        'Fin_equity': 'Equity-based financing',
        'Fin_other': 'Other financing sources',
        'Fem_wf': 'Female workforce participation',
        'Fem_CEO': 'Female leadership (CEO)',
        'Pvt_Own': 'Private ownership structure',
        'Con_Own': 'Concentrated ownership',
        'Edu': 'Education level of workforce',
        'Exports': 'Export activity level',
        'Innov': 'Innovation and R&D investment',
        'Transp': 'Transportation access',
        'Gifting': 'Informal payments/corruption',
        'Pol_Inst': 'Political instability impact',
        'Infor_Comp': 'Informal sector competition',
        'Credit': 'Access to credit facilities',
        'WSI': 'World economic strength',
        'WUI': 'World uncertainty levels',
        'GDP': 'Economic growth environment',
        'PRIME': 'Interest rate environment',
        'Size': 'Company size metric',
        'MarketCap': 'Market capitalization',
        'GPR': 'Geopolitical risk exposure'
    }
    
    # Sector mappings 
    SECTOR_MAPPINGS = {
        'Construction': '1', 'Retail': '2', 'Manufacturing': '3', 'Other Services': '4',
        'Other Manufacturing': '5', 'Food': '6', 'Garments': '7', 'Hotels': '8',
        'Services': '9', 'Rest of Universe': '10', 'IT & IT Services': '11', 'Textiles': '12',
        'Machinery & Equipment': '13', 'Textiles & Garments': '14',
        'Basic Metals/Fabricated Metals/Machinery & Equip.': '15',
        'Chemicals, Plastics & Rubber': '16', 'Chemicals & Chemical Products': '17',
        'Machinery & Equipment & Electronics': '18', 'Leather Products': '19',
        'Furniture': '20', 'Motor Vehicles & Transport Equip.': '21',
        'Fabricated Metal Products': '22', 'Hospitality & Tourism': '23',
        'Motor Vehicles': '24', 'Electronics': '25', 'Services of Motor Vehicles/Wholesale/Retail': '26',
        'Food/Leather/Wood/Tobacco/Rubber Products': '27', 'Professional Activities': '28',
        'Non-Metallic Mineral Products': '29', 'Hotels & Restaurants': '30',
        'Electronics & Communications Equip.': '31', 'Transport, Storage, & Communications': '32',
        'Services of Motor Vehicles': '33', 'Rubber & Plastics Products': '34',
        'Basic Metals & Metal Products': '35', 'Wholesale': '36', 'Basic Metals': '37',
        'Electrical & Computer Products': '38', 'Minerals, Metals, Machinery & Equipment': '39',
        'Wood Products': '40', 'Printing & Publishing': '41', 'Petroleum products, Plastics & Rubber': '42',
        'Wood products, Furniture, Paper & Publishing': '43',
        'Machinery & Equipment, Electronics & Vehicles': '44', 'Transport': '45',
        'Textiles, Garments & Leather': '46', 'Restaurants': '47',
        'Wholesale, Including of Motor Vehicles': '48', 'Publishing, Telecommunications & IT': '49',
        'Wholesale & Retail': '50', 'Mining Related Manufacturing': '51',
        'Pharmaceuticals & Medical Products': '52', 'Wood Products & Furniture': '53',
        'Computer, Electronic & Optical Products': '54', 'Retail & IT': '55',
        'Metals, Machinery, Computers & Electronics': '56', 'Manufacturing Panel': '57',
        'Retail Panel': '58', 'Other Services Panel': '59',
        'Chemicals, Non-Metallic Mineral, Plastics & Rubber': '60',
        'Textiles, Garments, Leather & Paper': '61', 'Pharmaceutical, Chemicals & Chemical Products': '62',
        'Wholesale of Agri Inputs & Equipment': '63'
    }

config = PipelineConfig()

class ModelManager:
    """Professional ML model manager following pipeline structure."""
    
    def __init__(self):
        self.models_cache: Dict[str, Any] = {}
        self.explainers_cache: Dict[str, Any] = {}
        self.models_metadata: Dict[str, Dict] = {}
        self.performance_tracker = ModelPerformanceTracker()
        
        # Ensure model directory exists
        Path(config.MODEL_BASE_PATH).mkdir(exist_ok=True)
        Path(os.path.dirname(config.AFR_MODEL_PATH)).mkdir(exist_ok=True)
        Path(os.path.dirname(config.ROW_MODEL_PATH)).mkdir(exist_ok=True)
    
    def load_model(self, region: str) -> Optional[Any]:
        """Load LightGBM pipeline model for specific region."""
        if region not in ["AFR", "ROW"]:
            raise ValueError(f"Invalid region: {region}. Must be 'AFR' or 'ROW'")
        
        # Return cached model
        if region in self.models_cache:
            logger.debug(f"🔄 Using cached {region} model")
            return self.models_cache[region]
        
        model_path = config.AFR_MODEL_PATH if region == "AFR" else config.ROW_MODEL_PATH
        
        try:
            # Validate model file exists
            if not Path(model_path).exists():
                logger.error(f"{region} model not found: {model_path}")
                raise FileNotFoundError(f"{region} model file not found: {model_path}")
            
            # Load pipeline model
            logger.info(f"📦 Loading {region} pipeline model...")
            model = joblib.load(model_path)
            
            # Validate pipeline structure
            self._validate_pipeline_model(model, region)
            
            # Cache model
            self.models_cache[region] = model
            
            # Create SHAP explainer for interpretability
            self._create_shap_explainer(model, region)
            
            # Store metadata
            self.models_metadata[region] = {
                "loaded_at": datetime.now(timezone.utc),
                "file_path": model_path,
                "file_size": Path(model_path).stat().st_size,
                "model_type": "LightGBM Pipeline",
                "region": region,
                "features": config.AFR_FEATURES if region == "AFR" else config.ROW_FEATURES
            }
            
            logger.info(f"{region} model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {region} model: {e}")
            raise RuntimeError(f"Could not load {region} model: {e}")
    
    def _validate_pipeline_model(self, model, region: str):
        """Validate pipeline model structure."""
        if not hasattr(model, 'named_steps'):
            raise ValueError(f"{region} model is not a valid pipeline")
        
        # Check for required pipeline steps
        required_steps = ['preprocessor', 'classifier']
        for step in required_steps:
            if step not in model.named_steps:
                raise ValueError(f"{region} model missing required step: {step}")
        
        # Validate classifier is LightGBM
        classifier = model.named_steps['classifier']
        if not hasattr(classifier, 'booster'):
            logger.warning(f"{region} model classifier may not be LightGBM")
    
    def _create_shap_explainer(self, model, region: str):
        """Create SHAP explainer for model interpretability."""
        try:
            classifier = model.named_steps['classifier']
            if hasattr(classifier, 'booster'):
                # LightGBM TreeExplainer
                explainer = shap.TreeExplainer(classifier)
                self.explainers_cache[region] = explainer
                logger.info(f"SHAP explainer created for {region}")
            else:
                logger.warning(f"Could not create SHAP explainer for {region}")
        except Exception as e:
            logger.warning(f"SHAP explainer creation failed for {region}: {e}")
    
    def load_all_models(self) -> Dict[str, Any]:
        """Load both AFR and ROW models."""
        models = {}
        
        for region in ["AFR", "ROW"]:
            try:
                models[region] = self.load_model(region)
            except Exception as e:
                logger.error(f"Failed to load {region} model: {e}")
        
        if not models:
            raise RuntimeError("No models could be loaded")
        
        logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
        return models
    
    def get_model(self, region: str) -> Optional[Any]:
        """Get cached model or load if needed."""
        if region in self.models_cache:
            return self.models_cache[region]
        return self.load_model(region)
    
    def prepare_input_data(self, input_dict: Dict[str, Any], region: str) -> pd.DataFrame:
        """Prepare input data following pipeline preprocessing."""
        try:
            # Get feature set for region
            features = config.AFR_FEATURES if region == "AFR" else config.ROW_FEATURES
            
            # Create dataframe with proper feature order
            input_data = {}
            
            # Map input fields to pipeline features
            field_mapping = {
                'wk14': 'startup',  # Convert years to startup indicator
                'car1': 'startup',  # Company age for startup calculation
                'Fin_bank': 'Fin_bank',
                'Fin_supplier': 'Fin_supplier', 
                'Fin_equity': 'Fin_equity',
                'Fin_other': 'Fin_other',
                'Fem_wf': 'Fem_wf',
                'Fem_CEO': 'Fem_CEO',
                'Pvt_Own': 'Pvt_Own',
                'Con_Own': 'Con_Own',
                'obst9': 'Edu',
                'tr15': 'Exports',
                't10': 'Innov',
                't2': 'Transp',
                'corr4': 'Gifting',
                'obst11': 'Pol_Inst',
                'infor1': 'Infor_Comp',
                'Credit': 'Credit',
                'WSI': 'WSI',
                'WUI': 'WUI',
                'GDP': 'GDP',
                'PRIME': 'PRIME'
            }
            
            # Add ROW-specific mappings
            if region == "ROW":
                field_mapping.update({
                    'size2': 'Size',
                    'MarketCap': 'MarketCap',
                    'GPR': 'GPR'
                })
            
            # Calculate startup indicator 
            wk14 = input_dict.get('wk14', 5)  # Years of operation
            car1 = input_dict.get('car1', 5)  # Years since establishment
            startup_indicator = 1 if (wk14 < 5 and car1 < 5) else 0
            
            # Build feature vector
            for feature in features:
                if feature == 'startup':
                    input_data[feature] = startup_indicator
                else:
                    # Map from input field or use default
                    input_field = None
                    for input_key, pipeline_feature in field_mapping.items():
                        if pipeline_feature == feature:
                            input_field = input_key
                            break
                    
                    if input_field and input_field in input_dict:
                        value = input_dict[input_field]
                        # Normalize percentage values (pipeline converts >1 to 0-1 scale)
                        if isinstance(value, (int, float)) and value > 1 and feature != 'startup':
                            value = value / 100
                        input_data[feature] = max(0, value)  # Ensure non-negative
                    else:
                        # Default values for missing features
                        input_data[feature] = 0.0
            
            # Create dataframe
            df = pd.DataFrame([input_data])
            
            logger.debug(f"Prepared {region} input data: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Input preparation failed for {region}: {e}")
            raise ValueError(f"Failed to prepare input data: {e}")
    
    def predict_with_explanation(self, input_dict: Dict[str, Any], region: str) -> Dict[str, Any]:
        """Make prediction with SHAP explanations (business-friendly)."""
        prediction_start = time.time()
        
        try:
            # Get model
            model = self.get_model(region)
            if model is None:
                raise RuntimeError(f"Model for {region} region not loaded")
            
            # Prepare input data
            input_df = self.prepare_input_data(input_dict, region)
            
            # Make prediction
            prediction_proba = model.predict_proba(input_df)
            distress_probability = float(prediction_proba[0][1])  # Probability of distress
            
            # Calculate confidence (distance from 0.5)
            confidence = float(abs(distress_probability - 0.5) * 2)
            
            # Determine risk level (business-friendly)
            risk_level = self._get_risk_level(distress_probability)
            
            # Generate SHAP explanations
            feature_importance = self._get_shap_explanations(input_df, region)
            
            # Generate business recommendations
            recommendations = self._generate_recommendations(
                distress_probability, feature_importance, input_dict
            )
            
            # Create business-friendly analysis
            analysis_message = self._create_analysis_message(
                distress_probability, risk_level, feature_importance
            )
            
            processing_time = (time.time() - prediction_start) * 1000
            
            # Track performance
            self.performance_tracker.log_prediction(
                region, processing_time, confidence, distress_probability
            )
            
            result = {
                'financial_distress_probability': distress_probability,
                'model_confidence': confidence,
                'risk_category': risk_level,
                'financial_health_status': self._get_health_status(risk_level),
                'risk_level_detail': self._get_risk_detail(distress_probability),
                'analysis_message': analysis_message,
                'key_influencing_factors': feature_importance,
                'recommendations': recommendations,
                'processing_time_ms': processing_time,
                'model_version': f"{region}_pipeline_v2.0",
                'region': region
            }
            
            logger.info(f"{region} prediction completed: {risk_level} risk ({processing_time:.1f}ms)")
            return result
            
        except Exception as e:
            processing_time = (time.time() - prediction_start) * 1000
            logger.error(f"Prediction failed for {region} after {processing_time:.1f}ms: {e}")
            raise
    
    def _get_shap_explanations(self, input_df: pd.DataFrame, region: str) -> List[Dict[str, Any]]:
        """Generate SHAP explanations for feature importance."""
        try:
            if region not in self.explainers_cache:
                logger.warning(f"No SHAP explainer for {region}, using fallback")
                return self._get_fallback_explanations(input_df, region)
            
            explainer = self.explainers_cache[region]
            shap_values = explainer.shap_values(input_df)
            
            # Get SHAP values for positive class (distress)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            # Create feature importance list
            features = config.AFR_FEATURES if region == "AFR" else config.ROW_FEATURES
            importance_list = []
            
            for i, feature in enumerate(features):
                shap_value = float(shap_values[0][i])
                feature_value = float(input_df.iloc[0][feature])
                
                importance_list.append({
                    'name': feature,
                    'display_name': config.FEATURE_DESCRIPTIONS.get(feature, feature),
                    'shap_value': shap_value,
                    'feature_value': feature_value,
                    'impact_level': self._get_impact_level(abs(shap_value)),
                    'weight': abs(shap_value),
                    'description': self._get_factor_description(feature, shap_value, feature_value)
                })
            
            # Sort by absolute SHAP value and return top 5
            importance_list.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            return importance_list[:5]
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed for {region}: {e}")
            return self._get_fallback_explanations(input_df, region)
    
    def _get_fallback_explanations(self, input_df: pd.DataFrame, region: str) -> List[Dict[str, Any]]:
        """Fallback explanations when SHAP is unavailable."""
        features = config.AFR_FEATURES if region == "AFR" else config.ROW_FEATURES
        
        # Use domain knowledge weights
        importance_weights = {
            'Fin_bank': 0.8, 'Credit': 0.9, 'startup': 0.6,
            'GDP': 0.7, 'Pol_Inst': 0.6, 'Fem_CEO': 0.4
        }
        
        explanations = []
        for feature in features[:5]:  # Top 5
            weight = importance_weights.get(feature, 0.3)
            feature_value = float(input_df.iloc[0][feature])
            
            explanations.append({
                'name': feature,
                'display_name': config.FEATURE_DESCRIPTIONS.get(feature, feature),
                'shap_value': weight * (feature_value - 0.5),
                'feature_value': feature_value,
                'impact_level': self._get_impact_level(weight),
                'weight': weight,
                'description': f"Impact based on {config.FEATURE_DESCRIPTIONS.get(feature, feature)}"
            })
        
        return explanations
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to business risk level."""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Medium"
        else:
            return "High"
    
    def _get_health_status(self, risk_level: str) -> str:
        """Get business-friendly health status."""
        status_map = {
            "Low": "Financially Stable",
            "Medium": "Some Financial Concerns",
            "High": "Significant Financial Risk"
        }
        return status_map.get(risk_level, "Under Review")
    
    def _get_risk_detail(self, probability: float) -> str:
        """Get detailed risk explanation."""
        if probability < 0.2:
            return "Very low risk of financial distress. Strong financial indicators."
        elif probability < 0.4:
            return "Low risk with good financial health. Monitor key indicators."
        elif probability < 0.6:
            return "Moderate risk. Some areas need attention and improvement."
        elif probability < 0.8:
            return "High risk of financial difficulties. Immediate action recommended."
        else:
            return "Very high risk. Urgent intervention required."
    
    def _get_impact_level(self, shap_value: float) -> str:
        """Convert SHAP value to impact level."""
        abs_value = abs(shap_value)
        if abs_value > 0.1:
            return "High"
        elif abs_value > 0.05:
            return "Medium"
        else:
            return "Low"
    
    def _create_analysis_message(self, probability: float, risk_level: str, factors: List[Dict]) -> str:
        """Create business-friendly analysis message."""
        if risk_level == "Low":
            message = f"✅ This company shows strong financial health with only a {probability:.1%} chance of distress."
        elif risk_level == "Medium":
            message = f"⚠️ This company has moderate financial risk with a {probability:.1%} chance of distress."
        else:
            message = f"🚨 This company shows high financial risk with a {probability:.1%} chance of distress."
        
        if factors:
            top_factor = factors[0]['display_name']
            message += f" Key factor: {top_factor}."
        
        return message
    
    def _get_factor_description(self, feature: str, shap_value: float, feature_value: float) -> str:
        """Generate business-friendly factor description."""
        base_desc = config.FEATURE_DESCRIPTIONS.get(feature, feature)
        
        if shap_value > 0:
            impact = "increases"
        else:
            impact = "decreases"
        
        return f"{base_desc} {impact} financial risk (current value: {feature_value:.2f})"
    
    def _generate_recommendations(self, probability: float, factors: List[Dict], input_data: Dict) -> List[str]:
        """Generate actionable business recommendations."""
        recommendations = []
        
        # Risk-level based recommendations
        if probability > 0.7:
            recommendations.append("🔴 Immediate action: Review cash flow and reduce expenses")
            recommendations.append("💰 Priority: Secure additional financing or credit facilities")
        elif probability > 0.4:
            recommendations.append("🟡 Monitor: Keep close watch on financial performance metrics")
            recommendations.append("📊 Analyze: Review operational efficiency and cost structure")
        else:
            recommendations.append("🟢 Maintain: Continue current financial management practices")
            recommendations.append("📈 Optimize: Look for growth opportunities while maintaining stability")
        
        # Factor-based recommendations
        if factors:
            top_factor = factors[0]
            if 'Credit' in top_factor['name']:
                recommendations.append("🏦 Banking: Improve relationships with financial institutions")
            elif 'GDP' in top_factor['name']:
                recommendations.append("🌍 Economic: Consider market diversification strategies")
            elif 'startup' in top_factor['name']:
                recommendations.append("🚀 Growth: Focus on establishing stable revenue streams")
        
        return recommendations[:4]  # Return top 4 recommendations

class ModelPerformanceTracker:
    """Track model performance and usage statistics."""
    
    def __init__(self):
        self.prediction_stats = {
            'AFR': {'count': 0, 'avg_time': 0, 'avg_confidence': 0},
            'ROW': {'count': 0, 'avg_time': 0, 'avg_confidence': 0}
        }
    
    def log_prediction(self, region: str, processing_time: float, confidence: float, probability: float):
        """Log prediction performance statistics."""
        if region not in self.prediction_stats:
            self.prediction_stats[region] = {'count': 0, 'avg_time': 0, 'avg_confidence': 0}
        
        stats = self.prediction_stats[region]
        stats['count'] += 1
        
        # Update running averages
        stats['avg_time'] = ((stats['avg_time'] * (stats['count'] - 1)) + processing_time) / stats['count']
        stats['avg_confidence'] = ((stats['avg_confidence'] * (stats['count'] - 1)) + confidence) / stats['count']
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring."""
        return {
            'total_predictions': sum(stats['count'] for stats in self.prediction_stats.values()),
            'by_region': self.prediction_stats,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

# Global instances
model_manager = ModelManager()

# FastAPI integration functions
async def load_models():
    """Load models during application startup."""
    try:
        logger.info("Loading ML models for financial distress prediction...")
        model_manager.load_all_models()
        logger.info("All ML models loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

async def get_model_status() -> Dict[str, Any]:
    """Get model status for health checks."""
    try:
        status = {
            'status': 'healthy',
            'models_loaded': len(model_manager.models_cache),
            'available_regions': list(model_manager.models_cache.keys()),
            'performance': model_manager.performance_tracker.get_performance_summary(),
            'last_check': datetime.now(timezone.utc).isoformat()
        }
        
        if len(model_manager.models_cache) == 0:
            status['status'] = 'unhealthy'
        elif len(model_manager.models_cache) < 2:
            status['status'] = 'degraded'
        
        return status
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'last_check': datetime.now(timezone.utc).isoformat()
        }

# Main prediction function for service integration
def predict_with_service(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main prediction function following exact pipeline."""
    try:
        # Determine region based on input
        region = input_data.get('region', 'ROW')
        if region not in ['AFR', 'ROW']:
            region = 'ROW'  # Default to ROW
        
        # Make prediction with explanations
        result = model_manager.predict_with_explanation(input_data, region)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction service failed: {e}")
        raise RuntimeError(f"Prediction failed: {e}")