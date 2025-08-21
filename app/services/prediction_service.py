# """
# Professional Financial Distress Prediction Service
# Business-focused ML service following pipeline structure with SHAP explanations.
# Designed for international conference presentation and production deployment.
# """

# import os
# import time
# import logging
# from typing import Dict, Any, List, Optional, Tuple
# from datetime import datetime, timezone
# from pathlib import Path

# import pandas as pd
# import numpy as np
# import shap
# import joblib
# from fastapi import HTTPException
# from dotenv import load_dotenv

# load_dotenv()
# logger = logging.getLogger(__name__)

# class PipelineConfig:
#     """Configuration matching exact ML pipeline structure."""
    
#     TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "data/DF_2025.xlsx")
#     MODEL_AFR_PATH = os.getenv("MODEL_AFR_PATH", "ml_pipeline/trained_models/afr_best_pipeline.joblib")
#     MODEL_ROW_PATH = os.getenv("MODEL_ROW_PATH", "ml_pipeline/trained_models/row_best_pipeline.joblib")
    
#     # Pipeline feature sets 
#     AFR_FEATURES = [
#         'startup', 'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf',
#         'Fem_CEO', 'Pvt_Own', 'Con_Own', 'Edu', 'Exports', 'Innov', 'Transp',
#         'Gifting', 'Pol_Inst', 'Infor_Comp', 'Credit', 'WSI', 'WUI', 'GDP', 'PRIME'
#     ]
    
#     ROW_FEATURES = AFR_FEATURES + ['Size', 'MarketCap', 'GPR']
    
#     # Sector mappings
#     SECTOR_MAPPINGS = {
#         'Construction': '1', 'Retail': '2', 'Manufacturing': '3', 'Other Services': '4',
#         'Other Manufacturing': '5', 'Food': '6', 'Garments': '7', 'Hotels': '8',
#         'Services': '9', 'Rest of Universe': '10', 'IT & IT Services': '11', 'Textiles': '12',
#         'Machinery & Equipment': '13', 'Textiles & Garments': '14',
#         'Basic Metals/Fabricated Metals/Machinery & Equip.': '15',
#         'Chemicals, Plastics & Rubber': '16', 'Chemicals & Chemical Products': '17',
#         'Machinery & Equipment & Electronics': '18', 'Leather Products': '19',
#         'Furniture': '20', 'Motor Vehicles & Transport Equip.': '21',
#         'Fabricated Metal Products': '22', 'Hospitality & Tourism': '23',
#         'Motor Vehicles': '24', 'Electronics': '25', 'Services of Motor Vehicles/Wholesale/Retail': '26',
#         'Food/Leather/Wood/Tobacco/Rubber Products': '27', 'Professional Activities': '28',
#         'Non-Metallic Mineral Products': '29', 'Hotels & Restaurants': '30',
#         'Electronics & Communications Equip.': '31', 'Transport, Storage, & Communications': '32',
#         'Services of Motor Vehicles': '33', 'Rubber & Plastics Products': '34',
#         'Basic Metals & Metal Products': '35', 'Wholesale': '36', 'Basic Metals': '37',
#         'Electrical & Computer Products': '38', 'Minerals, Metals, Machinery & Equipment': '39',
#         'Wood Products': '40', 'Printing & Publishing': '41', 'Petroleum products, Plastics & Rubber': '42',
#         'Wood products, Furniture, Paper & Publishing': '43',
#         'Machinery & Equipment, Electronics & Vehicles': '44', 'Transport': '45',
#         'Textiles, Garments & Leather': '46', 'Restaurants': '47',
#         'Wholesale, Including of Motor Vehicles': '48', 'Publishing, Telecommunications & IT': '49',
#         'Wholesale & Retail': '50', 'Mining Related Manufacturing': '51',
#         'Pharmaceuticals & Medical Products': '52', 'Wood Products & Furniture': '53',
#         'Computer, Electronic & Optical Products': '54', 'Retail & IT': '55',
#         'Metals, Machinery, Computers & Electronics': '56', 'Manufacturing Panel': '57',
#         'Retail Panel': '58', 'Other Services Panel': '59',
#         'Chemicals, Non-Metallic Mineral, Plastics & Rubber': '60',
#         'Textiles, Garments, Leather & Paper': '61', 'Pharmaceutical, Chemicals & Chemical Products': '62',
#         'Wholesale of Agri Inputs & Equipment': '63'
#     }
    
#     # African countries classification 
#     AFRICAN_COUNTRIES = {
#         'Angola', 'Bangladesh', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon',
#         'Central African Republic', 'Chad', 'Congo', "Cote d'Ivoire", 'DRC', 'Djibouti', 'Egypt',
#         'Equatorial Guinea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea',
#         'Lebanon', 'Lesotho', 'Liberia', 'Guineabissau', 'Kenya', 'Madagascar', 'Malawi',
#         'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger',
#         'Nigeria', 'Rwanda', 'Senegal', 'Seychelles', 'Sierra Leone', 'South Sudan',
#         'Southafrica', 'Sudan', 'Tanzania', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
#     }
    
#     # Business-friendly descriptions
#     BUSINESS_DESCRIPTIONS = {
#         'startup': 'Young company (less than 5 years operational)',
#         'Fin_bank': 'Traditional bank financing dependency',
#         'Fin_supplier': 'Supplier credit and trade financing',
#         'Fin_equity': 'Investor capital and equity funding',
#         'Fin_other': 'Alternative financing sources',
#         'Fem_wf': 'Female workforce participation',
#         'Fem_CEO': 'Female leadership representation',
#         'Pvt_Own': 'Private ownership concentration',
#         'Con_Own': 'Ownership concentration levels',
#         'Edu': 'Workforce education and skills',
#         'Exports': 'International market presence',
#         'Innov': 'Innovation and technology investment',
#         'Transp': 'Transportation and logistics access',
#         'Gifting': 'Governance and compliance quality',
#         'Pol_Inst': 'Political stability environment',
#         'Infor_Comp': 'Information access and competition',
#         'Credit': 'Credit market accessibility',
#         'WSI': 'Global economic strength indicators',
#         'WUI': 'Economic uncertainty measures',
#         'GDP': 'Regional economic growth',
#         'PRIME': 'Interest rate environment',
#         'Size': 'Company scale and capacity',
#         'MarketCap': 'Market valuation metrics',
#         'GPR': 'Geopolitical risk exposure'
#     }

# config = PipelineConfig()

# class ModelManager:
#     """Manage ML model loading and caching."""
    
#     def __init__(self):
#         self.models = {}
    
#     def load_model(self, region: str) -> Any:
#         """Load and cache the model for the specified region."""
#         if region not in self.models:
#             model_path = config.MODEL_AFR_PATH if region == 'AFR' else config.MODEL_ROW_PATH
#             if not Path(model_path).exists():
#                 raise HTTPException(
#                     status_code=503,
#                     detail=f"Model file for {region} region not found at {model_path}"
#                 )
#             try:
#                 self.models[region] = joblib.load(model_path)
#                 logger.info(f"Loaded model for {region} from {model_path}")
#             except Exception as e:
#                 logger.error(f"Failed to load model for {region}: {e}")
#                 raise HTTPException(
#                     status_code=503,
#                     detail=f"Failed to load model for {region} region"
#                 )
#         return self.models[region]
    
#     def get_model(self, region: str) -> Any:
#         """Get the model for the specified region."""
#         return self.load_model(region)

# model_manager = ModelManager()

# class PipelineDataProcessor:
#     """Data processor following ML pipeline preprocessing steps."""
    
#     @staticmethod
#     def validate_input_data(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
#         """Validate business input with user-friendly error messages."""
#         validated_data = data.copy()
#         warnings = []
#         errors = []
        
#         # Essential business fields validation
#         if not data.get('stra_sector'):
#             errors.append("Business sector is required - please select your industry")
        
#         if not data.get('wk14') or float(data.get('wk14', 0)) <= 0:
#             errors.append("Years of operation is required")
            
#         if not data.get('car1') or float(data.get('car1', 0)) <= 0:
#             errors.append("Company establishment year is required")
        
#         # Business logic validation
#         try:
#             wk14 = float(data.get('wk14', 0))
#             car1 = float(data.get('car1', 0))
            
#             if wk14 > car1:
#                 warnings.append("Years operating adjusted to match company age")
#                 validated_data['wk14'] = car1
                
#         except (ValueError, TypeError):
#             errors.append("Company age values must be valid numbers")
        
#         # Validate percentage fields
#         percentage_fields = [
#             'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fin_int',
#             'Fem_wf', 'Fem_CEO', 'Pvt_Own', 'Con_Own', 'obst9', 'tr15', 't10',
#             't2', 'corr4', 'obst11', 'infor1', 'Credit', 'GDP', 'PRIME'
#         ]
        
#         for field in percentage_fields:
#             if field in data and data[field] is not None:
#                 try:
#                     value = float(data[field])
#                     if value < 0:
#                         warnings.append(f"{field} adjusted from negative to zero")
#                         validated_data[field] = 0
#                     elif value > 100:
#                         warnings.append(f"{field} appears over 100% - please verify")
#                 except (ValueError, TypeError):
#                     warnings.append(f"{field} must be a valid number")
#                     validated_data[field] = 0
        
#         if errors:
#             raise HTTPException(
#                 status_code=422,
#                 detail={
#                     "error": "Invalid company data",
#                     "message": "Please correct the following issues:",
#                     "details": errors,
#                     "warnings": warnings
#                 }
#             )
        
#         return validated_data, warnings
    
#     @staticmethod
#     def preprocess_pipeline_data(data: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
#         """Preprocess data following the pipeline steps."""
#         try:
#             # Validate input first
#             validated_data, warnings = PipelineDataProcessor.validate_input_data(data)
            
#             # Determine region (exact pipeline logic)
#             country = validated_data.get('country2', '')
#             if country in config.AFRICAN_COUNTRIES:
#                 region = 'AFR'
#             else:
#                 region = 'ROW'
            
#             logger.info(f"Processing {region} region data following pipeline structure")
            
#             # Create dataframe
#             df = pd.DataFrame([validated_data])
            
#             # Rename columns (exact from pipeline)
#             column_mapping = {
#                 'fin1': 'Fin_int', 'fin2': 'Fin_bank', 'fin3': 'Fin_supplier', 
#                 'fin4': 'Fin_equity', 'fin5': 'Fin_other', 'gend2': 'Fem_wf', 
#                 'gend3': 'Fem_Wf_Non_Prod', 'gend4': 'Fem_CEO', 'gend6': 'Fem_Own', 
#                 'car3': 'For_Own', 'car2': 'Pvt_Own', 'car6': 'Con_Own',
#                 'obst9': 'Edu', 'tr15': 'Exports', 't10': 'Innov', 't2': 'Transp', 
#                 'corr4': 'Gifting', 'obst11': 'Pol_Inst', 'infor1': 'Infor_Comp', 
#                 'size2': 'Size'
#             }
            
#             for old_col, new_col in column_mapping.items():
#                 if old_col in df.columns:
#                     df[new_col] = df[old_col]
            
#             # Map sector 
#             sector_value = str(validated_data.get('stra_sector', 'Other Services'))
#             if sector_value in config.SECTOR_MAPPINGS:
#                 df['Sector'] = float(config.SECTOR_MAPPINGS[sector_value])
#             else:
#                 df['Sector'] = 4.0  # Default to Other Services
            
#             # Create distress and startup variables 
#             perf1 = df.get('perf1', pd.Series([0])).fillna(0).iloc[0]
#             obst1 = df.get('obst1', pd.Series([0])).fillna(0).iloc[0]
#             fin33 = df.get('fin33', pd.Series([0])).fillna(0).iloc[0]
#             fin16 = df.get('fin16', pd.Series([0])).fillna(0).iloc[0]
            
#             # Distress calculation 
#             df['distress'] = np.where(
#                 (perf1 < 0) & ((obst1 == 100) | (fin33 == 1) | (fin16 == 1)), 1, 0
#             )
            
#             # Startup calculation 
#             wk14 = df.get('wk14', pd.Series([5])).fillna(5).iloc[0]
#             car1 = df.get('car1', pd.Series([5])).fillna(5).iloc[0]
#             df['startup'] = np.where((wk14 < 5) & (car1 < 5), 1, 0)
            
#             # Fill NaN with 0 
#             df.fillna(0, inplace=True)
            
#             # Scale percentage features 
#             percentage_cols = [
#                 'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf', 'Fem_CEO',
#                 'Pvt_Own', 'Con_Own', 'Edu', 'Exports', 'Innov', 'Transp', 'Gifting',
#                 'Pol_Inst', 'Infor_Comp', 'Credit', 'PRIME', 'GDP'
#             ]
            
#             for col in percentage_cols:
#                 if col in df.columns:
#                     df[col] = df[col].apply(lambda x: x / 100 if x > 1 else x)
#                     df[col] = df[col].apply(lambda x: 0 if x < 0 else x)
            
#             # Scale MarketCap for ROW 
#             if region == 'ROW' and 'MarketCap' in df.columns:
#                 df['MarketCap'] = df['MarketCap'].apply(lambda x: x / 100 if x > 1 else x)
#                 df['MarketCap'] = df['MarketCap'].apply(lambda x: 0 if x < 0 else x)
            
#             # Select features based on region 
#             if region == 'AFR':
#                 feature_columns = config.AFR_FEATURES
#             else:
#                 feature_columns = config.ROW_FEATURES
            
#             # Ensure all required features exist
#             for feature in feature_columns:
#                 if feature not in df.columns:
#                     df[feature] = 0.0
            
#             # Select final features
#             processed_df = df[feature_columns].copy()
            
#             logger.info(f"Successfully processed {region} data with {len(feature_columns)} features")
#             return processed_df, region
            
#         except HTTPException:
#             raise
#         except Exception as e:
#             logger.error(f"Pipeline preprocessing failed: {e}")
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Unable to process company data: {str(e)}"
#             )

# class BusinessInsightGenerator:
#     """Generate business-friendly insights and explanations."""
    
#     @staticmethod
#     def calculate_risk_level(probability: float, confidence: float) -> Tuple[str, str, str]:
#         """Calculate business risk level with confidence adjustment."""
#         # Adjust for confidence
#         adjusted_prob = probability * max(0.7, confidence)
        
#         if adjusted_prob <= 0.2:
#             risk_level = "Low"
#             health_status = "Financially Stable"
#             detail = "Strong financial position with low distress risk"
#         elif adjusted_prob <= 0.4:
#             risk_level = "Medium"
#             health_status = "Generally Stable"
#             detail = "Good financial health with some areas to monitor"
#         else:
#             risk_level = "High"
#             health_status = "Requires Attention"
#             detail = "Elevated risk requiring strategic management"
        
#         return risk_level, health_status, detail
    
#     @staticmethod
#     def generate_analysis_message(probability: float, risk_level: str, region: str) -> str:
#         """Generate comprehensive business analysis message."""
#         market_context = "African market" if region == "AFR" else "global market"
        
#         if risk_level == "Low":
#             message = (
#                 f"Your company demonstrates strong financial health in the {market_context}. "
#                 f"With only a {probability:.1%} probability of financial distress, your business "
#                 f"shows solid performance across key indicators. This is an excellent foundation "
#                 f"for strategic growth and expansion initiatives."
#             )
#         elif risk_level == "Medium":
#             message = (
#                 f"Your company shows moderate financial health in the {market_context}. "
#                 f"While the {probability:.1%} distress probability indicates manageable risk, "
#                 f"there are opportunities to strengthen your financial position through "
#                 f"targeted improvements in key operational areas."
#             )
#         else:
#             message = (
#                 f"Your company faces elevated financial risk in the {market_context}. "
#                 f"The {probability:.1%} probability of distress signals the need for immediate "
#                 f"strategic attention. Focus on addressing key risk factors and implementing "
#                 f"comprehensive financial management strategies."
#             )
        
#         return message
    
#     @staticmethod
#     def generate_recommendations(risk_level: str, factors: List[Dict]) -> List[str]:
#         """Generate actionable business recommendations."""
#         recommendations = []
        
#         # Risk-level specific strategies
#         if risk_level == "High":
#             recommendations.extend([
#                 "Implement immediate cash flow monitoring and control systems",
#                 "Explore emergency funding options and credit facilities",
#                 "Conduct comprehensive operational cost review and reduction",
#                 "Consider strategic partnerships or investor involvement",
#                 "Engage crisis management professionals for guidance"
#             ])
#         elif risk_level == "Medium":
#             recommendations.extend([
#                 "Establish comprehensive financial monitoring dashboard",
#                 "Diversify revenue streams and customer base",
#                 "Build strategic cash reserves (3-6 months expenses)",
#                 "Optimize operational efficiency and cost structure",
#                 "Review and strengthen financial controls"
#             ])
#         else:  # Low risk
#             recommendations.extend([
#                 "Maintain current financial discipline while exploring growth",
#                 "Continue monitoring key performance indicators",
#                 "Consider strategic investments in innovation or expansion",
#                 "Build on existing competitive advantages",
#                 "Explore new market opportunities"
#             ])
        
#         # Factor-specific recommendations
#         if factors:
#             top_factor = factors[0].get('name', '')
#             if 'Fin_bank' in top_factor or 'Credit' in top_factor:
#                 recommendations.append("Strengthen banking relationships and explore diverse financing")
#             elif 'startup' in top_factor:
#                 recommendations.append("Focus on building consistent revenue and market credibility")
#             elif 'GDP' in top_factor or 'Pol_Inst' in top_factor:
#                 recommendations.append("Consider market diversification to reduce regional risk")
        
#         return recommendations[:6]  # Top 6 recommendations
    
#     @staticmethod
#     def prepare_visualization_data(probability: float, risk_level: str, factors: List[Dict]) -> Dict[str, Any]:
#         """Prepare data structures for frontend visualizations."""
        
#         # Risk gauge configuration
#         risk_gauge = {
#             'value': probability,
#             'level': risk_level,
#             'color': {
#                 'Low': '#22c55e',
#                 'Medium': '#f59e0b', 
#                 'High': '#ef4444'
#             }.get(risk_level, '#6b7280'),
#             'thresholds': [
#                 {'name': 'Low Risk', 'value': 0.2, 'color': '#22c55e'},
#                 {'name': 'Medium Risk', 'value': 0.4, 'color': '#f59e0b'},
#                 {'name': 'High Risk', 'value': 1.0, 'color': '#ef4444'}
#             ]
#         }
        
#         # Factor importance chart data
#         factor_chart = []
#         for i, factor in enumerate(factors[:5]):
#             factor_chart.append({
#                 'name': factor.get('name', f'Factor {i+1}'),
#                 'value': abs(factor.get('weight', 0)),
#                 'impact': factor.get('impact_level', 'Medium'),
#                 'description': factor.get('description', ''),
#                 'color': {
#                     'High': '#ef4444',
#                     'Medium': '#f59e0b',
#                     'Low': '#22c55e'
#                 }.get(factor.get('impact_level', 'Medium'), '#6b7280')
#             })
        
#         # Benchmark comparison
#         benchmark_data = {
#             'user_score': probability,
#             'industry_average': 0.25,
#             'market_average': 0.30,
#             'performance': 'Above Average' if probability < 0.25 else 'Below Average'
#         }
        
#         return {
#             'risk_gauge': risk_gauge,
#             'factor_chart': factor_chart,
#             'benchmark_comparison': benchmark_data,
#             'chart_ready': True
#         }

# class SHAPBusinessAnalyzer:
#     """SHAP analysis with business-friendly interpretations."""
    
#     @staticmethod
#     def analyze_feature_importance(model, processed_input: pd.DataFrame, num_factors: int = 5) -> List[Dict[str, Any]]:
#         """Extract SHAP values and convert to business insights using KernelExplainer."""
#         try:
#             # Get model components
#             classifier = model.named_steps['classifier']
#             preprocessor = model.named_steps['preprocessor']
            
#             # Transform input
#             transformed_input = preprocessor.transform(processed_input)
            
#             # Create SHAP KernelExplainer
#             explainer = shap.KernelExplainer(
#                 model.predict_proba,
#                 transformed_input,  # Use transformed input as background data
#                 link="logit"
#             )
#             shap_values = explainer.shap_values(transformed_input, nsamples=100)  # Adjust nsamples for performance
            
#             # Handle SHAP output format
#             if isinstance(shap_values, list):
#                 shap_values = shap_values[1]  # Positive class for binary classification
            
#             if shap_values.ndim > 1:
#                 shap_values = shap_values[0]  # First sample
            
#             # Get feature names
#             try:
#                 feature_names = preprocessor.get_feature_names_out()
#             except AttributeError:
#                 feature_names = processed_input.columns.tolist()
            
#             # Create importance ranking
#             importance_data = []
#             for i, (feature_name, shap_val) in enumerate(zip(feature_names, shap_values)):
                
#                 # Clean feature name
#                 clean_name = feature_name.replace('num__', '').replace('cat__', '')
#                 business_name = config.BUSINESS_DESCRIPTIONS.get(clean_name, clean_name.replace('_', ' ').title())
                
#                 # Determine impact level
#                 abs_shap = abs(shap_val)
#                 impact_level = "High" if abs_shap > 0.1 else "Medium" if abs_shap > 0.05 else "Low"
                
#                 # Get feature value
#                 feature_value = processed_input[clean_name].iloc[0] if clean_name in processed_input.columns else 0.0
                
#                 importance_data.append({
#                     'name': business_name,
#                     'impact_level': impact_level,
#                     'weight': float(abs_shap),
#                     'description': config.BUSINESS_DESCRIPTIONS.get(clean_name, f"This factor affects financial distress risk"),
#                     'shap_value': float(shap_val),
#                     'feature_value': float(feature_value)
#                 })
            
#             # Sort by absolute importance and return top factors
#             importance_data.sort(key=lambda x: x['weight'], reverse=True)
            
#             logger.info(f"SHAP analysis completed with {len(importance_data)} factors")
#             return importance_data[:num_factors]
            
#         except Exception as e:
#             logger.warning(f"SHAP analysis failed: {e}")
#             return SHAPBusinessAnalyzer._get_fallback_factors(processed_input, num_factors)
    
#     @staticmethod
#     def _get_fallback_factors(processed_input: pd.DataFrame, num_factors: int) -> List[Dict[str, Any]]:
#         """Provide fallback factors when SHAP fails."""
#         important_features = ['Fin_bank', 'Credit', 'startup', 'GDP', 'Pol_Inst']
        
#         factors = []
#         for i, feature in enumerate(important_features[:num_factors]):
#             if feature in processed_input.columns:
#                 value = processed_input[feature].iloc[0]
                
#                 factors.append({
#                     'name': config.BUSINESS_DESCRIPTIONS.get(feature, feature),
#                     'impact_level': 'High' if i < 2 else 'Medium',
#                     'weight': 0.1 * (1 - i * 0.02),
#                     'description': config.BUSINESS_DESCRIPTIONS.get(feature, 'Important business factor'),
#                     'shap_value': 0.1 * (1 - i * 0.02),
#                     'feature_value': float(value)
#                 })
        
#         return factors

# class FinancialDistressPredictionService:
#     """Main prediction service following exact ML pipeline."""
    
#     @staticmethod
#     def predict_financial_distress(input_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Generate comprehensive financial distress prediction."""
        
#         prediction_start = time.time()
        
#         try:
#             logger.info("Starting financial distress analysis...")
            
#             # Process input data following exact pipeline
#             processed_input, region = PipelineDataProcessor.preprocess_pipeline_data(input_data)
            
#             # Get appropriate model
#             model = model_manager.get_model(region)
#             if model is None:
#                 raise HTTPException(
#                     status_code=503,
#                     detail=f"Prediction model for {region} region is temporarily unavailable"
#                 )
            
#             # Make prediction using pipeline model
#             prediction_proba = model.predict_proba(processed_input)
#             distress_probability = float(prediction_proba[0][1])  # Probability of distress class
            
#             # Calculate model confidence
#             confidence = float(max(prediction_proba[0]))
            
#             # Generate business insights
#             risk_level, health_status, risk_detail = BusinessInsightGenerator.calculate_risk_level(
#                 distress_probability, confidence
#             )
            
#             # Create analysis message
#             analysis_message = BusinessInsightGenerator.generate_analysis_message(
#                 distress_probability, risk_level, region
#             )
            
#             # SHAP feature importance analysis
#             influencing_factors = SHAPBusinessAnalyzer.analyze_feature_importance(
#                 model, processed_input, num_factors=5
#             )
            
#             # Generate recommendations
#             recommendations = BusinessInsightGenerator.generate_recommendations(
#                 risk_level, influencing_factors
#             )
            
#             # Prepare visualization data for frontend
#             visualization_data = BusinessInsightGenerator.prepare_visualization_data(
#                 distress_probability, risk_level, influencing_factors
#             )
            
#             # Compile comprehensive result
#             result = {
#                 'financial_distress_probability': distress_probability,
#                 'model_confidence': confidence,
#                 'risk_category': risk_level,
#                 'financial_health_status': health_status,
#                 'risk_level_detail': risk_detail,
#                 'analysis_message': analysis_message,
#                 'key_influencing_factors': influencing_factors,
#                 'recommendations': recommendations,
#                 'visualization_data': visualization_data,
#                 'benchmark_comparisons': {
#                     'industry_avg': 0.25,
#                     'region_avg': 0.30
#                 }
#             }
            
#             processing_time = time.time() - prediction_start
#             logger.info(
#                 f"Analysis completed: {region} region, {risk_level} risk, "
#                 f"{distress_probability:.3f} probability, {processing_time:.3f}s"
#             )
            
#             return result
            
#         except HTTPException:
#             raise
#         except Exception as e:
#             processing_time = time.time() - prediction_start
#             logger.error(f"Prediction failed after {processing_time:.3f}s: {e}")
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Financial analysis failed: {str(e)}"
#             )

# def predict_with_service(input_data: Dict[str, Any]) -> Dict[str, Any]:
#     """Main prediction service function following exact pipeline."""
#     return FinancialDistressPredictionService.predict_financial_distress(input_data)

"""
Professional Financial Distress Prediction Service
Business-focused ML service following pipeline structure with SHAP explanations.
Designed for international conference presentation and production deployment.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import shap
import joblib
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class PipelineConfig:
    """Configuration matching exact ML pipeline structure."""
    
    TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "data/DF_2025.xlsx")
    MODEL_AFR_PATH = os.getenv("MODEL_AFR_PATH", "ml_pipeline/trained_models/afr_best_pipeline.joblib")
    MODEL_ROW_PATH = os.getenv("MODEL_ROW_PATH", "ml_pipeline/trained_models/row_best_pipeline.joblib")
    
    # Pipeline feature sets 
    AFR_FEATURES = [
        'startup', 'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf',
        'Fem_CEO', 'Pvt_Own', 'Con_Own', 'Edu', 'Exports', 'Innov', 'Transp',
        'Gifting', 'Pol_Inst', 'Infor_Comp', 'Credit', 'WSI', 'WUI', 'GDP', 'PRIME'
    ]
    
    ROW_FEATURES = AFR_FEATURES + ['Size', 'MarketCap', 'GPR']
    
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
    
    # African countries classification 
    AFRICAN_COUNTRIES = {
        'Angola', 'Bangladesh', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon',
        'Central African Republic', 'Chad', 'Congo', "Cote d'Ivoire", 'DRC', 'Djibouti', 'Egypt',
        'Equatorial Guinea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea',
        'Lebanon', 'Lesotho', 'Liberia', 'Guineabissau', 'Kenya', 'Madagascar', 'Malawi',
        'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger',
        'Nigeria', 'Rwanda', 'Senegal', 'Seychelles', 'Sierra Leone', 'South Sudan',
        'Southafrica', 'Sudan', 'Tanzania', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
    }
    
    # Business-friendly descriptions
    BUSINESS_DESCRIPTIONS = {
        'startup': 'Young company (less than 5 years operational)',
        'Fin_bank': 'Traditional bank financing dependency',
        'Fin_supplier': 'Supplier credit and trade financing',
        'Fin_equity': 'Investor capital and equity funding',
        'Fin_other': 'Alternative financing sources',
        'Fem_wf': 'Female workforce participation',
        'Fem_CEO': 'Female leadership representation',
        'Pvt_Own': 'Private ownership concentration',
        'Con_Own': 'Ownership concentration levels',
        'Edu': 'Workforce education and skills',
        'Exports': 'International market presence',
        'Innov': 'Innovation and technology investment',
        'Transp': 'Transportation and logistics access',
        'Gifting': 'Governance and compliance quality',
        'Pol_Inst': 'Political stability environment',
        'Infor_Comp': 'Information access and competition',
        'Credit': 'Credit market accessibility',
        'WSI': 'Global economic strength indicators',
        'WUI': 'Economic uncertainty measures',
        'GDP': 'Regional economic growth',
        'PRIME': 'Interest rate environment',
        'Size': 'Company scale and capacity',
        'MarketCap': 'Market valuation metrics',
        'GPR': 'Geopolitical risk exposure'
    }

config = PipelineConfig()

class ModelManager:
    """Manage ML model loading and caching."""
    
    def __init__(self):
        self.models = {}
    
    def load_model(self, region: str) -> Any:
        """Load and cache the model for the specified region."""
        if region not in self.models:
            model_path = config.MODEL_AFR_PATH if region == 'AFR' else config.MODEL_ROW_PATH
            if not Path(model_path).exists():
                raise HTTPException(
                    status_code=503,
                    detail=f"Model file for {region} region not found at {model_path}"
                )
            try:
                self.models[region] = joblib.load(model_path)
                logger.info(f"Loaded model for {region} from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model for {region}: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Failed to load model for {region} region"
                )
        return self.models[region]
    
    def get_model(self, region: str) -> Any:
        """Get the model for the specified region."""
        return self.load_model(region)

model_manager = ModelManager()

class PipelineDataProcessor:
    """Data processor following ML pipeline preprocessing steps."""
    
    @staticmethod
    def validate_input_data(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Validate business input with user-friendly error messages."""
        validated_data = data.copy()
        warnings = []
        errors = []
        
        # Essential business fields validation
        if not data.get('stra_sector'):
            errors.append("Business sector is required - please select your industry")
        
        if not data.get('wk14') or float(str(data.get('wk14', 0)).replace(",", "").strip() or 0) <= 0:
            errors.append("Years of operation is required")
            
        if not data.get('car1') or float(str(data.get('car1', 0)).replace(",", "").strip() or 0) <= 0:
            errors.append("Company establishment year is required")
        
        # Business logic validation
        try:
            wk14 = float(str(data.get('wk14', 0)).replace(",", "").strip() or 0)
            car1 = float(str(data.get('car1', 0)).replace(",", "").strip() or 0)
            
            if wk14 > car1:
                warnings.append("Years operating adjusted to match company age")
                validated_data['wk14'] = car1
                
        except (ValueError, TypeError):
            errors.append("Company age values must be valid numbers")
        
        # Validate percentage fields
        percentage_fields = [
            'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fin_int',
            'Fem_wf', 'Fem_CEO', 'Pvt_Own', 'Con_Own', 'obst9', 'tr15', 't10',
            't2', 'corr4', 'obst11', 'infor1', 'Credit', 'GDP', 'PRIME'
        ]
        
        for field in percentage_fields:
            if field in data and data[field] is not None:
                try:
                    # Convert to float explicitly before comparison
                    value = float(str(data[field]).replace(",", "").strip() or 0)
                    if value < 0:
                        warnings.append(f"{field} adjusted from negative to zero")
                        validated_data[field] = 0
                    elif value > 100:
                        warnings.append(f"{field} appears over 100% - please verify")
                except (ValueError, TypeError):
                    warnings.append(f"{field} must be a valid number")
                    validated_data[field] = 0
        
        if errors:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Invalid company data",
                    "message": "Please correct the following issues:",
                    "details": errors,
                    "warnings": warnings
                }
            )
        
        return validated_data, warnings
    
    @staticmethod
    def preprocess_pipeline_data(data: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        """Preprocess data following the pipeline steps."""
        try:
            # Validate input first
            validated_data, warnings = PipelineDataProcessor.validate_input_data(data)
            
            # Determine region (exact pipeline logic)
            country = validated_data.get('country2', '')
            if country in config.AFRICAN_COUNTRIES:
                region = 'AFR'
            else:
                region = 'ROW'
            
            logger.info(f"Processing {region} region data following pipeline structure")
            
            # Create dataframe
            df = pd.DataFrame([validated_data])
            
            # Clean numeric values - remove commas and convert to string then numeric
            for column in df.columns:
                if df[column].dtype == object:  # Only process string columns
                    df[column] = df[column].astype(str).str.replace(',', '').str.strip()
                    # Try to convert to numeric if possible
                    df[column] = pd.to_numeric(df[column], errors='ignore')
            
            # Rename columns (exact from pipeline)
            column_mapping = {
                'fin1': 'Fin_int', 'fin2': 'Fin_bank', 'fin3': 'Fin_supplier', 
                'fin4': 'Fin_equity', 'fin5': 'Fin_other', 'gend2': 'Fem_wf', 
                'gend3': 'Fem_Wf_Non_Prod', 'gend4': 'Fem_CEO', 'gend6': 'Fem_Own', 
                'car3': 'For_Own', 'car2': 'Pvt_Own', 'car6': 'Con_Own',
                'obst9': 'Edu', 'tr15': 'Exports', 't10': 'Innov', 't2': 'Transp', 
                'corr4': 'Gifting', 'obst11': 'Pol_Inst', 'infor1': 'Infor_Comp', 
                'size2': 'Size'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = df[old_col]
            
            # Map sector - ensure it's properly converted to string first
            sector_value = str(validated_data.get('stra_sector', 'Other Services'))
            if sector_value in config.SECTOR_MAPPINGS:
                df['Sector'] = float(config.SECTOR_MAPPINGS[sector_value])
            else:
                df['Sector'] = 4.0  # Default to Other Services
            
            # Convert numeric columns to float before comparison
            numeric_cols = ['perf1', 'obst1', 'fin33', 'fin16', 'wk14', 'car1']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Create distress and startup variables with proper numeric values
            perf1 = float(df.get('perf1', pd.Series([0])).fillna(0).iloc[0])
            obst1 = float(df.get('obst1', pd.Series([0])).fillna(0).iloc[0])
            fin33 = float(df.get('fin33', pd.Series([0])).fillna(0).iloc[0])
            fin16 = float(df.get('fin16', pd.Series([0])).fillna(0).iloc[0])
            
            # Distress calculation 
            df['distress'] = np.where(
                (perf1 < 0) & ((obst1 == 100) | (fin33 == 1) | (fin16 == 1)), 1, 0
            )
            
            # Startup calculation 
            wk14 = float(df.get('wk14', pd.Series([5])).fillna(5).iloc[0])
            car1 = float(df.get('car1', pd.Series([5])).fillna(5).iloc[0])
            df['startup'] = np.where((wk14 < 5) & (car1 < 5), 1, 0)
            
            # Fill NaN with 0 
            df.fillna(0, inplace=True)
            
            # Scale percentage features 
            percentage_cols = [
                'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf', 'Fem_CEO',
                'Pvt_Own', 'Con_Own', 'Edu', 'Exports', 'Innov', 'Transp', 'Gifting',
                'Pol_Inst', 'Infor_Comp', 'Credit', 'PRIME', 'GDP'
            ]
            
            for col in percentage_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    df[col] = df[col].apply(lambda x: float(x) / 100 if float(x) > 1 else float(x))
                    df[col] = df[col].apply(lambda x: 0 if float(x) < 0 else float(x))
            
            # Scale MarketCap for ROW 
            if region == 'ROW' and 'MarketCap' in df.columns:
                df['MarketCap'] = pd.to_numeric(df['MarketCap'], errors='coerce').fillna(0)
                df['MarketCap'] = df['MarketCap'].apply(lambda x: float(x) / 100 if float(x) > 1 else float(x))
                df['MarketCap'] = df['MarketCap'].apply(lambda x: 0 if float(x) < 0 else float(x))
            
            # Select features based on region 
            if region == 'AFR':
                feature_columns = config.AFR_FEATURES
            else:
                feature_columns = config.ROW_FEATURES
            
            # Ensure all required features exist
            for feature in feature_columns:
                if feature not in df.columns:
                    df[feature] = 0.0
                else:
                    # Ensure all feature columns are float
                    df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0.0)
            
            # Select final features
            processed_df = df[feature_columns].copy()
            
            logger.info(f"Successfully processed {region} data with {len(feature_columns)} features")
            return processed_df, region
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Pipeline preprocessing failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Unable to process company data: {str(e)}"
            )

class BusinessInsightGenerator:
    """Generate business-friendly insights and explanations."""
    
    @staticmethod
    def calculate_risk_level(probability: float, confidence: float) -> Tuple[str, str, str]:
        """Calculate business risk level with confidence adjustment."""
        # Adjust for confidence
        adjusted_prob = probability * max(0.7, confidence)
        
        if adjusted_prob <= 0.2:
            risk_level = "Low"
            health_status = "Financially Stable"
            detail = "Strong financial position with low distress risk"
        elif adjusted_prob <= 0.4:
            risk_level = "Medium"
            health_status = "Generally Stable"
            detail = "Good financial health with some areas to monitor"
        else:
            risk_level = "High"
            health_status = "Requires Attention"
            detail = "Elevated risk requiring strategic management"
        
        return risk_level, health_status, detail
    
    @staticmethod
    def generate_analysis_message(probability: float, risk_level: str, region: str) -> str:
        """Generate comprehensive business analysis message."""
        market_context = "African market" if region == "AFR" else "global market"
        
        if risk_level == "Low":
            message = (
                f"Your company demonstrates strong financial health in the {market_context}. "
                f"With only a {probability:.1%} probability of financial distress, your business "
                f"shows solid performance across key indicators. This is an excellent foundation "
                f"for strategic growth and expansion initiatives."
            )
        elif risk_level == "Medium":
            message = (
                f"Your company shows moderate financial health in the {market_context}. "
                f"While the {probability:.1%} distress probability indicates manageable risk, "
                f"there are opportunities to strengthen your financial position through "
                f"targeted improvements in key operational areas."
            )
        else:
            message = (
                f"Your company faces elevated financial risk in the {market_context}. "
                f"The {probability:.1%} probability of distress signals the need for immediate "
                f"strategic attention. Focus on addressing key risk factors and implementing "
                f"comprehensive financial management strategies."
            )
        
        return message
    
    @staticmethod
    def generate_recommendations(risk_level: str, factors: List[Dict]) -> List[str]:
        """Generate actionable business recommendations."""
        recommendations = []
        
        # Risk-level specific strategies
        if risk_level == "High":
            recommendations.extend([
                "Implement immediate cash flow monitoring and control systems",
                "Explore emergency funding options and credit facilities",
                "Conduct comprehensive operational cost review and reduction",
                "Consider strategic partnerships or investor involvement",
                "Engage crisis management professionals for guidance"
            ])
        elif risk_level == "Medium":
            recommendations.extend([
                "Establish comprehensive financial monitoring dashboard",
                "Diversify revenue streams and customer base",
                "Build strategic cash reserves (3-6 months expenses)",
                "Optimize operational efficiency and cost structure",
                "Review and strengthen financial controls"
            ])
        else:  # Low risk
            recommendations.extend([
                "Maintain current financial discipline while exploring growth",
                "Continue monitoring key performance indicators",
                "Consider strategic investments in innovation or expansion",
                "Build on existing competitive advantages",
                "Explore new market opportunities"
            ])
        
        # Factor-specific recommendations
        if factors:
            top_factor = factors[0].get('name', '')
            if 'Fin_bank' in top_factor or 'Credit' in top_factor:
                recommendations.append("Strengthen banking relationships and explore diverse financing")
            elif 'startup' in top_factor:
                recommendations.append("Focus on building consistent revenue and market credibility")
            elif 'GDP' in top_factor or 'Pol_Inst' in top_factor:
                recommendations.append("Consider market diversification to reduce regional risk")
        
        return recommendations[:6]  # Top 6 recommendations
    
    @staticmethod
    def prepare_visualization_data(probability: float, risk_level: str, factors: List[Dict]) -> Dict[str, Any]:
        """Prepare data structures for frontend visualizations."""
        
        # Risk gauge configuration
        risk_gauge = {
            'value': probability,
            'level': risk_level,
            'color': {
                'Low': '#22c55e',
                'Medium': '#f59e0b', 
                'High': '#ef4444'
            }.get(risk_level, '#6b7280'),
            'thresholds': [
                {'name': 'Low Risk', 'value': 0.2, 'color': '#22c55e'},
                {'name': 'Medium Risk', 'value': 0.4, 'color': '#f59e0b'},
                {'name': 'High Risk', 'value': 1.0, 'color': '#ef4444'}
            ]
        }
        
        # Factor importance chart data
        factor_chart = []
        for i, factor in enumerate(factors[:5]):
            factor_chart.append({
                'name': factor.get('name', f'Factor {i+1}'),
                'value': abs(factor.get('weight', 0)),
                'impact': factor.get('impact_level', 'Medium'),
                'description': factor.get('description', ''),
                'color': {
                    'High': '#ef4444',
                    'Medium': '#f59e0b',
                    'Low': '#22c55e'
                }.get(factor.get('impact_level', 'Medium'), '#6b7280')
            })
        
        # Benchmark comparison
        benchmark_data = {
            'user_score': probability,
            'industry_average': 0.25,
            'market_average': 0.30,
            'performance': 'Above Average' if probability < 0.25 else 'Below Average'
        }
        
        return {
            'risk_gauge': risk_gauge,
            'factor_chart': factor_chart,
            'benchmark_comparison': benchmark_data,
            'chart_ready': True
        }

class SHAPBusinessAnalyzer:
    """SHAP analysis with business-friendly interpretations."""
    
    @staticmethod
    def analyze_feature_importance(model, processed_input: pd.DataFrame, num_factors: int = 5) -> List[Dict[str, Any]]:
        """Extract SHAP values and convert to business insights using KernelExplainer."""
        try:
            # Get model components
            classifier = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocessor']
            
            # Transform input
            transformed_input = preprocessor.transform(processed_input)
            
            # Create SHAP KernelExplainer
            explainer = shap.KernelExplainer(
                model.predict_proba,
                transformed_input,  # Use transformed input as background data
                link="logit"
            )
            shap_values = explainer.shap_values(transformed_input, nsamples=100)  # Adjust nsamples for performance
            
            # Handle SHAP output format
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class for binary classification
            
            if shap_values.ndim > 1:
                shap_values = shap_values[0]  # First sample
            
            # Get feature names
            try:
                feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                feature_names = processed_input.columns.tolist()
            
            # Create importance ranking
            importance_data = []
            for i, (feature_name, shap_val) in enumerate(zip(feature_names, shap_values)):
                
                # Clean feature name
                clean_name = feature_name.replace('num__', '').replace('cat__', '')
                business_name = config.BUSINESS_DESCRIPTIONS.get(clean_name, clean_name.replace('_', ' ').title())
                
                # Determine impact level
                abs_shap = abs(shap_val)
                impact_level = "High" if abs_shap > 0.1 else "Medium" if abs_shap > 0.05 else "Low"
                
                # Get feature value
                feature_value = processed_input[clean_name].iloc[0] if clean_name in processed_input.columns else 0.0
                
                importance_data.append({
                    'name': business_name,
                    'impact_level': impact_level,
                    'weight': float(abs_shap),
                    'description': config.BUSINESS_DESCRIPTIONS.get(clean_name, f"This factor affects financial distress risk"),
                    'shap_value': float(shap_val),
                    'feature_value': float(feature_value)
                })
            
            # Sort by absolute importance and return top factors
            importance_data.sort(key=lambda x: x['weight'], reverse=True)
            
            logger.info(f"SHAP analysis completed with {len(importance_data)} factors")
            return importance_data[:num_factors]
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            return SHAPBusinessAnalyzer._get_fallback_factors(processed_input, num_factors)
    
    @staticmethod
    def _get_fallback_factors(processed_input: pd.DataFrame, num_factors: int) -> List[Dict[str, Any]]:
        """Provide fallback factors when SHAP fails."""
        important_features = ['Fin_bank', 'Credit', 'startup', 'GDP', 'Pol_Inst']
        
        factors = []
        for i, feature in enumerate(important_features[:num_factors]):
            if feature in processed_input.columns:
                value = processed_input[feature].iloc[0]
                
                factors.append({
                    'name': config.BUSINESS_DESCRIPTIONS.get(feature, feature),
                    'impact_level': 'High' if i < 2 else 'Medium',
                    'weight': 0.1 * (1 - i * 0.02),
                    'description': config.BUSINESS_DESCRIPTIONS.get(feature, 'Important business factor'),
                    'shap_value': 0.1 * (1 - i * 0.02),
                    'feature_value': float(value)
                })
        
        return factors

class FinancialDistressPredictionService:
    """Main prediction service following exact ML pipeline."""
    
    @staticmethod
    def predict_financial_distress(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive financial distress prediction."""
        
        prediction_start = time.time()
        
        try:
            logger.info("Starting financial distress analysis...")
            
            # Process input data following exact pipeline
            processed_input, region = PipelineDataProcessor.preprocess_pipeline_data(input_data)
            
            # Get appropriate model
            model = model_manager.get_model(region)
            if model is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Prediction model for {region} region is temporarily unavailable"
                )
            
            # Make prediction using pipeline model
            prediction_proba = model.predict_proba(processed_input)
            distress_probability = float(prediction_proba[0][1])  # Probability of distress class
            
            # Calculate model confidence
            confidence = float(max(prediction_proba[0]))
            
            # Generate business insights
            risk_level, health_status, risk_detail = BusinessInsightGenerator.calculate_risk_level(
                distress_probability, confidence
            )
            
            # Create analysis message
            analysis_message = BusinessInsightGenerator.generate_analysis_message(
                distress_probability, risk_level, region
            )
            
            # SHAP feature importance analysis
            influencing_factors = SHAPBusinessAnalyzer.analyze_feature_importance(
                model, processed_input, num_factors=5
            )
            
            # Generate recommendations
            recommendations = BusinessInsightGenerator.generate_recommendations(
                risk_level, influencing_factors
            )
            
            # Prepare visualization data for frontend
            visualization_data = BusinessInsightGenerator.prepare_visualization_data(
                distress_probability, risk_level, influencing_factors
            )
            
            # Compile comprehensive result
            result = {
                'financial_distress_probability': distress_probability,
                'model_confidence': confidence,
                'risk_category': risk_level,
                'financial_health_status': health_status,
                'risk_level_detail': risk_detail,
                'analysis_message': analysis_message,
                'key_influencing_factors': influencing_factors,
                'recommendations': recommendations,
                'visualization_data': visualization_data,
                'benchmark_comparisons': {
                    'industry_avg': 0.25,
                    'region_avg': 0.30
                }
            }
            
            processing_time = time.time() - prediction_start
            logger.info(
                f"Analysis completed: {region} region, {risk_level} risk, "
                f"{distress_probability:.3f} probability, {processing_time:.3f}s"
            )
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            processing_time = time.time() - prediction_start
            logger.error(f"Prediction failed after {processing_time:.3f}s: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Financial analysis failed: {str(e)}"
            )

def predict_with_service(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main prediction service function following exact pipeline."""
    return FinancialDistressPredictionService.predict_financial_distress(input_data)