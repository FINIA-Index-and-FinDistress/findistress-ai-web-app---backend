"""
Pydantic Schemas for Financial Distress Prediction API
Business-focused data validation aligned with ML pipeline structure.
"""

from pydantic import BaseModel, EmailStr, Field, ConfigDict, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import logging
from app.models.auth_schemas import Token

logger = logging.getLogger(__name__)

# Business-friendly enums
class RiskCategory(str, Enum):
    """Risk level classification for business users."""
    LOW = "Low"
    MEDIUM = "Medium" 
    HIGH = "High"

class ImpactLevel(str, Enum):
    """Factor impact level classification."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class RegionType(str, Enum):
    """Market regions matching ML pipeline."""
    AFR = "AFR"  # African markets
    ROW = "ROW"  # Rest of world markets

class UserRole(str, Enum):
    """User access roles."""
    USER = "user"
    ADMIN = "admin"
    ANALYST = "analyst"

class FinancialHealthStatus(str, Enum):
    """Overall financial health status."""
    STABLE = "Financially Stable"
    CONCERNS = "Some Financial Concerns"
    RISK = "Significant Financial Risk"

# USER MANAGEMENT SCHEMAS

class UserBase(BaseModel):
    """Base user information."""
    username: str = Field(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_]+$", 
                         description="Unique username for login")
    email: Optional[EmailStr] = Field(default=None, description="Email address for notifications")
    full_name: Optional[str] = Field(default=None, max_length=100, description="Full display name")
    
    # Business context
    company_name: Optional[str] = Field(default=None, max_length=200, description="Company name")
    industry: Optional[str] = Field(default=None, max_length=100, description="Industry sector")
    country: Optional[str] = Field(default=None, max_length=100, description="Operating country")

class UserCreate(UserBase):
    """User registration schema with validation."""
    password: str = Field(..., min_length=8, max_length=128, description="Secure password")
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Validate password meets security requirements."""
        errors = []
        
        if not any(c.isupper() for c in v):
            errors.append("Must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            errors.append("Must contain at least one lowercase letter")  
        if not any(c.isdigit() for c in v):
            errors.append("Must contain at least one number")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v):
            errors.append("Must contain at least one special character")
        
        if errors:
            raise ValueError(f"Password requirements: {', '.join(errors)}")
        
        return v

class UserResponse(UserBase):
    """User response for API responses."""
    id: int
    role: UserRole = Field(description="User access level")
    is_active: bool = Field(description="Account status")
    is_verified: bool = Field(description="Email verification status")
    last_login: Optional[datetime] = Field(description="Last login timestamp")
    created_at: datetime = Field(description="Account creation date")
    
    model_config = ConfigDict(from_attributes=True)

class UserUpdate(BaseModel):
    """User profile update schema."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    company_name: Optional[str] = Field(None, max_length=200)
    industry: Optional[str] = Field(None, max_length=100)
    country: Optional[str] = Field(None, max_length=100)

# # AUTHENTICATION SCHEMAS

# class Token(BaseModel):
#     """JWT authentication token response."""
#     access_token: str = Field(description="JWT access token")
#     token_type: str = Field(default="bearer", description="Token type")
#     refresh_token: str = Field(description="Refresh token for token renewal")
#     expires_in: int = Field(description="Token expiration time in seconds")

# class RefreshTokenRequest(BaseModel):
#     """Token refresh request."""
#     refresh_token: str = Field(description="Valid refresh token")

class UserLogin(BaseModel):
    """User login credentials."""
    username: str = Field(..., min_length=3, description="Username or email")
    password: str = Field(..., min_length=1, description="Account password")

# PREDICTION SCHEMAS - Aligned with ML Pipeline

class PredictionInput(BaseModel):
    """Company data input following exact ML pipeline structure."""
    model_config = ConfigDict(protected_namespaces=())
    
    # Core business identification
    stra_sector: str = Field(..., min_length=1, max_length=100, 
                            description="Business sector (e.g., Manufacturing, Retail, Services)")
    region: RegionType = Field(default=RegionType.ROW, description="Market region")
    country2: Optional[str] = Field(default=None, max_length=50, description="Operating country")
    
    # Company characteristics (pipeline: wk14, car1)
    wk14: float = Field(..., ge=0, le=50, description="Years of operation")
    car1: float = Field(..., ge=0, le=100, description="Years since company establishment")
    
    # Financial structure - percentages (pipeline: fin1-fin5)
    Fin_bank: Optional[float] = Field(default=0, ge=0, le=100, 
                                     description="Bank financing percentage")
    Fin_supplier: Optional[float] = Field(default=0, ge=0, le=100,
                                         description="Supplier credit percentage") 
    Fin_equity: Optional[float] = Field(default=0, ge=0, le=100,
                                       description="Equity financing percentage")
    Fin_other: Optional[float] = Field(default=0, ge=0, le=100,
                                      description="Other financing sources percentage")
    Fin_int: Optional[float] = Field(default=0, ge=0, le=100,
                                    description="Internal financing percentage")
    
    # Gender and diversity metrics (pipeline: gend2-gend6)
    Fem_wf: Optional[float] = Field(default=0, ge=0, le=100,
                                   description="Female workforce participation percentage")
    Fem_CEO: Optional[float] = Field(default=0, ge=0, le=100,
                                    description="Female leadership percentage")
    Fem_Wf_Non_Prod: Optional[float] = Field(default=0, ge=0, le=100,
                                            description="Female non-production workers percentage")
    Fem_Own: Optional[float] = Field(default=0, ge=0, le=100,
                                    description="Female ownership percentage")
    
    # Ownership structure (pipeline: car2, car3, car6)
    Pvt_Own: Optional[float] = Field(default=0, ge=0, le=100,
                                    description="Private ownership percentage")
    Con_Own: Optional[float] = Field(default=0, ge=0, le=100,
                                    description="Concentrated ownership percentage")
    For_Own: Optional[float] = Field(default=0, ge=0, le=100,
                                    description="Foreign ownership percentage")
    
    # Business obstacles and characteristics (pipeline: obst9, tr15, t10, t2, corr4, obst11, infor1)
    obst9: Optional[float] = Field(default=0, ge=0, le=100,
                                  description="Education/skills obstacle level (0-100)")
    tr15: Optional[float] = Field(default=0, ge=0, le=100,
                                 description="Export share of sales percentage")
    t10: Optional[float] = Field(default=0, ge=0, le=100,
                                description="Innovation and R&D investment level")
    t2: Optional[float] = Field(default=0, ge=0, le=100,
                               description="Transportation obstacle level")
    corr4: Optional[float] = Field(default=0, ge=0, le=100,
                                  description="Informal payments/corruption level")
    obst11: Optional[float] = Field(default=0, ge=0, le=100,
                                   description="Political instability impact level")
    infor1: Optional[float] = Field(default=0, ge=0, le=100,
                                   description="Informal sector competition level")
    
    # Macroeconomic environment (pipeline: Credit, WSI, WUI, GDP, PRIME)
    Credit: Optional[float] = Field(default=0, ge=0, le=100,
                                   description="Access to credit and financing")
    WSI: Optional[float] = Field(default=0, description="World Strength Index")
    WUI: Optional[float] = Field(default=0, description="World Uncertainty Index")
    GDP: Optional[float] = Field(default=0, description="GDP growth rate")
    PRIME: Optional[float] = Field(default=0, ge=0, description="Prime interest rate")
    
    # ROW-specific indicators (pipeline: size2, MarketCap, GPR)
    size2: Optional[float] = Field(default=None, ge=0, description="Company size metric")
    MarketCap: Optional[float] = Field(default=None, ge=0, description="Market capitalization")
    GPR: Optional[float] = Field(default=None, description="Geopolitical risk exposure")
    
    # Additional context (pipeline: fin16, fin33, obst1, perf1, year)
    fin16: Optional[float] = Field(default=0, description="Financial indicator 16")
    fin33: Optional[float] = Field(default=0, description="Financial indicator 33") 
    obst1: Optional[float] = Field(default=0, description="Main business obstacle")
    perf1: Optional[float] = Field(default=0, description="Performance indicator")
    year: Optional[int] = Field(default=2024, ge=2020, le=2030, description="Data year")
    
    @validator('region', pre=True)
    def validate_region(cls, v):
        """Validate and set region based on business logic."""
        if isinstance(v, str):
            return RegionType(v.upper())
        return v
    
    @validator('stra_sector')
    def validate_sector(cls, v):
        """Validate business sector is provided."""
        if not v or v.strip() == "":
            raise ValueError("Business sector is required")
        return v.strip()

class InfluencingFactor(BaseModel):
    """Factor influencing the prediction with business explanations."""
    name: str = Field(..., description="Technical feature name")
    display_name: Optional[str] = Field(None, description="Business-friendly name")
    impact_level: ImpactLevel = Field(description="Level of impact on risk")
    weight: float = Field(..., ge=0, le=1, description="Normalized importance weight")
    shap_value: Optional[float] = Field(None, description="SHAP explanation value")
    feature_value: Optional[float] = Field(None, description="Actual feature value")
    description: Optional[str] = Field(None, description="Business explanation of impact")

class PredictionOutput(BaseModel):
    """Comprehensive prediction results for business users."""
    model_config = ConfigDict(protected_namespaces=())
    
    # Core prediction results
    financial_distress_probability: float = Field(..., ge=0, le=1,
                                                 description="Probability of financial distress (0-100%)")
    model_confidence: float = Field(..., ge=0, le=1,
                                   description="Model confidence in prediction")
    risk_category: RiskCategory = Field(description="Overall risk level classification")
    
    # Business-friendly results
    financial_health_status: str = Field(description="Overall financial health assessment")
    risk_level_detail: str = Field(description="Detailed risk explanation")
    analysis_message: str = Field(description="Executive summary of analysis")
    
    # Explanations and insights
    key_influencing_factors: List[InfluencingFactor] = Field(
        default_factory=list, description="Top factors affecting the prediction"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Actionable business recommendations"
    )
    
    # Model metadata
    model_version: Optional[str] = Field(None, description="ML model version used")
    region: Optional[RegionType] = Field(None, description="Market region analyzed")
    processing_time_ms: Optional[float] = Field(None, description="Analysis processing time")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    
    # Frontend visualization data
    visualization_data: Optional[Dict[str, Any]] = Field(
        None, description="Chart and graph data for frontend display"
    )


# ANALYTICS AND DASHBOARD SCHEMAS

class DistressDistributionItem(BaseModel):
    """Risk distribution data for charts."""
    status: str = Field(..., description="Risk category name")
    count: int = Field(..., ge=0, description="Number of companies")
    percentage: float = Field(..., ge=0, le=100, description="Percentage of total")

class TopFeatureItem(BaseModel):
    """Top risk factors for analytics."""
    factor: str = Field(..., description="Risk factor name")
    display_name: Optional[str] = Field(None, description="Business-friendly name")
    impact: ImpactLevel = Field(description="Average impact level")
    frequency: Optional[int] = Field(None, description="How often this factor appears")

class PredictionTrendItem(BaseModel):
    """Prediction trends over time for charts."""
    period: str = Field(..., description="Time period (month/quarter)")
    distress_rate: float = Field(..., ge=0, le=100, description="Distress rate percentage")
    healthy_rate: float = Field(..., ge=0, le=100, description="Healthy rate percentage")
    total_analyzed: int = Field(..., ge=0, description="Total companies analyzed")
    avg_confidence: Optional[float] = Field(None, ge=0, le=1, description="Average confidence")

class SectorAnalysisItem(BaseModel):
    """Industry sector analysis data."""
    sector: str = Field(..., description="Business sector name")
    healthy_percentage: float = Field(..., ge=0, le=100, description="Healthy companies %")
    distressed_percentage: float = Field(..., ge=0, le=100, description="At-risk companies %")
    total_companies: int = Field(..., ge=0, description="Total companies in sector")
    avg_distress_probability: Optional[float] = Field(None, description="Average risk probability")

class ComparativeRiskFactor(BaseModel):
    """Comparative risk analysis vs market benchmarks."""
    factor: str = Field(..., description="Risk factor name")
    display_name: Optional[str] = Field(None, description="Business-friendly name")
    user_average: float = Field(description="User portfolio average")
    market_average: float = Field(description="Market benchmark average")
    risk_level: str = Field(..., pattern="^(Low|Medium|High|Neutral)$", description="Relative risk")
    deviation_percentage: Optional[float] = Field(None, description="% deviation from market")

class MLInsights(BaseModel):
    """Machine learning model insights and statistics."""
    overall_distress_rate: float = Field(..., ge=0, le=100, description="Market distress rate %")
    user_distress_rate: float = Field(..., ge=0, le=100, description="User portfolio distress rate %")
    predictions_made: int = Field(..., ge=0, description="Total predictions completed")
    model_accuracy: float = Field(..., ge=0, le=100, description="Model accuracy percentage")
    companies_analyzed: int = Field(..., ge=0, description="Companies in training dataset")
    risk_factors_count: int = Field(..., ge=0, description="Risk factors analyzed")
    last_model_update: Optional[datetime] = Field(None, description="Last model training date")

class DashboardData(BaseModel):
    """Complete dashboard analytics for frontend."""
    # Distribution charts
    user_distress_distribution: List[DistressDistributionItem] = Field(
        description="User portfolio risk distribution"
    )
    market_distress_distribution: List[DistressDistributionItem] = Field(
        description="Overall market risk distribution"
    )
    
    # Analytics charts
    top_risk_factors: List[TopFeatureItem] = Field(description="Most influential risk factors")
    prediction_trends: List[PredictionTrendItem] = Field(description="Risk trends over time")
    sector_analysis: List[SectorAnalysisItem] = Field(description="Industry sector breakdown")
    comparative_risk_factors: List[ComparativeRiskFactor] = Field(
        description="User vs market comparison"
    )
    
    # Summary insights
    ml_insights: MLInsights = Field(description="Model performance and insights")
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Dashboard generation time")
    data_freshness: Optional[str] = Field(None, description="How recent the data is")

# SYSTEM AND UTILITY SCHEMAS

class HealthCheck(BaseModel):
    """System health check response."""
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    timestamp: datetime = Field(description="Health check timestamp")
    service: str = Field(description="Service name")
    version: str = Field(description="API version")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime")
    
    # Component health
    components: Optional[Dict[str, Any]] = Field(None, description="Individual component status")

class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str = Field(description="Error type")
    message: str = Field(description="User-friendly error message")
    details: Optional[List[str]] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request tracking ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PredictionExportItem(BaseModel):
    """Individual prediction for data export."""
    prediction_id: int
    created_at: datetime
    risk_category: RiskCategory
    distress_probability: float
    model_confidence: float
    sector: Optional[str] = None
    region: Optional[RegionType] = None
    company_context: Optional[Dict[str, Any]] = None

class BulkPredictionRequest(BaseModel):
    """Bulk analysis request for multiple companies."""
    predictions: List[PredictionInput] = Field(..., max_items=50, description="Company data list")
    include_explanations: bool = Field(default=True, description="Include SHAP explanations")
    return_format: str = Field(default="json", pattern="^(json|csv|excel)$")

class BulkPredictionResponse(BaseModel):
    """Bulk analysis response."""
    total_requested: int = Field(description="Total analyses requested")
    successful_predictions: int = Field(description="Successfully completed")
    failed_predictions: int = Field(description="Failed analyses")
    results: List[PredictionOutput] = Field(description="Analysis results")
    processing_time_seconds: float = Field(description="Total processing time")
    download_url: Optional[str] = Field(None, description="Download link for results")

# FRONTEND VISUALIZATION SCHEMAS

class ChartDataPoint(BaseModel):
    """Individual data point for charts."""
    label: str = Field(description="Data point label")
    value: float = Field(description="Data point value")
    color: Optional[str] = Field(None, description="Chart color code")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional chart data")

class VisualizationData(BaseModel):
    """Structured data for frontend visualizations."""
    # Risk gauge data
    risk_gauge: Optional[Dict[str, Any]] = Field(None, description="Risk gauge configuration")
    
    # Chart datasets
    pie_chart_data: Optional[List[ChartDataPoint]] = Field(None, description="Pie chart data")
    bar_chart_data: Optional[List[ChartDataPoint]] = Field(None, description="Bar chart data")
    line_chart_data: Optional[List[Dict[str, Any]]] = Field(None, description="Time series data")
    
    # Comparison data
    benchmark_comparison: Optional[Dict[str, Any]] = Field(None, description="Benchmark comparison")
    
    # Interactive elements
    drill_down_data: Optional[Dict[str, Any]] = Field(None, description="Drill-down chart data")