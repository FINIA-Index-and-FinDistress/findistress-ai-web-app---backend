from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, Text, JSON, ForeignKey, Index, DateTime as SQLDateTime
from sqlalchemy.sql import func
from sqlalchemy.sql.sqltypes import Enum as SAEnum
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum
from pydantic import ConfigDict
import logging

logger = logging.getLogger(__name__)

# Enums for business clarity
class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    ANALYST = "analyst"

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class RegionType(str, Enum):
    AFR = "AFR"  # African markets
    ROW = "ROW"  # Rest of world

class PredictionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    PROCESSING = "processing"

# Database models
class User(SQLModel, table=True):
    """User accounts with comprehensive tracking."""
    __tablename__ = "users"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True, max_length=50, description="Unique username")
    email: str = Field(index=True, unique=True, max_length=255, description="Email address")
    full_name: str = Field(default="", max_length=100, description="Full display name")
    hashed_password: str = Field(max_length=255, description="Encrypted password")
    is_active: bool = Field(default=True, index=True, description="Account active status")
    role: UserRole = Field(
        default=UserRole.USER,
        sa_column=Column(SAEnum(UserRole, name="user_role"), default=UserRole.USER),
        description="User role"
    )
    is_verified: bool = Field(default=False, description="Email verification status")
    is_admin: bool = Field(default=False, index=True, description="Admin status")
    refresh_token_hash: Optional[str] = Field(default=None, max_length=255)
    refresh_token_expires_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(SQLDateTime(timezone=True), nullable=True)
    )
    last_login: Optional[datetime] = Field(
        default=None,
        sa_column=Column(SQLDateTime(timezone=True), nullable=True, index=True)
    )
    login_count: int = Field(default=0, description="Total login count")
    failed_login_attempts: int = Field(default=0, description="Failed login tracking")
    last_failed_login: Optional[datetime] = Field(
        default=None,
        sa_column=Column(SQLDateTime(timezone=True), nullable=True)
    )
    company_name: Optional[str] = Field(default=None, max_length=200, description="User's company")
    industry: Optional[str] = Field(default=None, max_length=100, description="Industry sector")
    country: Optional[str] = Field(default=None, max_length=100, description="Operating country")
    preferences: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLDateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(SQLDateTime(timezone=True), nullable=True, onupdate=func.now())
    )
    
    predictions: List["PredictionLog"] = Relationship(back_populates="user")
    audit_logs: List["AuditLog"] = Relationship(back_populates="user")
    
    __table_args__ = (
        Index('idx_user_active_role', 'is_active', 'role'),
        Index('idx_user_last_login', 'last_login'),
        Index('idx_user_company', 'company_name'),
    )

class PredictionLog(SQLModel, table=True):
    """Financial distress predictions following ML pipeline structure."""
    __tablename__ = "prediction_logs"
    
    model_config = ConfigDict(protected_namespaces=())
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
            index=True
        )
    )
    input_data: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    region: RegionType = Field(
        sa_column=Column(SAEnum(RegionType, name="region_type"), index=True),
        description="Market region (AFR/ROW)"
    )
    sector: str = Field(max_length=100, index=True, description="Business sector")
    company_size: Optional[str] = Field(default=None, max_length=50, description="Company size category")
    financial_distress_probability: float = Field(index=True, description="Probability of financial distress (0-1)")
    model_confidence: float = Field(description="Model confidence score (0-1)")
    risk_category: RiskLevel = Field(
        sa_column=Column(SAEnum(RiskLevel, name="risk_level"), index=True),
        description="Risk level classification"
    )
    financial_health_status: str = Field(max_length=100, description="Health status summary")
    risk_level_detail: str = Field(max_length=200, description="Detailed risk explanation")
    analysis_message: str = Field(sa_column=Column(Text), description="Business analysis summary")
    model_version: str = Field(default="2.0", max_length=20, description="ML model version used")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time")
    prediction_status: PredictionStatus = Field(
        sa_column=Column(SAEnum(PredictionStatus, name="prediction_status"), default=PredictionStatus.COMPLETED),
        description="Prediction status"
    )
    recommendations: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    benchmark_comparisons: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    visualization_data: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLDateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(SQLDateTime(timezone=True), nullable=True, onupdate=func.now())
    )
    
    user: Optional["User"] = Relationship(back_populates="predictions")
    influencing_factors: List["InfluencingFactorDB"] = Relationship(back_populates="prediction_log")
    
    __table_args__ = (
        Index('idx_prediction_user_created', 'user_id', 'created_at'),
        Index('idx_prediction_risk_region', 'risk_category', 'region'),
        Index('idx_prediction_sector_created', 'sector', 'created_at'),
        Index('idx_prediction_probability', 'financial_distress_probability'),
        Index('idx_prediction_status', 'prediction_status', 'created_at'),
    )

class InfluencingFactorDB(SQLModel, table=True):
    """SHAP values and feature importance from ML pipeline."""
    __tablename__ = "influencing_factors"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    prediction_log_id: int = Field(
        sa_column=Column(
            ForeignKey("prediction_logs.id", ondelete="CASCADE"),
            nullable=False,
            index=True
        )
    )
    name: str = Field(max_length=100, index=True, description="Feature name from ML pipeline")
    display_name: Optional[str] = Field(default=None, max_length=150, description="Business-friendly name")
    shap_value: float = Field(description="SHAP importance value")
    feature_value: float = Field(description="Raw feature value")
    impact_level: str = Field(max_length=20, index=True, description="Impact classification")
    weight: float = Field(description="Normalized weight for ranking")
    description: Optional[str] = Field(default=None, sa_column=Column(Text))
    recommendation: Optional[str] = Field(default=None, sa_column=Column(Text))
    percentile_rank: Optional[float] = Field(default=None, description="Percentile vs industry")
    benchmark_value: Optional[float] = Field(default=None, description="Industry average")
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLDateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(SQLDateTime(timezone=True), nullable=True, onupdate=func.now())
    )
    
    prediction_log: "PredictionLog" = Relationship(back_populates="influencing_factors")
    
    __table_args__ = (
        Index('idx_factor_prediction_weight', 'prediction_log_id', 'weight'),
        Index('idx_factor_name_impact', 'name', 'impact_level'),
        Index('idx_factor_shap_value', 'shap_value'),
    )

class AuditLog(SQLModel, table=True):
    """Comprehensive audit trail for compliance and monitoring."""
    __tablename__ = "audit_logs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
            index=True
        )
    )
    action: str = Field(max_length=100, index=True, description="Action type")
    resource: Optional[str] = Field(default=None, max_length=100, description="Affected resource")
    resource_id: Optional[str] = Field(default=None, max_length=100, description="Resource ID")
    ip_address: Optional[str] = Field(default=None, max_length=45, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, max_length=500, description="Browser/client info")
    endpoint: Optional[str] = Field(default=None, max_length=200, description="API endpoint")
    http_method: Optional[str] = Field(default=None, max_length=10, description="HTTP method")
    success: bool = Field(default=True, index=True, description="Operation success")
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))
    response_time_ms: Optional[float] = Field(default=None, description="Response time")
    additional_data: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLDateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(SQLDateTime(timezone=True), nullable=True, onupdate=func.now())
    )
    
    user: Optional["User"] = Relationship(back_populates="audit_logs")
    
    __table_args__ = (
        Index('idx_audit_user_action_created', 'user_id', 'action', 'created_at'),
        Index('idx_audit_ip_created', 'ip_address', 'created_at'),
        Index('idx_audit_success_created', 'success', 'created_at'),
        Index('idx_audit_endpoint_method', 'endpoint', 'http_method'),
    )

class SystemMetrics(SQLModel, table=True):
    """System performance metrics for monitoring dashboard."""
    __tablename__ = "system_metrics"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    metric_name: str = Field(max_length=100, index=True, description="Metric name")
    metric_value: float = Field(description="Metric value")
    metric_unit: Optional[str] = Field(default=None, max_length=20, description="Unit of measurement")
    category: str = Field(max_length=50, index=True, description="Metric category")
    subcategory: Optional[str] = Field(default=None, max_length=50, description="Metric subcategory")
    metric_metadata: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLDateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(SQLDateTime(timezone=True), nullable=True, onupdate=func.now())
    )
    
    __table_args__ = (
        Index('idx_metrics_name_created', 'metric_name', 'created_at'),
        Index('idx_metrics_category_created', 'category', 'created_at'),
        Index('idx_metrics_value', 'metric_value'),
    )

class ModelPerformance(SQLModel, table=True):
    """ML model performance tracking aligned with pipeline."""
    __tablename__ = "model_performance"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    model_name: str = Field(max_length=100, index=True, description="Model name (LGB_afr/LGB_row)")
    model_version: str = Field(max_length=20, index=True, description="Model version")
    region: RegionType = Field(
        sa_column=Column(SAEnum(RegionType, name="region_type"), index=True),
        description="Model region (AFR/ROW)"
    )
    accuracy: float = Field(description="Model accuracy")
    precision: float = Field(description="Precision score")
    recall: float = Field(description="Recall score")
    f1_score: float = Field(description="F1 score")
    auc_roc: Optional[float] = Field(default=None, description="AUC-ROC score")
    cv_accuracy_mean: Optional[float] = Field(default=None, description="Cross-validation accuracy")
    cv_accuracy_std: Optional[float] = Field(default=None, description="CV accuracy std dev")
    training_samples: int = Field(description="Training dataset size")
    test_samples: int = Field(description="Test dataset size")
    feature_count: int = Field(description="Number of features")
    adasyn_applied: bool = Field(default=True, description="ADASYN balancing applied")
    feature_set: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    model_path: Optional[str] = Field(default=None, max_length=500, description="Model file path")
    training_config: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    confusion_matrix: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLDateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(SQLDateTime(timezone=True), nullable=True, onupdate=func.now())
    )
    
    __table_args__ = (
        Index('idx_model_name_version_region', 'model_name', 'model_version', 'region'),
        Index('idx_model_accuracy', 'accuracy'),
        Index('idx_model_created', 'created_at'),
    )

class DataQualityCheck(SQLModel, table=True):
    """Data quality monitoring for training and input data."""
    __tablename__ = "data_quality_checks"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    check_type: str = Field(max_length=100, index=True, description="Type of quality check")
    data_source: str = Field(max_length=100, index=True, description="Data source checked")
    total_records: int = Field(description="Total records checked")
    passed_records: int = Field(description="Records passing quality checks")
    failed_records: int = Field(description="Records failing quality checks")
    quality_score: float = Field(description="Overall quality score (0-1)")
    issues_found: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    severity: str = Field(max_length=20, index=True, description="Issue severity level")
    check_config: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(SQLDateTime(timezone=True), nullable=False, server_default=func.now())
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(SQLDateTime(timezone=True), nullable=True, onupdate=func.now())
    )
    
    __table_args__ = (
        Index('idx_quality_type_severity', 'check_type', 'severity'),
        Index('idx_quality_source_created', 'data_source', 'created_at'),
        Index('idx_quality_score', 'quality_score'),
    )

# Pydantic models for API responses
class UserCreate(SQLModel):
    username: str
    email: str
    password: str
    full_name: str = ""

class UserResponse(SQLModel):
    id: int
    username: str
    email: str
    full_name: str
    is_active: bool
    is_admin: bool
    created_at: datetime

class PredictionInput(SQLModel):
    input_data: Dict[str, Any]
    region: RegionType = RegionType.AFR

class InfluencingFactor(SQLModel):
    name: str
    impact_level: str
    weight: float
    description: Optional[str] = None

class PredictionOutput(SQLModel):
    model_config = ConfigDict(protected_namespaces=())
    financial_distress_probability: float
    model_confidence: float
    risk_category: RiskLevel
    financial_health_status: str
    risk_level_detail: str
    analysis_message: str
    key_influencing_factors: List[InfluencingFactor]
    created_at: Optional[datetime] = None
    visualization_data: Optional[Dict[str, Any]] = None

class ComparativeRiskFactor(SQLModel):
    factor: str
    user_avg: float
    training_avg: float
    risk: str

class MLInsights(SQLModel):
    overallDistressRate: float
    userDistressRate: float
    predictions_made: int
    accuracy_rate: float
    companies_analyzed: int
    risk_factors: int

class DashboardData(SQLModel):
    distressDistribution: List[Dict[str, Any]]
    overallDistressDistribution: List[Dict[str, Any]]
    mlInsights: MLInsights
    topFeatures: List[Dict[str, str]]
    predictionTrends: List[Dict[str, Any]]
    sectorAnalysis: List[Dict[str, Any]]
    comparativeRiskFactors: List[ComparativeRiskFactor]