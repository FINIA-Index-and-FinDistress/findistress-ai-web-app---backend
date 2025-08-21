import os
import logging
import warnings
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status, Response, Query, BackgroundTasks, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import and_, func
from sqlalchemy.ext.asyncio import AsyncSession 
from sqlmodel import select
import jwt
from pydantic import BaseModel, ValidationError
# from app.auth.security import verify_refresh_token

from app.database.database import get_async_session, async_transactional
from app.database.models import (
    User, PredictionLog, InfluencingFactorDB, UserCreate, UserResponse, 
    PredictionInput, PredictionOutput, InfluencingFactor, DashboardData, 
    MLInsights, ComparativeRiskFactor, RiskLevel, RegionType
)
from app.models.auth_schemas import Token, RefreshTokenRequest
from app.auth.security import SecurityManager, authenticate_user, create_user_tokens, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_MINUTES
from app.services.prediction_service import predict_with_service, PipelineConfig

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

router = APIRouter(prefix="/api/v1", tags=["Financial Distress API"])

# REQUEST/RESPONSE MODELS 
class LoginRequest(BaseModel):
    """Login request model for JSON data."""
    username: str
    password: str

class RegisterRequest(BaseModel):
    """Registration request model for JSON data."""
    username: str
    password: str
    email: str = ""
    full_name: str = ""

class UserUpdateRequest(BaseModel):
    """User profile update request."""
    email: Optional[str] = None
    full_name: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None

class UserPreferencesRequest(BaseModel):
    """User preferences update request."""
    theme: Optional[str] = "light"
    notifications: Optional[bool] = True
    dashboard_layout: Optional[str] = "default"
    default_region: Optional[str] = "AFR"

class AnalyticsResponse(BaseModel):
    """Analytics data response."""
    total_predictions: int
    success_rate: float
    average_risk_score: float
    risk_distribution: List[Dict[str, Any]]
    monthly_trends: List[Dict[str, Any]]
    top_risk_factors: List[Dict[str, Any]]

class InsightsResponse(BaseModel):
    """Insights data response."""
    key_insights: List[str]
    recommendations: List[str]
    market_trends: List[Dict[str, Any]]
    risk_alerts: List[Dict[str, Any]]

class PredictionDetailResponse(BaseModel):
    """Detailed prediction response."""
    id: int
    financial_distress_probability: float
    model_confidence: float
    risk_category: str
    financial_health_status: str
    risk_level_detail: str
    analysis_message: str
    created_at: datetime
    input_data: Dict[str, Any]
    key_influencing_factors: List[InfluencingFactor]
    recommendations: List[str]
    benchmark_comparisons: Dict[str, Any]
    visualization_data: Dict[str, Any]

# 1. Add caching for training data processing
from functools import lru_cache
from typing import Optional

# OptimizedPipelineDataProcessor class 
class OptimizedPipelineDataProcessor:
    """Optimized data processor with caching."""
    def __init__(self):
        self._cached_data = None
        self._cache_timestamp = None
        self._processed_cache = None
    
    def load_training_data(self) -> pd.DataFrame:
        """Load training data. """
        return self.load_training_data_cached()
    
    @lru_cache(maxsize=1)
    def load_training_data_cached(self) -> pd.DataFrame:
        """Load training data with caching."""
        try:
            if not Path(config.TRAINING_DATA_PATH).exists():
                raise FileNotFoundError(f"Training data not found: {config.TRAINING_DATA_PATH}")
            
            # Check if cache is still valid (refresh every hour)
            if (self._cached_data is None or 
                self._cache_timestamp is None or 
                (datetime.now(timezone.utc) - self._cache_timestamp).seconds > 3600):
                
                self._cached_data = pd.read_excel(config.TRAINING_DATA_PATH)
                self._cache_timestamp = datetime.now(timezone.utc)
                logger.info(f"Loaded {len(self._cached_data)} training records")
            
            return self._cached_data
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise HTTPException(status_code=500, detail="Training data unavailable")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data following exact pipeline steps. """
        try:
            required_cols = [
                'idstd', 'country2', 'region', 'year', 'car1', 'fin1', 'fin2', 'fin3', 'fin4', 'fin5',
                'fin16', 'fin33', 'gend2', 'gend3', 'gend4', 'gend6', 'wk14', 'car3', 'car2', 'car6',
                'obst9', 'tr15', 't10', 't2', 'corr4', 'obst11', 'infor1', 'perf1', 'obst1', 'stra_sector',
                'GDP', 'Credit', 'MarketCap', 'WUI', 'GPR', 'PRIME', 'WSI', 'size2'
            ]
            
            df2 = df[required_cols].copy()
            
            df2 = df2.rename(columns={
                'fin1': 'Fin_int', 'fin2': 'Fin_bank', 'fin3': 'Fin_supplier', 'fin4': 'Fin_equity',
                'fin5': 'Fin_other', 'gend2': 'Fem_wf', 'gend3': 'Fem_Wf_Non_Prod', 'gend4': 'Fem_CEO',
                'gend6': 'Fem_Own', 'car3': 'For_Own', 'car2': 'Pvt_Own', 'car6': 'Con_Own',
                'obst9': 'Edu', 'tr15': 'Exports', 't10': 'Innov', 't2': 'Transp', 'corr4': 'Gifting',
                'obst11': 'Pol_Inst', 'infor1': 'Infor_Comp', 'size2': 'Size'
            })
            
            df2['Sector'] = df2['stra_sector'].map(config.SECTOR_MAPPINGS)
            
            # FCreate region2 column based on the pipeline logic
            african_countries = [
                'Angola', 'Bangladesh', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon',
                'Central African Republic', 'Chad', 'Congo', "Cote d'Ivoire", 'DRC', 'Djibouti', 'Egypt',
                'Equatorial Guinea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea',
                'Lebanon', 'Lesotho', 'Liberia', 'Guineabissau', 'Kenya', 'Madagascar', 'Malawi',
                'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger',
                'Nigeria', 'Rwanda', 'Senegal', 'Seychelles', 'Sierra Leone', 'South Sudan',
                'Southafrica', 'Sudan', 'Tanzania', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
            ]
            
            df2.loc[df2['country2'].isin(african_countries), 'region2'] = 'AFR'
            df2.loc[~df2['country2'].isin(african_countries), 'region2'] = 'ROW'
            
            df2_afr = df2[df2['region2'] == "AFR"]
            df2_row = df2[df2['region2'] != "AFR"]
            
            df3_afr = df2_afr[[
                'idstd', 'year', 'perf1', 'obst1', 'fin16', 'wk14', 'car1', 'fin33', 'Fin_bank',
                'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf', 'Fem_CEO', 'Pvt_Own', 'Con_Own',
                'Exports', 'Edu', 'Innov', 'Transp', 'Gifting', 'Pol_Inst', 'Infor_Comp', 'Sector',
                'Credit', 'WSI', 'WUI', 'GDP', 'PRIME', 'region2' 
            ]].copy()
            
            df3_row = df2_row[[
                'idstd', 'year', 'perf1', 'obst1', 'fin16', 'wk14', 'car1', 'fin33', 'Fin_bank',
                'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf', 'Fem_CEO', 'Pvt_Own', 'Con_Own',
                'Edu', 'Exports', 'Innov', 'Transp', 'Gifting', 'Pol_Inst', 'Infor_Comp', 'Size',
                'Credit', 'Sector', 'WUI', 'WSI', 'PRIME', 'MarketCap', 'GPR', 'GDP', 'region2'  
            ]].copy()
            
            for df_region in [df3_afr, df3_row]:
                df_region['distress'] = np.where(
                    (df_region['perf1'] < 0) & (
                        (df_region['obst1'] == 100) |
                        (df_region['fin33'] == 1) |
                        (df_region['fin16'] == 1)
                    ), 1, 0
                )
                df_region['startup'] = np.where(
                    (df_region['wk14'] < 5) & (df_region['car1'] < 5), 1, 0
                )
            
            df3_afr.fillna(0, inplace=True)
            df3_row.fillna(0, inplace=True)
            
            percentage_cols = [
                'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf', 'Fem_CEO',
                'Pvt_Own', 'Con_Own', 'Edu', 'Exports', 'Innov', 'Transp', 'Gifting',
                'Pol_Inst', 'Infor_Comp', 'Credit', 'PRIME', 'GDP'
            ]
            
            for col in percentage_cols:
                if col in df3_afr.columns:
                    df3_afr[col] = df3_afr[col].apply(lambda x: x / 100 if x > 1 else x)
                    df3_afr[col] = df3_afr[col].apply(lambda x: 0 if x < 0 else x)
                if col in df3_row.columns:
                    df3_row[col] = df3_row[col].apply(lambda x: x / 100 if x > 1 else x)
                    df3_row[col] = df3_row[col].apply(lambda x: 0 if x < 0 else x)
            
            if 'MarketCap' in df3_row.columns:
                df3_row['MarketCap'] = df3_row['MarketCap'].apply(lambda x: x / 100 if x > 1 else x)
                df3_row['MarketCap'] = df3_row['MarketCap'].apply(lambda x: 0 if x < 0 else x)
            
            # Return concatenated data with region2 column intact
            return pd.concat([df3_afr, df3_row], ignore_index=True)
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise HTTPException(status_code=500, detail="Data processing error")
    
    def get_processed_training_data(self) -> Optional[pd.DataFrame]:
        """Get processed training data with caching."""
        try:
            # Return cached processed data if available
            if self._processed_cache is not None:
                return self._processed_cache
            
            # Load and process data
            raw_data = self.load_training_data()
            self._processed_cache = self.preprocess_data(raw_data)
            
            return self._processed_cache
            
        except Exception as e:
            logger.warning(f"Could not process training data: {e}")
            return None
        
# CONFIGURATION AND PROCESSORS 
config = PipelineConfig()

# class PipelineDataProcessor:
#     """Data processor following pipeline steps."""
#     def __init__(self):
#         self._cached_data = None
#         self._cache_timestamp = None
    
#     def load_training_data(self) -> pd.DataFrame:
#         """Load training data following pipeline format."""
#         try:
#             if not Path(config.TRAINING_DATA_PATH).exists():
#                 raise FileNotFoundError(f"Training data not found: {config.TRAINING_DATA_PATH}")
            
#             if self._cached_data is None:
#                 self._cached_data = pd.read_excel(config.TRAINING_DATA_PATH)
#                 self._cache_timestamp = datetime.now(timezone.utc)
#                 logger.info(f"Loaded {len(self._cached_data)} training records")
            
#             return self._cached_data
            
#         except Exception as e:
#             logger.error(f"Failed to load training data: {e}")
#             raise HTTPException(status_code=500, detail="Training data unavailable")

# def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
#     """Preprocess data following exact pipeline steps - FIXED VERSION."""
#     try:
#         required_cols = [
#             'idstd', 'country2', 'region', 'year', 'car1', 'fin1', 'fin2', 'fin3', 'fin4', 'fin5',
#             'fin16', 'fin33', 'gend2', 'gend3', 'gend4', 'gend6', 'wk14', 'car3', 'car2', 'car6',
#             'obst9', 'tr15', 't10', 't2', 'corr4', 'obst11', 'infor1', 'perf1', 'obst1', 'stra_sector',
#             'GDP', 'Credit', 'MarketCap', 'WUI', 'GPR', 'PRIME', 'WSI', 'size2'
#         ]
        
#         df2 = df[required_cols].copy()
        
#         df2 = df2.rename(columns={
#             'fin1': 'Fin_int', 'fin2': 'Fin_bank', 'fin3': 'Fin_supplier', 'fin4': 'Fin_equity',
#             'fin5': 'Fin_other', 'gend2': 'Fem_wf', 'gend3': 'Fem_Wf_Non_Prod', 'gend4': 'Fem_CEO',
#             'gend6': 'Fem_Own', 'car3': 'For_Own', 'car2': 'Pvt_Own', 'car6': 'Con_Own',
#             'obst9': 'Edu', 'tr15': 'Exports', 't10': 'Innov', 't2': 'Transp', 'corr4': 'Gifting',
#             'obst11': 'Pol_Inst', 'infor1': 'Infor_Comp', 'size2': 'Size'
#         })
        
#         df2['Sector'] = df2['stra_sector'].map(config.SECTOR_MAPPINGS)
        
#         # Create region2 column based on the pipeline logic
#         african_countries = [
#             'Angola', 'Bangladesh', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon',
#             'Central African Republic', 'Chad', 'Congo', "Cote d'Ivoire", 'DRC', 'Djibouti', 'Egypt',
#             'Equatorial Guinea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea',
#             'Lebanon', 'Lesotho', 'Liberia', 'Guineabissau', 'Kenya', 'Madagascar', 'Malawi',
#             'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger',
#             'Nigeria', 'Rwanda', 'Senegal', 'Seychelles', 'Sierra Leone', 'South Sudan',
#             'Southafrica', 'Sudan', 'Tanzania', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
#         ]
        
#         df2.loc[df2['country2'].isin(african_countries), 'region2'] = 'AFR'
#         df2.loc[~df2['country2'].isin(african_countries), 'region2'] = 'ROW'
        
#         df2_afr = df2[df2['region2'] == "AFR"]
#         df2_row = df2[df2['region2'] != "AFR"]
        
#         df3_afr = df2_afr[[
#             'idstd', 'year', 'perf1', 'obst1', 'fin16', 'wk14', 'car1', 'fin33', 'Fin_bank',
#             'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf', 'Fem_CEO', 'Pvt_Own', 'Con_Own',
#             'Exports', 'Edu', 'Innov', 'Transp', 'Gifting', 'Pol_Inst', 'Infor_Comp', 'Sector',
#             'Credit', 'WSI', 'WUI', 'GDP', 'PRIME', 'region2'  
#         ]].copy()
        
#         df3_row = df2_row[[
#             'idstd', 'year', 'perf1', 'obst1', 'fin16', 'wk14', 'car1', 'fin33', 'Fin_bank',
#             'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf', 'Fem_CEO', 'Pvt_Own', 'Con_Own',
#             'Edu', 'Exports', 'Innov', 'Transp', 'Gifting', 'Pol_Inst', 'Infor_Comp', 'Size',
#             'Credit', 'Sector', 'WUI', 'WSI', 'PRIME', 'MarketCap', 'GPR', 'GDP', 'region2' 
#         ]].copy()
        
#         for df_region in [df3_afr, df3_row]:
#             df_region['distress'] = np.where(
#                 (df_region['perf1'] < 0) & (
#                     (df_region['obst1'] == 100) |
#                     (df_region['fin33'] == 1) |
#                     (df_region['fin16'] == 1)
#                 ), 1, 0
#             )
#             df_region['startup'] = np.where(
#                 (df_region['wk14'] < 5) & (df_region['car1'] < 5), 1, 0
#             )
        
#         df3_afr.fillna(0, inplace=True)
#         df3_row.fillna(0, inplace=True)
        
#         percentage_cols = [
#             'Fin_bank', 'Fin_supplier', 'Fin_equity', 'Fin_other', 'Fem_wf', 'Fem_CEO',
#             'Pvt_Own', 'Con_Own', 'Edu', 'Exports', 'Innov', 'Transp', 'Gifting',
#             'Pol_Inst', 'Infor_Comp', 'Credit', 'PRIME', 'GDP'
#         ]
        
#         for col in percentage_cols:
#             if col in df3_afr.columns:
#                 df3_afr[col] = df3_afr[col].apply(lambda x: x / 100 if x > 1 else x)
#                 df3_afr[col] = df3_afr[col].apply(lambda x: 0 if x < 0 else x)
#             if col in df3_row.columns:
#                 df3_row[col] = df3_row[col].apply(lambda x: x / 100 if x > 1 else x)
#                 df3_row[col] = df3_row[col].apply(lambda x: 0 if x < 0 else x)
        
#         if 'MarketCap' in df3_row.columns:
#             df3_row['MarketCap'] = df3_row['MarketCap'].apply(lambda x: x / 100 if x > 1 else x)
#             df3_row['MarketCap'] = df3_row['MarketCap'].apply(lambda x: 0 if x < 0 else x)
        
#         # Return concatenated data with region2 column intact
#         return pd.concat([df3_afr, df3_row], ignore_index=True)
        
#     except Exception as e:
#         logger.error(f"Preprocessing failed: {e}")
#         raise HTTPException(status_code=500, detail="Data processing error")

# data_processor = PipelineDataProcessor()
data_processor = OptimizedPipelineDataProcessor()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/token")

# AUTHENTICATION DEPENDENCIES
async def get_current_user(token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_async_session)):
    """Authenticate user with JWT token - Enhanced error handling."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        if not token:
            logger.warning("No token provided")
            raise credentials_exception
            
        secret_key = os.getenv("AUTH_SECRET_KEY")
        if not secret_key:
            logger.error("AUTH_SECRET_KEY environment variable not set")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error"
            )
        
        try:
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            raise credentials_exception
        
        username: str = payload.get("sub")
        if username is None:
            logger.warning("Token missing 'sub' claim")
            raise credentials_exception
        
        result = await session.execute(select(User).where(User.username == username))
        user = result.scalars().first()
        if user is None:
            logger.warning(f"User not found: {username}")
            raise credentials_exception
            
        if not user.is_active:
            logger.warning(f"Inactive user attempted access: {username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        logger.debug(f"User authenticated: {username}")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise credentials_exception

async def get_current_admin_user(current_user: User = Depends(get_current_user)):
    """Ensure user is admin."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Admin privileges required"
        )
    return current_user

# AUTHENTICATION ENDPOINTS
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    register_request: RegisterRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """Register new user account - JSON format."""
    try:
        logger.info(f"Registration attempt for user: {register_request.username}")
        
        if not register_request.username or not register_request.password:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Username and password are required"
            )
        
        result = await session.execute(select(User).where(User.username == register_request.username))
        if result.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        if register_request.email:
            result = await session.execute(select(User).where(User.email == register_request.email))
            if result.scalars().first():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
        
        hashed_password = SecurityManager.get_password_hash(register_request.password)
        user = User(
            username=register_request.username,
            email=register_request.email,
            full_name=register_request.full_name,
            hashed_password=hashed_password,
        )
        
        session.add(user)
        await session.commit()
        await session.refresh(user)
        
        logger.info(f"User registered successfully: {user.username}")
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_admin=user.is_admin,
            created_at=user.created_at
        )
        
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

# @router.post("/login", response_model=Token)
# async def login_for_access_token(
#     request: Request,
#     session: AsyncSession = Depends(get_async_session)
# ):
#     """User login with JWT tokens - supports JSON and form data."""
#     try:
#         content_type = request.headers.get("content-type", "").lower()
#         logger.info(f"Login attempt with content-type: {content_type}")

#         if "application/json" in content_type:
#             try:
#                 login_request = LoginRequest(**(await request.json()))
#             except (ValueError, ValidationError) as e:
#                 logger.error(f"Invalid JSON body: {e}")
#                 raise HTTPException(
#                     status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#                     detail="Invalid JSON format: username and password required"
#                 )
#         elif "application/x-www-form-urlencoded" in content_type:
#             body = await request.form()
#             login_request = LoginRequest(username=body.get("username", ""), password=body.get("password", ""))
#         else:
#             raise HTTPException(
#                 status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
#                 detail="Unsupported content type. Use application/json or application/x-www-form-urlencoded"
#             )

#         if not login_request.username or not login_request.password:
#             logger.warning("Missing username or password in login request")
#             raise HTTPException(
#                 status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#                 detail="Username and password are required"
#             )

#         user = await authenticate_user(login_request.username, login_request.password, session)
        
#         if not user:
#             logger.warning(f"Failed login attempt for user: {login_request.username}")
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Invalid credentials",
#                 headers={"WWW-Authenticate": "Bearer"},
#             )
        
#         if not user.is_active:
#             logger.warning(f"Inactive user login attempt: {login_request.username}")
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Account inactive"
#             )
        
#         token_data = await create_user_tokens(user, session)
#         await session.commit()
        
#         logger.info(f"User logged in successfully: {user.username}")
#         return Token(**token_data)
        
#     except HTTPException:
#         await session.rollback()
#         raise
#     except Exception as e:
#         await session.rollback()
#         logger.error(f"Login failed with exception: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Login system error: {str(e)}"
#         )

@router.post("/login", response_model=Token)
async def login_for_access_token(
    request: Request,
    session: AsyncSession = Depends(get_async_session)
):
    """User login with JWT tokens - supports JSON and form data."""
    try:
        content_type = request.headers.get("content-type", "").lower()
        logger.info(f"Login attempt with content-type: {content_type}")

        if "application/json" in content_type:
            try:
                login_data = await request.json()
                login_request = LoginRequest(
                    username=login_data.get("username", ""),
                    password=login_data.get("password", "")
                )
            except (ValueError, ValidationError) as e:
                logger.error(f"Invalid JSON body: {e}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid JSON format: username and password required"
                )
        elif "application/x-www-form-urlencoded" in content_type:
            body = await request.form()
            login_request = LoginRequest(
                username=body.get("username", ""), 
                password=body.get("password", "")
            )
        else:
            # Default to JSON if content-type is unclear
            try:
                login_data = await request.json()
                login_request = LoginRequest(
                    username=login_data.get("username", ""),
                    password=login_data.get("password", "")
                )
            except:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Unsupported content type. Use application/json or application/x-www-form-urlencoded"
                )

        if not login_request.username or not login_request.password:
            logger.warning("Missing username or password in login request")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Username and password are required"
            )

        user = await authenticate_user(login_request.username, login_request.password, session)
        
        if not user:
            logger.warning(f"Failed login attempt for user: {login_request.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            logger.warning(f"Inactive user login attempt: {login_request.username}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account inactive"
            )
        
        token_data = await create_user_tokens(user, session)
        await session.commit()
        
        logger.info(f"User logged in successfully: {user.username}")
        return Token(**token_data)
        
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Login failed with exception: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login system error: {str(e)}"
        )
    
@router.post("/token", response_model=Token)
async def login_oauth2_form(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_async_session)
):
    """OAuth2 compliant token endpoint (form data only)."""
    try:
        user = await authenticate_user(form_data.username, form_data.password, session)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account inactive"
            )
        
        token_data = await create_user_tokens(user, session)
        await session.commit()
        
        logger.info(f"User logged in via OAuth2: {user.username}")
        return Token(**token_data)
        
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"OAuth2 login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login system error"
        )


# @router.post("/refresh", response_model=Token)
# async def refresh_access_token(
#     refresh_request: RefreshTokenRequest,
#     session: AsyncSession = Depends(get_async_session)
# ):
#     """Refresh access token using valid refresh token."""
#     try:
#         logger.info("Token refresh request received")
        
#         if not refresh_request.refresh_token:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Refresh token is required"
#             )
        
#         # Verify refresh token and get user
#         user = await verify_refresh_token(refresh_request.refresh_token, session)
        
#         if not user:
#             logger.warning("Invalid or expired refresh token")
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Invalid or expired refresh token"
#             )
        
#         # Generate new tokens
#         token_data = await create_user_tokens(user, session)
#         await session.commit()
        
#         logger.info(f"Token refreshed successfully for user: {user.username}")
#         return Token(**token_data)
        
#     except HTTPException:
#         await session.rollback()
#         raise
#     except Exception as e:
#         await session.rollback()
#         logger.error(f"Token refresh failed: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Token refresh failed"
#         )

# USER PROFILE ENDPOINTS 
@router.get("/users/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_admin=current_user.is_admin,
        created_at=current_user.created_at
    )

@router.get("/users/me/profile")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get detailed user profile information."""
    try:
        return {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email or "",
            "full_name": current_user.full_name or "",
            "is_active": current_user.is_active,
            "is_admin": current_user.is_admin,
            "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
            "last_login": None,
            "profile_picture": None,
            "preferences": {
                "theme": "light",
                "notifications": True,
                "dashboard_layout": "default",
                "default_region": "AFR",
                "language": "en",
                "timezone": "UTC"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile"
        )

@router.put("/users/me", response_model=UserResponse)
@router.put("/users/me/profile")
@async_transactional
async def update_user_profile(
    update_request: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Update user profile information."""
    try:
        user_updated = False
        
        if update_request.email and update_request.email != current_user.email:
            result = await session.execute(
                select(User).where(and_(User.email == update_request.email, User.id != current_user.id))
            )
            if result.scalars().first():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already in use by another account"
                )
            current_user.email = update_request.email
            user_updated = True
        
        if update_request.full_name and update_request.full_name != current_user.full_name:
            current_user.full_name = update_request.full_name
            user_updated = True
        
        if update_request.new_password and update_request.current_password:
            if not SecurityManager.verify_password(update_request.current_password, current_user.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is incorrect"
                )
            
            if len(update_request.new_password) < 8:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="New password must be at least 8 characters long"
                )
            
            current_user.hashed_password = SecurityManager.get_password_hash(update_request.new_password)
            user_updated = True
        
        if user_updated:
            await session.commit()
            await session.refresh(current_user)
            logger.info(f"Profile updated for user: {current_user.username}")
        
        return UserResponse(
            id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            full_name=current_user.full_name,
            is_active=current_user.is_active,
            is_admin=current_user.is_admin,
            created_at=current_user.created_at
        )
        
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Profile update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )

@router.get("/users/me/preferences")
async def get_user_preferences(current_user: User = Depends(get_current_user)):
    """Get user preferences."""
    try:
        return {
            "theme": "light",
            "notifications": True,
            "dashboard_layout": "default",
            "default_region": "AFR",
            "language": "en",
            "timezone": "UTC",
            "email_notifications": True,
            "prediction_alerts": True,
            "auto_save": True,
            "data_retention_days": 365
        }
    except Exception as e:
        logger.error(f"Failed to get user preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve preferences"
        )

@router.put("/users/me/preferences")
async def update_user_preferences(
    preferences: UserPreferencesRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Update user preferences."""
    try:
        valid_themes = ["light", "dark", "auto"]
        valid_layouts = ["default", "compact", "detailed"]
        valid_regions = ["AFR", "ROW", "GLOBAL"]
        
        if preferences.theme and preferences.theme not in valid_themes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid theme. Must be one of: {', '.join(valid_themes)}"
            )
        
        if preferences.dashboard_layout and preferences.dashboard_layout not in valid_layouts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid layout. Must be one of: {', '.join(valid_layouts)}"
            )
        
        if preferences.default_region and preferences.default_region not in valid_regions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid region. Must be one of: {', '.join(valid_regions)}"
            )
        
        logger.info(f"Preferences updated for user: {current_user.username}")
        
        return {
            "message": "Preferences updated successfully",
            "preferences": {
                "theme": preferences.theme or "light",
                "notifications": preferences.notifications if preferences.notifications is not None else True,
                "dashboard_layout": preferences.dashboard_layout or "default",
                "default_region": preferences.default_region or "AFR"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preferences update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Preferences update failed"
        )

@router.get("/users/me/activity")
async def get_user_activity(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    limit: int = Query(default=50, le=100)
):
    """Get user activity log."""
    try:
        result = await session.execute(
            select(PredictionLog)
            .where(PredictionLog.user_id == current_user.id)
            .order_by(PredictionLog.created_at.desc())
            .limit(limit)
        )
        predictions = result.scalars().all()
        
        activities = []
        for pred in predictions:
            activities.append({
                "id": pred.id,
                "type": "prediction",
                "action": "Generated financial risk assessment",
                "timestamp": pred.created_at.isoformat(),
                "details": {
                    "risk_level": pred.risk_category.value,
                    "probability": pred.financial_distress_probability,
                    "confidence": pred.model_confidence
                }
            })
        
        return {
            "total_activities": len(activities),
            "activities": activities,
            "user_stats": {
                "total_predictions": len(predictions),
                "account_age_days": (datetime.now(timezone.utc) - current_user.created_at).days if current_user.created_at else 0,
                "last_activity": predictions[0].created_at.isoformat() if predictions else None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get user activity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve activity log"
        )

@router.delete("/users/me/account")
async def delete_user_account(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    confirmation: str = Query(..., description="Type 'DELETE' to confirm account deletion")
):
    """Delete user account (soft delete)."""
    try:
        if confirmation != "DELETE":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account deletion not confirmed. Type 'DELETE' to confirm."
            )
        
        current_user.is_active = False
        await session.commit()
        
        logger.info(f"Account deactivated for user: {current_user.username}")
        
        return {
            "message": "Account has been deactivated successfully",
            "account_id": current_user.id,
            "deactivated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Account deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deletion failed"
        )

# ADMIN ENDPOINTS 
@router.post("/admin/init-first-admin", response_model=UserResponse)
@async_transactional
async def initialize_first_admin(
    admin_request: RegisterRequest,
    session: AsyncSession = Depends(get_async_session)
):
    """Initialize the first admin user - Only works if no admins exist."""
    try:
        admin_count = await session.execute(
            select(func.count(User.id)).where(User.is_admin == True)
        )
        
        if admin_count.scalar() > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Admin users already exist. Use the regular admin creation endpoint."
            )
        
        result = await session.execute(select(User).where(User.username == admin_request.username))
        if result.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        hashed_password = SecurityManager.get_password_hash(admin_request.password)
        first_admin = User(
            username=admin_request.username,
            email=admin_request.email,
            full_name=admin_request.full_name or admin_request.username,
            hashed_password=hashed_password,
            is_admin=True,
            is_active=True
        )
        
        session.add(first_admin)
        await session.commit()
        await session.refresh(first_admin)
        
        logger.info(f"First admin user created: {first_admin.username}")
        
        return UserResponse(
            id=first_admin.id,
            username=first_admin.username,
            email=first_admin.email,
            full_name=first_admin.full_name,
            is_active=first_admin.is_active,
            is_admin=first_admin.is_admin,
            created_at=first_admin.created_at
        )
        
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"First admin creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="First admin creation failed"
        )

@router.post("/admin/create-admin", response_model=UserResponse)
@async_transactional
async def create_admin_user(
    admin_request: RegisterRequest,
    current_user: User = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Create a new admin user - Only existing admins can create new admins."""
    try:
        logger.info(f"Admin creation request from: {current_user.username}")
        
        if not admin_request.username or not admin_request.password:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Username and password are required"
            )
        
        result = await session.execute(select(User).where(User.username == admin_request.username))
        if result.scalars().first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        if admin_request.email:
            result = await session.execute(select(User).where(User.email == admin_request.email))
            if result.scalars().first():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
        
        hashed_password = SecurityManager.get_password_hash(admin_request.password)
        admin_user = User(
            username=admin_request.username,
            email=admin_request.email,
            full_name=admin_request.full_name,
            hashed_password=hashed_password,
            is_admin=True,
            is_active=True
        )
        
        session.add(admin_user)
        await session.commit()
        await session.refresh(admin_user)
        
        logger.info(f"Admin user created successfully: {admin_user.username}")
        return UserResponse(
            id=admin_user.id,
            username=admin_user.username,
            email=admin_user.email,
            full_name=admin_user.full_name,
            is_active=admin_user.is_active,
            is_admin=admin_user.is_admin,
            created_at=admin_user.created_at
        )
        
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Admin creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin creation failed"
        )

@router.post("/admin/promote-user/{user_id}", response_model=UserResponse)
@async_transactional
async def promote_user_to_admin(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Promote an existing user to admin role."""
    try:
        result = await session.execute(select(User).where(User.id == user_id))
        user_to_promote = result.scalars().first()
        
        if not user_to_promote:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        if user_to_promote.is_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User is already an admin"
            )
        
        user_to_promote.is_admin = True
        await session.commit()
        await session.refresh(user_to_promote)
        
        logger.info(f"User {user_to_promote.username} promoted to admin by {current_user.username}")
        
        return UserResponse(
            id=user_to_promote.id,
            username=user_to_promote.username,
            email=user_to_promote.email,
            full_name=user_to_promote.full_name,
            is_active=user_to_promote.is_active,
            is_admin=user_to_promote.is_admin,
            created_at=user_to_promote.created_at
        )
        
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"User promotion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User promotion failed"
        )

@router.post("/admin/demote-user/{user_id}", response_model=UserResponse)
@async_transactional
async def demote_admin_to_user(
    user_id: int,
    current_user: User = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Demote an admin user to regular user role."""
    try:
        result = await session.execute(select(User).where(User.id == user_id))
        user_to_demote = result.scalars().first()
        
        if not user_to_demote:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        if not user_to_demote.is_admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User is not an admin"
            )
        
        if user_to_demote.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot demote yourself"
            )
        
        admin_count = await session.execute(
            select(func.count(User.id)).where(User.is_admin == True)
        )
        if admin_count.scalar() <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot demote the last admin user"
            )
        
        user_to_demote.is_admin = False
        await session.commit()
        await session.refresh(user_to_demote)
        
        logger.info(f"Admin {user_to_demote.username} demoted to user by {current_user.username}")
        
        return UserResponse(
            id=user_to_demote.id,
            username=user_to_demote.username,
            email=user_to_demote.email,
            full_name=user_to_demote.full_name,
            is_active=user_to_demote.is_active,
            is_admin=user_to_demote.is_admin,
            created_at=user_to_demote.created_at
        )
        
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"User demotion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User demotion failed"
        )

@router.get("/admin/users", response_model=List[UserResponse])
@async_transactional
async def list_all_users(
    current_user: User = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_async_session),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, le=100)
):
    """List all users - Admin only."""
    try:
        result = await session.execute(
            select(User)
            .order_by(User.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        users = result.scalars().all()
        
        return [
            UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                is_admin=user.is_admin,
                created_at=user.created_at
            )
            for user in users
        ]
        
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user list"
        )

# PREDICTION ENDPOINTS 
@router.post("/predictions/predict", response_model=PredictionOutput)
@async_transactional
async def predict_financial_distress(
    prediction_input: PredictionInput,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Predict financial distress using ML pipeline."""
    start_time = datetime.now(timezone.utc)
    
    try:
        logger.info(f"Prediction request from user: {current_user.username}")
        
        input_data = prediction_input.input_data
        input_data['Sector'] = config.SECTOR_MAPPINGS.get(input_data.get('stra_sector', 'Retail'), '2')
        input_data['region'] = prediction_input.region
        prediction_result = predict_with_service(input_data)
        
        prediction_log = PredictionLog(
            user_id=current_user.id,
            input_data=input_data,
            financial_distress_probability=prediction_result['financial_distress_probability'],
            model_confidence=prediction_result['model_confidence'],
            risk_category=RiskLevel(prediction_result['risk_category']),
            financial_health_status=prediction_result['financial_health_status'],
            risk_level_detail=prediction_result['risk_level_detail'],
            analysis_message=prediction_result['analysis_message'],
            created_at=start_time,
            region=RegionType(prediction_input.region),
            sector=input_data['Sector'],
            model_version="2.0",
            processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            recommendations=prediction_result.get('recommendations', []),
            benchmark_comparisons=prediction_result.get('benchmark_comparisons', {}),
            visualization_data=prediction_result.get('visualization_data', {})
        )
        
        session.add(prediction_log)
        await session.commit()
        await session.refresh(prediction_log)
        
        for factor in prediction_result.get('key_influencing_factors', []):
            factor_entry = InfluencingFactorDB(
                prediction_log_id=prediction_log.id,
                name=factor['name'],
                impact_level=factor['impact_level'],
                weight=factor['weight'],
                description=factor.get('description', ''),
                shap_value=factor.get('shap_value', 0.0),
                feature_value=factor.get('feature_value', 0.0)
            )
            session.add(factor_entry)
        
        await session.commit()
        
        prediction_result['created_at'] = start_time
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"Prediction completed in {processing_time:.2f}s")
        
        return PredictionOutput(**prediction_result)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction service error: {str(e)}")

@router.get("/predictions/history")
@async_transactional
async def get_prediction_history(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    limit: int = Query(default=20, le=100)
):
    """Get user prediction history."""
    try:
        result = await session.execute(
            select(PredictionLog)
            .where(PredictionLog.user_id == current_user.id)
            .order_by(PredictionLog.created_at.desc())
            .limit(limit)
        )
        predictions = result.scalars().all()
        
        if not predictions:
            logger.info(f"No predictions found for user: {current_user.username}")
            # Return empty array for frontend .map() compatibility
            return []
        
        result_list = []
        
        for pred in predictions:
            try:
                # Get influencing factors with error handling
                factors_result = await session.execute(
                    select(InfluencingFactorDB).where(InfluencingFactorDB.prediction_log_id == pred.id)
                )
                factors = factors_result.scalars().all()
                
                # Ensure factors is always a list
                if factors is None:
                    factors = []
                
                # Ensure all required fields exist with safe defaults
                risk_category = pred.risk_category.value if pred.risk_category else "Unknown"
                
                # Prepare safe visualization data
                visualization_data = {
                    'risk_gauge': {
                        'value': float(pred.financial_distress_probability or 0),
                        'thresholds': {'low': 0.3, 'medium': 0.7, 'high': 1.0},
                        'color': _get_risk_color(risk_category)
                    },
                    'factor_chart': [
                        {
                            'factor': factor.name or 'Unknown Factor',
                            'impact': float(factor.weight or 0),
                            'description': factor.description or 'No description available'
                        }
                        for factor in factors[:5]  # Limit to top 5 factors
                    ],
                    'comparison_data': {
                        'user_score': float(pred.financial_distress_probability or 0),
                        'industry_avg': 0.25,
                        'region_avg': 0.30,
                        'benchmark': 'better' if (pred.financial_distress_probability or 1) < 0.3 else 'worse'
                    }
                }
                
                # Prepare safe prediction data - DIRECTLY COMPATIBLE WITH FRONTEND
                prediction_data = {
                    "id": pred.id,
                    "financial_distress_probability": float(pred.financial_distress_probability or 0),
                    "model_confidence": float(pred.model_confidence or 0),
                    "risk_category": risk_category,
                    "financial_health_status": pred.financial_health_status or "Unknown",
                    "risk_level_detail": pred.risk_level_detail or "No details available",
                    "analysis_message": pred.analysis_message or "No analysis available",
                    "created_at": pred.created_at.isoformat() if pred.created_at else datetime.now(timezone.utc).isoformat(),
                    "input_data": pred.input_data if pred.input_data else {},
                    "key_influencing_factors": [
                        {
                            "name": factor.name or 'Unknown Factor',
                            "impact_level": factor.impact_level or 'Unknown',
                            "weight": float(factor.weight or 0),
                            "description": factor.description or 'No description available'
                        } for factor in factors
                    ],
                    "recommendations": pred.recommendations if pred.recommendations else [
                        "Monitor key financial indicators regularly",
                        "Consider diversifying revenue streams",
                        "Maintain adequate cash flow reserves"
                    ],
                    "benchmark_comparisons": pred.benchmark_comparisons if pred.benchmark_comparisons else {
                        "industry_average": 0.25,
                        "sector_average": 0.30,
                        "region_average": 0.28
                    },
                    "visualization_data": visualization_data
                }
                result_list.append(prediction_data)
                
            except Exception as factor_error:
                logger.warning(f"Error processing prediction {pred.id}: {factor_error}")
                # Add minimal safe data for this prediction
                result_list.append({
                    "id": pred.id,
                    "financial_distress_probability": float(pred.financial_distress_probability or 0),
                    "model_confidence": float(pred.model_confidence or 0),
                    "risk_category": pred.risk_category.value if pred.risk_category else "Unknown",
                    "financial_health_status": pred.financial_health_status or "Unknown",
                    "risk_level_detail": "Error loading details",
                    "analysis_message": "Error loading analysis",
                    "created_at": pred.created_at.isoformat() if pred.created_at else datetime.now(timezone.utc).isoformat(),
                    "input_data": {},
                    "key_influencing_factors": [],
                    "recommendations": ["Unable to load recommendations"],
                    "benchmark_comparisons": {},
                    "visualization_data": {
                        'risk_gauge': {'value': 0, 'thresholds': {'low': 0.3, 'medium': 0.7, 'high': 1.0}, 'color': '#6b7280'},
                        'factor_chart': [],
                        'comparison_data': {'user_score': 0, 'industry_avg': 0.25, 'region_avg': 0.30, 'benchmark': 'unknown'}
                    }
                })
        
        logger.info(f"Retrieved {len(result_list)} predictions for user: {current_user.username}")
        
        # Return the array directly for frontend .map() compatibility
        return result_list
        
    except Exception as e:
        logger.error(f"Failed to get prediction history: {e}")
        # Return empty array instead of error object for frontend compatibility
        return []

@router.get("/predictions/{prediction_id}")
@async_transactional
async def get_prediction_details(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get detailed view of a specific prediction - FRONTEND COMPATIBLE."""
    try:
        # Validate prediction_id
        if not prediction_id or prediction_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid prediction ID"
            )
        
        # Get prediction with explicit error handling
        result = await session.execute(
            select(PredictionLog).where(
                and_(
                    PredictionLog.id == prediction_id, 
                    PredictionLog.user_id == current_user.id
                )
            )
        )
        prediction = result.scalars().first()
        
        if not prediction:
            logger.warning(f"Prediction {prediction_id} not found for user {current_user.username}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prediction with ID {prediction_id} not found"
            )
        
        # Get influencing factors with safe handling
        try:
            factors_result = await session.execute(
                select(InfluencingFactorDB).where(InfluencingFactorDB.prediction_log_id == prediction.id)
            )
            factors = factors_result.scalars().all()
            if factors is None:
                factors = []
        except Exception as factor_error:
            logger.warning(f"Error loading factors for prediction {prediction_id}: {factor_error}")
            factors = []
        
        # Prepare safe detailed response
        risk_category = prediction.risk_category.value if prediction.risk_category else "Unknown"
        
        # Return the prediction object directly (not wrapped)
        detailed_response = {
            "id": prediction.id,
            "financial_distress_probability": float(prediction.financial_distress_probability or 0),
            "model_confidence": float(prediction.model_confidence or 0),
            "risk_category": risk_category,
            "financial_health_status": prediction.financial_health_status or "Unknown",
            "risk_level_detail": prediction.risk_level_detail or "No details available",
            "analysis_message": prediction.analysis_message or "No analysis available",
            "created_at": prediction.created_at.isoformat() if prediction.created_at else datetime.now(timezone.utc).isoformat(),
            "region": prediction.region.value if prediction.region else "Unknown",
            "sector": prediction.sector or "Unknown",
            "model_version": prediction.model_version or "1.0",
            "processing_time_ms": float(prediction.processing_time_ms or 0),
            "input_data": prediction.input_data if prediction.input_data else {},
            "key_influencing_factors": [
                {
                    "name": factor.name or 'Unknown Factor',
                    "impact_level": factor.impact_level or 'Unknown',
                    "weight": float(factor.weight or 0),
                    "description": factor.description or 'No description available',
                    "shap_value": float(factor.shap_value or 0),
                    "feature_value": float(factor.feature_value or 0)
                } for factor in factors
            ],
            "recommendations": prediction.recommendations if prediction.recommendations else [
                "Monitor key financial indicators regularly",
                "Consider diversifying revenue streams",
                "Maintain adequate cash flow reserves",
                "Review and optimize operational efficiency"
            ],
            "benchmark_comparisons": prediction.benchmark_comparisons if prediction.benchmark_comparisons else {
                "industry_average": 0.25,
                "sector_average": 0.30,
                "region_average": 0.28,
                "percentile_rank": 65
            },
            "visualization_data": {
                "risk_gauge": {
                    "value": float(prediction.financial_distress_probability or 0),
                    "thresholds": {"low": 0.3, "medium": 0.7, "high": 1.0},
                    "color": _get_risk_color(risk_category)
                },
                "factor_importance": [
                    {
                        "factor": factor.name or 'Unknown Factor',
                        "importance": float(factor.weight or 0),
                        "value": float(factor.feature_value or 0),
                        "impact": factor.impact_level or 'Unknown'
                    } for factor in sorted(factors, key=lambda x: float(x.weight or 0), reverse=True)[:10]
                ],
                "risk_breakdown": {
                    "financial_metrics": float(prediction.financial_distress_probability or 0) * 0.4,
                    "market_conditions": float(prediction.financial_distress_probability or 0) * 0.3,
                    "operational_factors": float(prediction.financial_distress_probability or 0) * 0.2,
                    "external_factors": float(prediction.financial_distress_probability or 0) * 0.1
                }
            }
        }
        
        logger.info(f"Retrieved prediction details for ID: {prediction_id}")
        return detailed_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prediction details for ID {prediction_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve prediction details: {str(e)}"
        )

@router.delete("/predictions/{prediction_id}")
@async_transactional
async def delete_prediction(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Delete specific prediction - FRONTEND COMPATIBLE."""
    try:
        # Validate prediction_id
        if not prediction_id or prediction_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid prediction ID"
            )
        
        # Get prediction with ownership verification
        result = await session.execute(
            select(PredictionLog).where(
                and_(
                    PredictionLog.id == prediction_id, 
                    PredictionLog.user_id == current_user.id
                )
            )
        )
        prediction = result.scalars().first()
        
        if not prediction:
            logger.warning(f"Delete attempt failed - Prediction {prediction_id} not found for user {current_user.username}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"Prediction with ID {prediction_id} not found"
            )
        
        # Delete influencing factors first (foreign key constraint)
        try:
            factors_result = await session.execute(
                select(InfluencingFactorDB).where(InfluencingFactorDB.prediction_log_id == prediction.id)
            )
            factors = factors_result.scalars().all()
            
            factor_count = 0
            if factors:
                for factor in factors:
                    await session.delete(factor)
                    factor_count += 1
            
            logger.info(f"Deleted {factor_count} influencing factors for prediction {prediction_id}")
            
        except Exception as factor_error:
            logger.error(f"Error deleting factors for prediction {prediction_id}: {factor_error}")
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete prediction data"
            )
        
        # Delete the prediction
        try:
            await session.delete(prediction)
            await session.commit()
            
            logger.info(f"Prediction {prediction_id} successfully deleted by user {current_user.username}")
            
            # Return simple success response for frontend
            return {
                "success": True,
                "message": f"Prediction deleted successfully",
                "deleted_id": prediction_id
            }
            
        except Exception as delete_error:
            logger.error(f"Error deleting prediction {prediction_id}: {delete_error}")
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete prediction"
            )
        
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Delete prediction failed for ID {prediction_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete operation failed: {str(e)}"
        )
@router.delete("/predictions/clear")
@async_transactional
async def clear_predictions(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Clear all user predictions - FRONTEND COMPATIBLE."""
    try:
        # Get all user predictions
        result = await session.execute(
            select(PredictionLog).where(PredictionLog.user_id == current_user.id)
        )
        predictions = result.scalars().all()
        
        if not predictions:
            logger.info(f"No predictions to clear for user: {current_user.username}")
            return {
                "success": True,
                "message": "No predictions found to clear",
                "deleted_count": 0
            }
        
        total_predictions = len(predictions)
        total_factors = 0
        
        # Delete all influencing factors first
        try:
            for prediction in predictions:
                factors_result = await session.execute(
                    select(InfluencingFactorDB).where(InfluencingFactorDB.prediction_log_id == prediction.id)
                )
                factors = factors_result.scalars().all()
                
                if factors:
                    for factor in factors:
                        await session.delete(factor)
                        total_factors += 1
            
            logger.info(f"Deleted {total_factors} influencing factors")
            
        except Exception as factor_error:
            logger.error(f"Error deleting factors during clear: {factor_error}")
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear prediction data"
            )
        
        # Delete all predictions
        try:
            for prediction in predictions:
                await session.delete(prediction)
            
            await session.commit()
            
            logger.info(f"All {total_predictions} predictions cleared for user: {current_user.username}")
            
            return {
                "success": True,
                "message": f"Successfully cleared {total_predictions} predictions",
                "deleted_count": total_predictions
            }
            
        except Exception as delete_error:
            logger.error(f"Error clearing predictions: {delete_error}")
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear predictions"
            )
        
    except HTTPException:
        await session.rollback()
        raise
    except Exception as e:
        await session.rollback()
        logger.error(f"Clear predictions failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clear operation failed: {str(e)}"
        )

#  DASHBOARD AND ANALYTICS ENDPOINTS 
@router.get("/dashboard")
async def get_dashboard_data(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Dashboard Endpoint - High-level overview of financial health and risk profile
    Purpose: Provide at-a-glance overview serving as primary entry point for users
    """
    try:
        logger.info(f"Dashboard request from user: {current_user.username}")
        
        # Get user predictions
        user_predictions_result = await session.execute(
            select(PredictionLog).where(PredictionLog.user_id == current_user.id)
        )
        user_predictions = user_predictions_result.scalars().all()
        
        # Get all users' predictions for benchmarking
        all_predictions_result = await session.execute(
            select(PredictionLog).order_by(PredictionLog.created_at.desc())
        )
        all_predictions = all_predictions_result.scalars().all()
        
        # Load and process training data
        try:
            training_data = data_processor.load_training_data()
            processed_training_data = data_processor.preprocess_data(training_data)
        except Exception as e:
            logger.warning(f"Could not load training data: {e}")
            processed_training_data = None
        
        # Calculate Financial Health Score (0-100, inverted from distress probability)
        user_health_score = 100
        user_risk_category = "Low"
        latest_prediction = None
        
        if user_predictions:
            latest_prediction = max(user_predictions, key=lambda p: p.created_at)
            distress_prob = latest_prediction.financial_distress_probability or 0
            user_health_score = max(0, 100 - (distress_prob * 100))
            
            if user_health_score >= 80:
                user_risk_category = "Healthy"
            elif user_health_score >= 50:
                user_risk_category = "Moderate Risk"
            else:
                user_risk_category = "High Risk"
        
        # Risk Category Breakdown from user predictions
        risk_breakdown = {"Low": 0, "Medium": 0, "High": 0}
        if user_predictions:
            for pred in user_predictions:
                risk_cat = pred.risk_category.value if pred.risk_category else "Low"
                if risk_cat in risk_breakdown:
                    risk_breakdown[risk_cat] += 1
        
        total_user_preds = len(user_predictions)
        risk_breakdown_pct = {
            category: round((count / max(total_user_preds, 1)) * 100, 1)
            for category, count in risk_breakdown.items()
        }
        
        # Calculate industry benchmarking from all users
        benchmark_percentile = 50  # Default
        if all_predictions and user_predictions:
            user_avg_risk = sum(p.financial_distress_probability or 0 for p in user_predictions) / len(user_predictions)
            all_risks = [p.financial_distress_probability or 0 for p in all_predictions if p.financial_distress_probability is not None]
            
            if all_risks:
                better_than_count = sum(1 for risk in all_risks if user_avg_risk < risk)
                benchmark_percentile = round((better_than_count / len(all_risks)) * 100)
        
        # Key Risk Drivers - get top factors from latest prediction
        key_risk_drivers = []
        if latest_prediction:
            factors_result = await session.execute(
                select(InfluencingFactorDB)
                .where(InfluencingFactorDB.prediction_log_id == latest_prediction.id)
                .order_by(InfluencingFactorDB.weight.desc())
                .limit(5)
            )
            factors = factors_result.scalars().all()
            
            factor_descriptions = {
                'Fin_bank': 'Bank financing dependency affects liquidity and debt burden',
                'Credit': 'Credit accessibility impacts growth financing options',
                'startup': 'Early-stage companies face higher operational risks',
                'Fem_CEO': 'Female leadership correlates with better risk management',
                'Exports': 'International sales provide revenue diversification',
                'Innov': 'Innovation investment drives long-term sustainability',
                'Infor_Comp': 'Informal competition affects market positioning',
                'Pvt_Own': 'Private ownership structure impacts decision flexibility',
                'GDP': 'Economic growth environment affects business conditions'
            }
            
            for factor in factors:
                key_risk_drivers.append({
                    "factor": factor.name or "Unknown",
                    "impact": float(factor.weight or 0),
                    "impact_level": factor.impact_level or "Medium",
                    "description": factor_descriptions.get(factor.name, "Factor affecting financial risk"),
                    "direction": "increases" if (factor.weight or 0) > 0 else "decreases"
                })
        
        # Trend Overview - combine training data and user predictions over time
        trend_data = []
        
        # Add training data trends (yearly averages)
        if processed_training_data is not None:
            yearly_training_trends = processed_training_data.groupby('year')['distress'].mean()
            for year, distress_rate in yearly_training_trends.items():
                health_score = max(0, 100 - (distress_rate * 100))
                trend_data.append({
                    "period": str(year),
                    "health_score": round(health_score, 1),
                    "source": "industry_average",
                    "predictions_count": len(processed_training_data[processed_training_data['year'] == year])
                })
        
        # Add user prediction trends (monthly)
        if user_predictions:
            monthly_user_trends = {}
            for pred in user_predictions:
                month_key = pred.created_at.strftime('%Y-%m')
                if month_key not in monthly_user_trends:
                    monthly_user_trends[month_key] = []
                monthly_user_trends[month_key].append(pred.financial_distress_probability or 0)
            
            for month, risks in monthly_user_trends.items():
                avg_risk = sum(risks) / len(risks)
                health_score = max(0, 100 - (avg_risk * 100))
                trend_data.append({
                    "period": month,
                    "health_score": round(health_score, 1),
                    "source": "user_predictions",
                    "predictions_count": len(risks)
                })
        
        # Sort trend data by period
        trend_data.sort(key=lambda x: x["period"])
        
        return {
            "financial_health_snapshot": {
                "health_score": round(user_health_score, 1),
                "risk_category": user_risk_category,
                "score_change": _calculate_score_change(user_predictions),
                "color": _get_health_color(user_health_score),
                "latest_prediction_date": latest_prediction.created_at.isoformat() if latest_prediction else None
            },
            "risk_category_breakdown": {
                "user_distribution": [
                    {"category": "Low Risk", "percentage": risk_breakdown_pct["Low"], "count": risk_breakdown["Low"]},
                    {"category": "Medium Risk", "percentage": risk_breakdown_pct["Medium"], "count": risk_breakdown["Medium"]},
                    {"category": "High Risk", "percentage": risk_breakdown_pct["High"], "count": risk_breakdown["High"]}
                ],
                "benchmark_percentile": benchmark_percentile,
                "benchmark_message": f"Your startup performs better than {benchmark_percentile}% of similar businesses"
            },
            "key_risk_drivers": key_risk_drivers,
            "trend_overview": trend_data,
            "summary_stats": {
                "total_predictions": total_user_preds,
                "industry_comparisons": len(all_predictions),
                "training_data_points": len(processed_training_data) if processed_training_data is not None else 0,
                "last_updated": datetime.now(timezone.utc).isoformat()
            },
            "isEmpty": total_user_preds == 0,
            "dataQuality": "Good" if total_user_preds > 3 else "Fair" if total_user_preds > 0 else "No Data"
        }
        
    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        return {
            "financial_health_snapshot": {
                "health_score": 0,
                "risk_category": "Unknown",
                "score_change": 0,
                "color": "#6b7280"
            },
            "risk_category_breakdown": {"user_distribution": [], "benchmark_percentile": 50},
            "key_risk_drivers": [],
            "trend_overview": [],
            "summary_stats": {"total_predictions": 0},
            "isEmpty": True,
            "dataQuality": "Error",
            "error": str(e)
        }


@router.get("/analytics")
async def get_analytics_data_fixed(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session),
    days: int = Query(default=30, description="Number of days to analyze")
):
    """FIXED Analytics Endpoint with proper factor analysis data"""
    try:
        logger.info(f"Analytics request from user: {current_user.username} for {days} days")
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Get user predictions in date range
        user_predictions_result = await session.execute(
            select(PredictionLog)
            .where(and_(
                PredictionLog.user_id == current_user.id,
                PredictionLog.created_at >= start_date
            ))
            .order_by(PredictionLog.created_at.desc())
        )
        user_predictions = user_predictions_result.scalars().all()
        
        total_predictions = len(user_predictions)
        
        if total_predictions == 0:
            return {
                "isEmpty": True,
                "totalPredictions": 0,
                "period_days": days,
                "key_metrics": {
                    "total_predictions": 0,
                    "average_risk_score": 0.0,
                    "risk_distribution": [],
                    "data_quality": "No Data",
                    "health_score": 0
                },
                "risk_trend_analysis": [],
                "factor_contribution": [],
                "peer_comparison": {},
                "lastUpdated": datetime.now(timezone.utc).isoformat()
            }
        
        # Calculate key metrics
        risk_scores = [p.financial_distress_probability or 0 for p in user_predictions]
        average_risk_score = sum(risk_scores) / len(risk_scores)
        
        # Risk Distribution
        risk_counts = {"Low": 0, "Medium": 0, "High": 0}
        for pred in user_predictions:
            risk_cat = pred.risk_category.value if pred.risk_category else "Low"
            if risk_cat in risk_counts:
                risk_counts[risk_cat] += 1
        
        risk_distribution = [
            {"name": level, "value": count, "percentage": round((count / total_predictions) * 100, 1)}
            for level, count in risk_counts.items()
        ]
        
        # Monthly trends
        monthly_trends = {}
        for pred in user_predictions:
            month_key = pred.created_at.strftime('%Y-%m')
            if month_key not in monthly_trends:
                monthly_trends[month_key] = []
            monthly_trends[month_key].append(pred.financial_distress_probability or 0)
        
        trend_analysis = []
        for month, risks in sorted(monthly_trends.items()):
            avg_risk = sum(risks) / len(risks)
            health_score = max(0, 100 - (avg_risk * 100))
            trend_analysis.append({
                "period": month,
                "risk_score": avg_risk,
                "health_score": health_score,
                "prediction_count": len(risks)
            })
        
        # FIXED: Factor analysis - Properly extract and aggregate factor data
        factor_contributions = {}
        factor_count = 0
        
        for pred in user_predictions:
            # Get factors for this prediction
            factors_result = await session.execute(
                select(InfluencingFactorDB).where(InfluencingFactorDB.prediction_log_id == pred.id)
            )
            factors = factors_result.scalars().all()
            
            for factor in factors:
                factor_name = factor.name or "Unknown"
                weight = abs(factor.weight or 0)
                
                if factor_name not in factor_contributions:
                    factor_contributions[factor_name] = {
                        "weights": [],
                        "impact_level": factor.impact_level or "Medium",
                        "descriptions": []
                    }
                
                factor_contributions[factor_name]["weights"].append(weight)
                if factor.description:
                    factor_contributions[factor_name]["descriptions"].append(factor.description)
                factor_count += 1
        
        # Calculate factor contribution percentages
        factor_analysis = []
        total_weight = sum(sum(data["weights"]) for data in factor_contributions.values()) or 1
        
        for factor_name, data in factor_contributions.items():
            avg_weight = sum(data["weights"]) / len(data["weights"]) if data["weights"] else 0
            contribution_pct = (sum(data["weights"]) / total_weight) * 100
            
            # Get best description
            description = data["descriptions"][0] if data["descriptions"] else _get_factor_explanation(factor_name)
            
            factor_analysis.append({
                "factor": factor_name,
                "average_impact": avg_weight,
                "contribution_percentage": contribution_pct,
                "impact_level": data["impact_level"],
                "explanation": description,
                "frequency": len(data["weights"])
            })
        
        # Sort by contribution percentage
        factor_analysis.sort(key=lambda x: x["contribution_percentage"], reverse=True)
        
        logger.info(f"Factor analysis completed: {len(factor_analysis)} factors from {factor_count} total factor records")
        
        # Return FIXED structure
        return {
            "isEmpty": False,
            "totalPredictions": total_predictions,
            "period_days": days,
            "date_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "key_metrics": {
                "total_predictions": total_predictions,
                "average_risk_score": round(average_risk_score, 3),
                "risk_distribution": risk_distribution,
                "data_quality": "Good" if total_predictions > 5 else "Fair",
                "health_score": round(100 - (average_risk_score * 100), 1)
            },
            "risk_trend_analysis": trend_analysis,
            "factor_contribution": factor_analysis[:10],  # Top 10 factors
            "peer_comparison": {
                "overall": {
                    "comparison": "better" if average_risk_score < 0.3 else "worse",
                    "percentile": min(90, max(10, int((1 - average_risk_score) * 100)))
                }
            },
            "summary_insights": {
                "trend_direction": _calculate_trend_direction(risk_scores),
                "risk_stability": _calculate_risk_stability(risk_scores)
            },
            "lastUpdated": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analytics generation failed: {e}")
        return {
            "isEmpty": True,
            "error": str(e),
            "totalPredictions": 0,
            "period_days": days,
            "key_metrics": {
                "total_predictions": 0,
                "average_risk_score": 0.0,
                "risk_distribution": [],
                "data_quality": "Error",
                "health_score": 0
            },
            "factor_contribution": [],
            "dataQuality": "Error",
            "lastUpdated": datetime.now(timezone.utc).isoformat()
        }
    
@router.get("/debug/analytics")
async def debug_analytics(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Debug endpoint to see what analytics data is being returned"""
    try:
        # Get user predictions for debugging
        user_predictions_result = await session.execute(
            select(PredictionLog).where(PredictionLog.user_id == current_user.id)
        )
        user_predictions = user_predictions_result.scalars().all()
        
        # Return actual data structure that frontend receives
        debug_data = {
            "user_predictions_count": len(user_predictions),
            "sample_prediction": {
                "id": user_predictions[0].id if user_predictions else None,
                "risk_category": user_predictions[0].risk_category.value if user_predictions else None,
                "probability": user_predictions[0].financial_distress_probability if user_predictions else None
            } if user_predictions else None,
            "analytics_structure": {
                "isEmpty": len(user_predictions) == 0,
                "totalPredictions": len(user_predictions),
                "period_days": 30,
                "key_metrics": {
                    "total_predictions": len(user_predictions),
                    "average_risk_score": sum(p.financial_distress_probability or 0 for p in user_predictions) / max(len(user_predictions), 1),
                    "risk_distribution": [
                        {"name": "Low", "value": len([p for p in user_predictions if p.risk_category and p.risk_category.value == "Low"])},
                        {"name": "Medium", "value": len([p for p in user_predictions if p.risk_category and p.risk_category.value == "Medium"])},
                        {"name": "High", "value": len([p for p in user_predictions if p.risk_category and p.risk_category.value == "High"])}
                    ],
                    "data_quality": "Good" if len(user_predictions) > 5 else "Fair" if len(user_predictions) > 0 else "No Data"
                }
            }
        }
        
        return debug_data
        
    except Exception as e:
        logger.error(f"Debug analytics failed: {e}")
        return {"error": str(e), "user_id": current_user.id}

# @router.get("/insights")
# async def get_insights_data(
#     current_user: User = Depends(get_current_user),
#     session: AsyncSession = Depends(get_async_session)
# ):
#     """
#     Insights Endpoint - Actionable recommendations and risk alerts
#     Purpose: Provide actionable recommendations, risk alerts, and market context
#     """
#     try:
#         logger.info(f"Insights request from user: {current_user.username}")
        
#         # Get recent user predictions (last 10)
#         recent_predictions_result = await session.execute(
#             select(PredictionLog)
#             .where(PredictionLog.user_id == current_user.id)
#             .order_by(PredictionLog.created_at.desc())
#             .limit(10)
#         )
#         recent_predictions = recent_predictions_result.scalars().all()
        
#         # Load training data for context
#         try:
#             training_data = data_processor.load_training_data()
#             processed_training_data = data_processor.preprocess_data(training_data)
#         except Exception as e:
#             logger.warning(f"Could not load training data: {e}")
#             processed_training_data = None
        
#         if not recent_predictions:
#             return {
#                 "isEmpty": True,
#                 "actionable_recommendations": [
#                     {
#                         "title": "Start Your Financial Health Journey",
#                         "priority": "High",
#                         "action": "Conduct your first financial risk assessment",
#                         "reason": "Establish baseline understanding of your business risk profile",
#                         "implementation": "Use the Predict tab to analyze your company's financial data",
#                         "expected_impact": "Gain insights into key risk factors affecting your business"
#                     }
#                 ],
#                 "risk_alerts": [],
#                 "market_context": _get_market_context(),
#                 "insight_summary": {
#                     "total_insights": 1,
#                     "critical_risks": 0,
#                     "recommendations": 1,
#                     "alert_level": "None"
#                 },
#                 "dataQuality": "No Data",
#                 "lastUpdated": datetime.now(timezone.utc).isoformat()
#             }
        
#         # Analyze recent predictions for patterns
#         avg_risk = sum(p.financial_distress_probability or 0 for p in recent_predictions) / len(recent_predictions)
#         risk_trend = _calculate_trend_direction([p.financial_distress_probability or 0 for p in recent_predictions])
        
#         # Get top risk factors from recent predictions
#         all_factors = []
#         for pred in recent_predictions:
#             factors_result = await session.execute(
#                 select(InfluencingFactorDB).where(InfluencingFactorDB.prediction_log_id == pred.id)
#             )
#             factors = factors_result.scalars().all()
#             all_factors.extend(factors)
        
#         # Analyze factors
#         factor_analysis = {}
#         for factor in all_factors:
#             factor_name = factor.name or "Unknown"
#             weight = abs(factor.weight or 0)
            
#             if factor_name not in factor_analysis:
#                 factor_analysis[factor_name] = {
#                     "weights": [],
#                     "impact_level": factor.impact_level or "Medium"
#                 }
#             factor_analysis[factor_name]["weights"].append(weight)
        
#         # Calculate average impacts
#         top_risk_factors = []
#         for factor_name, data in factor_analysis.items():
#             avg_weight = sum(data["weights"]) / len(data["weights"])
#             top_risk_factors.append({
#                 "factor": factor_name,
#                 "average_impact": avg_weight,
#                 "impact_level": data["impact_level"],
#                 "frequency": len(data["weights"])
#             })
        
#         top_risk_factors.sort(key=lambda x: x["average_impact"], reverse=True)
#         top_3_factors = top_risk_factors[:3]
        
#         # Generate Actionable Recommendations
#         recommendations = _generate_recommendations(avg_risk, risk_trend, top_3_factors, processed_training_data)
        
#         # Generate Risk Alerts
#         risk_alerts = _generate_risk_alerts(avg_risk, risk_trend, top_3_factors)
        
#         # Market Context Analysis
#         market_context = _get_market_context_with_data(processed_training_data)
        
#         # Insight Summary
#         critical_risks = sum(1 for alert in risk_alerts if alert["severity"] == "Critical")
#         high_risks = sum(1 for alert in risk_alerts if alert["severity"] == "High")
        
#         alert_level = "Critical" if critical_risks > 0 else "High" if high_risks > 0 else "Medium" if risk_alerts else "Low"
        
#         insight_summary = {
#             "total_insights": len(recommendations),
#             "critical_risks": critical_risks,
#             "high_risks": high_risks,
#             "recommendations": len(recommendations),
#             "alert_level": alert_level,
#             "overall_risk_trend": risk_trend,
#             "avg_risk_score": round(avg_risk, 3),
#             "health_score": round(100 - (avg_risk * 100), 1),
#             "predictions_analyzed": len(recent_predictions)
#         }
        
#         return {
#             "isEmpty": False,
#             "actionable_recommendations": recommendations,
#             "risk_alerts": risk_alerts,
#             "market_context": market_context,
#             "insight_summary": insight_summary,
#             "key_factors_analysis": [
#                 {
#                     "factor": factor["factor"],
#                     "impact": round(factor["average_impact"], 3),
#                     "level": factor["impact_level"],
#                     "frequency": factor["frequency"],
#                     "explanation": _get_factor_explanation(factor["factor"])
#                 }
#                 for factor in top_3_factors
#             ],
#             "dataQuality": "Good" if len(recent_predictions) > 3 else "Fair",
#             "lastUpdated": datetime.now(timezone.utc).isoformat()
#         }
        
#     except Exception as e:
#         logger.error(f"Insights generation failed: {e}")
#         return {
#             "isEmpty": True,
#             "error": str(e),
#             "actionable_recommendations": [],
#             "risk_alerts": [],
#             "market_context": [],
#             "insight_summary": {"alert_level": "Error"},
#             "dataQuality": "Error",
#             "lastUpdated": datetime.now(timezone.utc).isoformat()}

# FIXED insights endpoints in your backend API

@router.get("/insights")
async def get_insights_data_fixed(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    FIXED Insights Endpoint - Always provides meaningful data
    """
    try:
        logger.info(f"Insights request from user: {current_user.username}")
        
        # Get recent user predictions (last 10)
        recent_predictions_result = await session.execute(
            select(PredictionLog)
            .where(PredictionLog.user_id == current_user.id)
            .order_by(PredictionLog.created_at.desc())
            .limit(10)
        )
        recent_predictions = recent_predictions_result.scalars().all()
        
        # FIXED: Always provide default recommendations even if no predictions
        if not recent_predictions:
            return {
                "isEmpty": False,  # Changed to False to show content
                "actionable_recommendations": [
                    {
                        "title": "Start Your Financial Health Journey",
                        "priority": "High",
                        "action": "Conduct your first financial risk assessment",
                        "reason": "Establish baseline understanding of your business risk profile",
                        "implementation": "Use the Predict tab to analyze your company's financial data",
                        "expected_impact": "Gain insights into key risk factors affecting your business"
                    },
                    {
                        "title": "Build Financial Data Foundation", 
                        "priority": "Medium",
                        "action": "Gather comprehensive financial and operational data",
                        "reason": "Accurate predictions require complete business information",
                        "implementation": "Collect 2-3 years of financial statements, operational metrics, and market data",
                        "expected_impact": "Enable more accurate risk assessments and better decision-making"
                    },
                    {
                        "title": "Establish Risk Management Framework",
                        "priority": "Medium", 
                        "action": "Create systematic approach to identify and monitor business risks",
                        "reason": "Proactive risk management prevents financial distress",
                        "implementation": "Document key risk factors, set monitoring schedules, define response procedures",
                        "expected_impact": "Early warning system for potential financial challenges"
                    }
                ],
                "risk_alerts": [],
                "market_context": _get_enhanced_market_context(),
                "insight_summary": {
                    "total_insights": 3,
                    "critical_risks": 0,
                    "recommendations": 3,
                    "alert_level": "None",
                    "health_score": 100  # Start with perfect score
                },
                "key_factors_analysis": _get_default_risk_factors(),
                "dataQuality": "Getting Started",
                "lastUpdated": datetime.now(timezone.utc).isoformat()
            }
        
        # FIXED: Enhanced data processing for existing predictions
        avg_risk = sum(p.financial_distress_probability or 0 for p in recent_predictions) / len(recent_predictions)
        risk_trend = _calculate_trend_direction([p.financial_distress_probability or 0 for p in recent_predictions])
        
        # Get top risk factors from recent predictions
        all_factors = []
        for pred in recent_predictions:
            factors_result = await session.execute(
                select(InfluencingFactorDB).where(InfluencingFactorDB.prediction_log_id == pred.id)
            )
            factors = factors_result.scalars().all()
            all_factors.extend(factors)
        
        # Analyze factors
        factor_analysis = {}
        for factor in all_factors:
            factor_name = factor.name or "Unknown"
            weight = abs(factor.weight or 0)
            
            if factor_name not in factor_analysis:
                factor_analysis[factor_name] = {
                    "weights": [],
                    "impact_level": factor.impact_level or "Medium"
                }
            factor_analysis[factor_name]["weights"].append(weight)
        
        # Calculate average impacts
        top_risk_factors = []
        for factor_name, data in factor_analysis.items():
            avg_weight = sum(data["weights"]) / len(data["weights"])
            top_risk_factors.append({
                "factor": factor_name,
                "average_impact": avg_weight,
                "impact_level": data["impact_level"],
                "frequency": len(data["weights"])
            })
        
        top_risk_factors.sort(key=lambda x: x["average_impact"], reverse=True)
        top_3_factors = top_risk_factors[:3]
        
        # FIXED: Enhanced recommendations generation - always provide recommendations
        recommendations = _generate_enhanced_recommendations(avg_risk, risk_trend, top_3_factors)
        
        # FIXED: Enhanced risk alerts generation
        risk_alerts = _generate_enhanced_risk_alerts(avg_risk, risk_trend, top_3_factors)
        
        # Enhanced Market Context
        market_context = _get_enhanced_market_context()
        
        # FIXED: Enhanced Insight Summary
        critical_risks = sum(1 for alert in risk_alerts if alert.get("severity") == "Critical")
        high_risks = sum(1 for alert in risk_alerts if alert.get("severity") == "High")
        
        alert_level = "Critical" if critical_risks > 0 else "High" if high_risks > 0 else "Medium" if risk_alerts else "Low"
        
        insight_summary = {
            "total_insights": len(recommendations) + len(risk_alerts),
            "critical_risks": critical_risks,
            "high_risks": high_risks,
            "recommendations": len(recommendations),
            "alert_level": alert_level,
            "overall_risk_trend": risk_trend,
            "avg_risk_score": round(avg_risk, 3),
            "health_score": round(100 - (avg_risk * 100), 1),
            "predictions_analyzed": len(recent_predictions)
        }
        
        # FIXED: Key factors analysis
        key_factors_analysis = [
            {
                "factor": factor["factor"],
                "impact": round(factor["average_impact"], 3),
                "level": factor["impact_level"],
                "frequency": factor["frequency"],
                "explanation": _get_factor_explanation(factor["factor"])
            }
            for factor in top_3_factors
        ]
        
        # FIXED: Return complete structure
        return {
            "isEmpty": False,
            "actionable_recommendations": recommendations,
            "risk_alerts": risk_alerts,
            "market_context": market_context,
            "insight_summary": insight_summary,
            "key_factors_analysis": key_factors_analysis,
            "dataQuality": "Good" if len(recent_predictions) > 3 else "Fair" if len(recent_predictions) > 0 else "Getting Started",
            "lastUpdated": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Insights generation failed: {e}")
        # FIXED: Return meaningful error response instead of empty
        return {
            "isEmpty": False,
            "error": str(e),
            "actionable_recommendations": [
                {
                    "title": "System Temporary Issue",
                    "priority": "Medium",
                    "action": "Please try refreshing the insights in a few moments",
                    "reason": "The AI insights system is temporarily processing your data",
                    "implementation": "Click the refresh button or navigate to another tab and return",
                    "expected_impact": "Access to personalized financial insights"
                }
            ],
            "risk_alerts": [],
            "market_context": _get_enhanced_market_context(),
            "insight_summary": {"alert_level": "System Processing", "total_insights": 1, "recommendations": 1, "critical_risks": 0},
            "key_factors_analysis": [],
            "dataQuality": "Processing",
            "lastUpdated": datetime.now(timezone.utc).isoformat()
        }

def _generate_enhanced_recommendations(avg_risk, risk_trend, top_factors):
    """FIXED: Enhanced recommendations that always provide value"""
    recommendations = []
    
    # ALWAYS provide basic financial health recommendations
    recommendations.append({
        "title": "Regular Financial Health Monitoring",
        "priority": "High",
        "action": "Establish monthly financial health check-ups using AI predictions",
        "reason": "Consistent monitoring enables early detection of financial risks",
        "implementation": "Schedule monthly predictions and track key metrics over time",
        "expected_impact": "Proactive risk management and improved financial stability"
    })
    
    # Risk level based recommendations
    if avg_risk > 0.7:
        recommendations.append({
            "title": "Immediate Risk Mitigation Required",
            "priority": "Critical",
            "action": "Implement emergency cash flow management and seek professional financial advisory",
            "reason": f"High financial distress probability of {avg_risk:.1%} indicates immediate attention needed",
            "implementation": "Contact business advisors, review all expenses, negotiate with creditors",
            "expected_impact": "Reduce immediate financial stress and stabilize operations"
        })
    elif avg_risk > 0.4:
        recommendations.append({
            "title": "Strengthen Financial Position",
            "priority": "High", 
            "action": "Diversify revenue streams and improve cash flow management",
            "reason": f"Moderate risk level of {avg_risk:.1%} suggests need for financial strengthening",
            "implementation": "Explore new markets, improve collection processes, optimize inventory",
            "expected_impact": "Improve financial stability and reduce risk exposure"
        })
    else:
        recommendations.append({
            "title": "Maintain Financial Excellence",
            "priority": "Medium",
            "action": "Continue current financial management practices while exploring growth opportunities",
            "reason": f"Low risk level of {avg_risk:.1%} indicates strong financial position",
            "implementation": "Focus on strategic investments, market expansion, operational efficiency",
            "expected_impact": "Sustainable growth while maintaining financial stability"
        })
    
    # Trend based recommendations
    if risk_trend == "increasing":
        recommendations.append({
            "title": "Address Rising Risk Trend",
            "priority": "High",
            "action": "Identify and address factors causing increasing financial risk",
            "reason": "Risk trend is increasing, requiring immediate corrective action",
            "implementation": "Review recent business decisions, analyze cost structures, assess market conditions",
            "expected_impact": "Halt risk progression and establish downward trend"
        })
    elif risk_trend == "stable":
        recommendations.append({
            "title": "Optimize Stable Performance",
            "priority": "Medium",
            "action": "Leverage stable risk profile to pursue strategic initiatives",
            "reason": "Stable risk trends provide foundation for strategic planning",
            "implementation": "Focus on long-term strategic projects, market expansion, innovation",
            "expected_impact": "Enhanced competitive position while maintaining stability"
        })
    
    # Factor-specific recommendations
    for factor in top_factors[:2]:  # Top 2 factors
        factor_name = factor["factor"]
        
        if factor_name == "Infor_Comp":
            recommendations.append({
                "title": "Combat Informal Competition",
                "priority": "Medium",
                "action": "Strengthen competitive positioning against informal competitors",
                "reason": "High informal competition reduces market share and pricing power",
                "implementation": "Improve value proposition, enhance customer service, leverage formal business advantages",
                "expected_impact": "Protect market position and maintain pricing integrity"
            })
        elif factor_name == "Fin_bank":
            recommendations.append({
                "title": "Optimize Bank Financing",
                "priority": "Medium", 
                "action": "Review and optimize banking relationships and debt structure",
                "reason": "High bank financing dependency increases interest rate and credit risk",
                "implementation": "Negotiate better terms, diversify funding sources, consider equity alternatives",
                "expected_impact": "Reduce financing costs and improve financial flexibility"
            })
        elif factor_name == "Credit":
            recommendations.append({
                "title": "Improve Credit Profile",
                "priority": "Medium",
                "action": "Enhance creditworthiness and access to financing",
                "reason": "Credit accessibility significantly impacts business growth and stability",
                "implementation": "Maintain strong financial records, build banking relationships, improve credit scores",
                "expected_impact": "Better financing terms and increased access to capital"
            })
    
    return recommendations[:6]  # Limit to top 6 recommendations

def _generate_enhanced_risk_alerts(avg_risk, risk_trend, top_factors):
    """FIXED: Enhanced risk alerts based on actual data"""
    alerts = []
    
    # Critical risk alerts
    if avg_risk > 0.8:
        alerts.append({
            "title": "Critical Financial Distress Risk",
            "severity": "Critical",
            "message": f"Extremely high financial distress probability of {avg_risk:.1%} detected",
            "impact": "Business survival at risk",
            "action": "Seek immediate professional financial assistance",
            "timeline": "Immediate action required"
        })
    elif avg_risk > 0.6:
        alerts.append({
            "title": "High Financial Risk",
            "severity": "High", 
            "message": f"High financial distress probability of {avg_risk:.1%} requires attention",
            "impact": "Significant business disruption possible",
            "action": "Implement risk mitigation strategies",
            "timeline": "Action needed within 30 days"
        })
    
    # Trend alerts
    if risk_trend == "increasing":
        alerts.append({
            "title": "Rising Risk Trend",
            "severity": "High",
            "message": "Financial risk is trending upward",
            "impact": "Deteriorating financial position",
            "action": "Investigate and address underlying causes",
            "timeline": "Review within 2 weeks"
        })
    
    # Factor-specific alerts
    for factor in top_factors[:2]:  # Top 2 factors only
        factor_name = factor["factor"]
        impact_level = factor["impact_level"]
        
        if impact_level == "High" and factor["average_impact"] > 0.05:  # Lowered threshold
            severity = "High" if factor["average_impact"] > 0.1 else "Medium"
            alerts.append({
                "title": f"High Impact from {factor_name}",
                "severity": severity,
                "message": f"{factor_name} is significantly contributing to financial risk",
                "impact": f"Contributing {factor['average_impact']:.1%} to overall risk score",
                "action": f"Focus on improving {factor_name.lower().replace('_', ' ')} metrics",
                "timeline": "Address within 60 days"
            })
    
    return alerts

def _get_enhanced_market_context():
    """Enhanced market context with more relevant information"""
    return [
        {
            "trend": "Economic Uncertainty",
            "impact": "High",
            "description": "Global economic volatility affecting business stability across regions",
            "recommendation": "Maintain flexible operations and adequate cash reserves",
            "source": "Global Economic Outlook"
        },
        {
            "trend": "Digital Transformation",
            "impact": "Medium",
            "description": "Businesses investing in technology showing improved resilience",
            "recommendation": "Consider digital investments to improve operational efficiency",
            "source": "Technology Adoption Studies"
        },
        {
            "trend": "Supply Chain Disruption", 
            "impact": "High",
            "description": "Ongoing supply chain challenges affecting multiple sectors globally",
            "recommendation": "Diversify suppliers and build strategic inventory buffers",
            "source": "Supply Chain Risk Reports"
        },
        {
            "trend": "Credit Market Tightening",
            "impact": "Medium",
            "description": "Banks becoming more selective in lending amid economic uncertainty",
            "recommendation": "Strengthen credit profiles and explore alternative financing",
            "source": "Financial Markets Analysis"
        },
        {
            "trend": "ESG Requirements",
            "impact": "Medium",
            "description": "Environmental, Social, and Governance factors increasingly impact business access to capital",
            "recommendation": "Develop ESG strategy and reporting capabilities",
            "source": "Sustainable Finance Reports"
        }
    ]

def _get_default_risk_factors():
    """Default risk factors for users with no predictions yet"""
    return [
        {
            "factor": "Cash Flow Management",
            "impact": 0.15,
            "level": "High",
            "frequency": 1,
            "explanation": "Effective cash flow management is critical for business survival and growth"
        },
        {
            "factor": "Market Position",
            "impact": 0.12,
            "level": "Medium",
            "frequency": 1,
            "explanation": "Strong market position provides competitive advantages and revenue stability"
        },
        {
            "factor": "Financial Planning",
            "impact": 0.10,
            "level": "Medium",
            "frequency": 1,
            "explanation": "Strategic financial planning enables proactive risk management and growth"
        }
    ]
    
# Enhanced insights endpoint
@router.get("/insights/fast")
async def get_insights_data_fast(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    FIXED Fast Insights Endpoint - Properly returns structured data
    """
    try:
        logger.info(f"Fast insights request from user: {current_user.username}")
        
        # Get recent user predictions (last 10)
        recent_predictions_result = await session.execute(
            select(PredictionLog)
            .where(PredictionLog.user_id == current_user.id)
            .order_by(PredictionLog.created_at.desc())
            .limit(10)
        )
        recent_predictions = recent_predictions_result.scalars().all()
        
        if not recent_predictions:
            return {
                "isEmpty": True,
                "actionable_recommendations": [
                    {
                        "title": "Start Your Financial Health Journey",
                        "priority": "High",
                        "action": "Conduct your first financial risk assessment",
                        "reason": "Establish baseline understanding of your business risk profile",
                        "implementation": "Use the Predict tab to analyze your company's financial data",
                        "expected_impact": "Gain insights into key risk factors affecting your business"
                    }
                ],
                "risk_alerts": [],
                "market_context": _get_market_context(),
                "insight_summary": {
                    "total_insights": 1,
                    "critical_risks": 0,
                    "recommendations": 1,
                    "alert_level": "None"
                },
                "key_factors_analysis": [],
                "dataQuality": "No Data",
                "lastUpdated": datetime.now(timezone.utc).isoformat()
            }
        
        # Analyze recent predictions for patterns
        avg_risk = sum(p.financial_distress_probability or 0 for p in recent_predictions) / len(recent_predictions)
        risk_trend = _calculate_trend_direction([p.financial_distress_probability or 0 for p in recent_predictions])
        
        # Get top risk factors from recent predictions
        all_factors = []
        for pred in recent_predictions:
            factors_result = await session.execute(
                select(InfluencingFactorDB).where(InfluencingFactorDB.prediction_log_id == pred.id)
            )
            factors = factors_result.scalars().all()
            all_factors.extend(factors)
        
        # Analyze factors
        factor_analysis = {}
        for factor in all_factors:
            factor_name = factor.name or "Unknown"
            weight = abs(factor.weight or 0)
            
            if factor_name not in factor_analysis:
                factor_analysis[factor_name] = {
                    "weights": [],
                    "impact_level": factor.impact_level or "Medium"
                }
            factor_analysis[factor_name]["weights"].append(weight)
        
        # Calculate average impacts
        top_risk_factors = []
        for factor_name, data in factor_analysis.items():
            avg_weight = sum(data["weights"]) / len(data["weights"])
            top_risk_factors.append({
                "factor": factor_name,
                "average_impact": avg_weight,
                "impact_level": data["impact_level"],
                "frequency": len(data["weights"])
            })
        
        top_risk_factors.sort(key=lambda x: x["average_impact"], reverse=True)
        top_3_factors = top_risk_factors[:3]
        
        # Generate Actionable Recommendations (without training data)
        recommendations = _generate_recommendations(avg_risk, risk_trend, top_3_factors, None)
        
        # Generate Risk Alerts
        risk_alerts = _generate_risk_alerts(avg_risk, risk_trend, top_3_factors)
        
        # Basic Market Context (without training data processing)
        market_context = _get_market_context()
        
        # Insight Summary
        critical_risks = sum(1 for alert in risk_alerts if alert.get("severity") == "Critical")
        high_risks = sum(1 for alert in risk_alerts if alert.get("severity") == "High")
        
        alert_level = "Critical" if critical_risks > 0 else "High" if high_risks > 0 else "Medium" if risk_alerts else "Low"
        
        insight_summary = {
            "total_insights": len(recommendations),
            "critical_risks": critical_risks,
            "high_risks": high_risks,
            "recommendations": len(recommendations),
            "alert_level": alert_level,
            "overall_risk_trend": risk_trend,
            "avg_risk_score": round(avg_risk, 3),
            "health_score": round(100 - (avg_risk * 100), 1),
            "predictions_analyzed": len(recent_predictions)
        }
        
        # FIXED: Return structure that matches frontend expectations
        return {
            "isEmpty": False,
            "actionable_recommendations": recommendations,
            "risk_alerts": risk_alerts,
            "market_context": market_context,
            "insight_summary": insight_summary,
            "key_factors_analysis": [
                {
                    "factor": factor["factor"],
                    "impact": round(factor["average_impact"], 3),
                    "level": factor["impact_level"],
                    "frequency": factor["frequency"],
                    "explanation": _get_factor_explanation(factor["factor"])
                }
                for factor in top_3_factors
            ],
            "dataQuality": "Good" if len(recent_predictions) > 3 else "Fair",
            "lastUpdated": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Fast insights generation failed: {e}")
        return {
            "isEmpty": True,
            "error": str(e),
            "actionable_recommendations": [],
            "risk_alerts": [],
            "market_context": [],
            "insight_summary": {"alert_level": "Error"},
            "key_factors_analysis": [],
            "dataQuality": "Error",
            "lastUpdated": datetime.now(timezone.utc).isoformat()
        }

    
# ================== UTILITY ENDPOINTS ==================
@router.get("/health")
async def health_check():
    """Health check endpoint for frontend connectivity testing."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "FinDistress AI API",
            "version": "2.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@router.post("/debug-request")
async def debug_request(request: Request):
    """Debug endpoint to see request data format."""
    try:
        content_type = request.headers.get("content-type", "").lower()
        authorization = request.headers.get("authorization", "No Authorization header")
        
        body = await request.body()
        body_str = body.decode() if body else None
        
        return {
            "method": request.method,
            "content_type": content_type,
            "headers": dict(request.headers),
            "authorization": authorization,
            "raw_body": body_str,
            "body_length": len(body) if body else 0,
            "query_params": dict(request.query_params),
            "path_params": dict(request.path_params)
        }
    except Exception as e:
        return {"error": str(e)}

def _calculate_score_change(predictions):
    """Calculate change in health score over time"""
    if len(predictions) < 2:
        return 0
    
    # Sort by date
    sorted_preds = sorted(predictions, key=lambda p: p.created_at)
    
    # Compare latest vs previous
    latest = sorted_preds[-1].financial_distress_probability or 0
    previous = sorted_preds[-2].financial_distress_probability or 0
    
    latest_health = 100 - (latest * 100)
    previous_health = 100 - (previous * 100)
    
    return round(latest_health - previous_health, 1)

def _get_health_color(health_score):
    """Get color for health score"""
    if health_score >= 80:
        return "#22c55e"  # Green
    elif health_score >= 50:
        return "#f59e0b"  # Yellow
    else:
        return "#ef4444"  # Red

def _get_factor_color(impact_level):
    """Get color for factor impact level"""
    colors = {
        "High": "#ef4444",
        "Medium": "#f59e0b", 
        "Low": "#22c55e"
    }
    return colors.get(impact_level, "#6b7280")

def _get_factor_explanation(factor_name):
    """Get explanation for risk factors"""
    explanations = {
        'Fin_bank': 'Bank financing dependency affects liquidity and interest burden',
        'Credit': 'Credit market accessibility determines growth financing options',
        'startup': 'Early-stage companies typically face higher operational uncertainty',
        'Fem_CEO': 'Female leadership often correlates with improved risk management',
        'Exports': 'Export revenue provides geographic diversification benefits',
        'Innov': 'Innovation investment drives competitive advantage and sustainability',
        'Infor_Comp': 'Informal competition affects pricing power and market share',
        'Pvt_Own': 'Private ownership structure impacts strategic flexibility',
        'GDP': 'Economic growth environment influences business opportunities',
        'Fin_supplier': 'Supplier financing affects working capital management',
        'Fin_equity': 'Equity financing provides stability without debt obligations',
        'Fem_wf': 'Diverse workforce brings varied perspectives and skills',
        'Pol_Inst': 'Political stability affects business continuity and planning'
    }
    return explanations.get(factor_name, 'Factor contributing to overall financial risk profile')

def _calculate_trend_direction(risk_scores):
    """Calculate if risk is increasing, decreasing, or stable"""
    if len(risk_scores) < 2:
        return "stable"
    
    # Compare first half vs second half
    mid_point = len(risk_scores) // 2
    first_half_avg = sum(risk_scores[:mid_point]) / mid_point if mid_point > 0 else 0
    second_half_avg = sum(risk_scores[mid_point:]) / (len(risk_scores) - mid_point)
    
    diff = second_half_avg - first_half_avg
    
    if abs(diff) < 0.05:
        return "stable"
    elif diff > 0:
        return "increasing"
    else:
        return "decreasing"

def _calculate_risk_stability(risk_scores):
    """Calculate risk stability/volatility"""
    if len(risk_scores) < 2:
        return "stable"
    
    import statistics
    std_dev = statistics.stdev(risk_scores)
    
    if std_dev < 0.1:
        return "stable"
    elif std_dev < 0.2:
        return "moderate"
    else:
        return "volatile"

def _get_performance_rating(user_risk, industry_avg):
    """Get performance rating compared to industry"""
    diff_pct = ((user_risk - industry_avg) / industry_avg) * 100 if industry_avg > 0 else 0
    
    if diff_pct < -20:
        return "Excellent"
    elif diff_pct < -10:
        return "Good"
    elif diff_pct < 10:
        return "Average"
    elif diff_pct < 20:
        return "Below Average"
    else:
        return "Poor"

def _generate_recommendations(avg_risk, risk_trend, top_factors, training_data):
    """Generate actionable recommendations based on analysis"""
    recommendations = []
    
    # Risk level based recommendations
    if avg_risk > 0.7:
        recommendations.append({
            "title": "Immediate Risk Mitigation Required",
            "priority": "Critical",
            "action": "Implement emergency cash flow management and seek professional financial advisory",
            "reason": f"High financial distress probability of {avg_risk:.1%} indicates immediate attention needed",
            "implementation": "Contact business advisors, review all expenses, negotiate with creditors",
            "expected_impact": "Reduce immediate financial stress and stabilize operations"
        })
    elif avg_risk > 0.4:
        recommendations.append({
            "title": "Strengthen Financial Position",
            "priority": "High", 
            "action": "Diversify revenue streams and improve cash flow management",
            "reason": f"Moderate risk level of {avg_risk:.1%} suggests need for financial strengthening",
            "implementation": "Explore new markets, improve collection processes, optimize inventory",
            "expected_impact": "Improve financial stability and reduce risk exposure"
        })
    
    # Trend based recommendations
    if risk_trend == "increasing":
        recommendations.append({
            "title": "Address Rising Risk Trend",
            "priority": "High",
            "action": "Identify and address factors causing increasing financial risk",
            "reason": "Risk trend is increasing, requiring immediate corrective action",
            "implementation": "Review recent business decisions, analyze cost structures, assess market conditions",
            "expected_impact": "Halt risk progression and establish downward trend"
        })
    
    # Factor-specific recommendations
    for factor in top_factors:
        factor_name = factor["factor"]
        
        if factor_name == "Infor_Comp":
            recommendations.append({
                "title": "Combat Informal Competition",
                "priority": "Medium",
                "action": "Strengthen competitive positioning against informal competitors",
                "reason": "High informal competition reduces market share and pricing power",
                "implementation": "Improve value proposition, enhance customer service, leverage formal business advantages",
                "expected_impact": "Protect market position and maintain pricing integrity"
            })
        elif factor_name == "Fin_bank":
            recommendations.append({
                "title": "Optimize Bank Financing",
                "priority": "Medium", 
                "action": "Review and optimize banking relationships and debt structure",
                "reason": "High bank financing dependency increases interest rate and credit risk",
                "implementation": "Negotiate better terms, diversify funding sources, consider equity alternatives",
                "expected_impact": "Reduce financing costs and improve financial flexibility"
            })
        elif factor_name == "Innov":
            recommendations.append({
                "title": "Increase Innovation Investment",
                "priority": "Medium",
                "action": "Allocate more resources to research, development, and innovation",
                "reason": "Low innovation investment limits competitive advantage and growth potential",
                "implementation": "Set R&D budget targets, explore government innovation grants, partner with research institutions",
                "expected_impact": "Enhance competitive position and long-term sustainability"
            })
        elif factor_name == "Exports":
            recommendations.append({
                "title": "Expand International Markets",
                "priority": "Medium",
                "action": "Develop export capabilities and international market presence",
                "reason": "Limited export activity reduces revenue diversification and growth opportunities",
                "implementation": "Research export markets, obtain necessary certifications, develop distribution channels",
                "expected_impact": "Diversify revenue streams and reduce domestic market dependency"
            })
        elif factor_name == "Fem_CEO":
            recommendations.append({
                "title": "Enhance Leadership Diversity",
                "priority": "Low",
                "action": "Promote diversity in leadership and management positions",
                "reason": "Diverse leadership teams often demonstrate better risk management and decision-making",
                "implementation": "Develop leadership pipeline, provide mentoring programs, establish diversity goals",
                "expected_impact": "Improve strategic decision-making and organizational resilience"
            })
    
    return recommendations[:6]  # Limit to top 6 recommendations

def _generate_risk_alerts(avg_risk, risk_trend, top_factors):
    """Generate risk alerts based on analysis"""
    alerts = []
    
    # Critical risk alerts
    if avg_risk > 0.8:
        alerts.append({
            "title": "Critical Financial Distress Risk",
            "severity": "Critical",
            "message": f"Extremely high financial distress probability of {avg_risk:.1%} detected",
            "impact": "Business survival at risk",
            "action": "Seek immediate professional financial assistance",
            "timeline": "Immediate action required"
        })
    elif avg_risk > 0.6:
        alerts.append({
            "title": "High Financial Risk",
            "severity": "High", 
            "message": f"High financial distress probability of {avg_risk:.1%} requires attention",
            "impact": "Significant business disruption possible",
            "action": "Implement risk mitigation strategies",
            "timeline": "Action needed within 30 days"
        })
    
    # Trend alerts
    if risk_trend == "increasing":
        alerts.append({
            "title": "Rising Risk Trend",
            "severity": "High",
            "message": "Financial risk is trending upward",
            "impact": "Deteriorating financial position",
            "action": "Investigate and address underlying causes",
            "timeline": "Review within 2 weeks"
        })
    
    # Factor-specific alerts
    for factor in top_factors[:2]:  # Top 2 factors only
        factor_name = factor["factor"]
        impact_level = factor["impact_level"]
        
        if impact_level == "High":
            severity = "High" if factor["average_impact"] > 0.3 else "Medium"
            alerts.append({
                "title": f"High Impact from {factor_name}",
                "severity": severity,
                "message": f"{factor_name} is significantly contributing to financial risk",
                "impact": f"Contributing {factor['average_impact']:.1%} to overall risk score",
                "action": f"Focus on improving {factor_name.lower()} metrics",
                "timeline": "Address within 60 days"
            })
    
    return alerts

def _get_market_context():
    """Get basic market context"""
    return [
        {
            "trend": "Economic Uncertainty",
            "impact": "High",
            "description": "Global economic volatility affecting business stability across regions",
            "recommendation": "Maintain flexible operations and adequate cash reserves",
            "source": "Global Economic Outlook"
        },
        {
            "trend": "Digital Transformation",
            "impact": "Medium",
            "description": "Businesses investing in technology showing improved resilience",
            "recommendation": "Consider digital investments to improve operational efficiency",
            "source": "Technology Adoption Studies"
        },
        {
            "trend": "Supply Chain Disruption", 
            "impact": "High",
            "description": "Ongoing supply chain challenges affecting multiple sectors globally",
            "recommendation": "Diversify suppliers and build strategic inventory buffers",
            "source": "Supply Chain Risk Reports"
        }
    ]

def _get_market_context_with_data(training_data):
    """Get market context enhanced with training data insights - FIXED"""
    base_context = _get_market_context()
    
    if training_data is not None:
        try:
            # Check if region2 column exists, if not create it
            if 'region2' not in training_data.columns:
                logger.warning("region2 column not found, creating it")
                african_countries = [
                    'Angola', 'Bangladesh', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon',
                    'Central African Republic', 'Chad', 'Congo', "Cote d'Ivoire", 'DRC', 'Djibouti', 'Egypt',
                    'Equatorial Guinea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea',
                    'Lebanon', 'Lesotho', 'Liberia', 'Guineabissau', 'Kenya', 'Madagascar', 'Malawi',
                    'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger',
                    'Nigeria', 'Rwanda', 'Senegal', 'Seychelles', 'Sierra Leone', 'South Sudan',
                    'Southafrica', 'Sudan', 'Tanzania', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
                ]
                
                # Create region2 based on country2 if it exists
                if 'country2' in training_data.columns:
                    training_data.loc[training_data['country2'].isin(african_countries), 'region2'] = 'AFR'
                    training_data.loc[~training_data['country2'].isin(african_countries), 'region2'] = 'ROW'
                else:
                    # Fallback: assign regions based on existing data distribution
                    training_data['region2'] = 'AFR'  # Default to AFR
            
            # Add insights from training data with safe column access
            if 'region2' in training_data.columns and 'distress' in training_data.columns:
                avg_distress_afr = training_data[training_data['region2'] == 'AFR']['distress'].mean()
                avg_distress_row = training_data[training_data['region2'] == 'ROW']['distress'].mean()
                
                # Sector analysis - use safe column access
                if 'stra_sector' in training_data.columns:
                    sector_risks = training_data.groupby('stra_sector')['distress'].mean().sort_values(ascending=False)
                    highest_risk_sector = sector_risks.index[0] if len(sector_risks) > 0 else "Unknown"
                else:
                    highest_risk_sector = "Manufacturing"  # Default sector
                
                base_context.extend([
                    {
                        "trend": "Regional Risk Patterns",
                        "impact": "Medium",
                        "description": f"AFR region shows {avg_distress_afr:.1%} average distress rate vs ROW at {avg_distress_row:.1%}",
                        "recommendation": "Consider regional factors in business planning and expansion",
                        "source": "Historical Business Data Analysis"
                    },
                    {
                        "trend": "Sector Risk Variation",
                        "impact": "Medium", 
                        "description": f"Highest risk observed in {highest_risk_sector} sector",
                        "recommendation": "Monitor sector-specific risk factors and adapt strategies accordingly",
                        "source": "Sector Performance Analysis"
                    }
                ])
            else:
                logger.warning("Required columns (region2, distress) not found for market context analysis")
                
        except Exception as e:
            logger.warning(f"Could not add training data context: {e}")
    
    return base_context

# ================== UTILITY FUNCTIONS CONTINUED ==================

def _get_risk_color(risk_category: str) -> str:
    """Get color for risk visualization."""
    colors = {'Low': '#22c55e', 'Medium': '#f59e0b', 'High': '#ef4444'}
    return colors.get(risk_category, '#6b7280')

def _generate_prediction_trends(predictions):
    """Generate prediction trends data."""
    if not predictions:
        return []
    
    monthly_data = {}
    for pred in predictions:
        month_key = pred.created_at.strftime('%Y-%m')
        if month_key not in monthly_data:
            monthly_data[month_key] = {'total': 0, 'high_risk': 0}
        
        monthly_data[month_key]['total'] += 1
        if pred.risk_category.value == 'High':
            monthly_data[month_key]['high_risk'] += 1
    
    return [
        {
            "period": month,
            "distressRate": round((data['high_risk'] / data['total']) * 100, 2),
            "totalAnalyzed": data['total']
        }
        for month, data in sorted(monthly_data.items())
    ]

def _generate_sector_analysis(predictions):
    """Generate sector analysis data."""
    return [
        {"sector": "Manufacturing", "distressed": 15.5, "healthy": 84.5, "total": 200},
        {"sector": "Retail", "distressed": 22.3, "healthy": 77.7, "total": 150},
        {"sector": "Services", "distressed": 18.7, "healthy": 81.3, "total": 300}
    ]

def _generate_comparative_factors(predictions):
    """Generate comparative risk factors."""
    return [
        {"factor": "Financial Leverage", "user_avg": 0.45, "training_avg": 0.40, "risk": "Medium"},
        {"factor": "Market Position", "user_avg": 0.65, "training_avg": 0.60, "risk": "Low"},
        {"factor": "Operational Efficiency", "user_avg": 0.55, "training_avg": 0.50, "risk": "Medium"},
        {"factor": "Industry Risk", "user_avg": 0.35, "training_avg": 0.40, "risk": "Low"}
    ]