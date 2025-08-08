"""
Authentication schemas for Financial Distress Prediction API
Separate auth models to avoid circular imports and conflicts
"""

from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    """User access roles."""
    USER = "user"
    ADMIN = "admin"
    ANALYST = "analyst"

# AUTHENTICATION SCHEMAS

class Token(BaseModel):
    """JWT authentication token response."""
    access_token: str = Field(..., description="JWT access token for API authentication")
    refresh_token: str = Field(..., description="Refresh token for token renewal")
    token_type: str = Field(default="bearer", description="Token type (always 'bearer')")
    expires_in: int = Field(..., description="Access token expiration time in seconds")
    scopes: Optional[List[str]] = Field(default=[], description="Token access scopes")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VybmFtZSIsImV4cCI6MTY3MzI4MzY0MH0...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VybmFtZSIsInR5cGUiOiJyZWZyZXNoIn0...",
                "token_type": "bearer",
                "expires_in": 1800,
                "scopes": ["user"]
            }
        }
    )

class TokenData(BaseModel):
    """Token payload data extracted from JWT."""
    username: Optional[str] = Field(None, description="Username from token")
    user_id: Optional[int] = Field(None, description="User ID from token")
    scopes: List[str] = Field(default_factory=list, description="Access scopes")
    exp: Optional[datetime] = Field(None, description="Token expiration time")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "john_doe",
                "user_id": 123,
                "scopes": ["user"],
                "exp": "2024-01-15T10:30:00Z"
            }
        }
    )

class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str = Field(..., description="Valid refresh token to exchange for new access token")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VybmFtZSIsInR5cGUiOiJyZWZyZXNoIn0..."
            }
        }
    )

class RefreshTokenResponse(BaseModel):
    """Response when refreshing access token."""
    access_token: str = Field(..., description="New JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="New token expiration time in seconds")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VybmFtZSIsImV4cCI6MTY3MzI4MzY0MH0...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }
    )

# USER AUTHENTICATION SCHEMAS

class UserLogin(BaseModel):
    """User login credentials."""
    username: str = Field(..., min_length=3, max_length=50, description="Username or email address")
    password: str = Field(..., min_length=1, description="Account password")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "john_doe",
                "password": "SecurePassword123!"
            }
        }
    )

class UserBase(BaseModel):
    """Base user information schema."""
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
    confirm_password: str = Field(..., description="Password confirmation")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "john_doe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "password": "SecurePassword123!",
                "confirm_password": "SecurePassword123!",
                "company_name": "Acme Corp",
                "industry": "Technology",
                "country": "Kenya"
            }
        }
    )

class UserResponse(UserBase):
    """User response for API responses."""
    id: int
    role: UserRole = Field(description="User access level")
    is_active: bool = Field(description="Account status")
    is_verified: bool = Field(description="Email verification status")
    is_admin: bool = Field(description="Admin privileges")
    last_login: Optional[datetime] = Field(description="Last login timestamp")
    login_count: int = Field(description="Total number of logins")
    created_at: datetime = Field(description="Account creation date")
    
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 123,
                "username": "john_doe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "role": "user",
                "is_active": True,
                "is_verified": True,
                "is_admin": False,
                "last_login": "2024-01-15T10:30:00Z",
                "login_count": 15,
                "created_at": "2024-01-01T08:00:00Z",
                "company_name": "Acme Corp",
                "industry": "Technology",
                "country": "Kenya"
            }
        }
    )

class UserUpdate(BaseModel):
    """User profile update schema."""
    email: Optional[EmailStr] = Field(None, description="New email address")
    full_name: Optional[str] = Field(None, max_length=100, description="Updated full name")
    company_name: Optional[str] = Field(None, max_length=200, description="Updated company name")
    industry: Optional[str] = Field(None, max_length=100, description="Updated industry")
    country: Optional[str] = Field(None, max_length=100, description="Updated country")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "newemail@example.com",
                "full_name": "John Smith",
                "company_name": "New Company Ltd",
                "industry": "Finance",
                "country": "Uganda"
            }
        }
    )

class PasswordChange(BaseModel):
    """Password change request."""
    current_password: str = Field(..., description="Current password for verification")
    new_password: str = Field(..., min_length=8, max_length=128, description="New secure password")
    confirm_new_password: str = Field(..., description="New password confirmation")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "current_password": "OldPassword123!",
                "new_password": "NewSecurePassword456@",
                "confirm_new_password": "NewSecurePassword456@"
            }
        }
    )

# SECURITY SCHEMAS

class LoginAttempt(BaseModel):
    """Login attempt tracking."""
    username: str = Field(description="Attempted username")
    success: bool = Field(description="Whether login succeeded")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    timestamp: datetime = Field(description="Attempt timestamp")
    failure_reason: Optional[str] = Field(None, description="Reason for failure if applicable")

class SecurityEvent(BaseModel):
    """Security-related events."""
    event_type: str = Field(description="Type of security event")
    user_id: Optional[int] = Field(None, description="Associated user ID")
    severity: str = Field(description="Event severity level")
    description: str = Field(description="Event description")
    metadata: Optional[dict] = Field(None, description="Additional event data")
    timestamp: datetime = Field(description="Event timestamp")

# TOKEN VALIDATION SCHEMAS

class TokenValidationRequest(BaseModel):
    """Request to validate a token."""
    token: str = Field(..., description="Token to validate")
    
class TokenValidationResponse(BaseModel):
    """Token validation result."""
    valid: bool = Field(description="Whether token is valid")
    expired: bool = Field(description="Whether token is expired")
    user_id: Optional[int] = Field(None, description="User ID if valid")
    username: Optional[str] = Field(None, description="Username if valid")
    scopes: List[str] = Field(default_factory=list, description="Token scopes if valid")
    expires_at: Optional[datetime] = Field(None, description="Token expiration time")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "valid": True,
                "expired": False,
                "user_id": 123,
                "username": "john_doe",
                "scopes": ["user"],
                "expires_at": "2024-01-15T12:30:00Z"
            }
        }
    )