import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
import logging

from app.database.database import get_async_session
from app.database.models import User

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# FIXED: Enhanced environment variables with better defaults
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY")
if not AUTH_SECRET_KEY:
    logger.error("AUTH_SECRET_KEY not set in environment variables")
    raise ValueError("AUTH_SECRET_KEY must be set in environment variables")

ALGORITHM = os.getenv("ALGORITHM", "HS256")
# CRITICAL FIX: Extended token expiration times to prevent frequent logouts
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))  # 2 hours instead of 30 minutes
REFRESH_TOKEN_EXPIRE_MINUTES = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", "10080"))  # 7 days

# Password hashing configuration
pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__rounds=12
)

# OAuth2 configuration
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/token",
    scopes={
        "user": "Basic user access",
        "admin": "Administrative access"
    }
)

class TokenData:
    def __init__(self, username: str, scopes: List[str] = None):
        self.username = username
        self.scopes = scopes or []

class SecurityManager:
    """Enhanced security management class with persistent tokens."""
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Generate password hash with enhanced security."""
        try:
            return pwd_context.hash(password)
        except Exception as e:
            logger.error(f"Error hashing password: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error processing password"
            )

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Error verifying password: {str(e)}")
            return False

    @staticmethod
    def create_access_token(
        data: dict, 
        expires_delta: Optional[timedelta] = None,
        scopes: List[str] = None
    ) -> tuple[str, datetime]:
        """Create JWT access token with enhanced payload."""
        try:
            to_encode = data.copy()
            if expires_delta:
                expire = datetime.now(timezone.utc) + expires_delta
            else:
                expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            
            # ENHANCED: Add more claims for better token management
            to_encode.update({
                "exp": expire,
                "iat": datetime.now(timezone.utc),
                "type": "access",
                "scopes": scopes or ["user"],
                "jti": secrets.token_urlsafe(16),  # Unique token ID
                "version": "2.0"  # Token version for future compatibility
            })
            
            encoded_jwt = jwt.encode(to_encode, AUTH_SECRET_KEY, algorithm=ALGORITHM)
            logger.debug(f"Created access token for user: {data.get('sub')} (expires: {expire})")
            return encoded_jwt, expire
        except Exception as e:
            logger.error(f"Error creating access token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating access token"
            )

    @staticmethod
    def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> tuple[str, datetime]:
        """Create JWT refresh token with enhanced security."""
        try:
            to_encode = data.copy()
            if expires_delta:
                expire = datetime.now(timezone.utc) + expires_delta
            else:
                expire = datetime.now(timezone.utc) + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
            
            # ENHANCED: Add security features to refresh token
            to_encode.update({
                "exp": expire,
                "iat": datetime.now(timezone.utc),
                "type": "refresh",
                "jti": secrets.token_urlsafe(16),
                "version": "2.0"
            })
            
            encoded_jwt = jwt.encode(to_encode, AUTH_SECRET_KEY, algorithm=ALGORITHM)
            logger.debug(f"Created refresh token for user: {data.get('sub')} (expires: {expire})")
            return encoded_jwt, expire
        except Exception as e:
            logger.error(f"Error creating refresh token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating refresh token"
            )

    @staticmethod
    def verify_token(
        token: str, 
        expected_type: str = "access",
        required_scopes: List[str] = None
    ) -> TokenData:
        """Verify JWT token with enhanced validation."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(token, AUTH_SECRET_KEY, algorithms=[ALGORITHM])
            logger.debug(f"Token payload: {payload}")
            
            # ENHANCED: Validate token type
            if payload.get("type") != expected_type:
                logger.warning(f"Invalid token type: expected {expected_type}, got {payload.get('type')}")
                raise credentials_exception
            
            # ENHANCED: Validate token version for compatibility
            token_version = payload.get("version", "1.0")
            if token_version not in ["1.0", "2.0"]:
                logger.warning(f"Unsupported token version: {token_version}")
                raise credentials_exception
            
            username: str = payload.get("sub")
            if username is None:
                logger.warning("Token missing 'sub' claim")
                raise credentials_exception
            
            token_scopes = payload.get("scopes", [])
            
            # ENHANCED: Scope validation
            if required_scopes:
                if not any(scope in token_scopes for scope in required_scopes):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
            
            return TokenData(username=username, scopes=token_scopes)
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except JWTError as e:
            logger.warning(f"JWT verification failed: {str(e)}")
            raise credentials_exception
        except Exception as e:
            logger.error(f"Unexpected error verifying token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error verifying token"
            )

# ENHANCED AUTHENTICATION DEPENDENCY FUNCTIONS

async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    session: AsyncSession = Depends(get_async_session)
):
    """Enhanced user authentication with better error handling and token validation."""
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
        
        # ENHANCED: Use SecurityManager for token verification
        try:
            token_data = SecurityManager.verify_token(token, expected_type="access")
            username = token_data.username
                
        except jwt.ExpiredSignatureError:
            logger.warning("Access token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired. Please refresh your session.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token. Please sign in again.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database with enhanced validation
        result = await session.execute(select(User).where(User.username == username))
        user = result.scalars().first()
        
        if user is None:
            logger.warning(f"User not found: {username}")
            raise credentials_exception
            
        if not user.is_active:
            logger.warning(f"Inactive user attempted access: {username}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive. Please contact support."
            )
        
        # ENHANCED: Update last activity timestamp
        try:
            user.last_login = datetime.now(timezone.utc)
            await session.commit()
        except Exception as e:
            logger.warning(f"Failed to update last login for {username}: {e}")
            # Don't fail authentication just because we can't update last login
        
        logger.debug(f"User authenticated successfully: {username}")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise credentials_exception

async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_async_session)
) -> Optional[User]:
    """Optional authentication - returns None if not authenticated."""
    if not token:
        return None
        
    try:
        return await get_current_user(token, session)
    except HTTPException:
        return None

async def get_current_admin_user(current_user: User = Depends(get_current_user)):
    """Ensure user is admin with enhanced validation."""
    if not current_user.is_admin:
        logger.warning(f"Non-admin user attempted admin access: {current_user.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Administrator privileges required"
        )
    return current_user

async def authenticate_user(username: str, password: str, session: AsyncSession) -> Optional[User]:
    """Enhanced user authentication with rate limiting consideration."""
    try:
        result = await session.execute(select(User).where(User.username == username))
        user = result.scalars().first()
        if not user:
            logger.warning(f"Authentication attempt for non-existent user: {username}")
            return None
        
        if not user.is_active:
            logger.warning(f"Authentication attempt for inactive user: {username}")
            return None
        
        if not SecurityManager.verify_password(password, user.hashed_password):
            logger.warning(f"Invalid password for user: {username}")
            return None
        
        logger.info(f"User authenticated successfully: {username}")
        return user
    except Exception as e:
        logger.error(f"Error authenticating user {username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication system error: {str(e)}"
        )

async def create_user_tokens(user: User, session: AsyncSession) -> Dict[str, Any]:
    """Enhanced token creation with persistent storage."""
    try:
        scopes = ["user"]
        is_admin = getattr(user, 'is_admin', False)
        if is_admin:
            scopes.append("admin")
        
        # ENHANCED: Create tokens with longer expiration
        access_token, access_expires = SecurityManager.create_access_token(
            data={
                "sub": user.username,
                "user_id": user.id,
                "email": user.email or "",
                "is_admin": is_admin
            },
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            scopes=scopes
        )
        
        refresh_token, refresh_expires = SecurityManager.create_refresh_token(
            data={
                "sub": user.username,
                "user_id": user.id
            },
            expires_delta=timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
        )
        
        # ENHANCED: Store refresh token hash for validation
        user.refresh_token_hash = SecurityManager.get_password_hash(refresh_token)
        user.refresh_token_expires_at = refresh_expires
        user.last_login = datetime.now(timezone.utc)
        
        # ENHANCED: Track login count
        if hasattr(user, 'login_count'):
            user.login_count = (user.login_count or 0) + 1
        
        session.add(user)
        await session.commit()
        
        logger.info(f"Tokens created for user: {user.username} (expires in {ACCESS_TOKEN_EXPIRE_MINUTES} minutes)")
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Return in seconds
            "refresh_expires_in": REFRESH_TOKEN_EXPIRE_MINUTES * 60,
            "scopes": scopes,
            "user_id": user.id,
            "username": user.username,
            "is_admin": is_admin
        }
        
    except Exception as e:
        logger.error(f"Error creating tokens for user {user.username}: {str(e)}")
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating authentication tokens: {str(e)}"
        )

async def verify_refresh_token(refresh_token: str, session: AsyncSession) -> Optional[User]:
    """Enhanced refresh token verification with security checks."""
    try:
        token_data = SecurityManager.verify_token(refresh_token, expected_type="refresh")
        
        result = await session.execute(select(User).where(User.username == token_data.username))
        user = result.scalars().first()
        if not user:
            logger.warning(f"User not found for refresh token: {token_data.username}")
            return None
        
        if not user.is_active:
            logger.warning(f"Inactive user refresh attempt: {user.username}")
            return None
        
        # ENHANCED: Verify stored refresh token hash
        if not user.refresh_token_hash:
            logger.warning(f"No refresh token hash stored for user: {user.username}")
            return None
        
        if not SecurityManager.verify_password(refresh_token, user.refresh_token_hash):
            logger.warning(f"Invalid refresh token for user: {user.username}")
            return None
        
        # ENHANCED: Check if refresh token has expired
        if user.refresh_token_expires_at and user.refresh_token_expires_at < datetime.now(timezone.utc):
            logger.warning(f"Expired refresh token for user: {user.username}")
            return None
        
        logger.info(f"Refresh token verified for user: {user.username}")
        return user
        
    except jwt.ExpiredSignatureError:
        logger.warning("Refresh token has expired")
        return None
    except Exception as e:
        logger.error(f"Error verifying refresh token: {str(e)}")
        return None

async def revoke_refresh_token(user: User, session: AsyncSession) -> bool:
    """Revoke user's refresh token for enhanced security."""
    try:
        user.refresh_token_hash = None
        user.refresh_token_expires_at = None
        session.add(user)
        await session.commit()
        
        logger.info(f"Refresh token revoked for user: {user.username}")
        return True
        
    except Exception as e:
        logger.error(f"Error revoking refresh token for user {user.username}: {str(e)}")
        await session.rollback()
        return False

# ENHANCED: Rate limiting and security monitoring
class EnhancedRateLimiter:
    """Enhanced in-memory rate limiter with security features."""
    def __init__(self):
        self.requests = {}
        self.blocked_ips = {}
        self.failed_attempts = {}
    
    def is_allowed(self, key: str, limit: int, window: int, track_failures: bool = False) -> bool:
        """Check if request is allowed with enhanced security tracking."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=window)
        
        # Check if IP is temporarily blocked
        if key in self.blocked_ips:
            if self.blocked_ips[key] > now:
                logger.warning(f"Blocked IP attempted access: {key}")
                return False
            else:
                del self.blocked_ips[key]
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove requests older than the window
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > window_start]
        
        if len(self.requests[key]) >= limit:
            logger.warning(f"Rate limit exceeded for {key}: {len(self.requests[key])} requests in {window}s")
            
            # ENHANCED: Block IP after repeated violations
            if track_failures:
                self.failed_attempts[key] = self.failed_attempts.get(key, 0) + 1
                if self.failed_attempts[key] >= 5:  # Block after 5 violations
                    self.blocked_ips[key] = now + timedelta(minutes=15)  # Block for 15 minutes
                    logger.warning(f"IP blocked for repeated violations: {key}")
            
            return False
        
        self.requests[key].append(now)
        return True

    def record_failed_login(self, ip: str):
        """Record failed login attempt for security monitoring."""
        now = datetime.now(timezone.utc)
        self.failed_attempts[ip] = self.failed_attempts.get(ip, 0) + 1
        
        # Block IP after 10 failed login attempts
        if self.failed_attempts[ip] >= 10:
            self.blocked_ips[ip] = now + timedelta(hours=1)  # Block for 1 hour
            logger.warning(f"IP blocked for excessive failed logins: {ip}")

    def get_remaining_requests(self, key: str, limit: int, window: int) -> int:
        """Get remaining requests allowed in the current window."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=window)
        
        if key not in self.requests:
            return limit
        
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > window_start]
        return limit - len(self.requests[key])

    def reset(self, key: str):
        """Reset rate limit for a specific key."""
        if key in self.requests:
            del self.requests[key]
        if key in self.failed_attempts:
            del self.failed_attempts[key]
        if key in self.blocked_ips:
            del self.blocked_ips[key]
        logger.debug(f"Rate limit and security data reset for {key}")

    def cleanup_old_entries(self):
        """Clean up old entries to prevent memory leaks."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24)
        
        # Clean up old requests
        for key in list(self.requests.keys()):
            self.requests[key] = [req_time for req_time in self.requests[key] if req_time > cutoff]
            if not self.requests[key]:
                del self.requests[key]
        
        # Clean up expired blocks
        for key in list(self.blocked_ips.keys()):
            if self.blocked_ips[key] < now:
                del self.blocked_ips[key]

# Global rate limiter instance
rate_limiter = EnhancedRateLimiter()

# ENHANCED: Security event logging
class SecurityEventLogger:
    """Log security-related events for monitoring."""
    
    @staticmethod
    def log_login_attempt(username: str, success: bool, ip_address: str = None, user_agent: str = None):
        """Log login attempts for security monitoring."""
        event_data = {
            "event_type": "login_attempt",
            "username": username,
            "success": success,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if success:
            logger.info(f"Successful login: {username} from {ip_address}")
        else:
            logger.warning(f"Failed login attempt: {username} from {ip_address}")
            if ip_address:
                rate_limiter.record_failed_login(ip_address)
    
    @staticmethod
    def log_token_refresh(username: str, success: bool):
        """Log token refresh attempts."""
        event_data = {
            "event_type": "token_refresh",
            "username": username,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if success:
            logger.info(f"Token refreshed successfully: {username}")
        else:
            logger.warning(f"Token refresh failed: {username}")
    
    @staticmethod
    def log_admin_action(admin_username: str, action: str, target_user: str = None):
        """Log administrative actions."""
        event_data = {
            "event_type": "admin_action",
            "admin_username": admin_username,
            "action": action,
            "target_user": target_user,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Admin action: {admin_username} performed {action} on {target_user}")

# Export the security event logger
security_logger = SecurityEventLogger()