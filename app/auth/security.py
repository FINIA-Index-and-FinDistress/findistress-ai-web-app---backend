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

# Environment variables
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY")
if not AUTH_SECRET_KEY:
    logger.error("AUTH_SECRET_KEY not set in environment variables")
    raise ValueError("AUTH_SECRET_KEY must be set in environment variables")

ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_MINUTES = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", "10080"))

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
    """Centralized security management class."""
    
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
        """Create JWT access token with scopes."""
        try:
            to_encode = data.copy()
            if expires_delta:
                expire = datetime.now(timezone.utc) + expires_delta
            else:
                expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            
            to_encode.update({
                "exp": expire,
                "type": "access",
                "scopes": scopes or ["user"],
                "iat": datetime.now(timezone.utc),
                "jti": secrets.token_urlsafe(16)
            })
            
            encoded_jwt = jwt.encode(to_encode, AUTH_SECRET_KEY, algorithm=ALGORITHM)
            logger.debug(f"Created access token for user: {data.get('sub')}")
            return encoded_jwt, expire
        except Exception as e:
            logger.error(f"Error creating access token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error creating access token"
            )

    @staticmethod
    def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> tuple[str, datetime]:
        """Create JWT refresh token."""
        try:
            to_encode = data.copy()
            if expires_delta:
                expire = datetime.now(timezone.utc) + expires_delta
            else:
                expire = datetime.now(timezone.utc) + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
            
            to_encode.update({
                "exp": expire,
                "type": "refresh",
                "iat": datetime.now(timezone.utc),
                "jti": secrets.token_urlsafe(16)
            })
            
            encoded_jwt = jwt.encode(to_encode, AUTH_SECRET_KEY, algorithm=ALGORITHM)
            logger.debug(f"Created refresh token for user: {data.get('sub')}")
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
        """Verify JWT token and extract data."""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(token, AUTH_SECRET_KEY, algorithms=[ALGORITHM])
            logger.debug(f"Token payload: {payload}")
            
            if payload.get("type") != expected_type:
                logger.warning(f"Invalid token type: expected {expected_type}, got {payload.get('type')}")
                raise credentials_exception
            
            username: str = payload.get("sub")
            if username is None:
                logger.warning("Token missing 'sub' claim")
                raise credentials_exception
            
            token_scopes = payload.get("scopes", [])
            
            if required_scopes:
                if not any(scope in token_scopes for scope in required_scopes):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
            
            return TokenData(username=username, scopes=token_scopes)
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {str(e)}")
            raise credentials_exception
        except Exception as e:
            logger.error(f"Unexpected error verifying token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error verifying token"
            )
        
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
import jwt
from app.database.models import User
from app.database.database import get_async_session

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/token")

async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    session: AsyncSession = Depends(get_async_session)
):
    """Enhanced user authentication with better error handling."""
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
        
        # Decode token with better error handling
        try:
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            username: str = payload.get("sub")
            if username is None:
                logger.warning("Token missing 'sub' claim")
                raise credentials_exception
                
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database
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
    """Ensure user is admin."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Admin privileges required"
        )
    return current_user

# async def get_current_user(
#     security_scopes: SecurityScopes,
#     token: str = Depends(oauth2_scheme),
#     session: AsyncSession = Depends(get_async_session)
# ) -> User:
#     """Get current authenticated user with scope validation."""
#     if security_scopes.scopes:
#         authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
#     else:
#         authenticate_value = "Bearer"
    
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": authenticate_value},
#     )
    
#     try:
#         token_data = SecurityManager.verify_token(
#             token, 
#             expected_type="access", 
#             required_scopes=security_scopes.scopes
#         )
        
#         result = await session.execute(select(User).where(User.username == token_data.username))
#         user = result.scalars().first()
#         if user is None:
#             logger.warning(f"User not found for username: {token_data.username}")
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="User not found"
#             )
        
#         logger.info(f"User authenticated successfully: {user.username}")
#         return user
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error getting current user: {str(e)}")
#         raise credentials_exception

# async def get_current_active_user(
#     current_user: User = Security(get_current_user, scopes=["user"])
# ) -> User:
#     """Get current active user."""
#     if not current_user.is_active:
#         logger.warning(f"Inactive user attempted access: {current_user.username}")
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="User account is inactive"
#         )
#     return current_user

async def get_current_admin_user(
    current_user: User = Security(get_current_user, scopes=["admin"])
) -> User:
    """Get current admin user."""
    if not current_user.is_active:
        logger.warning(f"Inactive admin attempted access: {current_user.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    is_admin = getattr(current_user, 'is_admin', False)
    if not is_admin:
        logger.warning(f"Non-admin user attempted admin access: {current_user.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required"
        )
    
    return current_user

async def authenticate_user(username: str, password: str, session: AsyncSession) -> Optional[User]:
    """Authenticate user credentials."""
    try:
        result = await session.execute(select(User).where(User.username == username))
        user = result.scalars().first()
        if not user:
            logger.warning(f"User not found: {username}")
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
            detail=f"Error authenticating user: {str(e)}"
        )

async def create_user_tokens(user: User, session: AsyncSession = Depends(get_async_session)) -> Dict[str, Any]:
    """Create access and refresh tokens for user."""
    try:
        scopes = ["user"]
        is_admin = getattr(user, 'is_admin', False)
        if is_admin:
            scopes.append("admin")
        
        access_token, access_expires = SecurityManager.create_access_token(
            data={"sub": user.username},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            scopes=scopes
        )
        
        refresh_token, refresh_expires = SecurityManager.create_refresh_token(
            data={"sub": user.username},
            expires_delta=timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
        )
        
        user.refresh_token_hash = SecurityManager.get_password_hash(refresh_token)
        user.refresh_token_expires_at = refresh_expires  # Keep timezone-aware datetime
        session.add(user)
        await session.commit()
        
        logger.info(f"Tokens created for user: {user.username}")
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "scopes": scopes
        }
        
    except Exception as e:
        logger.error(f"Error creating tokens for user {user.username}: {str(e)}")
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating authentication tokens: {str(e)}"
        )

async def verify_refresh_token(refresh_token: str, session: AsyncSession) -> Optional[User]:
    """Verify refresh token and return associated user."""
    try:
        token_data = SecurityManager.verify_token(refresh_token, expected_type="refresh")
        
        result = await session.execute(select(User).where(User.username == token_data.username))
        user = result.scalars().first()
        if not user:
            logger.warning(f"User not found for refresh token: {token_data.username}")
            return None
        
        if not user.refresh_token_hash:
            logger.warning(f"No refresh token hash for user: {user.username}")
            return None
        
        if not SecurityManager.verify_password(refresh_token, user.refresh_token_hash):
            logger.warning(f"Invalid refresh token for user: {user.username}")
            return None
        
        if user.refresh_token_expires_at and user.refresh_token_expires_at < datetime.now(timezone.utc):
            logger.warning(f"Expired refresh token for user: {user.username}")
            return None
        
        logger.info(f"Refresh token verified for user: {user.username}")
        return user
        
    except Exception as e:
        logger.error(f"Error verifying refresh token: {str(e)}")
        return None

class RateLimiter:
    """Simple in-memory rate limiter for API endpoints."""
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed based on rate limit."""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=window)
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove requests older than the window
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > window_start]
        
        if len(self.requests[key]) >= limit:
            logger.warning(f"Rate limit exceeded for {key}: {len(self.requests[key])} requests in {window}s")
            return False
        
        self.requests[key].append(now)
        return True

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
            logger.debug(f"Rate limit reset for {key}")

# import os
# import secrets
# from datetime import datetime, timedelta, timezone
# from typing import Optional, List, Dict, Any

# from jose import JWTError, jwt
# from passlib.context import CryptContext
# from fastapi import Depends, HTTPException, status, Security
# from fastapi.security import OAuth2PasswordBearer, SecurityScopes
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlmodel import select
# import logging

# from app.database.database import get_async_session
# from app.database.models import User

# from dotenv import load_dotenv
# load_dotenv()

# logger = logging.getLogger(__name__)

# # Environment variables
# AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY")
# if not AUTH_SECRET_KEY:
#     logger.error("AUTH_SECRET_KEY not set in environment variables")
#     raise ValueError("AUTH_SECRET_KEY must be set in environment variables")

# ALGORITHM = os.getenv("ALGORITHM", "HS256")
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
# REFRESH_TOKEN_EXPIRE_MINUTES = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", "10080"))  # 7 days

# # Password hashing configuration
# pwd_context = CryptContext(
#     schemes=["bcrypt"], 
#     deprecated="auto",
#     bcrypt__rounds=12
# )

# # OAuth2 configuration
# oauth2_scheme = OAuth2PasswordBearer(
#     tokenUrl="/api/v1/token",
#     scopes={
#         "user": "Basic user access",
#         "admin": "Administrative access"
#     }
# )

# class TokenData:
#     def __init__(self, username: str, scopes: List[str] = None):
#         self.username = username
#         self.scopes = scopes or []

# class SecurityManager:
#     """Centralized security management class."""
    
#     @staticmethod
#     def get_password_hash(password: str) -> str:
#         """Generate password hash with enhanced security."""
#         try:
#             return pwd_context.hash(password)
#         except Exception as e:
#             logger.error(f"Error hashing password: {str(e)}")
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Error processing password"
#             )

#     @staticmethod
#     def verify_password(plain_password: str, hashed_password: str) -> bool:
#         """Verify password against hash."""
#         try:
#             return pwd_context.verify(plain_password, hashed_password)
#         except Exception as e:
#             logger.error(f"Error verifying password: {str(e)}")
#             return False

#     @staticmethod
#     def create_access_token(
#         data: dict, 
#         expires_delta: Optional[timedelta] = None,
#         scopes: List[str] = None
#     ) -> tuple[str, datetime]:
#         """Create JWT access token with scopes."""
#         try:
#             to_encode = data.copy()
#             if expires_delta:
#                 expire = datetime.now(timezone.utc) + expires_delta
#             else:
#                 expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            
#             to_encode.update({
#                 "exp": expire,
#                 "type": "access",
#                 "scopes": scopes or ["user"],
#                 "iat": datetime.now(timezone.utc),
#                 "jti": secrets.token_urlsafe(16)
#             })
            
#             encoded_jwt = jwt.encode(to_encode, AUTH_SECRET_KEY, algorithm=ALGORITHM)
#             logger.debug(f"Created access token for user: {data.get('sub')}")
#             return encoded_jwt, expire
#         except Exception as e:
#             logger.error(f"Error creating access token: {str(e)}")
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Error creating access token"
#             )

#     @staticmethod
#     def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> tuple[str, datetime]:
#         """Create JWT refresh token."""
#         try:
#             to_encode = data.copy()
#             if expires_delta:
#                 expire = datetime.now(timezone.utc) + expires_delta
#             else:
#                 expire = datetime.now(timezone.utc) + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
            
#             to_encode.update({
#                 "exp": expire,
#                 "type": "refresh",
#                 "iat": datetime.now(timezone.utc),
#                 "jti": secrets.token_urlsafe(16)
#             })
            
#             encoded_jwt = jwt.encode(to_encode, AUTH_SECRET_KEY, algorithm=ALGORITHM)
#             logger.debug(f"Created refresh token for user: {data.get('sub')}")
#             return encoded_jwt, expire
#         except Exception as e:
#             logger.error(f"Error creating refresh token: {str(e)}")
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Error creating refresh token"
#             )

#     @staticmethod
#     def verify_token(
#         token: str, 
#         expected_type: str = "access",
#         required_scopes: List[str] = None
#     ) -> TokenData:
#         """Verify JWT token and extract data."""
#         credentials_exception = HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Could not validate credentials",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
        
#         try:
#             payload = jwt.decode(token, AUTH_SECRET_KEY, algorithms=[ALGORITHM])
#             logger.debug(f"Token payload: {payload}")
            
#             if payload.get("type") != expected_type:
#                 logger.warning(f"Invalid token type: expected {expected_type}, got {payload.get('type')}")
#                 raise credentials_exception
            
#             username: str = payload.get("sub")
#             if username is None:
#                 logger.warning("Token missing 'sub' claim")
#                 raise credentials_exception
            
#             token_scopes = payload.get("scopes", [])
            
#             if required_scopes:
#                 if not any(scope in token_scopes for scope in required_scopes):
#                     raise HTTPException(
#                         status_code=status.HTTP_403_FORBIDDEN,
#                         detail="Insufficient permissions"
#                     )
            
#             return TokenData(username=username, scopes=token_scopes)
            
#         except JWTError as e:
#             logger.warning(f"JWT verification failed: {str(e)}")
#             raise credentials_exception
#         except Exception as e:
#             logger.error(f"Unexpected error verifying token: {str(e)}")
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Error verifying token"
#             )

# # AUTHENTICATION DEPENDENCY FUNCTIONS

# async def get_current_user(
#     token: str = Depends(oauth2_scheme), 
#     session: AsyncSession = Depends(get_async_session)
# ):
#     """Enhanced user authentication with better error handling."""
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
    
#     try:
#         if not token:
#             logger.warning("No token provided")
#             raise credentials_exception
            
#         secret_key = os.getenv("AUTH_SECRET_KEY")
#         if not secret_key:
#             logger.error("AUTH_SECRET_KEY environment variable not set")
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Server configuration error"
#             )
        
#         # Decode token with better error handling
#         try:
#             payload = jwt.decode(token, secret_key, algorithms=["HS256"])
#             username: str = payload.get("sub")
#             if username is None:
#                 logger.warning("Token missing 'sub' claim")
#                 raise credentials_exception
                
#         except jwt.ExpiredSignatureError:
#             logger.warning("Token has expired")
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Token has expired",
#                 headers={"WWW-Authenticate": "Bearer"},
#             )
#         except jwt.InvalidTokenError as e:
#             logger.warning(f"Invalid token: {str(e)}")
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Invalid token",
#                 headers={"WWW-Authenticate": "Bearer"},
#             )
        
#         # Get user from database
#         result = await session.execute(select(User).where(User.username == username))
#         user = result.scalars().first()
        
#         if user is None:
#             logger.warning(f"User not found: {username}")
#             raise credentials_exception
            
#         if not user.is_active:
#             logger.warning(f"Inactive user attempted access: {username}")
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Inactive user"
#             )
        
#         logger.debug(f"User authenticated: {username}")
#         return user
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Authentication error: {str(e)}")
#         raise credentials_exception

# async def get_current_user_optional(
#     token: Optional[str] = Depends(oauth2_scheme),
#     session: AsyncSession = Depends(get_async_session)
# ) -> Optional[User]:
#     """Optional authentication - returns None if not authenticated."""
#     if not token:
#         return None
        
#     try:
#         return await get_current_user(token, session)
#     except HTTPException:
#         return None

# async def get_current_admin_user(current_user: User = Depends(get_current_user)):
#     """Ensure user is admin."""
#     if not current_user.is_admin:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN, 
#             detail="Admin privileges required"
#         )
#     return current_user

# async def authenticate_user(username: str, password: str, session: AsyncSession) -> Optional[User]:
#     """Authenticate user credentials."""
#     try:
#         result = await session.execute(select(User).where(User.username == username))
#         user = result.scalars().first()
#         if not user:
#             logger.warning(f"User not found: {username}")
#             return None
        
#         if not SecurityManager.verify_password(password, user.hashed_password):
#             logger.warning(f"Invalid password for user: {username}")
#             return None
        
#         logger.info(f"User authenticated successfully: {username}")
#         return user
#     except Exception as e:
#         logger.error(f"Error authenticating user {username}: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error authenticating user: {str(e)}"
#         )

# async def create_user_tokens(user: User, session: AsyncSession) -> Dict[str, Any]:
#     """Create access and refresh tokens for user."""
#     try:
#         scopes = ["user"]
#         is_admin = getattr(user, 'is_admin', False)
#         if is_admin:
#             scopes.append("admin")
        
#         access_token, access_expires = SecurityManager.create_access_token(
#             data={"sub": user.username, "user_id": user.id},
#             expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
#             scopes=scopes
#         )
        
#         refresh_token, refresh_expires = SecurityManager.create_refresh_token(
#             data={"sub": user.username, "user_id": user.id},
#             expires_delta=timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
#         )
        
#         # Update user's last login
#         user.last_login = datetime.now(timezone.utc)
        
#         logger.info(f"Tokens created for user: {user.username}")
#         return {
#             "access_token": access_token,
#             "refresh_token": refresh_token,
#             "token_type": "bearer",
#             "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
#             "scopes": scopes
#         }
        
#     except Exception as e:
#         logger.error(f"Error creating tokens for user {user.username}: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error creating authentication tokens: {str(e)}"
#         )

# async def verify_refresh_token(refresh_token: str, session: AsyncSession) -> Optional[User]:
#     """Verify refresh token and return associated user."""
#     try:
#         token_data = SecurityManager.verify_token(refresh_token, expected_type="refresh")
        
#         result = await session.execute(select(User).where(User.username == token_data.username))
#         user = result.scalars().first()
#         if not user:
#             logger.warning(f"User not found for refresh token: {token_data.username}")
#             return None
        
#         if not user.is_active:
#             logger.warning(f"Inactive user refresh attempt: {user.username}")
#             return None
        
#         logger.info(f"Refresh token verified for user: {user.username}")
#         return user
        
#     except Exception as e:
#         logger.error(f"Error verifying refresh token: {str(e)}")
#         return None

# class RateLimiter:
#     """Simple in-memory rate limiter for API endpoints."""
#     def __init__(self):
#         self.requests = {}
    
#     def is_allowed(self, key: str, limit: int, window: int) -> bool:
#         """Check if request is allowed based on rate limit."""
#         now = datetime.now(timezone.utc)
#         window_start = now - timedelta(seconds=window)
        
#         if key not in self.requests:
#             self.requests[key] = []
        
#         # Remove requests older than the window
#         self.requests[key] = [req_time for req_time in self.requests[key] if req_time > window_start]
        
#         if len(self.requests[key]) >= limit:
#             logger.warning(f"Rate limit exceeded for {key}: {len(self.requests[key])} requests in {window}s")
#             return False
        
#         self.requests[key].append(now)
#         return True

#     def get_remaining_requests(self, key: str, limit: int, window: int) -> int:
#         """Get remaining requests allowed in the current window."""
#         now = datetime.now(timezone.utc)
#         window_start = now - timedelta(seconds=window)
        
#         if key not in self.requests:
#             return limit
        
#         self.requests[key] = [req_time for req_time in self.requests[key] if req_time > window_start]
#         return limit - len(self.requests[key])

#     def reset(self, key: str):
#         """Reset rate limit for a specific key."""
#         if key in self.requests:
#             del self.requests[key]
#             logger.debug(f"Rate limit reset for {key}")