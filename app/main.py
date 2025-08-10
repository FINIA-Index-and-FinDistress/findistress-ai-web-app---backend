# import os
# import time
# import uuid
# import logging
# from contextlib import asynccontextmanager
# from typing import Dict, Any

# import uvicorn
# from fastapi import FastAPI, Request, HTTPException, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.trustedhost import TrustedHostMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.exceptions import RequestValidationError
# from pydantic import ValidationError
# from datetime import datetime, timezone

# from app.api.endpoints import router as api_router
# from app.database.database import create_db_and_tables, check_database_health
# from app.models.ml_models import load_models, get_model_status

# from dotenv import load_dotenv
# load_dotenv()

# def setup_logging():
#     """Configure application logging for production monitoring."""
#     log_level = os.getenv("LOG_LEVEL", "INFO").upper()
#     os.makedirs("logs", exist_ok=True)
    
#     logging.basicConfig(
#         level=getattr(logging, log_level),
#         format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s',
#         handlers=[
#             logging.FileHandler('logs/app.log'),
#             logging.StreamHandler()
#         ]
#     )

# setup_logging()
# logger = logging.getLogger(__name__)

# class RequestIDFilter(logging.Filter):
#     def filter(self, record):
#         record.request_id = getattr(record, 'request_id', 'startup')
#         return True

# logger.addFilter(RequestIDFilter())

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Application lifespan manager with comprehensive startup validation."""
#     startup_start_time = time.time()
    
#     try:
#         logger.info("Starting Financial Distress Prediction API...")
#         logger.info("Initializing database connection...")
#         await create_db_and_tables()
        
#         db_health = await check_database_health()
#         if db_health.get("status") != "healthy":
#             raise Exception(f"Database health check failed: {db_health}")
        
#         logger.info("Database initialized successfully")
        
#         logger.info("Loading financial prediction models...")
#         await load_models()
        
#         model_status = await get_model_status()
#         if model_status.get("status") not in ["healthy", "degraded"]:
#             raise Exception(f"Model loading failed: {model_status}")
        
#         logger.info("ML models loaded successfully")
        
#         startup_time = time.time() - startup_start_time
#         logger.info(f"Application startup completed in {startup_time:.2f} seconds")
        
#         yield
        
#     except Exception as e:
#         logger.error(f"Startup failed: {e}")
#         raise
#     finally:
#         logger.info("Financial Distress Prediction API shutting down...")
#         logger.info("Shutdown completed")

# app = FastAPI(
#     title="Financial Distress Prediction API",
#     description="AI-powered financial health assessment for businesses",
#     version="2.0.0",
#     lifespan=lifespan,
#     docs_url="/docs",
#     redoc_url="/redoc", 
#     openapi_url="/openapi.json",
#     contact={
#         "name": "Financial Distress Prediction Team",
#         "email": os.getenv("CONTACT_EMAIL", "support@financialhealth.ai")
#     },
#     license_info={
#         "name": "Proprietary License",
#         "url": "https://example.com/license"
#     }
# )

# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=[
#         "localhost", 
#         "127.0.0.1", 
#         "*.ngrok.io",
#         os.getenv("ALLOWED_HOST", "*")
#     ]
# )

# FRONTEND_URLS = os.getenv(
#     "FRONTEND_URLS", 
#     "https://findistress-web-app-frontend.netlify.app"
# ).split(",")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=FRONTEND_URLS,
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
#     allow_headers=["*"],
#     expose_headers=["X-Process-Time", "X-Request-ID"]
# )

# @app.middleware("http")
# async def add_request_tracking(request: Request, call_next):
#     """Add request ID, timing, and comprehensive logging."""
#     start_time = time.time()
#     request_id = str(uuid.uuid4())[:8]
    
#     request.state.request_id = request_id
    
#     origin = request.headers.get("origin")
#     logger.info(
#         f"{request.method} {request.url.path} from origin: {origin}",
#         extra={"request_id": request_id}
#     )
    
#     try:
#         response = await call_next(request)
        
#         process_time = time.time() - start_time
#         response.headers["X-Process-Time"] = f"{process_time:.4f}"
#         response.headers["X-Request-ID"] = request_id
        
#         logger.info(
#             f"{request.method} {request.url.path} → {response.status_code} ({process_time:.3f}s)",
#             extra={"request_id": request_id}
#         )
        
#         if process_time > 5.0:
#             logger.warning(
#                 f"Slow request: {request.method} {request.url.path} took {process_time:.2f}s",
#                 extra={"request_id": request_id}
#             )
        
#         return response
        
#     except Exception as e:
#         process_time = time.time() - start_time
#         logger.error(
#             f"Request failed: {request.method} {request.url.path} after {process_time:.3f}s - {str(e)}",
#             extra={"request_id": request_id}
#         )
#         raise
    

# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request: Request, exc: RequestValidationError):
#     """Handle input validation errors with user-friendly messages."""
#     request_id = getattr(request.state, "request_id", "unknown")
    
#     friendly_errors = []
#     for error in exc.errors():
#         field = " → ".join(str(loc) for loc in error["loc"])
#         msg = error["msg"]
        
#         if "field required" in msg:
#             friendly_errors.append(f"'{field}' is required")
#         elif "ensure this value" in msg:
#             friendly_errors.append(f"'{field}' value is invalid: {msg}")
#         else:
#             friendly_errors.append(f"'{field}': {msg}")
    
#     logger.warning(
#         f"Invalid input data: {friendly_errors}",
#         extra={"request_id": request_id}
#     )
    
#     return JSONResponse(
#         status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#         content={
#             "error": "Invalid Input Data",
#             "message": "Please check your input and try again. Some required information is missing or incorrect.",
#             "details": friendly_errors,
#             "request_id": request_id,
#             "help": "Ensure all required fields are filled out correctly."
#         },
#         headers={
#             "Access-Control-Allow-Origin": ", ".join(FRONTEND_URLS),
#             "Access-Control-Allow-Credentials": "true",
#             "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,PATCH",
#             "Access-Control-Allow-Headers": "*",
#             "Access-Control-Expose-Headers": "X-Process-Time,X-Request-ID"
#         }
#     )

# @app.exception_handler(HTTPException)
# async def http_exception_handler(request: Request, exc: HTTPException):
#     """Handle HTTP exceptions with consistent formatting and CORS headers."""
#     request_id = getattr(request.state, "request_id", "unknown")
    
#     logger.warning(
#         f"HTTP {exc.status_code}: {exc.detail}",
#         extra={"request_id": request_id}
#     )
    
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={
#             "error": "Request Failed",
#             "message": exc.detail,
#             "status_code": exc.status_code,
#             "request_id": request_id
#         },
#         headers={
#             "Access-Control-Allow-Origin": ", ".join(FRONTEND_URLS),
#             "Access-Control-Allow-Credentials": "true",
#             "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,PATCH",
#             "Access-Control-Allow-Headers": "*",
#             "Access-Control-Expose-Headers": "X-Process-Time,X-Request-ID"
#         }
#     )

# @app.exception_handler(ValidationError)
# async def pydantic_exception_handler(request: Request, exc: ValidationError):
#     """Handle Pydantic validation errors."""
#     request_id = getattr(request.state, "request_id", "unknown")
    
#     logger.warning(
#         f"Data validation failed: {exc}",
#         extra={"request_id": request_id}
#     )
    
#     return JSONResponse(
#         status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
#         content={
#             "error": "Data Validation Failed", 
#             "message": "The data you provided doesn't match the expected format.",
#             "details": exc.errors(),
#             "request_id": request_id
#         },
#         headers={
#             "Access-Control-Allow-Origin": ", ".join(FRONTEND_URLS),
#             "Access-Control-Allow-Credentials": "true",
#             "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,PATCH",
#             "Access-Control-Allow-Headers": "*",
#             "Access-Control-Expose-Headers": "X-Process-Time,X-Request-ID"
#         }
#     )

# @app.exception_handler(Exception)
# async def general_exception_handler(request: Request, exc: Exception):
#     """Handle unexpected exceptions with user-friendly messages."""
#     request_id = getattr(request.state, "request_id", "unknown")
    
#     logger.error(
#         f"Unexpected error: {str(exc)}",
#         extra={"request_id": request_id},
#         exc_info=True
#     )
    
#     return JSONResponse(
#         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#         content={
#             "error": "System Error",
#             "message": "An unexpected error occurred. Our team has been notified and is working to fix this issue.",
#             "request_id": request_id,
#             "support": "If this continues, please contact support with this request ID."
#         },
#         headers={
#             "Access-Control-Allow-Origin": ", ".join(FRONTEND_URLS),
#             "Access-Control-Allow-Credentials": "true",
#             "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,PATCH",
#             "Access-Control-Allow-Headers": "*",
#             "Access-Control-Expose-Headers": "X-Process-Time,X-Request-ID"
#         }
#     )

# app.include_router(api_router)

# @app.get("/", tags=["Welcome"])
# async def read_root() -> Dict[str, Any]:
#     """Welcome endpoint with system information."""
#     return {
#         "message": "Welcome to the Financial Distress Prediction API!",
#         "description": "AI-powered financial health assessment for businesses",
#         "version": "2.0.0",
#         "environment": os.getenv("ENVIRONMENT", "development"),
#         "documentation": {
#             "interactive_docs": "/docs",
#             "alternative_docs": "/redoc", 
#             "api_specification": "/openapi.json"
#         },
#         "quick_start": {
#             "step_1": "View API documentation at /docs",
#             "step_2": "Use /api/v1/predictions/predict to analyze company data",
#             "step_3": "Check /api/v1/analytics for insights and trends"
#         },
#         "features": [
#             "AI-powered financial distress prediction",
#             "Regional analysis (African & Global markets)",
#             "Real-time analytics and insights",
#             "User-friendly results for business users",
#             "Enterprise-grade security"
#         ]
#     }

# @app.get("/health", tags=["System Health"])
# async def health_check() -> Dict[str, Any]:
#     """Comprehensive system health check for monitoring."""
#     try:
#         health_start = time.time()
        
#         logger.info("Running health check...")
#         db_health = await check_database_health()
        
#         model_health = await get_model_status()
        
#         components_healthy = (
#             db_health.get("status") == "healthy" and
#             model_health.get("status") in ["healthy", "degraded"]
#         )
        
#         overall_status = "healthy" if components_healthy else "unhealthy"
#         health_time = time.time() - health_start
        
#         health_data = {
#             "status": overall_status,
#             "timestamp": datetime.now(timezone.utc).isoformat(),
#             "service": "financial-distress-prediction-api",
#             "version": "2.0.0",
#             "environment": os.getenv("ENVIRONMENT", "development"),
#             "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
#             "health_check_duration_ms": round(health_time * 1000, 2),
#             "components": {
#                 "database": {
#                     "status": db_health.get("status", "unknown"),
#                     "details": db_health
#                 },
#                 "ml_models": {
#                     "status": model_health.get("status", "unknown"), 
#                     "models_loaded": model_health.get("models_loaded", 0),
#                     "available_regions": model_health.get("available_regions", [])
#                 }
#             },
#             "endpoints": {
#                 "predictions": "/api/v1/predictions",
#                 "analytics": "/api/v1/analytics",
#                 "documentation": "/docs"
#             }
#         }
        
#         if overall_status == "healthy":
#             logger.info(f"Health check passed ({health_time:.3f}s)")
#         else:
#             logger.warning(f"Health check issues detected ({health_time:.3f}s)")
        
#         return health_data
        
#     except Exception as e:
#         logger.error(f"Health check failed: {e}")
#         return {
#             "status": "unhealthy",
#             "timestamp": datetime.now(timezone.utc).isoformat(),
#             "service": "financial-distress-prediction-api",
#             "error": str(e),
#             "message": "System health check failed"
#         }

# @app.on_event("startup")
# async def store_startup_time():
#     """Store application startup time."""
#     app.state.start_time = time.time()

# if __name__ == "__main__":
#     host = os.getenv("HOST", "0.0.0.0")
#     port = int(os.getenv("PORT", 8000))
#     reload = os.getenv("ENVIRONMENT", "development") == "development"
#     log_level = os.getenv("LOG_LEVEL", "info").lower()
    
#     uvicorn.run(
#         "main:app",
#         host=host,
#         port=port,
#         reload=reload,
#         log_level=log_level,
#         access_log=True,
#         use_colors=True,
#         workers=1
#     )

import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from datetime import datetime, timezone

from app.api.endpoints import router as api_router
from app.database.database import create_db_and_tables, check_database_health
from app.models.ml_models import load_models, get_model_status

from dotenv import load_dotenv
load_dotenv()

def setup_logging():
    """Configure application logging for production monitoring."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

class RequestIDFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'startup')
        return True

logger.addFilter(RequestIDFilter())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with comprehensive startup validation."""
    startup_start_time = time.time()
    
    try:
        logger.info("Starting Financial Distress Prediction API...")
        logger.info("Initializing database connection...")
        await create_db_and_tables()
        
        db_health = await check_database_health()
        if db_health.get("status") != "healthy":
            raise Exception(f"Database health check failed: {db_health}")
        
        logger.info("Database initialized successfully")
        
        logger.info("Loading financial prediction models...")
        await load_models()
        
        model_status = await get_model_status()
        if model_status.get("status") not in ["healthy", "degraded"]:
            raise Exception(f"Model loading failed: {model_status}")
        
        logger.info("ML models loaded successfully")
        
        startup_time = time.time() - startup_start_time
        logger.info(f"Application startup completed in {startup_time:.2f} seconds")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        logger.info("Financial Distress Prediction API shutting down...")
        logger.info("Shutdown completed")

app = FastAPI(
    title="Financial Distress Prediction API",
    description="AI-powered financial health assessment for businesses",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc", 
    openapi_url="/openapi.json",
    contact={
        "name": "Financial Distress Prediction Team",
        "email": os.getenv("CONTACT_EMAIL", "support@financialhealth.ai")
    },
    license_info={
        "name": "Proprietary License",
        "url": "https://example.com/license"
    }
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "localhost", 
        "127.0.0.1", 
        "*.ngrok.io",
        os.getenv("ALLOWED_HOST", "*")
    ]
)

FRONTEND_URLS = os.getenv(
    "FRONTEND_URLS", 
    "https://findistress-web-app-frontend.netlify.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)

@app.middleware("http")
async def add_request_tracking(request: Request, call_next):
    """Add request ID, timing, and comprehensive logging."""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    
    request.state.request_id = request_id
    
    origin = request.headers.get("origin")
    logger.info(
        f"{request.method} {request.url.path} from origin: {origin}",
        extra={"request_id": request_id}
    )
    
    try:
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Request-ID"] = request_id
        
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} ({process_time:.3f}s)",
            extra={"request_id": request_id}
        )
        
        if process_time > 5.0:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} took {process_time:.2f}s",
                extra={"request_id": request_id}
            )
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {request.method} {request.url.path} after {process_time:.3f}s - {str(e)}",
            extra={"request_id": request_id}
        )
        raise

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    if os.getenv("SECURITY_HEADERS_ENABLED", "false").lower() == "true":
        # Content Security Policy
        csp = os.getenv("CONTENT_SECURITY_POLICY", "default-src 'self'")
        response.headers["Content-Security-Policy"] = csp
        
        # HSTS
        hsts_max_age = os.getenv("HSTS_MAX_AGE", "31536000")
        response.headers["Strict-Transport-Security"] = f"max-age={hsts_max_age}; includeSubDomains"
        
        # X-Frame-Options
        x_frame_options = os.getenv("X_FRAME_OPTIONS", "DENY")
        response.headers["X-Frame-Options"] = x_frame_options
        
        # X-Content-Type-Options
        x_content_type = os.getenv("X_CONTENT_TYPE_OPTIONS", "nosniff")
        response.headers["X-Content-Type-Options"] = x_content_type
        
        # Additional security headers
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle input validation errors with user-friendly messages."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    friendly_errors = []
    for error in exc.errors():
        field = " → ".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        
        if "field required" in msg:
            friendly_errors.append(f"'{field}' is required")
        elif "ensure this value" in msg:
            friendly_errors.append(f"'{field}' value is invalid: {msg}")
        else:
            friendly_errors.append(f"'{field}': {msg}")
    
    logger.warning(
        f"Invalid input data: {friendly_errors}",
        extra={"request_id": request_id}
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Invalid Input Data",
            "message": "Please check your input and try again. Some required information is missing or incorrect.",
            "details": friendly_errors,
            "request_id": request_id,
            "help": "Ensure all required fields are filled out correctly."
        },
        headers={
            "Access-Control-Allow-Origin": ", ".join(FRONTEND_URLS),
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "X-Process-Time,X-Request-ID"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent formatting and CORS headers."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={"request_id": request_id}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "Request Failed",
            "message": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id
        },
        headers={
            "Access-Control-Allow-Origin": ", ".join(FRONTEND_URLS),
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "X-Process-Time,X-Request-ID"
        }
    )

@app.exception_handler(ValidationError)
async def pydantic_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        f"Data validation failed: {exc}",
        extra={"request_id": request_id}
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Data Validation Failed", 
            "message": "The data you provided doesn't match the expected format.",
            "details": exc.errors(),
            "request_id": request_id
        },
        headers={
            "Access-Control-Allow-Origin": ", ".join(FRONTEND_URLS),
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "X-Process-Time,X-Request-ID"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with user-friendly messages."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={"request_id": request_id},
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "System Error",
            "message": "An unexpected error occurred. Our team has been notified and is working to fix this issue.",
            "request_id": request_id,
            "support": "If this continues, please contact support with this request ID."
        },
        headers={
            "Access-Control-Allow-Origin": ", ".join(FRONTEND_URLS),
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "X-Process-Time,X-Request-ID"
        }
    )

app.include_router(api_router)

@app.get("/", tags=["Welcome"])
async def read_root() -> Dict[str, Any]:
    """Welcome endpoint with system information."""
    return {
        "message": "Welcome to the Financial Distress Prediction API!",
        "description": "AI-powered financial health assessment for businesses",
        "version": "2.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "documentation": {
            "interactive_docs": "/docs",
            "alternative_docs": "/redoc", 
            "api_specification": "/openapi.json"
        },
        "quick_start": {
            "step_1": "View API documentation at /docs",
            "step_2": "Use /api/v1/predictions/predict to analyze company data",
            "step_3": "Check /api/v1/analytics for insights and trends"
        },
        "features": [
            "AI-powered financial distress prediction",
            "Regional analysis (African & Global markets)",
            "Real-time analytics and insights",
            "User-friendly results for business users",
            "Enterprise-grade security"
        ]
    }

@app.get("/health", tags=["System Health"])
async def health_check() -> Dict[str, Any]:
    """Comprehensive system health check for monitoring."""
    try:
        health_start = time.time()
        
        logger.info("Running health check...")
        db_health = await check_database_health()
        
        model_health = await get_model_status()
        
        components_healthy = (
            db_health.get("status") == "healthy" and
            model_health.get("status") in ["healthy", "degraded"]
        )
        
        overall_status = "healthy" if components_healthy else "unhealthy"
        health_time = time.time() - health_start
        
        health_data = {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "financial-distress-prediction-api",
            "version": "2.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            "health_check_duration_ms": round(health_time * 1000, 2),
            "components": {
                "database": {
                    "status": db_health.get("status", "unknown"),
                    "details": db_health
                },
                "ml_models": {
                    "status": model_health.get("status", "unknown"), 
                    "models_loaded": model_health.get("models_loaded", 0),
                    "available_regions": model_health.get("available_regions", [])
                }
            },
            "endpoints": {
                "predictions": "/api/v1/predictions",
                "analytics": "/api/v1/analytics",
                "documentation": "/docs"
            }
        }
        
        if overall_status == "healthy":
            logger.info(f"Health check passed ({health_time:.3f}s)")
        else:
            logger.warning(f"Health check issues detected ({health_time:.3f}s)")
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "financial-distress-prediction-api",
            "error": str(e),
            "message": "System health check failed"
        }

@app.on_event("startup")
async def store_startup_time():
    """Store application startup time."""
    app.state.start_time = time.time()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True,
        use_colors=True,
        workers=1
    )