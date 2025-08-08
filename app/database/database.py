"""
Database configuration and management for the Financial Distress Prediction Service.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, AsyncGenerator
from functools import wraps

from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.exc import OperationalError
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration from environment."""
    
    # Connection settings
    ASYNC_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:Kenya%403140@localhost:5432/distress_predictions_db")
    
    # Pool settings  
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    
    # Performance settings
    DB_ECHO = os.getenv("DB_ECHO", "false").lower() == "true"
    DB_ECHO_POOL = os.getenv("DB_ECHO_POOL", "false").lower() == "true"
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

config = DatabaseConfig()

class DatabaseManager:
    """Enterprise database manager with async support and monitoring."""
    
    def __init__(self):
        self.async_database_url = self._get_async_database_url()
        self.async_engine = self._create_async_engine()
        self.async_session_maker = async_sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Monitoring metrics
        self._connection_count = 0
        self._query_count = 0
        self._start_time = time.time()
    
    def _get_async_database_url(self) -> str:
        """Get and validate async database URL."""
        database_url = config.ASYNC_DATABASE_URL
        if not database_url:
            raise ValueError("DATABASE_URL must be set in environment variables")
        
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
            logger.info("Converted postgres:// to postgresql+asyncpg://")
        
        return database_url
    
    def _create_async_engine(self):
        """Create async database engine with optimized settings."""
        try:
            engine = create_async_engine(
                self.async_database_url,
                pool_size=config.DB_POOL_SIZE,
                max_overflow=config.DB_MAX_OVERFLOW,
                pool_timeout=config.DB_POOL_TIMEOUT,
                pool_recycle=config.DB_POOL_RECYCLE,
                pool_pre_ping=True,
                echo=config.DB_ECHO,
                echo_pool=config.DB_ECHO_POOL,
                connect_args={
                    "timeout": 10,
                    "command_timeout": 30,
                    "server_settings": {
                        "application_name": f"financial_distress_api_{config.ENVIRONMENT}",
                        "timezone": "UTC"
                    }
                },
                isolation_level="READ_COMMITTED"
            )
            logger.info("Async database engine created with correct asyncpg parameters")
            return engine
        except Exception as e:
            logger.error(f"Failed to create async engine: {e}")
            raise

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with proper context management."""
        start_time = time.time()
        async with self.async_session_maker() as session:
            try:
                self._connection_count += 1
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Async session error: {e}")
                raise
            finally:
                duration = time.time() - start_time
                if duration > 1.0:
                    logger.warning(f"Slow async session operation: {duration:.2f}s")

    async def create_tables(self):
        """Create all database tables asynchronously."""
        try:
            logger.info("Creating database tables...")
            from app.database.models import (
                User, PredictionLog, InfluencingFactorDB, AuditLog,
                SystemMetrics, ModelPerformance, DataQualityCheck
            )
            async with self.async_engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            raise

    async def drop_tables(self):
        """Drop all tables (development only)."""
        if config.ENVIRONMENT == "production":
            raise RuntimeError("Cannot drop tables in production")
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.drop_all)
            logger.warning("All tables dropped")
        except Exception as e:
            logger.error(f"Table drop failed: {e}")
            raise

    async def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            async with self.async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_engine_info(self) -> Dict[str, Any]:
        """Get database engine metrics."""
        try:
            pool = self.async_engine.pool
            return {
                "database_url": self._mask_credentials(self.async_database_url),
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "total_connections": self._connection_count,
                "total_queries": self._query_count,
                "uptime_seconds": time.time() - self._start_time
            }
        except Exception as e:
            logger.error(f"Engine info failed: {e}")
            return {"error": str(e)}
    
    def _mask_credentials(self, url: str) -> str:
        """Mask database credentials for logging."""
        if "@" in url:
            return url.split("@")[-1]
        return "postgresql+asyncpg://***:***@localhost/***"

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            async with self.async_session_maker() as session:
                # Database size
                size_query = text("SELECT pg_size_pretty(pg_database_size(current_database())) as size")
                size_result = await session.execute(size_query)
                db_size = size_result.scalar()
                
                # Table statistics (FIXED: Use relname instead of tablename)
                tables_query = text("""
                    SELECT schemaname, relname AS tablename, n_live_tup AS records, n_dead_tup AS dead_records
                    FROM pg_stat_user_tables
                    ORDER BY n_live_tup DESC
                """)
                tables_result = await session.execute(tables_query)
                table_stats = [dict(row._mapping) for row in tables_result]
                
                # Connection statistics
                conn_query = text("""
                    SELECT COUNT(*) AS total, 
                           COUNT(*) FILTER (WHERE state = 'active') AS active,
                           COUNT(*) FILTER (WHERE state = 'idle') AS idle
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                """)
                conn_result = await session.execute(conn_query)
                conn_stats = dict(conn_result.first()._mapping)
                
                return {
                    "database_size": db_size,
                    "table_statistics": table_stats,
                    "connection_statistics": conn_stats,
                    "engine_info": self.get_engine_info()
                }
        except Exception as e:
            logger.warning(f"Database stats failed: {e}")
            return {"error": str(e), "engine_info": self.get_engine_info()}

# Global database manager
db_manager = DatabaseManager()

# FIXED: FastAPI dependency that properly uses the async context manager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for async sessions."""
    async with db_manager.get_async_session() as session:
        yield session

# Application startup
async def create_db_and_tables():
    """Initialize database - called at startup."""
    await db_manager.create_tables()

async def check_database_health() -> Dict[str, Any]:
    """Health check for monitoring."""
    try:
        is_healthy = await db_manager.test_connection()
        engine_info = db_manager.get_engine_info()
        stats = await db_manager.get_database_stats() if is_healthy else {"warning": "Limited stats - connection failed"}
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "connection_test": "passed" if is_healthy else "failed",
            "engine_info": engine_info,
            "database_stats": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@asynccontextmanager
async def async_database_transaction():
    """Context manager for async transactions."""
    async with db_manager.get_async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async transaction failed: {e}")
            raise

def async_transactional(func):
    """
    Fixed decorator for async transactions that doesn't interfere with FastAPI dependency injection.
    Only adds transaction handling, doesn't modify the session dependency.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Check if session is already provided via dependency injection
        if 'session' in kwargs:
            session = kwargs['session']
            try:
                result = await func(*args, **kwargs)
                # Only commit if the session is still active and not already committed
                if session.in_transaction():
                    await session.commit()
                return result
            except Exception as e:
                if session.in_transaction():
                    await session.rollback()
                raise
        else:
            # If no session provided, create one with transaction
            async with async_database_transaction() as session:
                kwargs['session'] = session
                return await func(*args, **kwargs)
    
    return wrapper