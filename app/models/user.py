from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    email: str
    full_name: str = Field(default='')
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    refresh_token_hash: Optional[str] = None
    refresh_token_expires_at: Optional[datetime] = None