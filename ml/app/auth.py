from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel

from app.config import (
    AUTH_ENABLED,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    JWT_USERS,
)

ROLE_ORDER = {"viewer": 1, "operator": 2, "admin": 3}
bearer_scheme = HTTPBearer(auto_error=False)


class AuthUser(BaseModel):
    username: str
    role: str


def _load_users() -> Dict[str, Dict[str, str]]:
    try:
        users = json.loads(JWT_USERS)
    except json.JSONDecodeError as exc:
        raise RuntimeError("JWT_USERS must be valid JSON") from exc

    if not isinstance(users, dict):
        raise RuntimeError("JWT_USERS must be a JSON object")

    return users


def authenticate_user(username: str, password: str) -> Optional[AuthUser]:
    users = _load_users()
    user = users.get(username)

    if not user:
        return None

    expected_password = str(user.get("password", ""))
    if not secrets.compare_digest(password, expected_password):
        return None

    role = str(user.get("role", "viewer"))
    if role not in ROLE_ORDER:
        role = "viewer"

    return AuthUser(username=username, role=role)


def create_access_token(user: AuthUser) -> str:
    expires_at = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user.username,
        "role": user.role,
        "exp": expires_at,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> AuthUser:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    username = payload.get("sub")
    role = payload.get("role", "viewer")

    if not username or role not in ROLE_ORDER:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthUser(username=username, role=role)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> AuthUser:
    if not AUTH_ENABLED:
        return AuthUser(username="demo", role="admin")

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decode_access_token(credentials.credentials)


def require_role(required_role: str):
    def dependency(user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if ROLE_ORDER[user.role] < ROLE_ORDER[required_role]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' or higher required",
            )
        return user

    return dependency


def login_with_form(form_data: OAuth2PasswordRequestForm = Depends()) -> AuthUser:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_user_from_websocket_token(token: Optional[str]) -> AuthUser:
    if not AUTH_ENABLED:
        return AuthUser(username="demo", role="admin")

    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")

    return decode_access_token(token)
