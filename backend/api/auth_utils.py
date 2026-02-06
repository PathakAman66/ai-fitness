import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from passlib.context import CryptContext
import jwt

DATA_DIR = Path(__file__).parent.parent / "data"
USERS_FILE = DATA_DIR / "users.json"

# Secret key for JWT (in production, this should be an env variable)
SECRET_KEY = "super-secret-key-change-me-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Ensure data directory exists
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True)

class UserManager:
    """Secure user manager with hashing and JWT"""
    
    @staticmethod
    def _load_users():
        if not USERS_FILE.exists():
            return {}
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _save_users(users):
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)

    @staticmethod
    def verify_password(plain_password, hashed_password):
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password):
        return pwd_context.hash(password)

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @classmethod
    def create_user(cls, username, email, password, profile_data=None):
        users = cls._load_users()
        
        # Check if username or email already exists
        if username in users:
            return False, "Username already exists"
        
        for user in users.values():
            if user.get("email") == email:
                return False, "Email already registered"
            
        # Hash the password
        hashed_password = cls.get_password_hash(password)
        
        users[username] = {
            "username": username,
            "email": email,
            "password": hashed_password,
            "profile": profile_data or {},
            "created_at": datetime.now().isoformat()
        }
        
        cls._save_users(users)
        return True, "User created successfully"

    @classmethod
    def authenticate(cls, username, password):
        users = cls._load_users()
        
        if username not in users:
            return False, None
            
        user = users[username]
        if not cls.verify_password(password, user["password"]):
            return False, None
            
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = cls.create_access_token(
            data={"sub": username, "email": user["email"]},
            expires_delta=access_token_expires
        )
        
        # Return user info plus token
        user_response = user.copy()
        del user_response["password"] # Don't return the hash
        user_response["access_token"] = access_token
        
        return True, user_response

