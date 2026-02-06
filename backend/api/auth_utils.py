import json
import os
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
USERS_FILE = DATA_DIR / "users.json"

# Ensure data directory exists
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True)

class UserManager:
    """Simple file-based user manager"""
    
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

    @classmethod
    def create_user(cls, username, email, password, profile_data=None):
        users = cls._load_users()
        
        if username in users:
            return False, "Username already exists"
            
        # In a real app, hash the password!
        # For this demo, we store plaintext (NOT SECURE)
        users[username] = {
            "username": username,
            "email": email,
            "password": password, # TODO: Hash this
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
        if user["password"] == password:
            return True, user
            
        return False, None
