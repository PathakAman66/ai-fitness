import streamlit as st
from styles.theme import COLORS, TYPOGRAPHY, SPACING, BORDER_RADIUS, SHADOWS

def render_auth_header():
    """
    Renders fixed-position Login and Sign Up buttons in the top-right corner.
    Features premium glassmorphism styling and hover effects.
    
    Checks query params to trigger the Authentication Dialog.
    """
    # Initialize session state for auth modal
    if "show_auth_modal" not in st.session_state:
        st.session_state.show_auth_modal = False
        st.session_state.auth_modal_tab = "login"

    # Check for auth trigger from query params
    if "auth_action" in st.query_params:
        action = st.query_params["auth_action"]
        st.session_state.show_auth_modal = True
        st.session_state.auth_modal_tab = action
        # Clear the param immediately
        st.query_params.clear()
        st.rerun()

    # Note: The auth dialog is now rendered by render_auth_modal_if_needed() 
    # to avoid duplicate dialog errors

    # If authenticated, show user profile instead of login buttons
    if st.session_state.get("is_authenticated", False):
        username = st.session_state.user.get("username", "User")
        st.markdown(
            f"""
            <style>
            .auth-header-container {{
                position: fixed;
                top: 20px;
                right: 250px;
                z-index: 10000;
                display: flex;
                gap: 12px;
                align-items: center;
            }}
            .user-profile {{
                display: flex;
                align-items: center;
                gap: 10px;
                background: rgba(15, 23, 42, 0.6);
                padding: 8px 20px;
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
            }}
            .user-greeting {{
                color: #94a3af;
                font-family: {TYPOGRAPHY['font_family_primary']};
                font-size: 0.85rem;
                font-weight: 400;
                margin-right: 4px;
            }}
            .user-name {{
                color: #e2e8f0;
                font-family: {TYPOGRAPHY['font_family_primary']};
                font-size: 0.95rem;
                font-weight: 600;
            }}
            </style>
            <div class="auth-header-container">
                <div class="user-profile">
                    <span class="user-greeting">Hi,</span>
                    <span class="user-name">{username}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Render login/signup buttons if not authenticated
    st.markdown(
        f"""
        <style>
        .auth-header-container {{
            position: fixed;
            top: 20px;
            right: 250px;
            z-index: 10000;
            display: flex;
            gap: 12px;
            align-items: center;
        }}

        .auth-btn {{
            font-family: {TYPOGRAPHY['font_family_primary']};
            font-size: 0.9rem;
            font-weight: 600;
            padding: 8px 20px;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-decoration: none !important;
            display: flex;
            align-items: center;
            gap: 8px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .auth-btn-login {{
            background: rgba(15, 23, 42, 0.6);
            color: #cbd5e1;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}

        .auth-btn-login:hover {{
            background: rgba(30, 41, 59, 0.9);
            color: #f8fafc;
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 10px 20px -3px rgba(0, 0, 0, 0.3);
            border-color: rgba(99, 102, 241, 0.5);
        }}

        .auth-btn-signup {{
            background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['accent']});
            color: white;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}

        .auth-btn-signup:hover {{
            filter: brightness(1.2);
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 12px 30px rgba(37, 99, 235, 0.6);
        }}
        
        /* Mobile adjustment */
        @media (max-width: 640px) {{
            .auth-header-container {{
                top: 10px;
                left: 10px;
                gap: 8px;
            }}
            .auth-btn {{
                padding: 6px 14px;
                font-size: 0.8rem;
            }}
        }}
        </style>

        <div class="auth-header-container">
            <a href="?auth_action=login" target="_self" class="auth-btn auth-btn-login">
                <span class="material-icons" style="font-size: 18px;">login</span>
                Log In
            </a>
            <a href="?auth_action=signup" target="_self" class="auth-btn auth-btn-signup">
                <span class="material-icons" style="font-size: 18px;">person_add</span>
                Sign Up
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
