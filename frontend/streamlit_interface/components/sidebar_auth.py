"""
Sidebar authentication component
Displays login/signup buttons at the top of the sidebar
"""

import streamlit as st
from styles.theme import COLORS

def render_sidebar_auth():
    """
    Renders authentication buttons at the top of the sidebar.
    Shows login/signup buttons if not authenticated, or user profile if authenticated.
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
        st.query_params.clear()
        st.rerun()
    
    # Custom CSS for sidebar auth buttons
    st.markdown(
        f"""
        <style>
        .sidebar-auth-container {{
            padding: 0.5rem 0 1rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 1rem;
        }}
        
        .sidebar-auth-btn {{
            width: 100%;
            padding: 0.5rem 1rem;
            margin: 0.25rem 0;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(59, 130, 246, 0.1);
            color: #3b82f6;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
            text-decoration: none;
            display: block;
        }}
        
        .sidebar-auth-btn:hover {{
            background: rgba(59, 130, 246, 0.2);
            border-color: rgba(59, 130, 246, 0.4);
            transform: translateY(-1px);
        }}
        
        .sidebar-auth-btn.signup {{
            background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['accent']});
            color: white;
            border: none;
        }}
        
        .sidebar-auth-btn.signup:hover {{
            filter: brightness(1.1);
        }}
        
        .sidebar-user-profile {{
            padding: 0.75rem;
            background: rgba(16, 185, 129, 0.1);
            border-radius: 8px;
            border: 1px solid rgba(16, 185, 129, 0.2);
            margin-bottom: 1rem;
        }}
        
        .sidebar-user-name {{
            font-weight: 600;
            color: #10b981;
            margin-bottom: 0.25rem;
        }}
        
        .sidebar-user-email {{
            font-size: 0.85rem;
            color: rgba(255, 255, 255, 0.6);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Check if user is authenticated
    if st.session_state.get("is_authenticated", False):
        user = st.session_state.get("user", {})
        username = user.get("username", "User")
        email = user.get("email", "")
        
        # Show user profile
        st.markdown(
            f"""
            <div class="sidebar-user-profile">
                <div class="sidebar-user-name">👤 {username}</div>
                <div class="sidebar-user-email">{email}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True, type="secondary", key="sidebar_logout_btn"):
            st.session_state.is_authenticated = False
            st.session_state.user = {}
            st.session_state.access_token = None
            st.rerun()
    else:
        # Show login/signup buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔑 Login", use_container_width=True, key="sidebar_login_btn"):
                st.session_state.show_auth_modal = True
                st.session_state.auth_modal_tab = "login"
                st.rerun()
        
        with col2:
            if st.button("✨ Sign Up", use_container_width=True, type="primary", key="sidebar_signup_btn"):
                st.session_state.show_auth_modal = True
                st.session_state.auth_modal_tab = "signup"
                st.rerun()
        
        # AI Chatbot Help Button
        st.markdown(
            f"""
            <style>
            .ai-chatbot-btn {{
                width: 100%;
                padding: 0.75rem 1rem;
                margin-top: 0.75rem;
                border-radius: 12px;
                background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.15));
                border: 1px solid rgba(139, 92, 246, 0.3);
                color: #a78bfa;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                text-align: center;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }}
            
            .ai-chatbot-btn:hover {{
                background: linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(59, 130, 246, 0.25));
                border-color: rgba(139, 92, 246, 0.5);
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
            }}
            
            .ai-chatbot-icon {{
                font-size: 1.2rem;
            }}
            
            .ai-chatbot-text {{
                display: flex;
                flex-direction: column;
                align-items: flex-start;
            }}
            
            .ai-chatbot-title {{
                font-size: 0.95rem;
                font-weight: 600;
            }}
            
            .ai-chatbot-subtitle {{
                font-size: 0.75rem;
                color: rgba(167, 139, 250, 0.7);
                font-weight: 400;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("🤖 AI Chatbot - 24/7 Help", use_container_width=True, key="sidebar_chatbot_btn", type="secondary"):
            # Open chatbot dialog
            st.session_state.show_chatbot = True
            st.rerun()
        
        st.markdown("---")


def render_chatbot_if_needed():
    """
    Renders the chatbot dialog if the show_chatbot flag is set.
    This should be called ONCE per page, similar to auth modal.
    """
    if st.session_state.get("show_chatbot", False):
        from components.chatbot_dialog import chatbot_dialog
        chatbot_dialog()


def render_auth_modal_if_needed():
    """
    Renders the auth modal dialog if the show_auth_modal flag is set.
    This should be called ONCE per page, typically in the main app or page file.
    """
    if st.session_state.get("show_auth_modal", False):
        from components.auth_modal import auth_dialog
        auth_dialog(initial_mode=st.session_state.get("auth_modal_tab", "login"))
