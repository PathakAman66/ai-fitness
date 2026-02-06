import streamlit as st
from styles.theme import COLORS

@st.dialog("Welcome to AI Fitness")
def auth_dialog(initial_mode="login"):
    """
    Renders the authentication dialog with Login and Sign Up tabs.
    
    Args:
        initial_mode (str): 'login' or 'signup' to determine default view (if controllable)
                            or just for context.
                            Note: st.tabs doesn't support programmatic indexing yet, 
                            so we'll stick to standard tabs or use a radio if strict default is needed.
                            For now, tabs are more "UI friendly" as requested.
    """
    
    # Custom styling for the dialog content
    st.markdown(
        f"""
        <style>
        .stTextInput input {{
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: white !important;
        }}
        .stTextInput input:focus {{
            border-color: {COLORS['primary']} !important;
            box-shadow: 0 0 0 1px {COLORS['primary']} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Tabs for separating Login and Sign Up
    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    # Initialize API Client
    from services.api_client import get_api_client
    api_client = get_api_client()

    with tab_login:
        st.header("Welcome Back")
        st.write("Enter your credentials to access your workout history.")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)
            if submitted:
                if username and password:
                    # Call Backend API
                    with st.spinner("Authenticating..."):
                        response = api_client.login(username, password)
                    
                    if response.get("success"):
                        st.success(f"Welcome back, {username}!")
                        st.session_state.user = response.get("user")
                        st.session_state.is_authenticated = True
                        st.session_state.show_auth_modal = False
                        st.query_params.clear()
                        st.rerun()
                    else:
                        st.error(response.get("message", "Login failed"))
                else:
                    st.error("Please enter both username and password.")

    with tab_signup:
        st.header("Create Account")
        st.write("Join us to start your AI-powered fitness journey.")
        
        with st.form("signup_form"):
            new_username = st.text_input("Username", placeholder="Choose a username")
            email = st.text_input("Email", placeholder="name@example.com")
            
            # Fitness Details
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=10, max_value=100, value=25)
                height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            with col2:
                weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            fitness_goal = st.selectbox("Fitness Goal", [
                "Weight Loss", 
                "Muscle Gain", 
                "Endurance", 
                "Flexibility", 
                "General Fitness"
            ])
            
            new_password = st.text_input("Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            
            submitted = st.form_submit_button("Sign Up", type="primary", use_container_width=True)
            if submitted:
                if new_username and email and new_password:
                    if new_password == confirm_password:
                        # Call Backend API
                        user_data = {
                            "username": new_username,
                            "email": email,
                            "password": new_password,
                            "age": age,
                            "height": height,
                            "weight": weight,
                            "gender": gender,
                            "goal": fitness_goal
                        }
                        
                        with st.spinner("Creating account..."):
                            response = api_client.signup(user_data)
                        
                        if response.get("success"):
                            st.success(f"Account created for {new_username}! Logging you in...")
                            st.session_state.user = response.get("user")
                            st.session_state.is_authenticated = True
                            st.session_state.show_auth_modal = False
                            st.query_params.clear()
                            st.rerun()
                        else:
                            st.error(response.get("message", "Signup failed"))
                    else:
                        st.error("Passwords do not match.")
                else:
                    st.error("Please fill in all required fields.")

    # Note: st.dialog automatically includes a close button (X) in the top right.
