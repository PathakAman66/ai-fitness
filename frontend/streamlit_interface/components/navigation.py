"""
Navigation component for the AI Fitness Trainer Streamlit application.

This module provides a sidebar navigation menu with icons and labels for all
four pages (Home, Workout, History, Stats). It handles active page highlighting
and navigation state management using Streamlit's session_state.
"""

import streamlit as st
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils.icons import render_icon_with_text, get_icon_name
from styles.theme import COLORS, TYPOGRAPHY, SPACING, BORDER_RADIUS, TRANSITIONS


# Page configuration
PAGES = [
    {"key": "home", "label": "Home", "icon": "home"},
    {"key": "workout", "label": "Workout", "icon": "workout"},
    {"key": "history", "label": "History", "icon": "history"},
    {"key": "stats", "label": "Stats", "icon": "stats"},
]


class Navigation:
    """
    Navigation component for multi-page Streamlit application.
    
    Provides sidebar navigation with icon-based menu items, active page
    highlighting, and state management.
    """
    
    @staticmethod
    def initialize_state() -> None:
        """
        Initialize navigation state in session_state if not already present.
        
        Sets default values for navigation-related state variables.
        """
        if "current_page" not in st.session_state:
            st.session_state.current_page = "home"
    
    @staticmethod
    def get_current_page() -> str:
        """
        Get the currently active page identifier.
        
        Returns:
            Current page key (e.g., 'home', 'workout', 'history', 'stats')
        """
        Navigation.initialize_state()
        return st.session_state.current_page
    
    @staticmethod
    def set_current_page(page_key: str) -> None:
        """
        Set the currently active page.
        
        Args:
            page_key: Page identifier to set as active
        """
        Navigation.initialize_state()
        st.session_state.current_page = page_key
    
    @staticmethod
    def render_sidebar_nav() -> None:
        """
        Render navigation menu in the sidebar with icons and labels.
        
        Displays all four pages with appropriate icons, highlights the active
        page, and handles navigation clicks.
        """
        Navigation.initialize_state()
        
        # Inject navigation styles
        Navigation._inject_nav_styles()
        
        with st.sidebar:
            # App title/logo section
            st.markdown(
                f"""
                <div style="
                    padding: {SPACING['lg']} {SPACING['md']};
                    margin-bottom: {SPACING['xl']};
                    text-align: center;
                    border-bottom: 2px solid var(--nav-border);
                ">
                    <h1 style="
                        font-family: {TYPOGRAPHY['font_family_primary']};
                        font-size: {TYPOGRAPHY['font_size_2xl']};
                        font-weight: {TYPOGRAPHY['font_weight_bold']};
                        color: {COLORS['primary']};
                        margin: 0;
                        line-height: {TYPOGRAPHY['line_height_tight']};
                    ">
                        AI Fitness Trainer
                    </h1>
                    <p style="
                        font-family: {TYPOGRAPHY['font_family_primary']};
                        font-size: {TYPOGRAPHY['font_size_sm']};
                        color: var(--nav-text-secondary);
                        margin: {SPACING['xs']} 0 0 0;
                    ">
                        Your Personal Workout Assistant
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Navigation menu
            st.markdown(
                f"""
                <div style="
                    padding: 0 {SPACING['sm']};
                ">
                    <p class="nav-section-label" style="
                        font-family: {TYPOGRAPHY['font_family_primary']};
                        font-size: {TYPOGRAPHY['font_size_xs']};
                        font-weight: {TYPOGRAPHY['font_weight_semibold']};
                        color: var(--nav-text-secondary);
                        text-transform: uppercase;
                        letter-spacing: {TYPOGRAPHY['letter_spacing_wide']};
                        margin-bottom: {SPACING['sm']};
                    ">
                        Navigation
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Render navigation items
            for page in PAGES:
                Navigation._render_nav_item(
                    page_key=page["key"],
                    label=page["label"],
                    icon_category_key=page["icon"]
                )
            
            # Render social links
            Navigation.render_social_links()
    
    @staticmethod
    def _render_nav_item(page_key: str, label: str, icon_category_key: str) -> None:
        """
        Render a single navigation menu item.
        
        Args:
            page_key: Unique identifier for the page
            label: Display label for the navigation item
            icon_category_key: Key to look up icon in navigation icons
        """
        current_page = Navigation.get_current_page()
        is_active = current_page == page_key
        
        # Get icon name
        icon_name = get_icon_name("navigation", icon_category_key)
        
        # Determine styling based on active state
        if is_active:
            icon_color = COLORS['text_inverse']
        else:
            icon_color = "var(--nav-text-secondary)"
        
        # Create unique key for button
        button_key = f"nav_button_{page_key}"
        
        # Render navigation item as a button-like element
        # Using columns to create clickable area
        col1, col2 = st.columns([0.15, 0.85])
        
        with col1:
            # Icon
            st.markdown(
                f"""
                <div class="nav-item-icon" style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: {SPACING['sm']};
                ">
                    <span class="material-icons" style="
                        font-size: 24px;
                        color: {icon_color};
                    ">{icon_name}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            # Label with button
            if st.button(
                label,
                key=button_key,
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                Navigation.set_current_page(page_key)
                st.rerun()
        
        # Add spacing between items
        st.markdown(f'<div style="margin-bottom: {SPACING["xs"]};"></div>', unsafe_allow_html=True)

    @staticmethod
    def render_social_links() -> None:
        """
        Render social media links in the sidebar.
        """
        st.markdown(
            f"""
            <div style="
                margin-top: {SPACING['xl']};
                padding-top: {SPACING['lg']};
                border-top: 1px solid var(--nav-border);
            ">
                <p style="
                    font-family: {TYPOGRAPHY['font_family_primary']};
                    font-size: {TYPOGRAPHY['font_size_xs']};
                    font-weight: {TYPOGRAPHY['font_weight_semibold']};
                    color: var(--nav-text-secondary);
                    text-transform: uppercase;
                    letter-spacing: {TYPOGRAPHY['letter_spacing_wide']};
                    margin-bottom: {SPACING['sm']};
                ">
                    Connect
                </p>
                <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                    <a href="https://github.com/Suvam18" target="_blank" style="text-decoration: none; color: var(--nav-text-secondary); transition: color 0.3s;">
                        <i class="fab fa-github" style="font-size: 20px;"></i>
                    </a>
                    <a href="https://www.linkedin.com/in/suvam-chatterjee-ba6b922b2?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank" style="text-decoration: none; color: var(--nav-text-secondary); transition: color 0.3s;">
                        <i class="fab fa-linkedin" style="font-size: 20px;"></i>
                    </a>
                    <a href="https://www.instagram.com/suv_am28/" target="_blank" style="text-decoration: none; color: var(--nav-text-secondary); transition: color 0.3s;">
                        <i class="fab fa-instagram" style="font-size: 20px;"></i>
                    </a>
                    <a href="https://www.facebook.com" target="_blank" style="text-decoration: none; color: var(--nav-text-secondary); transition: color 0.3s;">
                        <i class="fab fa-facebook" style="font-size: 20px;"></i>
                    </a>
                    <a href="https://twitter.com" target="_blank" style="text-decoration: none; color: var(--nav-text-secondary); transition: color 0.3s;">
                        <i class="fab fa-twitter" style="font-size: 20px;"></i>
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    @staticmethod
    def highlight_active_page(page_name: str) -> None:
        """
        Apply active styling to the specified page in navigation.
        
        This method updates the session state to mark the given page as active,
        which will be reflected in the navigation menu on the next render.
        
        Args:
            page_name: Page identifier to highlight
        """
        Navigation.set_current_page(page_name)
    
    @staticmethod
    def _inject_nav_styles() -> None:
        """
        Inject custom CSS styles for navigation components.
        
        Applies hover states, transitions, and responsive styling to
        navigation menu items with dark mode support.
        """
        st.markdown(
            f"""
            <style>
            /* CSS Variables for dark mode support */
            :root {{
                --nav-bg: {COLORS['background']};
                --nav-text: {COLORS['text_primary']};
                --nav-text-secondary: {COLORS['text_secondary']};
                --nav-border: {COLORS['border']};
                --nav-hover-bg: {COLORS['background_secondary']};
            }}
            
            /* Dark mode support via media query */
            @media (prefers-color-scheme: dark) {{
                :root {{
                    --nav-bg: #0f172a;
                    --nav-text: #f9fafb;
                    --nav-text-secondary: #9ca3af;
                    --nav-border: #374151;
                    --nav-hover-bg: #1e293b;
                }}
            }}
            
            /* Dark mode support via Streamlit's data-theme attribute */
            [data-testid="stAppViewContainer"][data-theme="dark"] {{
                --nav-bg: #0f172a;
                --nav-text: #f9fafb;
                --nav-text-secondary: #9ca3af;
                --nav-border: #374151;
                --nav-hover-bg: #1e293b;
            }}
            
            /* Navigation container styling */
            [data-testid="stSidebar"] {{
                background-color: var(--nav-bg) !important;
            }}
            
            /* Fix sidebar content background */
            [data-testid="stSidebar"] > div:first-child {{
                background-color: var(--nav-bg) !important;
            }}
            
            /* Navigation header text colors */
            [data-testid="stSidebar"] h1 {{
                color: {COLORS['primary']} !important;
            }}
            
            [data-testid="stSidebar"] p {{
                color: var(--nav-text-secondary) !important;
            }}
            
            /* Navigation section label */
            [data-testid="stSidebar"] .nav-section-label {{
                color: var(--nav-text-secondary) !important;
            }}
            
            /* Border color in dark mode */
            [data-testid="stSidebar"] > div > div {{
                border-color: var(--nav-border) !important;
            }}
            
            /* Navigation item hover effects */
            .nav-item-icon {{
                transition: transform {TRANSITIONS['fast']};
            }}
            
            .nav-item-icon:hover {{
                transform: scale(1.1);
            }}
            
            /* Icon colors in dark mode */
            [data-testid="stAppViewContainer"][data-theme="dark"] .nav-item-icon .material-icons {{
                color: var(--nav-text-secondary) !important;
            }}
            
            /* Button styling for navigation */
            [data-testid="stSidebar"] button {{
                font-family: {TYPOGRAPHY['font_family_primary']};
                font-size: {TYPOGRAPHY['font_size_base']};
                border-radius: {BORDER_RADIUS['md']};
                transition: all {TRANSITIONS['normal']};
                text-align: left;
                padding: {SPACING['sm']} {SPACING['md']};
            }}
            
            /* Secondary button styling */
            [data-testid="stSidebar"] button[kind="secondary"] {{
                background-color: transparent !important;
                color: var(--nav-text) !important;
                border: 1px solid transparent !important;
            }}
            
            /* Secondary button hover state */
            [data-testid="stSidebar"] button[kind="secondary"]:hover {{
                background-color: var(--nav-hover-bg) !important;
                border-color: var(--nav-border) !important;
            }}
            
            /* Primary button (active page) styling */
            [data-testid="stSidebar"] button[kind="primary"] {{
                background-color: {COLORS['primary']} !important;
                color: {COLORS['text_inverse']} !important;
                border: none !important;
            }}
            
            [data-testid="stSidebar"] button[kind="primary"]:hover {{
                background-color: {COLORS['primary_dark']} !important;
            }}
            
            /* Active page icon color */
            [data-testid="stSidebar"] button[kind="primary"] ~ div .material-icons {{
                color: {COLORS['text_inverse']} !important;
            }}
            
            /* Remove default Streamlit button focus outline */
            [data-testid="stSidebar"] button:focus {{
                box-shadow: 0 0 0 2px {COLORS['primary_light']} !important;
            }}
            
            /* Responsive navigation for mobile */
            @media (max-width: 640px) {{
                /* Compact sidebar title */
                [data-testid="stSidebar"] h1 {{
                    font-size: {TYPOGRAPHY['font_size_xl']};
                }}
                
                [data-testid="stSidebar"] p {{
                    font-size: {TYPOGRAPHY['font_size_xs']};
                }}
                
                /* Smaller navigation buttons */
                [data-testid="stSidebar"] button {{
                    font-size: {TYPOGRAPHY['font_size_sm']};
                    padding: {SPACING['xs']} {SPACING['sm']};
                }}
                
                /* Smaller icons */
                .nav-item-icon .material-icons {{
                    font-size: 20px !important;
                }}
                
                /* Reduce spacing */
                [data-testid="stSidebar"] > div {{
                    padding: {SPACING['sm']} {SPACING['xs']};
                }}
            }}
            
            /* Tablet responsive navigation */
            @media (max-width: 768px) {{
                /* Adjust sidebar width */
                [data-testid="stSidebar"] {{
                    min-width: 200px;
                }}
                
                /* Compact navigation items */
                [data-testid="stSidebar"] button {{
                    padding: {SPACING['sm']};
                }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
