"""
Custom CSS styling for the AI Fitness Trainer Streamlit application.

This module provides CSS injection functionality to apply professional
styling including hover states, responsive design, and custom components.
"""

import streamlit as st
from .theme import (
    COLORS, TYPOGRAPHY, SPACING, BORDER_RADIUS, 
    SHADOWS, TRANSITIONS, BREAKPOINTS, GOOGLE_FONTS_URL
)


def get_custom_css() -> str:
    """
    Generate custom CSS for the application.
    
    Returns:
        str: Complete CSS string to be injected into Streamlit pages
    """
    css = f"""
    <style>
    /* Import Google Fonts */
    @import url('{GOOGLE_FONTS_URL}');
    /* Import FontAwesome */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    /* Global Styles */
    * {{
        box-sizing: border-box;
    }}
    
    html, body, [class*="css"] {{
        font-family: {TYPOGRAPHY['font_family_primary']};
        color: {COLORS['text_primary']};
    }}
    
    /* Main Container with Gradient Background */
    .main {{
        background: {COLORS['background']};
        background-image: {COLORS['background_gradient']};
        background-attachment: fixed;
        padding: {SPACING['lg']};
        min-height: 100vh;
    }}
    
    /* Streamlit Header - Transparent */
    header {{
        background-color: transparent !important;
    }}
    
    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Typography Styles */
    h1 {{
        font-size: clamp({TYPOGRAPHY['font_size_3xl']}, 5vw, {TYPOGRAPHY['font_size_5xl']});
        font-weight: {TYPOGRAPHY['font_weight_bold']};
        color: {COLORS['text_primary']};
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
        margin-bottom: {SPACING['xl']};
        letter-spacing: {TYPOGRAPHY['letter_spacing_tight']};
    }}
    
    h2 {{
        font-size: clamp({TYPOGRAPHY['font_size_xl']}, 3vw, {TYPOGRAPHY['font_size_3xl']});
        font-weight: {TYPOGRAPHY['font_weight_semibold']};
        color: {COLORS['text_primary']};
        margin-bottom: {SPACING['lg']};
    }}
    
    h3 {{
        font-size: clamp({TYPOGRAPHY['font_size_lg']}, 2.5vw, {TYPOGRAPHY['font_size_2xl']});
        font-weight: {TYPOGRAPHY['font_weight_semibold']};
        color: {COLORS['text_primary']};
        margin-bottom: {SPACING['md']};
    }}
    
    p {{
        font-size: clamp({TYPOGRAPHY['font_size_sm']}, 1.5vw, {TYPOGRAPHY['font_size_base']});
        font-weight: {TYPOGRAPHY['font_weight_normal']};
        color: {COLORS['text_secondary']};
        line-height: {TYPOGRAPHY['line_height_relaxed']};
        margin-bottom: {SPACING['md']};
    }}
    
    /* Button Styles - Neon Glow */
    .stButton > button {{
        background: linear-gradient(135deg, #c2410c, #dc2626);
        color: {COLORS['text_inverse']};
        border: none;
        border-radius: {BORDER_RADIUS['full']};
        padding: {SPACING['md']} {SPACING['xl']};
        font-size: {TYPOGRAPHY['font_size_base']};
        font-weight: {TYPOGRAPHY['font_weight_semibold']};
        cursor: pointer;
        transition: all {TRANSITIONS['normal']};
        box-shadow: 0 0 15px rgba(194, 65, 12, 0.3);
        text-transform: uppercase;
        letter-spacing: {TYPOGRAPHY['letter_spacing_wide']};
        width: 100%;
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 0 25px rgba(194, 65, 12, 0.5);
        filter: brightness(1.15);
    }}
    
    /* Glassmorphism Card Styles */
    .glass-card {{
        background: {COLORS['card_background']};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {COLORS['border']};
        border-radius: {BORDER_RADIUS['2xl']};
        padding: {SPACING['xl']};
        box-shadow: {SHADOWS['lg']};
        transition: all {TRANSITIONS['normal']};
    }}
    
    .glass-card:hover {{
        background: {COLORS['card_hover']};
        border-color: {COLORS['border_glow']};
        box-shadow: {COLORS['primary_glow']}, {SHADOWS['xl']};
        transform: translateY(-5px);
    }}
    
    /* Metric Card Styles */
    [data-testid="stMetricValue"] {{
        color: {COLORS['primary_light']};
        text-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {COLORS['text_tertiary']};
    }}
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['background']} !important;
        border-right: 1px solid {COLORS['border']};
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .animate-fade-in {{
        animation: fadeIn 0.8s ease-out forwards;
    }}
    
    .animate-slide-up {{
        animation: slideUp 0.6s ease-out forwards;
    }}
    
    /* Material Icon Styles */
    .material-icons {{
        font-family: 'Material Icons';
        font-weight: normal;
        font-style: normal;
        display: inline-block;
        line-height: 1;
        text-transform: none;
        letter-spacing: normal;
        word-wrap: normal;
        white-space: nowrap;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }}
    
    .icon-glow-primary {{
        text-shadow: {COLORS['primary_glow']};
    }}
    
    .icon-glow-secondary {{
        text-shadow: {COLORS['secondary_glow']};
    }}
    
    .icon-glow-accent {{
        text-shadow: {COLORS['accent_glow']};
    }}
    
    /* Responsive Adjustments */
    @media (max-width: {BREAKPOINTS['tablet']}) {{
        .main {{
            padding: {SPACING['md']};
        }}
        .glass-card {{
            padding: {SPACING['lg']};
        }}
    }}
    /* History Page Styles */
    .history-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .history-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.03), transparent);
        transform: translateX(-100%);
        transition: 0.5s;
    }

    .history-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(139, 92, 246, 0.3);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    .history-card:hover::before {
        transform: translateX(100%);
    }

    .history-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 1rem;
    }

    .history-title {
        color: #f3f4f6;
        font-size: 1.1rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .history-date {
        color: #9ca3af;
        font-size: 0.85rem;
    }

    .history-stats {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
    }

    .stat-item {
        background: rgba(0, 0, 0, 0.2);
        padding: 0.8rem;
        border-radius: 12px;
        text-align: center;
    }

    .stat-label {
        color: #9ca3af;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }

    .stat-value {
        color: #e5e7eb;
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }

    .status-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .status-completed {
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .status-active {
        background: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    .status-incomplete {
        background: rgba(245, 158, 11, 0.1);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    /* Footer Styles */
    .footer {{
        background-color: #2d2d2d;
        color: #ffffff;
        padding: 3rem 1rem;
        margin-top: 5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .footer-content {{
        max-width: 1200px;
        margin: 0 auto;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
    }}
    
    .footer-section h3 {{
        color: #8b5cf6;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        margin-top: 0;
    }}
    
    .footer-section p {{
        color: #9ca3af;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }}
    
    .footer-social-links {{
        display: flex;
        gap: 1.2rem;
        margin-top: 1rem;
    }}
    
    .footer-social-links a {{
        color: #e5e7eb;
        text-decoration: none;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.1);
    }}
    
    .footer-social-links a:hover {{
        background: #8b5cf6;
        color: white;
        transform: translateY(-3px);
    }}
    
    .footer-bottom {{
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #6b7280;
        font-size: 0.85rem;
    }}
    
    .footer-badges {{
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 0.8rem;
    }}
    
    .footer-badge {{
        background: rgba(139, 92, 246, 0.1);
        color: #a78bfa;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
    }}
    </style>
    """
    return css


def inject_custom_css() -> None:
    """
    Inject custom CSS into the current Streamlit page.
    
    This function should be called at the beginning of each page
    to apply consistent styling across the application.
    """
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    # Also inject responsive utilities
    try:
        from utils.responsive import inject_responsive_utilities
        inject_responsive_utilities()
    except ImportError:
        # Fallback for when running from different contexts
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        from utils.responsive import inject_responsive_utilities
        inject_responsive_utilities()



def apply_page_config(
    page_title: str = "AI Fitness Trainer",
    page_icon: str = "💪",
    layout: str = "wide",
    initial_sidebar_state: str = "expanded"
) -> None:
    """
    Apply Streamlit page configuration with custom settings.
    
    Args:
        page_title: Title displayed in browser tab
        page_icon: Icon displayed in browser tab
        layout: Page layout ("centered" or "wide")
        initial_sidebar_state: Initial sidebar state ("expanded" or "collapsed")
    """
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state
    )
