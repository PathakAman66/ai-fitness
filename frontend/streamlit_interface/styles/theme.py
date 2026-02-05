"""
Theme configuration for the AI Fitness Trainer Streamlit application.

This module defines the professional color scheme, typography, and spacing
constants used throughout the application to ensure visual consistency.
"""

# Color Scheme - Professional and Modern
COLORS = {
    # Primary colors - Neon Blue
    "primary": "#2563eb",  # Blue - main brand color
    "primary_dark": "#1d4ed8",  # Darker blue for hover states
    "primary_light": "#3b82f6",  # Lighter blue for accents
    "primary_glow": "0 0 20px rgba(37, 99, 235, 0.5)",
    
    # Secondary colors - Neon Green
    "secondary": "#10b981",  # Green - success/positive feedback
    "secondary_dark": "#059669",
    "secondary_light": "#34d399",
    "secondary_glow": "0 0 20px rgba(16, 185, 129, 0.5)",
    
    # Accent colors - Neon Purple/Orange
    "accent": "#8b5cf6",  # Purple - highlights
    "accent_dark": "#7c3aed",
    "accent_light": "#a78bfa",
    "accent_orange": "#f97316",
    "accent_glow": "0 0 20px rgba(139, 92, 246, 0.5)",
    
    # Semantic colors
    "success": "#10b981",  # Green
    "warning": "#f59e0b",  # Amber
    "error": "#ef4444",  # Red
    "info": "#3b82f6",  # Blue
    
    # Neutral colors - Dark Theme
    "background": "#0f172a",  # Deep Navy/Black
    "background_gradient": "linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)",
    "background_secondary": "rgba(30, 41, 59, 0.5)",  # Semi-transparent dark
    "background_tertiary": "rgba(51, 65, 85, 0.5)",  # Lighter semi-transparent
    
    # Text colors - High Contrast
    "text_primary": "#f8fafc",  # Off-white
    "text_secondary": "#cbd5e1",  # Light gray
    "text_tertiary": "#94a3b8",  # Medium gray
    "text_inverse": "#020617",  # Dark text for light backgrounds
    
    # Border colors
    "border": "rgba(148, 163, 184, 0.1)",  # Subtle border
    "border_dark": "rgba(148, 163, 184, 0.2)",  # Stronger border
    "border_glow": "rgba(59, 130, 246, 0.5)",
    
    # Card and surface colors
    "card_background": "rgba(30, 41, 59, 0.4)",  # Glassmorphism base
    "card_hover": "rgba(51, 65, 85, 0.6)",
    "surface": "#1e293b",
}

# Typography - Using Inter as primary font with fallbacks
TYPOGRAPHY = {
    # Font families
    "font_family_primary": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif",
    "font_family_secondary": "'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "font_family_mono": "'Fira Code', 'Courier New', monospace",
    
    # Font sizes
    "font_size_xs": "0.75rem",    # 12px
    "font_size_sm": "0.875rem",   # 14px
    "font_size_base": "1rem",     # 16px
    "font_size_lg": "1.125rem",   # 18px
    "font_size_xl": "1.25rem",    # 20px
    "font_size_2xl": "1.5rem",    # 24px
    "font_size_3xl": "1.875rem",  # 30px
    "font_size_4xl": "2.25rem",   # 36px
    "font_size_5xl": "3rem",      # 48px
    
    # Font weights
    "font_weight_light": "300",
    "font_weight_normal": "400",
    "font_weight_medium": "500",
    "font_weight_semibold": "600",
    "font_weight_bold": "700",
    
    # Line heights
    "line_height_tight": "1.25",
    "line_height_normal": "1.5",
    "line_height_relaxed": "1.75",
    
    # Letter spacing
    "letter_spacing_tight": "-0.025em",
    "letter_spacing_normal": "0",
    "letter_spacing_wide": "0.025em",
}

# Spacing - Consistent spacing scale
SPACING = {
    "xs": "0.25rem",   # 4px
    "sm": "0.5rem",    # 8px
    "md": "1rem",      # 16px
    "lg": "1.5rem",    # 24px
    "xl": "2rem",      # 32px
    "2xl": "2.5rem",   # 40px
    "3xl": "3rem",     # 48px
    "4xl": "4rem",     # 64px
    "5xl": "5rem",     # 80px
}

# Border Radius - Rounded corners
BORDER_RADIUS = {
    "none": "0",
    "sm": "0.375rem",   # 6px
    "md": "0.75rem",    # 12px
    "lg": "1rem",       # 16px
    "xl": "1.5rem",     # 24px
    "2xl": "2rem",      # 32px
    "full": "9999px",   # Fully rounded
}

# Shadows - Elevation system
SHADOWS = {
    "none": "none",
    "sm": "0 1px 2px 0 rgba(0, 0, 0, 0.3)",
    "md": "0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)",
    "lg": "0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2)",
    "xl": "0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2)",
    "2xl": "0 25px 50px -12px rgba(0, 0, 0, 0.5)",
    "glow": "0 0 15px rgba(37, 99, 235, 0.3)",
}

# Transitions - Smooth animations
TRANSITIONS = {
    "fast": "150ms ease-in-out",
    "normal": "250ms ease-in-out",
    "slow": "350ms ease-in-out",
}

# Breakpoints - Responsive design
BREAKPOINTS = {
    "mobile": "640px",
    "tablet": "768px",
    "desktop": "1024px",
    "wide": "1280px",
}

# Z-index layers
Z_INDEX = {
    "base": "0",
    "dropdown": "1000",
    "sticky": "1100",
    "overlay": "1200",
    "modal": "1300",
    "popover": "1400",
    "tooltip": "1500",
}

# Google Fonts URL
GOOGLE_FONTS_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto:wght@300;400;500;700&display=swap"
