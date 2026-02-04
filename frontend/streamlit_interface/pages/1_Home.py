"""
Home page for the AI Fitness Trainer Streamlit application.

This page serves as the main landing page with a hero section, modern dark theme,
featuring glassmorphism cards and neon accents.
"""

import streamlit as st
import sys
from pathlib import Path
import base64

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from styles.custom_css import inject_custom_css, apply_page_config
from utils.icons import inject_material_icons_cdn
from utils.state_manager import StateManager
from components.navigation import Navigation
from components.hero_carousel import render_hero_carousel
from components.footer import render_footer


def get_base64_image(image_path):
    """Convert image to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def render_media(media_path):
    """Render image or video based on file extension."""
    file_ext = Path(media_path).suffix.lower()
    
    if file_ext in ['.mp4', '.webm', '.ogg', '.mov']:
        # Render video
        base64_video = get_base64_image(media_path)
        return f"""
        <video autoplay loop muted playsinline style="width: 100%; height: 100%; object-fit: cover; border-radius: 12px; transform: scale(1);">
            <source src="data:video/{file_ext[1:]};base64,{base64_video}" type="video/{file_ext[1:]}">
            Your browser does not support the video tag.
        </video>
        """
    else:
        # Render image
        base64_img = get_base64_image(media_path)
        return f'<img src="data:image/png;base64,{base64_img}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 12px; transform: scale(1);">'


def main():
    """Main function to render the Home page."""
    # Apply page configuration
    apply_page_config(
        page_title="Home - AI Fitness Trainer",
        page_icon="💪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS and Material Icons
    inject_custom_css()
    inject_material_icons_cdn()

    # Initialize session state
    StateManager.initialize_all()

    # Set current page
    Navigation.set_current_page("home")

    # Render navigation
    Navigation.render_sidebar_nav()

    # Hero Carousel Section
    render_hero_carousel()

    # Features Section
    st.markdown(
        """
        <div class="animate-slide-up" style="animation-delay: 0.2s; text-align: center; margin-bottom: 3rem;">
            <div style="display: inline-block; background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 20px; padding: 0.5rem 1.5rem; margin-bottom: 1rem;">
                <span style="color: #a78bfa; font-size: 0.9rem; font-weight: 600;">✨ Powerful Features</span>
            </div>
            <h2 style='color: #ffffff; font-size: 3rem; font-weight: 800; margin-bottom: -10px; line-height: 1.2;'>Everything You Need for</h2>
            <h2 style='background: linear-gradient(to right, #a78bfa, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline-block; font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;'>Fitness Success</h2>
            <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 1rem;">AI-powered tools built with advanced computer vision and machine learning for smart fitness tracking</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Custom CSS for alternating feature layout
    st.markdown(
        """
        <style>
        .feature-row {
            display: flex;
            align-items: center;
            gap: 3rem;
            margin-bottom: 4rem;
            opacity: 0;
            transform: translateY(30px);
            animation: slideUpFade 0.6s ease-out forwards;
        }
        
        .feature-row:nth-child(1) {
            animation-delay: 0.2s;
        }
        
        .feature-row:nth-child(2) {
            animation-delay: 0.4s;
        }
        
        .feature-row:nth-child(3) {
            animation-delay: 0.6s;
        }
        
        .feature-row:nth-child(4) {
            animation-delay: 0.8s;
        }
        
        .feature-content {
            flex: 1;
            padding: 1rem;
            height: 400px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .feature-video {
            flex: 1;
            background: rgba(20, 25, 35, 0.8);
            border: 1px solid rgba(249, 115, 22, 0.3);
            border-radius: 16px;
            padding: 1rem;
            height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .feature-video:hover {
            border-color: rgba(249, 115, 22, 0.8);
            box-shadow: 0 0 40px rgba(249, 115, 22, 0.4), 0 10px 50px rgba(0, 0, 0, 0.5);
            transform: translateY(-5px);
        }
        
        .feature-video::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #f97316, #ff6b6b, #8b5cf6, #2563eb);
            border-radius: 16px;
            opacity: 0;
            z-index: -1;
            transition: opacity 0.4s ease;
        }
        
        .feature-video:hover::before {
            opacity: 0.15;
        }
        
        /* Image wrapper and styling */
        .feature-video .image-wrapper {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        
        .feature-video img,
        .feature-video video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 12px;
            transform: scale(1);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            opacity: 0;
            animation: fadeInImage 0.8s ease-out forwards;
        }
        
        @keyframes fadeInImage {
            from {
                opacity: 0;
                transform: scale(0.95) translateY(20px);
            }
            to {
                opacity: 1;
                transform: scale(1) translateY(0);
            }
        }
        
        /* Staggered animation delays for each feature */
        .feature-content:nth-child(1) ~ div .feature-video img,
        .feature-content:nth-child(1) ~ div .feature-video video {
            animation-delay: 0.3s;
        }
        
        div:has(> .feature-video) {
            animation-delay: 0s;
        }
        
        .feature-video:hover img,
        .feature-video:hover video {
            transform: scale(1.1);
        }
        
        .feature-icon-large {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: rgba(249, 115, 22, 0.1);
            border: 3px solid rgba(249, 115, 22, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
            transition: all 0.4s ease;
        }
        
        .feature-content:hover .feature-icon-large {
            transform: scale(1.1) rotate(5deg);
            box-shadow: 0 0 30px rgba(249, 115, 22, 0.4);
        }
        
        .feature-title {
            font-size: 2rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 1rem;
            background: linear-gradient(to right, #f97316, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        }
        
        .feature-description {
            font-size: 1.1rem;
            color: #94a3b8;
            line-height: 1.8;
            margin-bottom: 1.5rem;
        }

        /* Exercise Card Styling */
        .exercise-card-container {
            background: rgba(30, 41, 59, 0.7);
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.4s ease;
            height: 420px; /* Uniform Height */
            display: flex;
            flex-direction: column;
            margin-bottom: 1rem;
        }
        
        .exercise-card-container:hover {
            transform: translateY(-10px);
            border-color: rgba(249, 115, 22, 0.5);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }
        
        .exercise-image-wrapper {
            height: 180px;
            width: 100%;
            position: relative;
            overflow: hidden;
            flex-shrink: 0;
        }
        
        .exercise-image-wrapper img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.6s ease;
        }
        
        .exercise-card-container:hover .exercise-image-wrapper img {
            transform: scale(1.1);
        }
        
        .exercise-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(249, 115, 22, 0.9);
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            z-index: 2;
        }
        
        .exercise-card-content {
            padding: 1.5rem;
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        .exercise-card-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: white;
            margin-bottom: 0.5rem;
        }
        
        .exercise-card-desc {
            color: #94a3b8;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-bottom: auto; /* Pushes stats down */
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .exercise-stats {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            font-size: 0.8rem;
            color: #cbd5e1;
        }
        
        .stat-item {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        /* Custom Button Styling */
        div.stButton > button {
            background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        div.stButton > button:hover {
            opacity: 0.9;
            transform: scale(1.02);
            box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3);
        }
        
        .feature-list {
            list-style: none;
            padding: 0;
        }
        
        .feature-list li {
            color: #cbd5e1;
            margin-bottom: 0.75rem;
            padding-left: 1.5rem;
            position: relative;
        }
        
        .feature-list li::before {
            content: '✓';
            position: absolute;
            left: 0;
            color: #f97316;
            font-weight: bold;
        }
        
        @media (max-width: 768px) {
            .feature-row {
                flex-direction: column;
            }
            
            .feature-video {
                height: 300px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    features = [
        {
            "icon": "videocam",
            "title": "Real-Time Analysis",
            "description": "Get instant feedback on your form using advanced computer vision and pose detection technology.",
            "points": [
                "AI-powered pose detection with 33 body landmarks",
                "Real-time form correction and feedback",
                "Accurate rep counting with motion tracking",
                "Live performance metrics display"
            ],
            "image": "C:/Users/User/.gemini/antigravity/brain/9b58a6c5-afe4-4a04-8170-f53d9a1b246b/pose_detection_demo_1769528268875.png"
        },
        {
            "icon": "trending_up",
            "title": "Track Progress",
            "description": "Monitor your workout history and performance trends over time with detailed analytics.",
            "points": [
                "Comprehensive workout history tracking",
                "Visual progress charts and graphs",
                "Performance trend analysis",
                "Personal records and achievements"
            ],
            "image": "C:/Users/User/.gemini/antigravity/brain/9b58a6c5-afe4-4a04-8170-f53d9a1b246b/progress_tracking_dashboard_1769528437503.png"
        },
        {
            "icon": "emoji_events",
            "title": "Achieve Goals",
            "description": "Set targets, track achievements, and celebrate milestones as you improve.",
            "points": [
                "Customizable fitness goals",
                "Daily and weekly target tracking",
                "Achievement badges and rewards",
                "Milestone celebration notifications"
            ],
            "image": "C:/Users/User/.gemini/antigravity/brain/9b58a6c5-afe4-4a04-8170-f53d9a1b246b/achievement_goals_system_1769528563258.png"
        },
        {
            "icon": "psychology",
            "title": "AI Coaching",
            "description": "Personalized recommendations powered by machine learning algorithms.",
            "points": [
                "Smart workout recommendations",
                "Adaptive difficulty adjustments",
                "Form improvement suggestions",
                "Personalized training plans"
            ],
            "image": "C:/Users/User/.gemini/antigravity/brain/9b58a6c5-afe4-4a04-8170-f53d9a1b246b/ai_coaching_interface_1769528735677.png"
        }
    ]

    for i, feature in enumerate(features):
        # Alternate layout: odd indices (0,2) = text left, even indices (1,3) = text right
        if i % 2 == 0:
            # Text on LEFT, Video on RIGHT
            col1, col2 = st.columns([1, 1], gap="large")
            with col1:
                st.markdown(
                    f"""
                    <div class="feature-content">
                        <div class="feature-icon-large">
                            <span class="material-icons" style='font-size: 3.5rem; color: #f97316;'>{feature['icon']}</span>
                        </div>
                        <h3 class="feature-title">{feature['title']}</h3>
                        <p class="feature-description">{feature['description']}</p>
                        <ul class="feature-list">
                            {''.join([f'<li>{point}</li>' for point in feature['points']])}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="feature-video">
                        <div class="image-wrapper">
                            {render_media(feature['image'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            # Video on LEFT, Text on RIGHT
            col1, col2 = st.columns([1, 1], gap="large")
            with col1:
                st.markdown(
                    f"""
                    <div class="feature-video">
                        <div class="image-wrapper">
                            {render_media(feature['image'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"""
                    <div class="feature-content">
                        <div class="feature-icon-large">
                            <span class="material-icons" style='font-size: 3.5rem; color: #f97316;'>{feature['icon']}</span>
                        </div>
                        <h3 class="feature-title">{feature['title']}</h3>
                        <p class="feature-description">{feature['description']}</p>
                        <ul class="feature-list">
                            {''.join([f'<li>{point}</li>' for point in feature['points']])}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Add space between feature rows
        st.markdown("<div style='margin-bottom: 5rem;'></div>", unsafe_allow_html=True)

    # Start Workout Section
    st.markdown(
        """
        <div class="animate-slide-up" style="animation-delay: 0.6s; margin-top: 5rem; text-align: center; margin-bottom: 3rem;">
            <div style="display: inline-block; background: rgba(249, 115, 22, 0.1); border: 1px solid rgba(249, 115, 22, 0.3); border-radius: 20px; padding: 0.5rem 1.5rem; margin-bottom: 1rem;">
                <span style="color: #fb923c; font-size: 0.9rem; font-weight: 600;">💪 Start Your Workout</span>
            </div>
            <h2 style='color: #ffffff; font-size: 3rem; font-weight: 800; margin-bottom: -10px; line-height: 1.2;'>Choose Your</h2>
            <h2 style='background: linear-gradient(to right, #f97316, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline-block; font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;'>Exercise Journey</h2>
            <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 1rem;">Select an exercise below and let AI guide your perfect form</p>
        </div>
        """,
        unsafe_allow_html=True
    )


    exercises = [
        {
            "type": "bicep_curl",
            "display_name": "Bicep Curls",
            "description": "Build arm strength with proper curl form and technique",
            "image": "https://images.pexels.com/photos/1229356/pexels-photo-1229356.jpeg?auto=compress&cs=tinysrgb&w=800",
            "difficulty": "Beginner",
            "duration": "15 min",
            "calories": "120 kcal"
        },
        {
            "type": "squat",
            "display_name": "Squats",
            "description": "Strengthen legs and core with perfect squat technique",
            "image": "https://images.pexels.com/photos/1954524/pexels-photo-1954524.jpeg?auto=compress&cs=tinysrgb&w=800",
            "difficulty": "Intermediate",
            "duration": "20 min",
            "calories": "200 kcal"
        },
        {
            "type": "push_up",
            "display_name": "Push-ups",
            "description": "Develop upper body strength with controlled movements",
            "image": "https://images.pexels.com/photos/176782/pexels-photo-176782.jpeg?auto=compress&cs=tinysrgb&w=800",
            "difficulty": "Intermediate",
            "duration": "10 min",
            "calories": "150 kcal"
        },
        {
            "type": "plank",
            "display_name": "Planks",
            "description": "Build core stability and endurance with proper form",
            "image": "https://images.pexels.com/photos/1552104/pexels-photo-1552104.jpeg?auto=compress&cs=tinysrgb&w=800",
            "difficulty": "Advanced",
            "duration": "5 min",
            "calories": "80 kcal"
        },
    ]

    col1, col2, col3, col4 = st.columns(4)

    for i, (col, exercise) in enumerate(zip([col1, col2, col3, col4], exercises)):
        with col:
            st.markdown(
                f"""
                <div class="exercise-card-container">
                    <div class="exercise-image-wrapper">
                        <span class="exercise-badge">{exercise['difficulty']}</span>
                        <img src="{exercise['image']}" alt="{exercise['display_name']}">
                    </div>
                    <div class="exercise-card-content">
                        <div class="exercise-card-title">{exercise['display_name']}</div>
                        <div class="exercise-card-desc">{exercise['description']}</div>
                        <div class="exercise-stats">
                            <div class="stat-item">
                                <span class="material-icons" style="font-size: 14px; color: #f97316;">schedule</span>
                                {exercise['duration']}
                            </div>
                            <div class="stat-item">
                                <span class="material-icons" style="font-size: 14px; color: #f97316;">local_fire_department</span>
                                {exercise['calories']}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Button outside the card container HTML but visually connected via CSS
            if st.button(
                f"Start {exercise['display_name']}",
                key=f"start_{exercise['type']}",
                use_container_width=True,
            ):
                StateManager.set_current_page("workout")
                st.switch_page("pages/2_Workout.py")

    # Render Footer
    render_footer()


if __name__ == "__main__":
    main()
