
import streamlit as st
import streamlit.components.v1 as components

def render_hero_carousel():
    """
    Renders a full-width, responsive, auto-sliding hero carousel using HTML/CSS/JS.
    Features premium HD images curated for AI Fitness theme with enhanced animations.
    """
    
    # Curated Premium HD Images - AI Fitness Theme
    # Slide 1: AI Pose Analysis - Tech/Futuristic Gym
    img_ai_analysis = "https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?q=80&w=3870&auto=format&fit=crop"
    
    # Slide 2: Strength Training - Intense Workout
    img_strength = "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?q=80&w=3870&auto=format&fit=crop"
    
    # Slide 3: Cardio Performance - Dynamic Motion
    img_cardio = "https://images.unsplash.com/photo-1476480862126-209bfaa8edc8?q=80&w=3870&auto=format&fit=crop"
    
    carousel_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

            :root {{
                --primary: #00f2ff;
                --secondary: #00ff9d;
                --accent: #ff0055;
                --text: #ffffff;
                --overlay-bg: linear-gradient(180deg, rgba(0,0,0,0.1) 0%, rgba(0,0,0,0.4) 50%, #0e1117 100%);
            }}

            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}

            body {{
                background-color: transparent;
                overflow: hidden;
                font-family: 'Roboto', sans-serif;
            }}

            .carousel-container {{
                position: relative;
                width: 100%;
                height: 85vh; /* Full viewport height */
                min-height: 600px;
                border-radius: 0 0 20px 20px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}

            .slide {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                opacity: 0;
                transition: opacity 1.2s ease-in-out, transform 8s ease-out;
                z-index: 1;
                transform: scale(1.1);
            }}

            .slide.active {{
                opacity: 1;
                z-index: 2;
                transform: scale(1);
                animation: slideZoom 8s ease-out forwards;
            }}

            @keyframes slideZoom {{
                0% {{ transform: scale(1.1); }}
                100% {{ transform: scale(1); }}
            }}

            .slide-img {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                position: absolute;
                top: 0;
                left: 0;
            }}

            .overlay {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: var(--overlay-bg);
                z-index: 3;
            }}

            .content {{
                position: absolute;
                bottom: 100px;
                left: 60px;
                z-index: 4;
                color: var(--text);
                max-width: 800px;
                opacity: 0;
                transform: translateY(30px);
                transition: all 0.8s ease-out 0.3s;
            }}

            .slide.active .content {{
                opacity: 1;
                transform: translateY(0);
            }}

            h1 {{
                font-family: 'Orbitron', sans-serif;
                font-size: 4rem;
                font-weight: 700;
                margin-bottom: 1rem;
                text-transform: uppercase;
                text-shadow: 0 0 20px rgba(0, 242, 255, 0.5);
                background: linear-gradient(90deg, #fff, var(--primary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                line-height: 1.1;
                display: inline-block;
            }}

            p {{
                font-size: 1.5rem;
                margin-bottom: 2rem;
                font-weight: 300;
                text-shadow: 0 2px 4px rgba(0,0,0,0.8);
                max-width: 600px;
            }}

            .btn-cta {{
                display: inline-block;
                padding: 16px 40px;
                background: rgba(0, 242, 255, 0.1);
                border: 2px solid var(--primary);
                color: var(--primary);
                font-family: 'Orbitron', sans-serif;
                font-size: 1.1rem;
                font-weight: 700;
                text-decoration: none;
                text-transform: uppercase;
                border-radius: 4px;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }}

            .btn-cta:hover {{
                background: var(--primary);
                color: #000;
                box-shadow: 0 0 30px var(--primary);
            }}

            /* UI Overlays */
            .ui-overlay {{
                position: absolute;
                top: 40px;
                right: 40px;
                z-index: 4;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }}

            .ui-widget {{
                background: rgba(0, 0, 0, 0.7);
                border-right: 3px solid var(--primary);
                padding: 12px 20px;
                border-radius: 4px 0 0 4px;
                font-family: 'Orbitron', sans-serif;
                font-size: 0.9rem;
                color: #fff;
                backdrop-filter: blur(5px);
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 10px;
                transform: translateX(20px);
                opacity: 0;
                transition: all 0.5s ease 0.5s;
            }}

            .slide.active .ui-widget {{
                transform: translateX(0);
                opacity: 1;
            }}
            
            .highlight {{
                color: var(--primary);
                font-weight: bold;
            }}

            /* Indicators */
            .indicators {{
                position: absolute;
                bottom: 40px;
                left: 50%;
                transform: translateX(-50%);
                z-index: 5;
                display: flex;
                gap: 15px;
            }}

            .indicator {{
                width: 60px;
                height: 4px;
                background: rgba(255,255,255,0.2);
                border-radius: 2px;
                cursor: pointer;
                transition: all 0.3s ease;
            }}

            .indicator.active {{
                background: var(--primary);
                box-shadow: 0 0 15px var(--primary);
                width: 80px;
            }}

            @media (max-width: 768px) {{
                h1 {{
                    font-size: 2.5rem;
                }}
                .content {{
                    bottom: 80px;
                    left: 20px;
                    right: 20px;
                }}
                .btn-cta {{
                    padding: 12px 30px;
                    font-size: 1rem;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="carousel-container">
            <!-- Slide 1 -->
            <div class="slide active">
                <img src="{img_ai_analysis}" class="slide-img" alt="AI Analysis">
                <div class="overlay"></div>
                <div class="ui-overlay">
                    <div class="ui-widget">AI STATUS <span class="highlight">ACTIVE</span></div>
                    <div class="ui-widget">POSE ACCURACY <span class="highlight">99.2%</span></div>
                </div>
                <div class="content">
                    <h1>AI-Powered<br>Fitness</h1>
                    <p>Real-time pose detection and biomechanical analysis. Train smarter with intelligent feedback.</p>
                    <a href="?auth_action=signup" target="_self" class="btn-cta">Get Started</a>
                </div>
            </div>

            <!-- Slide 2 -->
            <div class="slide">
                <img src="{img_strength}" class="slide-img" alt="Strength">
                <div class="overlay"></div>
                <div class="ui-overlay">
                    <div class="ui-widget">REPS COUNTED <span class="highlight">12/15</span></div>
                    <div class="ui-widget">FORM SCORE <span class="highlight">94%</span></div>
                </div>
                <div class="content">
                    <h1>Perfect<br>Form</h1>
                    <p>Automatic rep counting with real-time form corrections. Build strength safely and effectively.</p>
                    <a href="?nav=workout" target="_self" class="btn-cta">Start Workout</a>
                </div>
            </div>

            <!-- Slide 3 -->
            <div class="slide">
                <img src="{img_cardio}" class="slide-img" alt="Cardio">
                <div class="overlay"></div>
                <div class="ui-overlay">
                    <div class="ui-widget">HEART RATE <span class="highlight">156 BPM</span></div>
                    <div class="ui-widget">CALORIES <span class="highlight">420 KCAL</span></div>
                </div>
                <div class="content">
                    <h1>Track<br>Progress</h1>
                    <p>Monitor your performance metrics in real-time. Achieve your fitness goals faster.</p>
                    <a href="?nav=stats" target="_self" class="btn-cta">View Stats</a>
                </div>
            </div>

            <!-- Indicators -->
            <div class="indicators">
                <div class="indicator active" onclick="setSlide(0)"></div>
                <div class="indicator" onclick="setSlide(1)"></div>
                <div class="indicator" onclick="setSlide(2)"></div>
            </div>
        </div>

        <script>
            let slides = document.querySelectorAll('.slide');
            let indicators = document.querySelectorAll('.indicator');
            let currentSlide = 0;
            const slideInterval = 3000; // 3 seconds

            function nextSlide() {{
                slides[currentSlide].classList.remove('active');
                indicators[currentSlide].classList.remove('active');
                
                currentSlide = (currentSlide + 1) % slides.length;
                
                slides[currentSlide].classList.add('active');
                indicators[currentSlide].classList.add('active');
            }}

            function setSlide(index) {{
                slides[currentSlide].classList.remove('active');
                indicators[currentSlide].classList.remove('active');
                
                currentSlide = index;
                
                slides[currentSlide].classList.add('active');
                indicators[currentSlide].classList.add('active');
            }}

            let slideTimer = setInterval(nextSlide, slideInterval);

            // Optional: Pause on hover
            const container = document.querySelector('.carousel-container');
            container.addEventListener('mouseenter', () => clearInterval(slideTimer));
            container.addEventListener('mouseleave', () => slideTimer = setInterval(nextSlide, slideInterval));
        </script>
    </body>
    </html>
    """

    # Embed the HTML component
    # Increased height to 800 for full immersion
    components.html(carousel_html, height=800, scrolling=False)

if __name__ == "__main__":
    render_hero_carousel()
