import streamlit as st

def render_footer():
    """
    Render a professional footer for the Streamlit application.
    Matches the design of the web interface footer.
    """
    # Minified HTML to prevent Markdown code block interpretation
    footer_html = """<div class="footer"><div class="footer-content"><div class="footer-section"><h3>🏋️ AI Fitness Trainer</h3><p>Your personal AI-powered fitness coach. Detects your form in real-time and provides instant feedback to help you workout safely and effectively. Powered by MediaPipe and advanced computer vision.</p></div><div class="footer-section"><h3>Connect With Us</h3><p>Follow our journey and stay updated with the latest features.</p><div class="footer-social-links"><a href="https://github.com/Suvam18" target="_blank" title="GitHub"><i class="fab fa-github"></i></a><a href="https://www.linkedin.com/in/suvam-chatterjee-ba6b922b2?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank" title="LinkedIn"><i class="fab fa-linkedin"></i></a><a href="https://www.instagram.com/suv_am28/" target="_blank" title="Instagram"><i class="fab fa-instagram"></i></a><a href="https://www.facebook.com" target="_blank" title="Facebook"><i class="fab fa-facebook"></i></a><a href="https://twitter.com" target="_blank" title="Twitter"><i class="fab fa-twitter"></i></a></div></div><div class="footer-section"><h3>Quick Links</h3><p><a href="#" style="color: #9ca3af; text-decoration: none;">Features</a></p><p><a href="#" style="color: #9ca3af; text-decoration: none;">How it works</a></p><p><a href="#" style="color: #9ca3af; text-decoration: none;">Privacy Policy</a></p></div></div><div class="footer-bottom">&copy; 2024 AI Fitness Trainer. Built with Python, OpenCV, and MediaPipe.<div class="footer-badges"><span class="footer-badge">Python 3.8+</span><span class="footer-badge">OpenCV 4.8</span><span class="footer-badge">MediaPipe 0.10</span></div></div></div>"""
    
    st.markdown(footer_html, unsafe_allow_html=True)
