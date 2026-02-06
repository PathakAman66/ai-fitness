"""
AI Chatbot Dialog Component
Provides 24/7 fitness assistance through an interactive chat interface
"""

import streamlit as st
from datetime import datetime

@st.dialog("🤖 AI Fitness Assistant - 24/7 Help", width="large")
def chatbot_dialog():
    """
    Renders an interactive AI chatbot dialog for fitness assistance.
    """
    
    # Initialize chat history in session state
    if "chatbot_messages" not in st.session_state:
        st.session_state.chatbot_messages = [
            {
                "role": "assistant",
                "content": "👋 Hi! I'm your AI Fitness Assistant. I'm here to help you 24/7 with:\n\n"
                          "💪 Workout recommendations\n"
                          "🍎 Nutrition advice\n"
                          "📊 Progress tracking tips\n"
                          "🎯 Goal setting strategies\n\n"
                          "How can I assist you today?"
            }
        ]
    
    # Custom styling
    st.markdown(
        """
        <style>
        .chatbot-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 1rem;
            background: rgba(15, 23, 42, 0.3);
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        
        .user-message {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.2));
            padding: 0.75rem 1rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            border-left: 3px solid #3b82f6;
        }
        
        .assistant-message {
            background: rgba(139, 92, 246, 0.15);
            padding: 0.75rem 1rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            border-left: 3px solid #8b5cf6;
        }
        
        .message-time {
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.5);
            margin-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Display chat messages
    st.markdown("### Chat History")
    
    for message in st.session_state.chatbot_messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(
                f"""
                <div class="user-message">
                    <strong>You:</strong><br>
                    {content}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="assistant-message">
                    <strong>🤖 AI Assistant:</strong><br>
                    {content}
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Chat input
    st.markdown("---")
    
    with st.form("chatbot_input_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your message:",
            placeholder="Ask me anything about fitness, workouts, nutrition, or your progress...",
            height=100,
            key="chatbot_input"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            submitted = st.form_submit_button("Send 💬", type="primary", use_container_width=True)
        
        with col2:
            if st.form_submit_button("Clear Chat 🗑️", use_container_width=True):
                st.session_state.chatbot_messages = [
                    {
                        "role": "assistant",
                        "content": "Chat cleared! How can I help you?"
                    }
                ]
                st.rerun()
        
        with col3:
            if st.form_submit_button("Close ❌", use_container_width=True):
                st.session_state.show_chatbot = False
                st.rerun()
        
        if submitted and user_input.strip():
            # Add user message
            st.session_state.chatbot_messages.append({
                "role": "user",
                "content": user_input.strip()
            })
            
            # Generate AI response (simple rule-based for now)
            response = generate_chatbot_response(user_input.strip())
            
            # Add assistant response
            st.session_state.chatbot_messages.append({
                "role": "assistant",
                "content": response
            })
            
            st.rerun()
    
    # Quick action buttons
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💪 Workout Tips", use_container_width=True, key="quick_workout"):
            st.session_state.chatbot_messages.append({
                "role": "user",
                "content": "Give me some workout tips"
            })
            st.session_state.chatbot_messages.append({
                "role": "assistant",
                "content": "Here are some essential workout tips:\n\n"
                          "1. **Warm up properly** - 5-10 minutes of light cardio\n"
                          "2. **Focus on form** - Quality over quantity\n"
                          "3. **Progressive overload** - Gradually increase intensity\n"
                          "4. **Rest & recovery** - Muscles grow during rest\n"
                          "5. **Stay hydrated** - Drink water before, during, and after\n\n"
                          "What specific workout are you planning?"
            })
            st.rerun()
    
    with col2:
        if st.button("🍎 Nutrition Guide", use_container_width=True, key="quick_nutrition"):
            st.session_state.chatbot_messages.append({
                "role": "user",
                "content": "Tell me about nutrition"
            })
            st.session_state.chatbot_messages.append({
                "role": "assistant",
                "content": "Nutrition fundamentals for fitness:\n\n"
                          "🥩 **Protein**: 1.6-2.2g per kg body weight\n"
                          "🍚 **Carbs**: Primary energy source, adjust based on activity\n"
                          "🥑 **Fats**: 20-35% of total calories\n"
                          "💧 **Water**: 3-4 liters daily\n\n"
                          "**Meal timing**: Eat protein within 2 hours post-workout\n\n"
                          "What's your fitness goal? I can provide specific nutrition advice!"
            })
            st.rerun()
    
    with col3:
        if st.button("📊 Track Progress", use_container_width=True, key="quick_progress"):
            st.session_state.chatbot_messages.append({
                "role": "user",
                "content": "How do I track my progress?"
            })
            st.session_state.chatbot_messages.append({
                "role": "assistant",
                "content": "Track your fitness progress effectively:\n\n"
                          "📸 **Progress photos** - Weekly front/side/back\n"
                          "⚖️ **Body measurements** - Weight, body fat %, muscle mass\n"
                          "💪 **Strength gains** - Track weights and reps\n"
                          "⏱️ **Performance metrics** - Speed, endurance, flexibility\n"
                          "📝 **Workout logs** - Use our History page!\n\n"
                          "Consistency is key - track at the same time weekly!"
            })
            st.rerun()


def generate_chatbot_response(user_message: str) -> str:
    """
    Generates a response based on user input.
    Simple rule-based system for now - can be upgraded to AI model later.
    """
    message_lower = user_message.lower()
    
    # Workout-related queries
    if any(word in message_lower for word in ["workout", "exercise", "training", "routine"]):
        return ("Great question about workouts! 💪\n\n"
                "I recommend checking out our **Workout** page where you can:\n"
                "- Get AI-powered exercise recommendations\n"
                "- Track your sets and reps\n"
                "- Receive real-time form feedback\n\n"
                "What type of workout are you interested in? (strength, cardio, flexibility)")
    
    # Nutrition queries
    elif any(word in message_lower for word in ["nutrition", "diet", "food", "eat", "meal"]):
        return ("Nutrition is crucial for fitness success! 🍎\n\n"
                "Key principles:\n"
                "- **Protein**: Essential for muscle recovery\n"
                "- **Carbs**: Fuel your workouts\n"
                "- **Healthy fats**: Support hormone production\n"
                "- **Hydration**: Critical for performance\n\n"
                "What's your specific nutrition question?")
    
    # Progress tracking
    elif any(word in message_lower for word in ["progress", "track", "history", "stats"]):
        return ("Tracking progress is essential! 📊\n\n"
                "Check out our:\n"
                "- **History** page - View all your past workouts\n"
                "- **Stats** page - Analyze your performance trends\n\n"
                "Remember: Progress isn't always linear. Stay consistent!")
    
    # Motivation
    elif any(word in message_lower for word in ["motivat", "inspire", "tired", "lazy", "quit"]):
        return ("You've got this! 🔥\n\n"
                "Remember why you started:\n"
                "- Every workout counts\n"
                "- Progress takes time\n"
                "- You're stronger than you think\n"
                "- Small steps lead to big changes\n\n"
                "What's your fitness goal? Let's work towards it together!")
    
    # Greetings
    elif any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
        return ("Hello! 👋 Great to see you!\n\n"
                "I'm here to help with:\n"
                "- Workout planning\n"
                "- Nutrition advice\n"
                "- Progress tracking\n"
                "- Motivation & tips\n\n"
                "What would you like to know?")
    
    # Default response
    else:
        return (f"Thanks for your message! 🤖\n\n"
                f"I'm here to help with fitness-related questions. You can ask me about:\n\n"
                f"💪 Workouts and exercises\n"
                f"🍎 Nutrition and diet\n"
                f"📊 Progress tracking\n"
                f"🎯 Goal setting\n\n"
                f"Try using the Quick Actions buttons below for instant help!")
