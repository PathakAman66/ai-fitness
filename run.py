#!/usr/bin/env python3
"""
AI Fitness Trainer - Main Entry Point
Updated with Web Interface Options
"""
import argparse
import sys
import os
import subprocess

def run_desktop_mode():
    """Run in desktop mode with OpenCV window"""
    print("Starting AI Fitness Trainer in Desktop Mode...")
    subprocess.run([sys.executable, "run_fitness_trainer.py"])

def run_enhanced_desktop():
    """Run enhanced desktop version"""
    print("Starting Enhanced AI Fitness Trainer...")
    subprocess.run([sys.executable, "enhanced_trainer.py"])

def run_web_mode():
    """Run web interface launcher"""
    subprocess.run([sys.executable, "web/run_website.py"])

def main():
    parser = argparse.ArgumentParser(description='AI Fitness Trainer')
    parser.add_argument('--mode', choices=['desktop', 'enhanced', 'web', 'ui'], 
                       default='desktop', help='Run mode')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏋️ AI FITNESS TRAINER - Choose Your Interface")
    print("=" * 60)
    
    # If no mode specified, show interactive menu
    if args.mode == 'desktop':
        run_desktop_mode()
    elif args.mode == 'enhanced':
        run_enhanced_desktop()
    elif args.mode == 'web' or args.mode == 'ui':
        run_web_mode()
    else:
        # Interactive mode selection
        print("Available Interfaces:")
        print("1. 🖥️  Desktop App (OpenCV) - Full AI Features")
        print("2. 💪 Enhanced Desktop - Advanced Analytics")
        print("3. 🌐 Web Interface - Professional Website")
        print("4. 📊 Progress Dashboard - View Your Data")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_desktop_mode()
        elif choice == "2":
            run_enhanced_desktop()
        elif choice == "3":
            run_web_mode()
        elif choice == "4":
            subprocess.run([sys.executable, "progress_dashboard.py"])
        else:
            print("Invalid choice. Running desktop mode...")
            run_desktop_mode()

if __name__ == "__main__":
    main()