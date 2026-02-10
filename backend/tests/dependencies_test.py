"""
Simple test for AI Fitness Trainer - Minimal dependencies
"""
import cv2

def test_camera():
    """Test if camera works"""
    print("Testing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return False
        
    ret, frame = cap.read()
    if ret:
        print("✅ Camera working - frame captured")
        cv2.imshow('Test Camera - Press any key to close', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Could not read frame from camera")
        
    cap.release()
    return ret

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    
    try:
        import mediapipe
        print("✅ MediaPipe imported successfully")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
        
    try:
        import numpy
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
        
    try:
        import pygame
        print("✅ Pygame imported successfully")
    except ImportError as e:
        print(f"❌ Pygame import failed: {e}")
        
    return True

if __name__ == "__main__":
    print("🧪 AI Fitness Trainer - Quick Test")
    print("=" * 40)
    
    test_imports()
    print()
    test_camera()
    
    print("\n🎯 If both tests pass, the main app should work!")