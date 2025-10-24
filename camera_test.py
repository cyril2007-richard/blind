import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {index}")
        return False
    
    print(f"Successfully opened camera {index}")
    
    # Try to read a few frames
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame {i} from camera {index}")
            cap.release()
            return False
        print(f"Successfully read frame {i} from camera {index}")
    
    print(f"Camera {index} test successful!")
    cap.release()
    return True

if __name__ == "__main__":
    for i in range(4):
        print(f"\n--- Testing camera index {i} ---")
        if test_camera(i):
            print(f"--- Camera index {i} is working ---")