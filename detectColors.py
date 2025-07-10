import cv2
import numpy as np

def detect_pixel_colors_hsv(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Could not read the image.")
        return
    
    # Convert BGR to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Get image dimensions
    height, width, _ = img.shape
    
    # Create a copy of the image to draw on
    display_img = img.copy()
    
    # Function to get color name from HSV values
    def get_color_name_hsv(hsv):
        hue, saturation, value = hsv
        
        # Define HSV ranges for different colors
        if value < 30:
            return "Black"
        elif saturation < 50 and value > 200:
            return "White"
        elif saturation < 50:
            return "Gray"
        elif 0 <= hue < 10 or 170 <= hue < 180:
            return "Red"
        elif 10 <= hue < 25:
            return "Orange"
        elif 25 <= hue < 35:
            return "Yellow"
        elif 35 <= hue < 80:
            return "Green"
        elif 80 <= hue < 100:
            return "Teal"
        elif 100 <= hue < 130:
            return "Blue"
        elif 130 <= hue < 150:
            return "Purple"
        elif 150 <= hue < 170:
            return "Pink"
        else:
            return "Unknown"
    
    # Function to handle mouse clicks
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get BGR and HSV values
            bgr = img[y, x]
            hsv = hsv_img[y, x]
            
            # Convert BGR to RGB for display purposes
            rgb = bgr[::-1]
            
            # Get color name
            color_name = get_color_name_hsv(hsv)
            
            # Display information
            print(f"Pixel at ({x}, {y}):")
            print(f"BGR values: {bgr}")
            print(f"HSV values: {hsv}")
            print(f"RGB values: {rgb}")
            print(f"Color: {color_name}")
            print("-" * 30)
            
            # Draw a circle and text on the display image
            cv2.circle(display_img, (x, y), 3, (0, 0, 255), -1)
            text = f"{color_name} {rgb}"
            cv2.putText(display_img, text, (x+10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Show the image with the marker
            cv2.imshow('Image', display_img)
    
    # Display the image and set mouse callback
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', click_event)
    
    print("Click on any pixel to see its color in HSV space. Press 'q' to quit.")
    
    # Wait for a key press to exit
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    image_path = input("Enter the path to your image file: ")
    detect_pixel_colors_hsv(image_path)