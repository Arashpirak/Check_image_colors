import cv2
import numpy as np
from skimage import filters, morphology

# Global variables
img = None
gray = None
result_window = 'Wrinkle Detection'
control_window = 'Controls'
filter_states = {}
trackbar_params = {}

def nothing(x):
    pass

# Filter functions
def apply_gaussian_blur(img, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def apply_median_blur(img, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(img, kernel_size)

def apply_bilateral_filter(img, d, sigma_color, sigma_space):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def apply_canny_edge(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def apply_sobel_edge(img, ksize):
    # Ensure ksize is odd and within the valid range [1, 31]
    ksize = max(1, ksize) # Ensure minimum size is 1
    ksize = min(31, ksize) # Ensure maximum size is 31
    if ksize % 2 == 0:
        ksize += 1 # Ensure ksize is odd
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = np.hypot(sobelx, sobely)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return sobel

def apply_laplacian_edge(img, ksize):
    # Ensure ksize is odd and within the valid range [1, 31] for Sobel-based Laplacian
    # Laplacian ksize must be 1, 3, 5, 7, etc. up to 31.
    ksize = max(1, ksize) # Ensure minimum size is 1
    ksize = min(31, ksize) # Ensure maximum size is 31
    if ksize % 2 == 0:
        ksize += 1 # Ensure ksize is odd
        
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return laplacian

def apply_scharr_edge(img):
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharr = np.hypot(scharrx, scharry)
    scharr = cv2.normalize(scharr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return scharr

def apply_frangi_filter(img, sigma_min, sigma_max):
    frangi = filters.frangi(img, sigmas=range(sigma_min, sigma_max + 1), black_ridges=False)
    frangi = cv2.normalize(frangi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return frangi

def apply_hessian_filter(img, sigma):
    hessian = filters.hessian(img, sigmas=[sigma], mode='constant', cval=0)
    hessian = cv2.normalize(hessian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return hessian

def apply_morphological_refinement(img, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    refined = morphology.thin(img, max_iter=1)
    refined = cv2.normalize(refined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return refined

# Update image based on trackbar settings
def update_image():
    global img, gray
    result = gray.copy()
    
    # Apply preprocessing filters sequentially
    if filter_states['Gaussian']:
        kernel_size = cv2.getTrackbarPos('Gauss Kernel', control_window)
        result = apply_gaussian_blur(result, kernel_size)
    
    if filter_states['Median']:
        kernel_size = cv2.getTrackbarPos('Median Kernel', control_window)
        result = apply_median_blur(result, kernel_size)
    
    if filter_states['Bilateral']:
        d = cv2.getTrackbarPos('Bilateral D', control_window)
        sigma_color = cv2.getTrackbarPos('Bilateral SigmaColor', control_window)
        sigma_space = cv2.getTrackbarPos('Bilateral SigmaSpace', control_window)
        result = apply_bilateral_filter(result, d, sigma_color, sigma_space)
    
    # Collect edge detection results
    edge_results = []
    
    if filter_states['Canny']:
        low_threshold = cv2.getTrackbarPos('Canny Low', control_window)
        high_threshold = cv2.getTrackbarPos('Canny High', control_window)
        edge_results.append(apply_canny_edge(result, low_threshold, high_threshold))
    
    if filter_states['Sobel']:
        ksize = cv2.getTrackbarPos('Sobel Kernel', control_window)
        edge_results.append(apply_sobel_edge(result, ksize))
    
    if filter_states['Laplacian']:
        ksize = cv2.getTrackbarPos('Laplacian Kernel', control_window)
        edge_results.append(apply_laplacian_edge(result, ksize))
    
    if filter_states['Scharr']:
        edge_results.append(apply_scharr_edge(result))
    
    if filter_states['Frangi']:
        sigma_min = cv2.getTrackbarPos('Frangi Sigma Min', control_window)
        sigma_max = cv2.getTrackbarPos('Frangi Sigma Max', control_window)
        edge_results.append(apply_frangi_filter(result, sigma_min, sigma_max))
    
    if filter_states['Hessian']:
        sigma = cv2.getTrackbarPos('Hessian Sigma', control_window)
        edge_results.append(apply_hessian_filter(result, sigma))
    
    # Combine edge detection results
    if edge_results:
        combined_edges = edge_results[0]
        for edge in edge_results[1:]:
            combined_edges = cv2.bitwise_or(combined_edges, edge)
        result = combined_edges
    else:
        # If no edge filters are applied, keep the preprocessed image
        result = result
    
    # Apply morphological refinement
    if filter_states['Morphological']:
        kernel_size = cv2.getTrackbarPos('Morph Kernel', control_window)
        result = apply_morphological_refinement(result, kernel_size)
    
    # Display the result in the result window
    cv2.imshow(result_window, result)
    return result

def main():
    global img, gray, filter_states, trackbar_params, result_window, control_window
    
    # Load image (replace 'skin_image.jpg' with your image path)
    img = cv2.imread(r'W:\Dell desk\amirmoheb\skinenhancment\test-filters.jpg')
    if img is None:
        print("Error: Could not load image")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize filter states
    filter_states['Gaussian'] = 0
    filter_states['Median'] = 0
    filter_states['Bilateral'] = 0
    filter_states['Canny'] = 0
    filter_states['Sobel'] = 0
    filter_states['Laplacian'] = 0
    filter_states['Scharr'] = 0
    filter_states['Frangi'] = 0
    filter_states['Hessian'] = 0
    filter_states['Morphological'] = 0
    
    # Create windows
    cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
    
    # Create a blank image for the control window to make it visible
    control_img = np.zeros((100, 500), np.uint8)
    cv2.imshow(control_window, control_img)
    
    # Trackbars for enabling/disabling filters
    cv2.createTrackbar('Gaussian', control_window, 0, 1, lambda x: update_filter_state('Gaussian', x))
    cv2.createTrackbar('Median', control_window, 0, 1, lambda x: update_filter_state('Median', x))
    cv2.createTrackbar('Bilateral', control_window, 0, 1, lambda x: update_filter_state('Bilateral', x))
    cv2.createTrackbar('Canny', control_window, 0, 1, lambda x: update_filter_state('Canny', x))
    cv2.createTrackbar('Sobel', control_window, 0, 1, lambda x: update_filter_state('Sobel', x))
    cv2.createTrackbar('Laplacian', control_window, 0, 1, lambda x: update_filter_state('Laplacian', x))
    cv2.createTrackbar('Scharr', control_window, 0, 1, lambda x: update_filter_state('Scharr', x))
    cv2.createTrackbar('Frangi', control_window, 0, 1, lambda x: update_filter_state('Frangi', x))
    cv2.createTrackbar('Hessian', control_window, 0, 1, lambda x: update_filter_state('Hessian', x))
    cv2.createTrackbar('Morphological', control_window, 0, 1, lambda x: update_filter_state('Morphological', x))
    
    # Trackbars for filter parameters
    cv2.createTrackbar('Gauss Kernel', control_window, 5, 15, nothing)
    cv2.createTrackbar('Median Kernel', control_window, 5, 15, nothing)
    cv2.createTrackbar('Bilateral D', control_window, 5, 15, nothing)
    cv2.createTrackbar('Bilateral SigmaColor', control_window, 75, 150, nothing)
    cv2.createTrackbar('Bilateral SigmaSpace', control_window, 75, 150, nothing)
    cv2.createTrackbar('Canny Low', control_window, 78, 200, nothing)
    cv2.createTrackbar('Canny High', control_window, 222, 300, nothing)
    cv2.createTrackbar('Sobel Kernel', control_window, 3, 7, nothing)
    cv2.createTrackbar('Laplacian Kernel', control_window, 3, 7, nothing)
    cv2.createTrackbar('Frangi Sigma Min', control_window, 1, 5, nothing)
    cv2.createTrackbar('Frangi Sigma Max', control_window, 3, 10, nothing)
    cv2.createTrackbar('Hessian Sigma', control_window, 1, 5, nothing)
    cv2.createTrackbar('Morph Kernel', control_window, 3, 7, nothing)
    
    # Initial image display
    update_image()
    
    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == 13: # Enter key to find and draw contours
            # Get the current result image
            current_result_image = update_image()
            # Find and draw contours on the result image
            find_and_draw_contours(current_result_image)
        else:
            # Update the image based on trackbar changes
            update_image()
    
    cv2.destroyAllWindows()

def update_filter_state(filter_name, value):
    filter_states[filter_name] = bool(value)
    update_image()

def find_and_draw_contours(result_image):
    # Convert the grayscale result to color to draw red contours
    result_color = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)

    # Apply binary thresholding to the result image to get a binary image
    # You might need to adjust the threshold value based on your results
    _, binary_image = cv2.threshold(result_image, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold', binary_image)

    # Find contours in the binary image
    # cv2.findContours modifies the input image, so use a copy
    contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the color image in red (BGR: 0, 0, 255)
    cv2.drawContours(result_color, contours, -1, (0, 0, 255), 1) # -1 means draw all contours, 2 is thickness

    # Display the image with contours in a new window
    cv2.imshow('Contours', result_color)


if __name__ == '__main__':
    main()
