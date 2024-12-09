import cv2
import numpy as np

# Global variables for drawing
polygon_points = []
image_copy = None

def draw_polygon(event, x, y, flags, param):
    global polygon_points, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse click to add a point
        polygon_points.append((x, y))
        cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)  # Draw the point
        cv2.imshow("Draw Polygon", image_copy)
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right mouse click to finish the polygon
        if len(polygon_points) > 2:  # At least three points to form a polygon
            cv2.polylines(image_copy, [np.array(polygon_points)], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.imshow("Draw Polygon", image_copy)

def polygon_coordinate_extractor(image_path):
    global polygon_points, image_copy

    # Clear previous points
    polygon_points = []

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return []

    image_copy = image.copy()  # Create a copy for drawing

    # Set up window and mouse callback
    cv2.namedWindow("Draw Polygon")
    cv2.setMouseCallback("Draw Polygon", draw_polygon)

    # Display the image and wait for interaction
    print("Left-click to add points, right-click to finish the polygon.")
    cv2.imshow("Draw Polygon", image)
    cv2.waitKey(0)  # Wait until user presses a key
    cv2.destroyAllWindows()

    return polygon_points  # Return the drawn polygon points

if __name__ == "__main__":
    train_image_path = "/Users/abubakershabbir/Desktop/CityUniversity/Engineering Programming/Coursework 2/epm102_p2_dataset/img_train.jpg"
    
    # Extract polygon for the train image
    train_polygon = polygon_coordinate_extractor(train_image_path)
    print("Train Image Polygon Coordinates:", train_polygon)

    