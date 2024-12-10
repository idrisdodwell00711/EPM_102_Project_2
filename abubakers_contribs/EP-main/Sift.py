import cv2
import numpy as np

# HSI Preprocessing Class
# HSI gave better results in matching
class HSIConverter:
    @staticmethod
    def convert_to_hsi(image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsi = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        return image_hsi

    @staticmethod
    # Extract intensity channel from HSI image
    def get_intensity_channel(image_hsi):
        return image_hsi[:, :, 2]  

# Feature Detector Class
class SIFTDetector:
    def __init__(self):
        # SIFT with the octave and contrast threshold given in the documentation. The nfeatures was found by trial and error.
        self.detector = cv2.SIFT_create(2000, 3, 0.09)

    def detect_and_compute(self, image, mask=None):
        return self.detector.detectAndCompute(image, mask)

# Matcher Class
class FLANNMatcher:
    
    def __init__(self):
        # FLANN with KDTree
        index_params = dict(algorithm=1, trees=5)  
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, desc1, desc2):
        # Setting K nearest neighbours to 2
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        # Lowe's ratio test to remove bad matches in a robust manner. The constant, 0.85 gave the best all round results.
        good_matches = [m for m, n in matches if m.distance < 0.85 * n.distance] 
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        return good_matches

# Object Detection Processor
class ObjectDetectionProcessor:
    def __init__(self, detector, matcher):
        self.detector = detector
        self.matcher = matcher
    # Using a mask on the training image to 'crop' the target object
    def mask_train_image(self, train_img, train_bbox):
        mask = np.zeros(train_img.shape[:2], dtype=np.uint8)
        polygon = np.array(train_bbox, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        masked_train_img = cv2.bitwise_and(train_img, train_img, mask=mask)
        return masked_train_img, mask

    def compute_homography_and_transform(self, train_bbox, train_keypoints, train_descriptors, query_image, query_keypoints, query_descriptors):
        matches = self.matcher.match(train_descriptors, query_descriptors)
        print(f"Number of matches: {len(matches)}")

        if len(matches) < 5:
            print("Not enough matches for reliable homography.")
            return None, None, None
        # Obtaining the corresponding keypoints from the good matches
        src_pts = np.float32([train_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([query_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Removing outliers with Prosac and computing the homography matrix. The delta for Prosac is set to 4.5.
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO, 4.5)

        if homography is not None:
            train_bbox = np.float32(train_bbox).reshape(-1, 1, 2)
            # Using the homography matrix to perform a perspective transform
            query_bbox = cv2.perspectiveTransform(train_bbox, homography)
            return query_bbox, matches, mask
        else:
            print("Homography computation failed.")
            return None, None, None

    def visualize_matches(self, train_masked_img, query_img, train_bbox, query_bbox, matches, train_keypoints, query_keypoints, object_name, query_idx):
        query_img_with_bbox = query_img.copy()
        if query_bbox is not None:
            query_bbox = np.int32(query_bbox)
            cv2.polylines(query_img_with_bbox, [query_bbox], isClosed=True, color=(0, 255, 0), thickness=3)
            label_pos = (query_bbox[0][0][0], query_bbox[0][0][1] - 10)
            cv2.putText(query_img_with_bbox, object_name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        match_img = cv2.drawMatches(
            train_masked_img, train_keypoints, query_img_with_bbox, query_keypoints, matches, None,
            matchColor=(255, 0, 0), singlePointColor=None, flags=2
        )
        # Prints out the X and Y coordinates of the object detected
        print(query_keypoints[0].pt, query_keypoints[1].pt)
        cv2.imshow(f"Query {query_idx + 1} - Matches and Detection: {object_name}", match_img)

    def process_query_images(self, query_images, train_image, train_bboxes, object_names, query_objects):
        train_img_color = cv2.imread(train_image)
        train_hsi = HSIConverter.convert_to_hsi(train_img_color)
        train_intensity = HSIConverter.get_intensity_channel(train_hsi)

        for query_idx, (query_path, relevant_objects) in enumerate(zip(query_images, query_objects)):
            print(f"Processing Query {query_idx + 1}: {query_path}")
            query_img_color = cv2.imread(query_path)
            query_hsi = HSIConverter.convert_to_hsi(query_img_color)
            query_intensity = HSIConverter.get_intensity_channel(query_hsi)

            query_keypoints, query_descriptors = self.detector.detect_and_compute(query_intensity)

            for obj_idx in relevant_objects:
                train_bbox = train_bboxes[obj_idx]
                object_name = object_names[obj_idx]

                train_masked_img, mask = self.mask_train_image(train_img_color, train_bbox)
                train_keypoints, train_descriptors = self.detector.detect_and_compute(train_intensity, mask)
                print(f"Object '{object_name}': Detected {len(train_keypoints)} keypoints in the masked train image.")

                if train_descriptors is None or query_descriptors is None:
                    print(f"Object '{object_name}': No descriptors found for matching.")
                    continue

                query_bbox, matches, mask = self.compute_homography_and_transform(
                    train_bbox, train_keypoints, train_descriptors, query_intensity, query_keypoints, query_descriptors
                )

                if query_bbox is not None:
                    print(f"Object '{object_name}': Found {len(matches)} matches.")
                    self.visualize_matches(
                        train_masked_img, query_img_color, train_bbox, query_bbox, matches, train_keypoints, query_keypoints, object_name, query_idx
                    )
                else:
                    print(f"Object '{object_name}': No valid matches or homography found.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Main Program
if __name__ == "__main__":
    query_images = [
        "./img_query_1.jpg",
        "./img_query_2.jpg",
        "./img_query_3.jpg"
    ]
    train_image = "./img_train.jpg"
    
    # Coordinates in the training image for each object
    train_bboxes = [
        [(609, 28), (305, 232), (666, 654), (933, 341)],  # Controller
        [(266, 85), (247, 353), (69, 345), (87, 45)],    # Tablets
        [(605, 483), (602, 699), (402, 712), (395, 484)],  # Airpods
        [(327, 430), (327, 711), (50, 712), (66, 421)],  # Coin
    ]

    object_names = ["Controller", "Tablets", "Airpods", "Coin"]
    query_objects = [[2], [1, 2], [0, 1, 2, 3]]

    detector = SIFTDetector()
    matcher = FLANNMatcher()

    processor = ObjectDetectionProcessor(detector, matcher)
    processor.process_query_images(query_images, train_image, train_bboxes, object_names, query_objects)