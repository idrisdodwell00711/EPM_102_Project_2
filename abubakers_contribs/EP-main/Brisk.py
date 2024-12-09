import cv2
import numpy as np

# HSI Preprocessing Class
class HSIConverter:
    @staticmethod
    def convert_to_hsi(image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsi = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        return image_hsi

    @staticmethod
    def get_intensity_channel(image_hsi):
        return image_hsi[:, :, 2]  # Extract intensity channel

# Feature Detector Class
class BRISKDetector:
    def __init__(self):
        self.detector = cv2.BRISK_create()

    def detect_and_compute(self, image, mask=None):
        return self.detector.detectAndCompute(image, mask)

# Matcher Class
class BFHammingMatcher:
    def __init__(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, desc1, desc2):
        matches = self.matcher.match(desc1, desc2)
        return sorted(matches, key=lambda x: x.distance)

# Object Detection Processor
class ObjectDetectionProcessor:
    def __init__(self, detector, matcher):
        self.detector = detector
        self.matcher = matcher

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

        src_pts = np.float32([train_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([query_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if homography is not None:
            train_bbox = np.float32(train_bbox).reshape(-1, 1, 2)
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
        "/Users/abubakershabbir/Desktop/CityUniversity/Engineering Programming/Coursework 2/epm102_p2_dataset/img_query_1.jpg",
        "/Users/abubakershabbir/Desktop/CityUniversity/Engineering Programming/Coursework 2/epm102_p2_dataset/img_query_2.jpg",
        "/Users/abubakershabbir/Desktop/CityUniversity/Engineering Programming/Coursework 2/epm102_p2_dataset/img_query_3.jpg"
    ]
    train_image = "/Users/abubakershabbir/Desktop/CityUniversity/Engineering Programming/Coursework 2/epm102_p2_dataset/img_train.jpg"

    train_bboxes = [
        [(609, 28), (305, 232), (666, 654), (933, 341)],  # Controller
        [(266, 85), (247, 353), (69, 345), (87, 45)],    # Tablets
        [(605, 483), (602, 699), (402, 712), (395, 484)],  # Airpods
        [(327, 430), (327, 711), (50, 712), (66, 421)],  # Coin
    ]

    object_names = ["Controller", "Tablets", "Airpods", "Coin"]
    query_objects = [[2], [1, 2], [0, 1, 2, 3]]

    detector = BRISKDetector()
    matcher = BFHammingMatcher()

    processor = ObjectDetectionProcessor(detector, matcher)
    processor.process_query_images(query_images, train_image, train_bboxes, object_names, query_objects)