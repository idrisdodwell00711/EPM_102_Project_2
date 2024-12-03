import cv2 
import numpy as np 

# Takes a training img and testing img as inputs and outputs the object matches between the two.
class ObjectDetector:
    def __init__(self, detector_type, matcher_type, img1, img2):
        self.detector_type = detector_type
        if detector_type == "sift":
            self.detector = cv2.SIFT_create(1000, 3, 0.09)
        elif detector_type == "orb":
            self.detector = cv2.ORB_create(nfeatures=500)
        else:
            raise ValueError("Detector type unknown.")
        
        self.img1 = img1
        self.img2 = img2
        
        gray_img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        gray_img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        
        keypoints1, descriptors1 = self.detector.detectAndCompute(gray_img1, None)
        keypoints2, descriptors2 = self.detector.detectAndCompute(gray_img2, None)
        
        if self.detector_type == "sift":
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)  # Use NORM_L2 for SIFT
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Use NORM_HAMMING for ORB
            
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        # Sort matches by distance
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        
        # Storing coordinates of points corresponding to the matches found in both the images
        src_pts = []
        dst_pts = []
    
        for Match in good_matches:
            src_pts.append(keypoints1[Match[0].queryIdx].pt)
            dst_pts.append(keypoints2[Match[0].trainIdx].pt)

        # Compute homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
# img1 = cv2.imread( './img_query_1.jpg')

# cv2.imshow('img1', img1 )

# cv2.waitKey(0)
# cv2.destroyAllWindows()
