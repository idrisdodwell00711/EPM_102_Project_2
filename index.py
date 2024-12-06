import cv2 
import numpy as np
from matplotlib import pyplot as plt

# Takes a training img and testing img as inputs and outputs the object matches between the two.
class ObjectDetector:
    def __init__(self, detector_type, img1, img2):
        self.detector_type = detector_type
        if detector_type == "sift":
            self.detector = cv2.SIFT_create(1000, 3, 0.09)
        elif detector_type == "orb":
            self.detector = cv2.ORB_create(nfeatures=5000)
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
        if len(good_matches)> 10:
            for Match in good_matches:
                 src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                 dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

            # Compute homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h,w = img1.shape[:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,H)
            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        else:
             print( "Not enough matches are found - {}/{}".format(len(good_matches), 10) )
             matchesMask = None
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        img3 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,good_matches,None,**draw_params)
        plt.imshow(img3, 'gray'),plt.show()
img2 = cv2.imread( './img_query_1.jpg')
img1 = cv2.imread( './img_train.jpg')

detect = ObjectDetector(detector_type='sift', img1=img1, img2=img2)

# cv2.imshow('img1', img1 )

# cv2.waitKey(0)
# cv2.destroyAllWindows()
