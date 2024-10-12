import numpy as np
import cv2
import matplotlib.pyplot as plt


class SIFTMatcher:
    def __init__(self, max_features=1000, contrast_threshold=0.05, edge_threshold=20):
        self.max_features = max_features
        self.detector = cv2.SIFT_create(max_features, contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold)

    def extract_sift_features(self, rgb_image):
        """Extract SIFT features from an RGB image."""
        grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(grayscale_image, None)
        return keypoints[:self.max_features], descriptors[:self.max_features]

    def load_images(self, image_path_1, image_path_2):
        """Load images from specified paths."""
        self.image_1 = cv2.imread(image_path_1)
        self.image_2 = cv2.imread(image_path_2)

        self.keypoints_1, self.descriptors_1 = self.extract_sift_features(self.image_1)
        self.keypoints_2, self.descriptors_2 = self.extract_sift_features(self.image_2)

    def match_features(self):
        """Match features between the two images."""
        brute_force_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        opencv_matches = brute_force_matcher.match(self.descriptors_1, self.descriptors_2)
        self.initial_matches = np.array([[match.queryIdx, match.trainIdx] for match in opencv_matches])

    def find_fundamental_matrix(self):
        """Find the fundamental matrix using RANSAC."""
        keypoints_1_positions = np.array([kp.pt for kp in self.keypoints_1])
        keypoints_2_positions = np.array([kp.pt for kp in self.keypoints_2])

        self.fundamental_matrix, self.inlier_mask = cv2.findFundamentalMat(
            keypoints_1_positions[self.initial_matches[:, 0]],
            keypoints_2_positions[self.initial_matches[:, 1]],
            cv2.USAC_MAGSAC,
            ransacReprojThreshold=0.25,
            confidence=0.99999,
            maxIters=10000
        )
        self.inlier_mask = self.inlier_mask.ravel().astype(bool)

    def draw_matches(self, max_matches=None):
        """Draw matches between the two images."""
        concatenated_image = cv2.hconcat([self.image_1, self.image_2])

        if max_matches is not None:
            self.initial_matches = self.initial_matches[:max_matches]

        for i, match in enumerate(self.initial_matches):
            point_1 = tuple(np.round(self.keypoints_1[match[0]].pt).astype(int))
            point_2 = tuple(np.round(self.keypoints_2[match[1]].pt).astype(int) + np.array([self.image_1.shape[1], 0]))

            line_color = (0, 255, 0) if self.inlier_mask[i] else (255, 0, 0)

            cv2.circle(concatenated_image, point_1, 5, (255, 0, 0), -1)
            cv2.circle(concatenated_image, point_2, 5, (255, 0, 0), -1)
            cv2.line(concatenated_image, point_1, point_2, line_color, 2)

        plt.figure(figsize=(10, 6))
        plt.imshow(concatenated_image)
        plt.axis('off')
        plt.title(f'Matches: {len(self.initial_matches)}')
        plt.show()

    def plot_images_with_keypoints(self):
        """Plot images with detected keypoints."""
        image_1_with_keypoints = cv2.drawKeypoints(self.image_1, self.keypoints_1, outImage=None,
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        image_2_with_keypoints = cv2.drawKeypoints(self.image_2, self.keypoints_2, outImage=None,
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].imshow(image_1_with_keypoints)
        axes[0].axis('off')
        axes[0].set_title('Image 1 Keypoints')

        axes[1].imshow(image_2_with_keypoints)
        axes[1].axis('off')
        axes[1].set_title('Image 2 Keypoints')

        plt.tight_layout()
        plt.show()

    def process_images(self, image_path_1, image_path_2, max_matches=20):
        """Load images, match features, find the fundamental matrix, and visualize results."""
        self.load_images(image_path_1, image_path_2)
        self.match_features()
        self.find_fundamental_matrix()
        self.plot_images_with_keypoints()
        self.draw_matches(max_matches)