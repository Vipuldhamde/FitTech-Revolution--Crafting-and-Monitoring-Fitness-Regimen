# Import necessary libraries
import cv2  # OpenCV for computer vision operations
import mediapipe as mp  # MediaPipe for pose estimation
import time  # To calculate FPS
import math  # To perform mathematical operations like angle and distance

# Define a class to detect and process human pose
class poseDetector():
    def __init__(self):
        # Initialize MediaPipe Pose class with custom parameters
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=False,            # Process video stream (not static images)
            model_complexity=0,                 # Lightweight model for faster performance
            smooth_landmarks=True,              # Smoothen landmark jitter
            enable_segmentation=False,          # Do not use segmentation mask
            smooth_segmentation=True,           # If enabled, smooth segmentation mask
            min_detection_confidence=0.5,       # Minimum confidence to detect person
            min_tracking_confidence=0.5         # Minimum confidence to track person
        )
        self.mpDraw = mp.solutions.drawing_utils  # Utility for drawing landmarks

    def findPose(self, img, draw=True):
        """
        Detect pose landmarks in an image and optionally draw them.

        Parameters:
            img: Input image (BGR format)
            draw: Boolean flag to draw pose landmarks

        Returns:
            img: Image with pose landmarks drawn (if draw=True)
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        self.results = self.pose.process(imgRGB)       # Perform pose detection

        # If landmarks are found, draw them on the image
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):
        """
        Extract (x, y) coordinates of all landmarks detected in the pose.

        Parameters:
            img: Image to extract landmarks from
            draw: Boolean flag to draw circles on landmarks

        Returns:
            lmList: List of landmarks with their ID and pixel coordinates
        """
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coords to pixel values
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)  # Draw green circle
        return self.lmList

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
        Calculate Euclidean distance between two landmarks.

        Parameters:
            p1, p2: Tuples of pixel coordinates (x, y)
            img: Image to draw the distance visualization (optional)
            color: Color of drawing
            scale: Scale factor for drawing elements

        Returns:
            length: Euclidean distance
            info: Tuple with coordinates used
            img: Image with drawings (if img is not None)
        """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Midpoint
        length = math.hypot(x2 - x1, y2 - y1)   # Euclidean distance
        info = (x1, y1, x2, y2, cx, cy)

        # Draw distance representation if image is provided
        if img is not None:
            cv2.circle(img, (x1, y1), 10, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (cx, cy), 10, color, cv2.FILLED)

        return length, info, img

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Calculate the angle between three points using their pixel coordinates.

        Parameters:
            img: Image to draw the angle visualization
            p1, p2, p3: Indices of the landmarks from lmList
            draw: Boolean flag to draw angle lines and text

        Returns:
            angle: Angle in degrees
        """
        # Get coordinates from the landmark list
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Use arctangent to find angle between three points
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw angle visualizations
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            for x, y in [(x1, y1), (x2, y2), (x3, y3)]:
                cv2.circle(img, (x, y), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x, y), 5, (255, 0, 0), 2)
            cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return angle

# Main function to run pose detection in webcam feed
def main():
    cap = cv2.VideoCapture(0)  # Start webcam capture
    pTime = 0                  # Previous timestamp for FPS calculation
    detector = poseDetector()  # Initialize pose detector

    while True:
        success, img = cap.read()              # Read frame from webcam
        img = cv2.resize(img, (900, 600))      # Resize for consistent display
        detector.findPose(img)                 # Detect pose
        lmList = detector.getPosition(img)     # Get landmark positions
        print(lmList)                          # Print landmark positions (for debugging)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        # Show the processed frame
        cv2.imshow("Image", img)
        cv2.waitKey(1)  # Wait for 1ms before moving to next frame

# Entry point of the script
if __name__ == '__main__':
    main()
