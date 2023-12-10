# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import pyrealsense2 as rs
import numpy as np
import time

# ANSI colors
RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
BLUE = '\033[34m'
RESET = '\033[37m'

# initialise realsense2
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
 
frame_count = 0 # frame count to use in sampling

while True:
    start_time = time.time()

    # Store next frameset for later processing:
    frameset = pipeline.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    color_image = np.asanyarray(color_frame.get_data())
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # detect multiple faces
    if len(rects) > 1:
        print(f'{YELLOW}Multiple Faces Detected!{RESET}')

    '''
    elif aligned_depth_frame.get_distance(320, 240) < 0.2:
        print(f'{YELLOW}Too Close To The Camera!{RESET}')
    '''

    if len(rects) < 1:
        print(f'{BLUE}No Faces Detected!{RESET}')

    else:
        sample_limit = 10 # sampling frame rate

        # Define the dimensions of the 2D array
        rows = 9 # how many landmarks
        columns = sample_limit # how many data points

        # Initialize an empty 2D array with zeros
        if frame_count == 0:
            depth_list = [[0] * columns for _ in range(rows)]
        elif frame_count > sample_limit-1:
            frame_count = 0
        
        # print(depth_list)

        m_avg_left_ear = 0
        m_avg_right_ear = 0
        m_avg_nose = 0

        nose_to_left_ear_depth = 0
        nose_to_right_ear_depth = 0

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # print(shape)
        
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            landmark_count = 0

            for (x, y) in shape:
                try:
                    cv2.circle(color_image, (x, y), 2, (0, 255, 0), -1)
                    
                    choosen_landmarks = [1, 2, 3, 15, 16, 17, 30, 31, 34]
                    if landmark_count in choosen_landmarks:
                        depth_value = aligned_depth_frame.get_distance(x, y)
                        try:
                            if landmark_count == 1:
                                depth_list[0][frame_count] = depth_value
                            elif landmark_count == 2:
                                depth_list[1][frame_count] = depth_value
                            elif landmark_count == 3:
                                depth_list[2][frame_count] = depth_value
                            elif landmark_count == 15:
                                depth_list[3][frame_count] = depth_value
                            elif landmark_count == 16:
                                depth_list[4][frame_count] = depth_value
                            elif landmark_count == 17:
                                depth_list[5][frame_count] = depth_value
                            elif landmark_count == 30:
                                depth_list[6][frame_count] = depth_value
                            elif landmark_count == 31:
                                depth_list[7][frame_count] = depth_value
                            elif landmark_count == 34:
                                depth_list[8][frame_count] = depth_value

                        # depth_value = aligned_depth_frame.get_distance(x, y)
                        # depth_list.append(depth_value)
                        # print(f"Landmark:{landmark_count} X:{x} Y:{y} Depth: {depth_value}")
                        except IndexError:
                            print(f'{YELLOW}Index Error!{RESET}')
                    
                    landmark_count += 1

                    # print(len(depth_list))
                    # print(depth_list)
                    # print('==========================')

                    avg_depth_list = []

                    if frame_count == sample_limit-1:

                        for item in depth_list:
                            avg_value = np.average(item)
                            avg_depth_list.append(avg_value)

                        # print(avg_depth_list)
                        # print('===================================')
                        # print('===================================')

                        avg_left_ear = np.average(avg_depth_list[:3])
                        avg_right_ear = np.average(avg_depth_list[3:6])
                        avg_nose = np.average(avg_depth_list[6:9])

                        #print(f'Left Ear AVG:{avg_left_ear} Nose AVG:{avg_nose} Right Ear AVG:{avg_right_ear}')

                        nose_to_left_ear_depth = avg_left_ear - avg_nose
                        nose_to_right_ear_depth = avg_right_ear - avg_nose

                        # print(f'Nose to Right Ear:{nose_to_right_ear_depth} Nose to Left Ear:{nose_to_left_ear_depth}')
                        # print(f'Right Gap{nose_to_right_ear_depth} Left Gap{nose_to_left_ear_depth}')

                        if ((0.04 < nose_to_left_ear_depth) and (0.04 < nose_to_right_ear_depth)): #< 0.15 < 0.15
                            print(f"{GREEN}Human Detected!{RESET}")
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            # print(f"Elapsed time: {elapsed_time} seconds")
                        else:
                            print(f"{RED}Not a real human!{RESET}")

                except RuntimeError:
                    print(f'{YELLOW}Error! Landmarks coordinates out of range!{RESET}')

            frame_count += 1

    # show the output image with the face detections + facial landmarks
    # cv2.imshow("RGB", color_image)
    # cv2.imshow("Depth", colorized_depth)
    blend = cv2.addWeighted(color_image,0.7,colorized_depth,0.5,0)
    cv2.imshow("Blended", blend)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

# Cleanup:
pipeline.stop()
print("Stopped!")