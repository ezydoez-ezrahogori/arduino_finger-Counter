import cv2
import mediapipe as mp
import numpy as np
import time
import serial



# Change this to your port

my_port = "use your port"




ser = serial.Serial(my_port, 9600, timeout=1)

# Flush the buffer of any additional information.
ser.flush()

#

# if serial communication has been established
if ser.name:
    # get the name of the port being used
    port = ser.name

    # print a status message
    print(f'Serial comms established on port: {port}')


# Define the colours
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

# to draw the landmarks
mp_draw = mp.solutions.drawing_utils

# Set the drawing specs.
# the defaults also look fine.
draw_specs = mp_draw.DrawingSpec(color=WHITE_COLOR, thickness=2, circle_radius=2)

mp_pose = mp.solutions.hands
pose = mp_pose.Hands(static_image_mode=False,
                     max_num_hands=1,
                     min_detection_confidence=0.85,
                     min_tracking_confidence=0.5)

# Create a video capture object


# Do this when using a Webcam
# Using webcam index 0
cap = cv2.VideoCapture(0)
video_width = 640
video_height = 480
cap.set(3, video_width)  # 3 is the id for width
cap.set(4, video_height)  # 4 is the id for height

# A video is just a sequence of images.
# Create a loop to loop through each image.

start_time = 0

while True:
    # 'success' is a boolean that indicates if the image was read.
    # 'image' is the image frame that was read.
    success, image = cap.read()

    # flip the image so that it's like looking in a mirror.
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB.
    # Note that we don't display this RGB image. We display
    # the original BGR image.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # process the image and store the results in an object
    results = pose.process(image_rgb)

    # draw a rectangle on the screen
    cv2.rectangle(image, (25, 130), (100, 200), BLUE_COLOR, cv2.FILLED)

    # This is how to create a black background and
    # a white background.
    h, w, c = image.shape
    black_image = np.zeros((h, w, c))
    white_image = np.zeros((h, w, c)) + 255

    # Choose to display the original image or show the hand landmarks
    # against a black or a white background.
    display_image = image
    # display_image = black_image
    # display_image = white_image

    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:

        # We can have multiple hands.
        # Therefore, we need to loop through the hands
        for hand_landmarks in results.multi_hand_landmarks:

            x_list = []
            y_list = []
            fingers_up = [0, 0, 0, 0, 0]

            # draw the lanmarks on the image as dots only
            # mp_draw.draw_landmarks(display_image, hand_landmarks)

            # draw dots and the connections between them
            mp_draw.draw_landmarks(display_image, hand_landmarks, mp_pose.HAND_CONNECTIONS,
                                   draw_specs, draw_specs)

            # Get each separate point.
            # We can put the ids into a list and return them.
            for id, lm in enumerate(hand_landmarks.landmark):
                # These values have been normalized from 0 to 1.
                # We need to convert them to coordinates.
                # print(lm)

                # convert to coordinates
                h, w, c = image.shape
                x = int(lm.x * w)
                y = int(lm.y * h)

                # print(id, x, y)

                # add the x and y coords to a list
                x_list.append(x)
                y_list.append(y)

            # print(y_list)


            # thumb
            if x_list[4] <= x_list[5]:
                fingers_up[0] = 1

            if y_list[8] <= y_list[6]:
                fingers_up[1] = 1

            if y_list[12] <= y_list[10]:
                fingers_up[2] = 1

            if y_list[16] <= y_list[14]:
                fingers_up[3] = 1

            if y_list[20] < y_list[18]:
                fingers_up[4] = 1

            # print the list
            print(fingers_up)

            # sum the list to get the number of fingers that are up
            num_fingers_up = sum(fingers_up)

            # Example:

            if ser.name:

                if num_fingers_up == 1:
                    # Send a communication to the Arduino.
                    # Note that we add a newline character because
                    # the Arduino code reads up to a newline character.
                    ser.write(b"one\n")

                if num_fingers_up == 2:
                    ser.write(b"two\n")

            # write the number of fingers onto the webccam image
            cv2.putText(image, str(num_fingers_up), (50, 180), cv2.FONT_HERSHEY_PLAIN, 3,
                        WHITE_COLOR, 3)

    # Get the frame rate
    current_time = time.time()
    fps = 1 / (current_time - start_time)
    start_time = current_time

    # Uncomment this line to display the frame rate on the image (top left corner)
    # cv2.putText(display_image, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255,255,255), 3)

    # Display the webcam video frame
    cv2.imshow('Video', display_image)

    # Press q on the keyboard to close the video
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break