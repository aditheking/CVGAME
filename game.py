import cv2
import time
import random
import mediapipe as mp
import math
import numpy as np
import pygame

# Initialize Pygame mixer for sound effects
pygame.mixer.init()

# Load sound effects
slice_sound = pygame.mixer.Sound("audio/slice.mp3")
game_over_sound = pygame.mixer.Sound("audio/gameover.mp3")

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Game variables
curr_Frame = 0
prev_Frame = 0
next_Time_to_Spawn = 0
Speed = [0, 5]
Fruit_Size = 64  # Set the size of the fruit
Spawn_Rate = 1
Score = 0
Lives = 15
Difficulty_level = 1
game_Over = False
slicing_effects = []
Fruits = []# List to hold slicing effects

# Load and resize fruit images
fruit_images = {
    "apple": cv2.imread("icon/red-apple.png"),
    "banana": cv2.imread("icon/banana.png"),
    "orange": cv2.imread("icon/tangerine.png"),
}

# Resize fruit images to Fruit_Size
for key in fruit_images:
    fruit_images[key] = cv2.resize(fruit_images[key], (Fruit_Size, Fruit_Size))

def Spawn_Fruits():
    fruit_type = random.choice(list(fruit_images.keys()))
    fruit_image = fruit_images[fruit_type]
    random_x = random.randint(15, 600)
    Fruits.append({
        "Image": fruit_image,
        "Curr_position": [random_x, 440],
        "Size": fruit_image.shape[:2],  # Get the size of the image
        "Type": fruit_type
    })

def Fruit_Movement(Fruits, speed):
    global Lives
    fruits_to_remove = []  # List to collect fruits to remove
    for fruit in Fruits:
        # Update fruit position
        fruit["Curr_position"][0] += speed[0]
        fruit["Curr_position"][1] -= speed[1]

        # Get current position
        img_position = (int(fruit["Curr_position"][0]), int(fruit["Curr_position"][1]))

        # Check if the fruit is out of bounds
        if (img_position[1] < 0 or img_position[0] < 0 or
            img_position[0] + Fruit_Size > img.shape[1] or
            img_position[1] + Fruit_Size > img.shape[0]):
            if fruit["Curr_position"][1] < 20 or fruit["Curr_position"][0] > 650:
                Lives -= 1
                fruits_to_remove.append(fruit)  # Collect fruit to remove
            continue  # Skip drawing if out of bounds

        # Draw the fruit image
        img[img_position[1]:img_position[1] + Fruit_Size, img_position[0]:img_position[0] + Fruit_Size] = fruit["Image"]

    # Remove collected fruits
    for fruit in fruits_to_remove:
        Fruits.remove(fruit)

def distance(a, b):
    return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

def show_menu():
    while True:
        menu_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(menu_img, "FRUIT CUT GAME", (380, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(menu_img, "Press 'S' to Start", (420, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(menu_img, "Press 'Q' to Quit", (420, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("img", menu_img)
        key = cv2.waitKey(10)
        if key == ord('s'):
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

# Slicing Effect Class
class SlicingEffect:
    def __init__(self, position, fruit_type):
        self.position = position
        self.fruit_type = fruit_type
        self.split_offset = 20  # Offset for the split effect
        self.duration = 0.3  # Duration of the effect
        self.start_time = time.time()

    def draw(self, img):
        elapsed = time.time() - self.start_time
        if elapsed < self.duration:
            x, y = self.position
            # Draw two halves of the fruit
            fruit_half = fruit_images[self.fruit_type]
            left_half = cv2.flip(fruit_half[:, :Fruit_Size//2], 1)
            right_half = fruit_half[:, Fruit_Size//2:]

            # Move the halves apart
            left_x = x - self.split_offset * (elapsed / self.duration)
            right_x = x + self.split_offset * (elapsed / self.duration)

            img[y:y + Fruit_Size, int(left_x):int(left_x) + Fruit_Size//2] = left_half
            img[y:y + Fruit_Size, int(right_x):int(right_x) + Fruit_Size//2] = right_half

# Show the menu before starting the game
show_menu()

# Set up the video capture
cap = cv2.VideoCapture(0)

# Set full screen
cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("skipping frame")
        continue
    h, w, c = img.shape
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_drawing_styles.get_default_hand_landmarks_style(),
                                       mp_drawing_styles.get_default_hand_connections_style())
            for id, lm in enumerate(hand_landmarks.landmark):
                if id == 8:  # Index finger tip
                    index_pos = (int(lm.x * w), int(lm.y * h))
                    fruits_to_remove = []  # List to collect fruits to remove
                    for fruit in Fruits:
                        d = distance(index_pos, fruit["Curr_position"])
                        if d < Fruit_Size:
                            Score += 100
                            slice_sound.play()  # Play slicing sound
                            fruits_to_remove.append(fruit)  # Collect fruit to remove
                            slicing_effects.append(SlicingEffect(fruit["Curr_position"], fruit["Type"]))  # Add slicing effect

                    # Remove collected fruits
                    for fruit in fruits_to_remove:
                        Fruits[:] = [f for f in Fruits if f["Curr_position"] != fruit["Curr_position"]]

    # Draw slicing effects
    for effect in slicing_effects:
        effect.draw(img)

    # Clear effects that have finished
    slicing_effects = [effect for effect in slicing_effects if time.time() - effect.start_time < effect.duration]

    if Score % 1000 == 0 and Score != 0:
        Difficulty_level = int(Score / 1000) + 1
        Spawn_Rate = Difficulty_level * 4 / 5
        Speed[0] = Speed[0] * Difficulty_level
        Speed[1] = int(5 * Difficulty_level / 2)

    if Lives <= 0:
        game_Over = True

    curr_Frame = time.time()
    delta_Time = curr_Frame - prev_Frame
    FPS = int(1 / delta_Time)
    cv2.putText(img, "FPS : " + str(FPS), (int(w * 0.82), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 0), 2)
    cv2.putText(img, "Score: " + str(Score), (int(w * 0.35), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
    cv2.putText(img, "Level: " + str(Difficulty_level), (int(w * 0.01), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 150), 5)
    cv2.putText(img, "Lives remaining : " + str(Lives), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    prev_Frame = curr_Frame

    if not game_Over:
        if time.time() > next_Time_to_Spawn:
            Spawn_Fruits()
            next_Time_to_Spawn = time.time() + (1 / Spawn_Rate)
        Fruit_Movement(Fruits, Speed)
    else:
        game_over_sound.play()  # Play game over sound
        cv2.putText(img, "GAME OVER", (int(w * 0.1), int(h * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        Fruits.clear()

    cv2.imshow("img", img)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
