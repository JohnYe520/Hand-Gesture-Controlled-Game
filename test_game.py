import pygame
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from PIL import Image
import numpy as np
import sys

# Testing the model with moving green cube using webcam 

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GESTURES = ['down', 'left', 'right', 'stop', 'up', 'zero']
CONF_THRESHOLD = 0.50

model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(GESTURES))

state_dict = torch.load("best_model_perclass.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("[INFO] Loaded best_model_perclass.pth")

# Transform the image to the same as the training data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 640, 480
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Game (Per-Class Model)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 26)

player = pygame.Rect(WIDTH // 2, HEIGHT // 2, 40, 40)
player_color = (0, 255, 0)
move_dist = 40

current_gesture = "none"
conf = 0.0

# Capture frame on demand
def capture_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam could not be opened.")
        return None
    
    print("[INFO] Webcam active. Pose gesture and press SPACE")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        cv2.imshow("Pose gesture, press SPACE to capture", frame)
        k = cv2.waitKey(1)

        if k == 32:  # SPACE
            captured = frame.copy()
            cv2.destroyAllWindows()
            cap.release()
            return captured
        
        if k == 27:  # ESC to cancel
            cv2.destroyAllWindows()
            cap.release()
            return None

# Predict the gesture from the frame
def predict(frame):
    global conf

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    conf = float(np.max(probs))
    idx = int(np.argmax(probs))

    if conf < CONF_THRESHOLD:
        return "none"
    return GESTURES[idx]

# Game loop
running = True

while running:
    clock.tick(30)
    win.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # SPACE → capture gesture
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            frame = capture_frame()
            if frame is not None:
                current_gesture = predict(frame)

                if current_gesture == "up":
                    player.y -= move_dist
                elif current_gesture == "down":
                    player.y += move_dist
                elif current_gesture == "left":
                    player.x -= move_dist
                elif current_gesture == "right":
                    player.x += move_dist
                elif current_gesture == "zero":
                    player.x = WIDTH // 2
                    player.y = HEIGHT // 2
                elif current_gesture == "stop":
                    pass  # pause or do nothing

                player.x = np.clip(player.x, 0, WIDTH - player.width)
                player.y = np.clip(player.y, 0, HEIGHT - player.height)

    pygame.draw.rect(win, player_color, player)

    win.blit(font.render(f"Gesture: {current_gesture.upper()}", True, (255,255,255)), (20,20))
    win.blit(font.render(f"Conf: {conf:.2f}", True, (180,180,180)), (20,55))
    win.blit(font.render("Press SPACE to capture gesture", True, (120,160,255)), (20, HEIGHT-40))

    pygame.display.flip()

pygame.quit()
sys.exit()
