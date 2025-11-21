import pygame
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from PIL import Image
import numpy as np
import sys
import time
import random
import os

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CLASSES = ['down', 'left', 'right', 'stop', 'up', 'zero']
DIR_GESTURES = ['up', 'down', 'left', 'right']
CONF_THRESHOLD = 0.50

model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(MODEL_CLASSES))

state_dict = torch.load("best_model_perclass.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("[INFO] Loaded best_model_perclass.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

GESTURE_SYMBOL = {
    'up': '↑',
    'down': '↓',
    'left': '←',
    'right': '→'
}

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Battle Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 28)
small_font = pygame.font.SysFont("Arial", 22)

# Load icons (REMOVED)
player_img = None
enemy_img = None

# Removed enemy_file logic
enemy_name = "ENEMY"

player_pos = (WIDTH // 3, HEIGHT // 2 + 30)
enemy_pos = (2 * WIDTH // 3, HEIGHT // 2 + 30)

HP_BAR_WIDTH = 220
HP_BAR_HEIGHT = 25
player_hp = 1.0
enemy_hp = 1.0

ROUND_TIME = 120
start_time = time.time()

current_gesture = "none"
conf = 0.0

SEQ_LENGTH = 4
current_sequence = []
sequence_index = 0
game_over = False
game_result = ""


# Helper functions
def draw_health_bars(surface, player_hp, enemy_hp):
    x_margin = 40
    y_bar = 30

    pygame.draw.rect(surface, (80, 80, 80),
                     (x_margin, y_bar, HP_BAR_WIDTH, HP_BAR_HEIGHT))
    pygame.draw.rect(surface, (80, 80, 80),
                     (WIDTH - x_margin - HP_BAR_WIDTH, y_bar,
                      HP_BAR_WIDTH, HP_BAR_HEIGHT))

    pygame.draw.rect(surface, (0, 200, 255),
                     (x_margin, y_bar, int(HP_BAR_WIDTH * player_hp), HP_BAR_HEIGHT))
    pygame.draw.rect(surface, (255, 80, 80),
                     (WIDTH - x_margin - HP_BAR_WIDTH, y_bar,
                      int(HP_BAR_WIDTH * enemy_hp), HP_BAR_HEIGHT))

    player_label = small_font.render("PLAYER HP", True, (255, 255, 255))
    surface.blit(player_label, (x_margin, y_bar - 24))

    enemy_label = small_font.render(f"{enemy_name}'s HP", True, (255, 255, 255))
    surface.blit(enemy_label, (WIDTH - x_margin - HP_BAR_WIDTH, y_bar - 24))


def draw_timer(surface, remaining_time):
    time_text = font.render(f"{int(remaining_time):02d}s", True, (255, 255, 255))
    surface.blit(time_text, (WIDTH // 2 - time_text.get_width() // 2, 70))


def draw_characters(surface):
    pygame.draw.circle(surface, (0, 150, 255), player_pos, 50, 6)

    enemy_rect = pygame.Rect(0, 0, 100, 100)
    enemy_rect.center = enemy_pos
    pygame.draw.rect(surface, (255, 80, 80), enemy_rect, 6)


def new_sequence():
    return [random.choice(DIR_GESTURES) for _ in range(SEQ_LENGTH)]


def draw_sequence(surface, sequence, current_idx):
    panel_height = 80
    panel_rect = pygame.Rect(40, HEIGHT - panel_height - 30, WIDTH - 80, panel_height)
    pygame.draw.rect(surface, (0, 80, 0), panel_rect, border_radius=10)
    pygame.draw.rect(surface, (0, 180, 0), panel_rect, 3, border_radius=10)

    inst = small_font.render(
        "Perform these gestures in order (press SPACE to capture):",
        True, (230, 255, 230)
    )
    surface.blit(inst, (panel_rect.x + 20, panel_rect.y + 8))

    gap = 80
    start_x = panel_rect.x + 60
    y = panel_rect.y + 40

    for i, g in enumerate(sequence):
        symbol = GESTURE_SYMBOL[g]
        color = (255, 255, 0) if i == current_idx else (255, 255, 255)
        text = font.render(symbol, True, color)
        surface.blit(text, (start_x + i * gap, y))


def capture_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam could not be opened.")
        return None

    print("[INFO] Pose gesture & press SPACE to capture.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        cv2.imshow("Press SPACE to capture, ESC to cancel", frame)
        k = cv2.waitKey(1)

        if k == 32:
            captured = frame.copy()
            cv2.destroyAllWindows()
            cap.release()
            return captured
        elif k == 27:
            cv2.destroyAllWindows()
            cap.release()
            return None


def predict_gesture(frame):
    global conf

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()

    conf = float(np.max(probs))
    idx = int(np.argmax(probs))

    if conf < CONF_THRESHOLD:
        return "none"

    gesture_name = MODEL_CLASSES[idx]
    return gesture_name if gesture_name in DIR_GESTURES else "none"


def apply_damage(success, player_hp, enemy_hp):
    dmg = 0.2
    if success:
        enemy_hp = max(0.0, enemy_hp - dmg)
    else:
        player_hp = max(0.0, player_hp - dmg)
    return player_hp, enemy_hp


# Initialize sequence
current_sequence = new_sequence()
sequence_index = 0

# Main loop
running = True
while running:
    dt = clock.tick(30)

    remaining_time = max(0.0, ROUND_TIME - (time.time() - start_time))

    if remaining_time <= 0 and not game_over:
        if player_hp > enemy_hp:
            game_result = "TIME UP – YOU WIN!"
        elif enemy_hp > player_hp:
            game_result = "TIME UP – YOU LOSE!"
        else:
            game_result = "TIME UP – DRAW!"
        game_over = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if not game_over and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                frame = capture_frame()

                if frame is not None:
                    predicted = predict_gesture(frame)
                    current_gesture = predicted
                    required = current_sequence[sequence_index]

                    print(f"[DEBUG] Required={required}, Predicted={predicted}, conf={conf:.2f}")

                    if predicted == required:
                        sequence_index += 1

                        if sequence_index >= SEQ_LENGTH:
                            player_hp, enemy_hp = apply_damage(True, player_hp, enemy_hp)
                            current_sequence = new_sequence()
                            sequence_index = 0

                    elif predicted != "none":
                        print("[INFO] Wrong gesture, try again!")

                if not game_over and (player_hp <= 0 or enemy_hp <= 0):
                    if player_hp <= 0 and enemy_hp <= 0:
                        game_result = "BOTH KO – DRAW!"
                    elif enemy_hp <= 0:
                        game_result = "YOU WIN!"
                    else:
                        game_result = "YOU LOSE!"
                    game_over = True

    win.fill((10, 10, 20))

    draw_health_bars(win, player_hp, enemy_hp)
    draw_timer(win, remaining_time)
    draw_characters(win)
    draw_sequence(win, current_sequence, sequence_index)

    vs_text = font.render("VS", True, (255, 255, 255))
    win.blit(vs_text, (WIDTH // 2 - vs_text.get_width() // 2, 20))

    debug_text = small_font.render(
        f"Last Gesture: {current_gesture.upper()}  Conf: {conf:.2f}",
        True, (200, 200, 200)
    )
    win.blit(debug_text, (40, HEIGHT - 120))

    if game_over:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        win.blit(overlay, (0, 0))

        result_text = font.render(game_result, True, (255, 255, 255))
        info_text = small_font.render("Press ESC to quit.",
                                      True, (220, 220, 220))
        win.blit(result_text, (WIDTH // 2 - result_text.get_width() // 2,
                               HEIGHT // 2 - 30))
        win.blit(info_text, (WIDTH // 2 - info_text.get_width() // 2,
                             HEIGHT // 2 + 10))

    pygame.display.flip()

    if pygame.key.get_pressed()[pygame.K_ESCAPE]:
        running = False

pygame.quit()
sys.exit()
