import pygame
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Arcade Car Racing Game")

# Clock to control frame rate
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (169, 169, 169)
GREEN = (0, 255, 0)

# Track Boundaries
track_image = pygame.image.load("track.png")  # Add your custom track image here
track_image = pygame.transform.scale(track_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

# Car class
class Car:
    def __init__(self, x, y, image_path):
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (40, 80))  # Car size
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.max_speed = 8
        self.acceleration = 0.2
        self.deceleration = 0.1
        self.friction = 0.05
        self.handling = 3

    def accelerate(self):
        if self.speed < self.max_speed:
            self.speed += self.acceleration

    def decelerate(self):
        if self.speed > 0:
            self.speed -= self.deceleration

    def apply_friction(self):
        if self.speed > 0:
            self.speed -= self.friction
        if self.speed < 0:
            self.speed = 0

    def turn_left(self):
        self.angle += self.handling

    def turn_right(self):
        self.angle -= self.handling

    def move(self):
        radian_angle = math.radians(self.angle)
        self.x += self.speed * math.cos(radian_angle)
        self.y -= self.speed * math.sin(radian_angle)

    def draw(self):
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        car_rect = rotated_image.get_rect(center=(self.x, self.y))
        screen.blit(rotated_image, car_rect.topleft)

# Initialize player car
player_car = Car(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100, "car.png")  # Replace "car.png" with your car image

# Lap Counter
laps = 0
num_laps = 5
lap_font = pygame.font.SysFont(None, 36)

# Track Boundaries
track_rect = pygame.Rect(50, 50, SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100)

def check_lap_completion(car, prev_x):
    global laps
    if prev_x > SCREEN_WIDTH // 2 and car.x <= SCREEN_WIDTH // 2:
        laps += 1

def draw_track():
    screen.blit(track_image, (0, 0))

# Main game loop
running = True
prev_x = player_car.x
while running:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw track
    draw_track()

    # Player input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        player_car.accelerate()
    if keys[pygame.K_DOWN]:
        player_car.decelerate()
    if keys[pygame.K_LEFT]:
        player_car.turn_left()
    if keys[pygame.K_RIGHT]:
        player_car.turn_right()

    # Move and draw the player's car
    player_car.move()
    player_car.apply_friction()  # Apply friction
    player_car.draw()

    # Check for lap completion
    check_lap_completion(player_car, prev_x)
    prev_x = player_car.x

    # Display lap counter
    lap_text = lap_font.render(f"Laps: {laps}/{num_laps}", True, BLACK)
    screen.blit(lap_text, (50, 50))

    # Check if race is finished
    if laps >= num_laps:
        finish_text = lap_font.render("Race Finished!", True, GREEN)
        screen.blit(finish_text, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2))

    # Update the display
    pygame.display.update()

    # Cap the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
