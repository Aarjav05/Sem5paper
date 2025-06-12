import pygame
import sys
import math

# Initialise Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Stickman Game")

# Colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Stickman properties
stickman1_pos = [WIDTH // 2 - 100, HEIGHT // 2]  # Initial position for Stickman 1
stickman1_scale = 1
stickman1_angle = 0

stickman2_pos = [WIDTH // 2 + 100, HEIGHT // 2]  # Initial position for Stickman 2
stickman2_scale = 1
stickman2_angle = 0


# Define stickman parts as lines
def draw_stickman(surface, position, scale, angle):
    # Body coordinates
    head_radius = 10 * scale
    body_length = 40 * scale
    arm_length = 20 * scale
    leg_length = 30 * scale

    # Head
    pygame.draw.circle(surface, BLACK, position, int(head_radius))

    # Body
    body_end = (position[0], position[1] + int(body_length))
    pygame.draw.line(surface, BLACK, position, body_end, 2)

    # Arms
    left_arm_end = (
        position[0] - int(arm_length * math.cos(math.radians(angle))),
        position[1] + int(arm_length * math.sin(math.radians(angle)))
    )
    right_arm_end = (
        position[0] + int(arm_length * math.cos(math.radians(angle))),
        position[1] + int(arm_length * math.sin(math.radians(angle)))
    )
    pygame.draw.line(surface, BLACK, position, left_arm_end, 2)
    pygame.draw.line(surface, BLACK, position, right_arm_end, 2)

    # Legs
    left_leg_end = (
        position[0] - int(leg_length * math.cos(math.radians(angle))),
        position[1] + int(body_length + leg_length * math.sin(math.radians(angle)))
    )
    right_leg_end = (
        position[0] + int(leg_length * math.cos(math.radians(angle))),
        position[1] + int(body_length + leg_length * math.sin(math.radians(angle)))
    )
    pygame.draw.line(surface, BLACK, body_end, left_leg_end, 2)
    pygame.draw.line(surface, BLACK, body_end, right_leg_end, 2)


# Main game loop
clock = pygame.time.Clock()

while True:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Get pressed keys
    keys = pygame.key.get_pressed()

    # Stickman 1 Controls (WASD, Q/E for rotation, + / - for scaling)
    if keys[pygame.K_a]:  # Move left
        stickman1_pos[0] -= 5
    if keys[pygame.K_d]:  # Move right
        stickman1_pos[0] += 5
    if keys[pygame.K_w]:  # Move up
        stickman1_pos[1] -= 5
    if keys[pygame.K_s]:  # Move down
        stickman1_pos[1] += 5
    if keys[pygame.K_q]:  # Rotate left
        stickman1_angle += 1
    if keys[pygame.K_e]:  # Rotate right
        stickman1_angle -= 1
    if keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]:  # Scale up
        stickman1_scale += 0.01
    if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:  # Scale down
        stickman1_scale = max(0.1, stickman1_scale - 0.01)

    # Stickman 2 Controls (Arrow Keys, O/P for rotation, [/] for scaling)
    if keys[pygame.K_LEFT]:  # Move left
        stickman2_pos[0] -= 5
    if keys[pygame.K_RIGHT]:  # Move right
        stickman2_pos[0] += 5
    if keys[pygame.K_UP]:  # Move up
        stickman2_pos[1] -= 5
    if keys[pygame.K_DOWN]:  # Move down
        stickman2_pos[1] += 5
    if keys[pygame.K_o]:  # Rotate left
        stickman2_angle += 1
    if keys[pygame.K_p]:  # Rotate right
        stickman2_angle -= 1
    if keys[pygame.K_LEFTBRACKET]:  # Scale down
        stickman2_scale = max(0.1, stickman2_scale - 0.01)
    if keys[pygame.K_RIGHTBRACKET]:  # Scale up
        stickman2_scale += 0.01

    # Draw stickmen
    draw_stickman(screen, stickman1_pos, stickman1_scale, stickman1_angle)
    draw_stickman(screen, stickman2_pos, stickman2_scale, stickman2_angle)

    pygame.display.flip()
    clock.tick(60)
