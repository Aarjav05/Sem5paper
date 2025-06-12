import pygame
import math

# Initialize Pygame
pygame.init()

# Screen constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
CENTER_X = SCREEN_WIDTH // 2
CENTER_Y = SCREEN_HEIGHT // 2

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)  # Sun color
BLUE = (0, 0, 255)  # Earth color
RED = (255, 0, 0)  # Mars color
ORANGE = (255, 165, 0)  # Mercury color
GREEN = (0, 255, 0)  # Venus color
BROWN = (165, 42, 42)  # Jupiter color
GRAY = (169, 169, 169)  # Saturn color

# Screen setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2D Solar System Simulation with Elliptical Orbits")

# Clock to control frame rate
clock = pygame.time.Clock()

# Font for planet names
font = pygame.font.SysFont(None, 20)

# Planet data: (semi-major axis, semi-minor axis, color, orbital speed, size)
PLANETS = [
    {"name": "Mercury", "semi_major_axis": 50, "semi_minor_axis": 30, "color": ORANGE, "speed": 0.05, "size": 5},
    {"name": "Venus", "semi_major_axis": 90, "semi_minor_axis": 70, "color": GREEN, "speed": 0.03, "size": 8},
    {"name": "Earth", "semi_major_axis": 140, "semi_minor_axis": 110, "color": BLUE, "speed": 0.02, "size": 10},
    {"name": "Mars", "semi_major_axis": 190, "semi_minor_axis": 150, "color": RED, "speed": 0.015, "size": 7},
    {"name": "Jupiter", "semi_major_axis": 250, "semi_minor_axis": 200, "color": BROWN, "speed": 0.01, "size": 15},
    {"name": "Saturn", "semi_major_axis": 320, "semi_minor_axis": 260, "color": GRAY, "speed": 0.008, "size": 13},
]

# Zoom and speed controls
zoom_factor = 1
orbit_speed_factor = 1

# Trail data
MAX_TRAIL_LENGTH = 100
trails = {planet["name"]: [] for planet in PLANETS}

def draw_sun():
    pygame.draw.circle(screen, YELLOW, (CENTER_X, CENTER_Y), 30)

def draw_planet(planet, angle):
    # Calculate the planet's elliptical position using parametric equations
    semi_major_axis = planet["semi_major_axis"] * zoom_factor
    semi_minor_axis = planet["semi_minor_axis"] * zoom_factor

    x = CENTER_X + semi_major_axis * math.cos(angle)
    y = CENTER_Y + semi_minor_axis * math.sin(angle)

    # Draw the planet
    pygame.draw.circle(screen, planet["color"], (int(x), int(y)), planet["size"])

    # Draw planet's name
    name_text = font.render(planet["name"], True, WHITE)
    screen.blit(name_text, (int(x) + planet["size"], int(y)))

    # Save the position to the trail
    trails[planet["name"]].append((x, y))
    if len(trails[planet["name"]]) > MAX_TRAIL_LENGTH:
        trails[planet["name"]].pop(0)

    # Draw the trail
    for trail_pos in trails[planet["name"]]:
        pygame.draw.circle(screen, planet["color"], (int(trail_pos[0]), int(trail_pos[1])), 2)

def main():
    global zoom_factor, orbit_speed_factor

    # Initial angles for each planet
    planet_angles = [0 for _ in PLANETS]

    running = True
    while running:
        screen.fill(BLACK)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # User input for controlling zoom and speed
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            zoom_factor += 0.01  # Zoom in
        if keys[pygame.K_DOWN]:
            zoom_factor -= 0.01  # Zoom out
        if keys[pygame.K_RIGHT]:
            orbit_speed_factor += 0.01  # Increase orbit speed
        if keys[pygame.K_LEFT]:
            orbit_speed_factor -= 0.01  # Decrease orbit speed

        # Draw the sun
        draw_sun()

        # Update and draw planets with trails and names
        for i, planet in enumerate(PLANETS):
            planet_angles[i] += planet["speed"] * orbit_speed_factor  # Update the angle
            draw_planet(planet, planet_angles[i])

        # Update the display
        pygame.display.update()

        # Cap the frame rate
        clock.tick(60)

    # Quit Pygame
    pygame.quit()

if __name__ == "__main__":
    main()
