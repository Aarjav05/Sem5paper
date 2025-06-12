import pygame
import time
import random
import os

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 150, 0)
RED = (255, 0, 0)

# Snake parameters
snake_block = 10
initial_speed = 15
speed_increase_length = 5  # Increase speed every 5 units of length

# Font styles
font_style = pygame.font.SysFont("bahnschrift", 25)
score_font = pygame.font.SysFont("comicsansms", 35)

# High Score File
high_score_file = "high_score.txt"


def read_high_score():
    if os.path.exists(high_score_file):
        with open(high_score_file, 'r') as file:
            return int(file.read().strip())
    return 0


def write_high_score(score):
    with open(high_score_file, 'w') as file:
        file.write(str(score))


def your_score(score):
    value = score_font.render("Score: " + str(score), True, BLACK)
    screen.blit(value, [0, 0])


def display_high_score(score):
    value = score_font.render("High Score: " + str(score), True, BLACK)
    screen.blit(value, [SCREEN_WIDTH - 200, 0])


def our_snake(snake_block, snake_list):
    for i, x in enumerate(snake_list):
        color = GREEN if i == len(snake_list) - 1 else DARK_GREEN  # Darker color for the body
        pygame.draw.rect(screen, color, [x[0], x[1], snake_block, snake_block])


def message(msg, color):
    mesg = font_style.render(msg, True, color)
    screen.blit(mesg, [SCREEN_WIDTH / 6, SCREEN_HEIGHT / 3])


def display_instructions():
    screen.fill(WHITE)
    instructions = [
        "Welcome to Snake Game!",
        "Control the snake using the arrow keys.",
        "Eat the red food to grow.",
        "Avoid hitting the walls or yourself.",
        "Press C to play again after losing.",
        "Press Q to quit the game."
    ]

    y_offset = 50
    for line in instructions:
        msg = font_style.render(line, True, BLACK)
        screen.blit(msg, [SCREEN_WIDTH / 8, y_offset])
        y_offset += 30

    pygame.display.update()
    time.sleep(3)  # Display instructions for 3 seconds


def draw_food(foodx, foody, pulse):
    if pulse:
        pygame.draw.circle(screen, RED, (int(foodx + snake_block / 2), int(foody + snake_block / 2)), 10)
    else:
        pygame.draw.rect(screen, RED, [foodx, foody, snake_block, snake_block])


def game_loop():
    game_over = False
    game_close = False

    x1 = SCREEN_WIDTH / 2
    y1 = SCREEN_HEIGHT / 2

    x1_change = 0
    y1_change = 0

    snake_List = []
    Length_of_snake = 1

    # Initial speed
    snake_speed = initial_speed

    # Food coordinates
    foodx = round(random.randrange(0, SCREEN_WIDTH - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, SCREEN_HEIGHT - snake_block) / 10.0) * 10.0

    pulse = True  # Pulse effect for food
    pulse_timer = 0

    # Load high score
    high_score = read_high_score()

    while not game_over:

        while game_close == True:
            screen.fill(WHITE)
            message("You Lost! Press C-Play Again or Q-Quit", RED)
            your_score(Length_of_snake - 1)
            display_high_score(high_score)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        game_loop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -snake_block
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = snake_block
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -snake_block
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = snake_block
                    x1_change = 0

        # Update snake position
        if x1 >= SCREEN_WIDTH or x1 < 0 or y1 >= SCREEN_HEIGHT or y1 < 0:
            game_close = True

        x1 += x1_change
        y1 += y1_change
        screen.fill(WHITE)

        # Draw food with pulse effect
        pulse_timer += 1
        pulse = (pulse_timer // 10) % 2 == 0  # Pulse effect every few frames
        draw_food(foodx, foody, pulse)

        snake_Head = []
        snake_Head.append(x1)
        snake_Head.append(y1)
        snake_List.append(snake_Head)
        if len(snake_List) > Length_of_snake:
            del snake_List[0]

        for x in snake_List[:-1]:
            if x == snake_Head:
                game_close = True

        our_snake(snake_block, snake_List)
        your_score(Length_of_snake - 1)
        display_high_score(high_score)

        # Increase speed based on length
        if Length_of_snake % speed_increase_length == 0 and Length_of_snake > 1:
            snake_speed += 1  # Increase speed

        # Control the speed of the snake
        pygame.time.Clock().tick(snake_speed)

        # Check if the snake eats the food
        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, SCREEN_WIDTH - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, SCREEN_HEIGHT - snake_block) / 10.0) * 10.0
            Length_of_snake += 1

            # Update high score if needed
            if Length_of_snake - 1 > high_score:
                high_score = Length_of_snake - 1
                write_high_score(high_score)

    pygame.quit()
    quit()


# Display instructions before starting the game
display_instructions()

# Start the game loop
game_loop()
