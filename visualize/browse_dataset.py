import pygame
import numpy as np
import argparse

import visualize.draw as draw

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

WIDTH, HEIGHT = 600, 600
FPS = 60

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 24)


def draw_image(image, x, y, size, is_rgb):
    if is_rgb:
        rows, cols = 32, 32
        cell_w = size / cols
        cell_h = size / rows
        for row in range(rows):
            for col in range(cols):
                R = int(image[row * cols + col] * 255)
                G = int(image[rows * cols + row * cols + col] * 255)
                B = int(image[2 * rows * cols + row * cols + col] * 255)
                pygame.draw.rect(screen, (R, G, B),
                                 pygame.Rect(x + col * cell_w, y + row * cell_h, cell_w + 1, cell_h + 1))
    else:
        rows, cols = 28, 28
        cell_w = size / cols
        cell_h = size / rows
        for row in range(rows):
            for col in range(cols):
                v = int(image[row * cols + col] * 255)
                pygame.draw.rect(screen, (v, v, v),
                                 pygame.Rect(x + col * cell_w, y + row * cell_h, cell_w + 1, cell_h + 1))


def browse(dataset, randomize):
    if dataset == "CIFAR10":
        import setup_datasets.CIFAR10 as DATA
        class_names = DATA.class_names
        is_rgb = True
    elif dataset == "MNIST":
        import setup_datasets.MNIST as DATA
        class_names = None
        is_rgb = False
    else:
        import setup_datasets.FASHION_MNIST as DATA
        class_names = DATA.class_names
        is_rgb = False

    x, y = DATA.x_test, DATA.y_test

    if randomize:
        idx = np.random.permutation(x.shape[0])
        x, y = x[idx], y[idx]

    index = 0
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT}))
    has_changed = True
    running = True

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    index = (index + 1) % len(x)
                    has_changed = True
                elif event.key == pygame.K_LEFT:
                    index = (index - 1) % len(x)
                    has_changed = True

                if has_changed:
                    screen.fill(WHITE)
                    draw.draw_input(x[index], WIDTH // 4, HEIGHT // 4, WIDTH // 2, is_rgb)

                    label = int(y[index])
                    label_str = f"{label}" if class_names is None else f"{label} - {class_names[label]}"
                    text = font.render(label_str, True, BLACK)
                    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 8))

                    pygame.display.update()
                    has_changed = False

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MNIST")
    parser.add_argument("--randomize", default="0")
    args = parser.parse_args()

    browse(args.dataset, args.randomize == "1")