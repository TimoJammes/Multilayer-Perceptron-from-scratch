from collections import defaultdict
import numpy as np
from PIL import Image

import pygame

import network.neuralNet as NN

def to_grid(x, y, start_x, start_y, cell_size):
    
    return (x-start_x) // cell_size, (y-start_y) // cell_size

def center_points(dic):

    center_of_mass = np.array([0, 0])
    mass_sum = 0
    
    for p in dic:
        
        center_of_mass = center_of_mass + np.array(p) * dic[p]
        mass_sum += dic[p]
    
    center_of_mass = center_of_mass / mass_sum

    center_of_mass = np.round(center_of_mass)

    dist = np.array([13, 13]) - center_of_mass

    # newPoints = deepcopy(points)
    new_points = defaultdict(lambda: 0)

    for p in dic:
        new_coord = np.array(p) + dist
        
        if new_coord[0] >= 0 and new_coord[0] < 28 and new_coord[1] >= 0 and new_coord[1] < 28:
            new_points[(new_coord[0], new_coord[1])] = dic[p]

    return new_points

def MNISTIFY(arr):
    arr = np.array(arr, dtype=np.float32)  # shape (28,28)
    
    # 1. Find bounding box
    ys, xs = np.nonzero(arr)
    if len(xs) == 0: 
        return np.zeros((28, 28))
    
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    cropped = arr[min_y:max_y+1, min_x:max_x+1]

    # 2. Resize to fit 20×20
    pil_img = Image.fromarray(cropped * 255)
    w, h = cropped.shape[::-1]
    scale = 20 / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 3. Paste into 28×28 canvas
    new_img = Image.new("L", (28, 28))
    new_img.paste(resized, ((28 - new_w)//2, (28 - new_h)//2))
    arr = np.array(new_img, dtype=np.float32)
    
    # 4. Center by center-of-mass
    cy, cx = np.array(np.nonzero(arr)).mean(axis=1)
    shift_x = int(round(13.5 - cx))
    shift_y = int(round(13.5 - cy))
    arr = np.roll(arr, shift_y, axis=0)
    arr = np.roll(arr, shift_x, axis=1)
    
    return arr / 255.0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
def user_input_test(net: NN.NeuralNet):
    
    import visualize.draw as draw
    
    grid_start_x = draw.WIDTH // 6
    grid_start_y = grid_start_x
    
    grid_size = draw.WIDTH - grid_start_x * 2
    
    cell_size = grid_size / 28
    
    points = defaultdict(lambda: 0)
    
    draw_border_rect = pygame.Rect(grid_start_x, grid_start_y, grid_size, grid_size)
    
    draw_background_color = draw.WHITE
    draw_color = draw.BLACK
    right_x = draw.WIDTH - grid_start_x+10
    y_start = 40
    line_spacing = 35
    
    prediction_rect = pygame.Rect(right_x, y_start, draw.WIDTH-right_x, draw.HEIGHT-y_start)
    
    MNIST_grid_start_x = draw.WIDTH *7 // 8
    MNIST_grid_start_y = draw.HEIGHT - draw.HEIGHT // 8
    MNIST_grid_size = draw.WIDTH // 8
    MNIST_draw_border_rect = draw.pygame.Rect(MNIST_grid_start_x, MNIST_grid_start_y, MNIST_grid_size, MNIST_grid_size)

    # Initialize font
    pygame.font.init()
    # font = pygame.font.SysFont("Arial", 28)       # main text
    small_font = pygame.font.SysFont("Arial", 22) # smaller text
    
    
    testing = True
    
    drawing = False
    
    brush_radius = 15
    r2 = brush_radius**2
    
    grid = np.zeros((grid_size, grid_size))
    
    draw.screen.fill(draw.GREY)
    draw.pygame.draw.rect(draw.screen, draw.DARK_GREY, draw_border_rect, 2)
    draw.pygame.draw.rect(draw.screen, draw_background_color, draw_border_rect)
    
    for i in range(10):
        # Highlight the predicted class
        color = (0, 0, 0)
        
        txt = small_font.render(f"{i}: {0:.3f}", True, color)
        draw.screen.blit(txt, (right_x, y_start + 2*line_spacing + i * line_spacing))
    draw.pygame.display.update()
    
    was_drawing_last_frame = False
    last_draw_pos: np.ndarray = np.array([0, 0]) #to avoid possibly unbound errors
    
    while testing:
        draw.clock.tick(draw.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                testing = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    
                    grid = np.zeros((grid_size, grid_size))
                    draw.screen.fill(draw.GREY)
                    draw.pygame.draw.rect(draw.screen, draw.DARK_GREY, draw_border_rect, 2)
                    draw.pygame.draw.rect(draw.screen, draw_background_color, draw_border_rect)

                    for i in range(10):
                        # Highlight the predicted class
                        color = (0, 0, 0)
                        
                        txt = small_font.render(f"{i}: {0:.3f}", True, color)
                        draw.screen.blit(txt, (right_x, y_start + 2*line_spacing + i * line_spacing))
                        
                    draw.pygame.display.update()
                    
                elif event.key == pygame.K_RETURN:
                    drawing = not(drawing)
            
        if draw.pygame.mouse.get_pressed()[0]:
            
            
            x, y = draw.pygame.mouse.get_pos()
                    
            if x >= grid_start_x and x < grid_start_x + grid_size:
                if y >= grid_start_y and y < grid_start_y + grid_size:
                    
                    pos = np.array([x, y])
                    
                    sampling = [pos]
                    if was_drawing_last_frame: 
                        dist = np.linalg.norm(pos-last_draw_pos)
                        if dist > brush_radius:
                            
                            sampling = np.linspace(last_draw_pos, pos, int(dist//brush_radius+3))
                            
                            
                    for x, y in sampling:
                        x, y = int(x), int(y)
                        for dx in range(-brush_radius, brush_radius + 1):
                            dy_max = int((r2 - dx * dx) ** 0.5)
                            for dy in range(-dy_max, dy_max + 1):
                                if x+dx >= grid_start_x and x+dx < grid_start_x + grid_size:
                                    if y+dy >= grid_start_y and y+dy < grid_start_y + grid_size:
                                        grid[y+dy-grid_start_y, x+dx-grid_start_x] = 1
                                        draw.pygame.draw.rect(draw.screen, draw_color, draw.pygame.Rect(x+dx, y+dy, 1, 1))
                    
                    
                    MNIST_grid_values = MNISTIFY(grid)
                    MNIST_grid_values = MNIST_grid_values.reshape(28*28)
                    
                    features = np.array(MNIST_grid_values)
                    
                        
                
                
                    output = net.inference_feedforward(features)
                    
                    print_output = [round(float(a[0]), 3) for a in output]

                    # draw.screen.fill(draw.GREY)
                    draw.pygame.draw.rect(draw.screen, draw.GREY, pygame.Rect(0, 0, draw.WIDTH, grid_start_y))
                    # draw.pygame.draw.rect(draw.screen, drawBackgroundColor, drawBorderRect)
                    
                    draw.pygame.draw.rect(draw.screen, draw.GREY, prediction_rect)
                    
                    draw.pygame.draw.rect(draw.screen, draw.DARK_GREY, MNIST_draw_border_rect, 2)                
                    draw.draw_input(MNIST_grid_values, MNIST_grid_start_x, MNIST_grid_start_y, MNIST_grid_size, RGB=False)

                    target_text = draw.font.render(f"Predicting: {np.argmax(print_output)}", True, (0, 0, 0))
                    draw.screen.blit(target_text, (draw.WIDTH//2-target_text.get_size()[0]//2, y_start))
                    #Draw 10 output probabilities
                    for i, prob in enumerate(print_output):
                        # Highlight the predicted class
                        color = (0, 0, 0)
                        if i == int(np.argmax(print_output)):
                            color = (0, 0, 255)  # blue for prediction
                        
                        txt = small_font.render(f"{i}: {prob:.3f}", True, color)
                        draw.screen.blit(txt, (right_x, y_start + 2*line_spacing + i * line_spacing))

                    #Update display
                    pygame.display.update()
            
            was_drawing_last_frame = True
            last_draw_pos = np.array([x, y])
        
        else:
            was_drawing_last_frame = False
        
    pygame.quit()



if __name__ == "__main__":

    import argparse

    from testing.import_params import NN_from_params
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", default=None)
    parser.add_argument("--params_folder", default="FINAL_PARAMS")
    args = parser.parse_args()

    net = NN_from_params("MNIST", args.params_folder)

    user_input_test(net)