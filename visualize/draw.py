import pygame

#region colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (169, 169, 169)
DARK_GREY = (105, 105, 105)



#region pygame vars
WIDTH, HEIGHT = 600, 600
AREA = (WIDTH, HEIGHT)
FPS = 60
mainloop = True
screen = pygame.display.set_mode(AREA)
clock = pygame.time.Clock()
#endregion

pygame.font.init()
font = pygame.font.SysFont("Arial", 28)
small_font = pygame.font.SysFont("Arial", 22)

def draw_input(features, startX, startY, size, RGB=False):

    if RGB:
        cols = int((features.size/3) ** (0.5)) #assumes NxNx3 square input
    else:
        cols = int(features.size ** (0.5)) #assumes NxN square input
    rows = cols
    
    cellWidth = size / cols
    cellHeight = size / rows
    
    for y in range(rows):
        for x in range(cols):
            
            if RGB:
                R = features[y*cols+x] * 255
                G = features[rows*cols+y*cols+x] * 255
                B = features[2*rows*cols+y*cols+x] * 255
                color = (R, G, B)
            else:
                color = [features[y*cols+x] * 255] * 3
                
            # print(color)
            # color = ((y*numCellsPerRow+x)/(numCellsPerRow**2)*255, 0, 0)
            # print(color)
            pygame.draw.rect(screen, color, pygame.Rect(startX + x*cellWidth, startY + y*cellHeight, cellWidth+1, cellHeight+1))
