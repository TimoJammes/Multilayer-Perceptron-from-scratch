import pygame

#region colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREY = (169, 169, 169)
FIRE = (252, 82, 0)
BROWN = (90, 56, 37)
LIGHT_BLUE = (137, 207, 240)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
PINK = (255, 192, 203)
CYAN = (0, 255, 255)
DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (144, 238, 144)
DARK_BLUE = (0, 0, 139)
LIGHT_GREY = (211, 211, 211)
DARK_GREY = (105, 105, 105)
MAGENTA = (255, 0, 255)
NAVY = (0, 0, 128)
TEAL = (0, 128, 128)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)
MAROON = (128, 0, 0)
OLIVE = (128, 128, 0)
INDIGO = (75, 0, 130)
TURQUOISE = (64, 224, 208)
#endregion

#region pygame vars
WIDTH, HEIGHT = 600, 600
AREA = (WIDTH, HEIGHT)
FPS = 30
mainloop = True
screen = pygame.display.set_mode(AREA)
clock = pygame.time.Clock()
#endregion

pygame.font.init()

font = pygame.font.SysFont('Arial', 15)


# initLoss = None

# lineColor = RED
train_loss_color = BLUE
train_avg_color = BLUE
test_loss_color = RED
test_avg_color = RED
loss_width = 2
avg_width = 1

horiz_border_size = 10
vert_border_size = 10


start_y = HEIGHT * 1/6

usable_width = WIDTH - horiz_border_size * 2
usable_height = HEIGHT - vert_border_size - start_y


last_train_loss_pos = -1
last_test_loss_pos = -1

max_train_loss = -1
max_test_loss = -1

init_train_avg = -1
init_test_avg = -1

last_train_avg_pos = -1
last_test_avg_pos = -1

train_loss_list = []
test_loss_list = []

has_quit = False

def setup():
    
    screen.fill(WHITE)
    
    pygame.draw.line(screen, GREY, (horiz_border_size, start_y), (horiz_border_size, HEIGHT-vert_border_size))
    pygame.draw.line(screen, GREY, (horiz_border_size, HEIGHT-vert_border_size), (WIDTH-horiz_border_size, HEIGHT-vert_border_size))
    
# def rescale(epochs):
    
#     setup()
#     for i in range(len(train_loss_list)-1):
        
#         train_loss_pos_1 = start_y + usable_height * (1 - train_loss_list[i][0]/max_train_loss)
#         train_loss_pos_2 = start_y + usable_height * (1 - train_loss_list[i+1][0]/max_train_loss)
        
#         x1 = horiz_border_size + (train_loss_list[i][1] / epochs) * usable_width
#         x2 = horiz_border_size + ((train_loss_list[i][1]+1) / epochs) * usable_width

#         pygame.draw.line(screen, train_line_color, (x1, train_loss_pos_1), (x2, train_loss_pos_2), lineWidth)

#         if len(test_loss_list) > 0:
#             test_loss_pos_1 = start_y + usable_height * (1 - test_loss_list[i][0]/max_test_loss)
#             test_loss_pos_2 = start_y + usable_height * (1 - test_loss_list[i+1][0]/max_test_loss)
            
#             # x1 = horiz_border_size + (test_loss_list[i][1] / epochs) * usable_width
#             # x2 = horiz_border_size + (test_loss_list[i+1][1] / epochs) * usable_width

#             pygame.draw.line(screen, test_line_color, (x1, test_loss_pos_1), (x2, test_loss_pos_2), lineWidth)

def update_screen(train_loss, test_loss, epochs, curr_epoch, train_avg, test_avg):
    
    global last_train_loss_pos, last_test_loss_pos, max_train_loss, max_test_loss, train_loss_list, test_loss_list, has_quit, start_y
    global init_test_avg, last_test_avg_pos, init_train_avg, last_train_avg_pos
    
    if has_quit:
        return
    
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            has_quit = True
            return
    
        elif event.type == pygame.WINDOWFOCUSLOST:
            pass
        elif event.type == pygame.WINDOWFOCUSGAINED:
            pass
    
    
    if max_train_loss is -1:
        
        x1 = horiz_border_size + ((curr_epoch - 1) / epochs) * usable_width
        # x2 = horiz_border_size + (epoch / epochs) * usable_width
        max_train_loss = train_loss
        last_train_loss_pos = start_y + usable_height * (1 - train_loss/max_train_loss)
        
        pygame.draw.line(screen, train_loss_color, (x1, start_y), (x1, start_y), loss_width)
        # train_loss_list.append((train_loss, 0))
        
        init_train_avg = train_avg
        last_train_avg_pos = start_y
        pygame.draw.line(screen, train_avg_color, (x1, start_y), (x1, start_y), avg_width)
        
        if test_loss is not -1:
            max_test_loss = test_loss
            last_test_loss_pos = start_y + usable_height * (1 - test_loss/max_test_loss)
            # test_loss_list.append((test_loss, 0))
            pygame.draw.line(screen, test_loss_color, (x1, start_y), (x1, start_y), loss_width)

            init_test_avg = test_avg
            last_test_avg_pos = start_y
            pygame.draw.line(screen, test_avg_color, (x1, start_y), (x1, start_y), avg_width)

        pygame.display.update()
        return
    
    # if test_loss is not None and test_loss > max_test_loss:# or test_loss > max_test_loss:
    #     # max_test_loss = max(max_test_loss, test_loss)
    #     max_test_loss = test_loss
    #     # start_y = vert_border_size * max_test_loss / test_loss_list[0][0]
        
    #     rescale(epochs)

    x1 = horiz_border_size + ((curr_epoch - 2) / epochs) * usable_width
    x2 = horiz_border_size + ((curr_epoch-1) / epochs) * usable_width
    # train_loss_list.append((train_loss, epoch))
    train_loss_pos = start_y + usable_height * (1 - train_loss/max_train_loss)
    
    
    pygame.draw.line(screen, train_loss_color, (x1, last_train_loss_pos), (x2, train_loss_pos), loss_width)
    last_train_loss_pos = train_loss_pos    
    
    train_avg_pos = start_y + usable_height * (train_avg-init_train_avg)/(1 - init_train_avg)
    
    pygame.draw.line(screen, train_avg_color, (x1, last_train_avg_pos), (x2, train_avg_pos), avg_width)
    last_train_avg_pos = train_avg_pos  
    
    if test_loss is not None:
        # test_loss_list.append((test_loss, epoch))
        test_loss_pos = start_y + usable_height * (1 - test_loss/max_test_loss)
        
        pygame.draw.line(screen, test_loss_color, (x1, last_test_loss_pos), (x2, test_loss_pos), loss_width)
        last_test_loss_pos = test_loss_pos    
        
        test_avg_pos = start_y + usable_height * (test_avg-init_test_avg)/(1 - init_test_avg)
        
        # print((test_avg-init_test_avg)/(100 - init_test_avg))
        pygame.draw.line(screen, test_avg_color, (x1, last_test_avg_pos), (x2, test_avg_pos), avg_width)
        last_test_avg_pos = test_avg_pos    
    
    pygame.display.update()
