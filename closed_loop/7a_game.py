import pygame
import time
import random
from pylsl import StreamInlet, resolve_stream
import os

pygame.init()


# -------------------INIT BCI STUFF-------------------
#streams = resolve_stream()
#inlet = StreamInlet(streams[0])
#sig_tot = ''
# ----------------------------------------------------

# colors
blue = (0,0,255)
green = (0,255,0)
red = (255,0,0)
white = (255, 255, 255)
black = (0, 0, 0)
bright_red = (95,0,0)
bright_green = (0,95,0)
gray  = (128, 128, 128)#(0,0,255)

# display coordinates
display_width = 1440
display_height = 900


os.environ['SDL_VIDEO_WINDOW_POS'] = "1440, 0"

# game display
#pygame.display.set_caption('THE DODGE GAME')
gameDisplay = pygame.display.set_mode((display_width,display_height), pygame.FULLSCREEN)
clock = pygame.time.Clock()

# Car
car_width = 40


# car method for the position of the car
def car(object_x, object_y, object_width, object_height, color):
    pygame.draw.rect(gameDisplay, color , [object_x, object_y, object_width, object_height])

# bot image method
def objects(object_x, object_y, object_width, object_height, color):
    pygame.draw.rect(gameDisplay, color , [object_x, object_y, object_width, object_height])

def objects2(object_x, object_y, object_width, object_height, color):
    pygame.draw.rect(gameDisplay, color , [object_x, object_y, object_width, object_height])

def objects3(object_x, object_y, object_width, object_height, color):
    pygame.draw.rect(gameDisplay, color , [object_x, object_y, object_width, object_height])

#Score Function
def objects_dodged(count):
    font = pygame.font.SysFont(None,40)
    text = font.render("Score : "+str(count),True,gray)
    gameDisplay.blit(text,(0, 0))

# Game Menu
def game_intro():
    intro = True
    while intro :
        for event in pygame.event.get():
            #print(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        gameDisplay.fill(black)
        largeText = pygame.font.Font('freesansbold.ttf',30)
        TextSurf,TextRect = text_objects("Ready to start?",largeText)
        TextRect.center =((display_width/2),(display_height/2))
        gameDisplay.blit(TextSurf,TextRect)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            game_loop()
        #button("Go!",100,450,100,60,white,gray,game_loop)
        #button("Quit",500,450,100,60,white,gray,game_quit)

        pygame.display.update()
        clock.tick(20)

# game loop
def game_loop():
    # position of car
    x = (display_width * 0.4)
    y = (display_height * 0.75)

    x_left = 0
    x_right = 0
    
    # obects parameters
    object_speed = 75
    object_height = 100
    object_width1, object_width2, object_width3 = 150, 150, 150
    car_height = 40
    car_width = 60
    object_start_x = random.randrange(object_width1//2, display_width - (object_width1//2))
    object_start_y = -900
    object2_start_x = random.randrange(object_width2//2, display_width - (object_width2//2))
    object2_start_y = -2250
    object3_start_x = random.randrange(object_width3//2, display_width - (object_width3//2))
    object3_start_y = -3375

    #Objects Dodged
    dodged = 0

    # Game Logic
    game_exit = False

    while not game_exit:
        for event in pygame.event.get():                            # For closing the game
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:                        # when a key is pressed down
                if event.key == pygame.K_LEFT:                      # left arrow key
                    x_left = -40
                if event.key == pygame.K_RIGHT:                     # right arrow key
                    x_right = +40

            if event.type == pygame.KEYUP:                          # when the key is released
                if event.key == pygame.K_LEFT:                      # left arrow key
                    x_left = 0
                if event.key == pygame.K_RIGHT:                     # right arrow key
                    x_right = 0
                

        # -----------ADD BCI STREAM CODE HERE-----------------
        #prediction = 0
        #sample, timestamp = inlet.pull_sample()
        # ----------------------------------------------------
        '''
        prediction = random.randrange(0,3)
        print(prediction)
        if prediction == 0:
            x_right = 0
        elif prediction == 1: #right arm
            x_right = 1
            x_left = 0
        elif prediction == 2:
            x_left = -1
            x_right = 0
        '''


        # change the position of the car
        x += x_left
        x += x_right

        if y-car_height < (object_start_y + object_height - object_speed):                            # object crash logic
            if (x-car_width > object_start_x and x < object_start_x + object_width1 or x+car_width > object_start_x and x + car_width < object_start_x+object_width1):
                crashed(dodged)
                game_quit(dodged)
        elif y-car_height < (object2_start_y  + object_height - object_speed):   
            if (x-car_width > object2_start_x and x < object2_start_x + object_width2 or x+car_width > object2_start_x and x + car_width < object2_start_x+object_width2):
                crashed(dodged)
                game_quit(dodged)  
        elif y-car_height < (object3_start_y  + object_height - object_speed):   
            if (x-car_width > object3_start_x and x < object3_start_x + object_width3 or x+car_width > object3_start_x and x + car_width < object3_start_x+object_width3):
                crashed(dodged)
                game_quit(dodged)

        # black background
        gameDisplay.fill(black)

        # objects location
        objects(object_start_x, object_start_y, object_width1, object_height, gray)
        objects2(object2_start_x, object2_start_y, object_width2, object_height, gray)
        objects3(object3_start_x, object3_start_y, object_width3, object_height, gray)
        object_start_y += object_speed
        object2_start_y += object_speed
        object3_start_y += object_speed

        car(x, y, car_width, car_height, gray)
        objects_dodged(dodged)
        
        if x > (display_width - car_width) or x < 0:                    # if the car goes outside the boundary
            crashed(dodged)
            game_quit(dodged)

        if object_start_y-object_height > y:                               # object repeats itself
            object_start_y = 0 - object_height
            object_start_x = random.randrange(object_width1//2, display_width - (object_width1)//2)
            dodged += 1
            object_width1 += 2

        if object2_start_y-object_height > y:                               # object repeats itself
            object2_start_x = random.randrange(object_width2//2, display_width - (object_width2)//2)
            object2_start_y = 0 - object_height
            dodged += 1
            object_width2 += 2

        if object3_start_y-object_height > y:                             # object repeats itself
            object3_start_x = random.randrange(object_width3//2, display_width - (object_width3)//2)
            object3_start_y = 0 - object_height
            dodged += 1
            object_width3 += 2


        pygame.display.update()
        clock.tick(2)

# Quit Game
def game_quit(dodged) :
    print(f'Total score: {dodged}')
    #TODO put dodged amount in csv of txt file or just note it down
    # note down experience as well: did you felt in control? was it frustrating? would you play a game with BCI instead of keyboard?
    pygame.quit()
    quit()


# crashed method
def crashed(dodged):
   crashed_message(f'Crashed! Total score: {dodged}')

# text objects
def text_objects(message, font):
    text = font.render(message, True, gray)
    return text, text.get_rect()


# crashed message
def crashed_message(message):
    large_text = pygame.font.Font('freesansbold.ttf', 75)
    text_surface, text_rectangle = text_objects(message, large_text)
    text_rectangle.center = ((display_width/2), (display_height/2))
    gameDisplay.blit(text_surface, text_rectangle)
    pygame.display.update()
    time.sleep(2)
    #game_loop()


game_intro()
game_loop()