import pygame
import random
import time
pygame.init()

WIDTH = 800
HEIGHT = 700

pygame.display.set_caption('Frog Catcher')
screen = pygame.display.set_mode((WIDTH, HEIGHT))
frog = pygame.image.load("assets/frog.png")
frog = pygame.transform.scale(frog, (200, 200))
eating_frog = pygame.image.load("assets/eating-frog.png")
eating_frog = pygame.transform.scale(eating_frog, (200, 200))

insects = [
            pygame.transform.rotate(
            pygame.transform.scale(
            pygame.image.load(f"assets/poochi ({i}).png"), (100,100)),180)
            for i in range(1, 10)
        ]
bg_frames = [
    pygame.transform.scale(
    pygame.image.load(f"assets/bg-frames/bg ({i}).jpg"),
    (WIDTH, HEIGHT))
    for i in range(1, 51)
]

GAME = True
frog_x = 300
insect_x = random.randint(-30, 630)
insect_y = -60
SCORE = 0
frame_i = 0

font = pygame.font.Font('freesansbold.ttf', 32)
text = font.render('SCORE: 0', True, (255,255,255))
textRect = text.get_rect()
textRect.center = (90,20)
insect = random.choice(insects)
eaten = False

while GAME:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            GAME = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                GAME = False
            if event.key == pygame.K_RIGHT:
                frog_x = frog_x + 30
                if frog_x > 770:
                    frog_x = -150
            if event.key == pygame.K_LEFT:
                frog_x = frog_x - 30
                if frog_x < -170:
                    frog_x = 750

    screen.fill((255, 255, 255))
    screen.blit(bg_frames[frame_i], (0, 0))
    frame_i += 1
    if frame_i >= len(bg_frames):
        frame_i = 0
    screen.blit(insect, (insect_x, insect_y))
    screen.blit(text, textRect)
    insect_y += 1

    if(insect_y > 727.5):
        insect_x = random.randint(-30, 630)
        insect_y = -60
        insect = random.choice(insects)
    if(insect_x + 80 > frog_x and insect_x < frog_x + 100 and insect_y + 10 > 470):
        SCORE += 1
        text = font.render(f'SCORE: {SCORE}', True, (255,255,255))
        textRect = text.get_rect()
        textRect.center = (90,20)
        insect_x = random.randint(-30, 630)
        insect_y = -60
        insect = random.choice(insects)
        eaten = True
    if eaten:
        screen.blit(eating_frog, (frog_x, 470))
        eaten = False
        pygame.display.flip()
        time.sleep(0.3)
    
    screen.blit(frog, (frog_x, 470))
    pygame.display.flip()