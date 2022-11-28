#Sources:
#For future reference: https://opensource.com/article/18/5/pygame-enemy ( How to expand on mobs in pygame)

#import libraries
import pygame as pg
from pygame.sprite import Sprite
import random

from random import randint

vec = pg.math.Vector2

#game layout
WIDTH = 360
HEIGHT = 450
FPS = 100

#player settings
PLAYER_GRAV = 0.9
HEALTH = 100



# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (100, 0 , 0)

#rules and objectives
def main1():
    x = input("Hey, what is your name?")
    print("Hey!!!", x,)
    print("Welcome to Dodge.IO")
    print("Your goal as the player is to dodge the incoming mobs")
    print("Each hit mob results in a 10 health deduction; You start with 100 health")
    print("This is still a developing version of the game")
    print("Good luck and have fun!!!")

main1()

#function for text
def draw_text(text, size, color, x, y):
        font_name = pg.font.match_font('arial')
        font = pg.font.Font(font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midtop = (x, y)
        screen.blit(text_surface, text_rect)
    
def colorbyte():
    return random.randint(0,255)


#class for player and its controls
class Player(Sprite):
    def __init__(self):
        Sprite.__init__(self)
        #self.image = (screen, WHITE, ((95, 120), (80, 180), (110, 180)))
        self.image = pg.Surface((35, 35))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        #self.rect.center = (WIDTH, HEIGHT/2)
        self.pos = vec(250, HEIGHT/2)
        self.vel = vec(0,0)
        self.acc = vec(0,0)

#controls of the player
    def controls(self):
        keys = pg.key.get_pressed()
        if keys[pg.K_d]:
            self.acc.x = 0.75
        if keys[pg.K_SPACE]:
            self.acc.x = -0.3
        if keys[pg.K_a]:
            self.acc.x = -0.75

    def update(self):
        self.acc = vec(0,PLAYER_GRAV)
        self.controls()
        # friction
        self.acc.x += self.vel.x * -0.1
        # self.acc.y += self.vel.y * -0.1
        self.vel += self.acc
        self.pos += self.vel + 0.5 * self.acc
        # self.rect.x += self.xvel
        # self.rect.y += self.yvel
        self.rect.midbottom = self.pos
        if HEALTH < 0: 
            print("you lost, the game is over")
            pg.quit


#class for the platforms and mobs
class Platform(Sprite):
    def __init__(self, x, y, w, h):
        Sprite.__init__(self)
        self.image = pg.Surface((w, h))
        self.image.fill(GREY)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

#class for the mobs in the game
class Mob(Sprite):
    def __init__(self, x, y, w, h, color):
        Sprite.__init__(self)
        self.image = pg.Surface((w, h))
        self.color = color
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
    #fuction to update mob position
    def update(self):
        x = randint(0, 2)
        self.rect.y += x


pg.init()
pg.mixer.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("My Game...")
clock = pg.time.Clock()

#grouping the classes
all_sprites = pg.sprite.Group()
all_plats = pg.sprite.Group()
mobs = pg.sprite.Group()


# instantiate classes
player = Player()
plat = Platform(0, 420, WIDTH*10, 10)

#creating mobs
x = 0
while x < 30:
    m = Mob(randint(0, WIDTH), randint(-300, 0), 30, 30, (colorbyte(),colorbyte(),colorbyte()))
    all_sprites.add(m)
    mobs.add(m)
    x +=1 

# add player to all sprites group
all_sprites.add(player)
all_plats.add(plat)

# add platform to all sprites group
all_sprites.add(plat)


#looping the game
running = True
while running:
    # keep the loop running using clock
    -clock.tick(FPS)

    hits = pg.sprite.spritecollide(player, all_plats, False)
    if hits:
        player.pos.y = hits[0].rect.top
        player.vel.y = 0


    mobhits = pg.sprite.spritecollide(player, mobs, True)
    if mobhits:
        screen.fill(RED)
        HEALTH -= 10
        


    for event in pg.event.get():
        # check for closed window
        if event.type == pg.QUIT:
            running = False

    if HEALTH <= 0:
        print('You have died')
        pg.quit()


    # update all sprites
    all_sprites.update()

    # draw the background screen
    screen.fill(BLACK)
    # draw text
    draw_text("HEALTH " + str(HEALTH), 22, WHITE, WIDTH / 2, HEIGHT / 24)
    # draw all sprites
    all_sprites.draw(screen)

    # buffer - after drawing everything, flip display
    pg.display.flip()

pg.quit()