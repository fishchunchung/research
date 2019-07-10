import sys,os
import pygame
import numpy as np
import datetime
import random
import math
from pygame.locals import *
from pygame.color import THECOLORS
import time

GRID_DIM = (20,10)

class Window:
    def __init__(self, screen_dim):
        self.width_px = screen_dim[0]
        self.height_px = screen_dim[1]

        self.surface = pygame.display.set_mode(screen_dim)
        self.black_update()
        
    def update_title(self, title):
        pygame.display.set_caption(title)
        self.title = title
        
    def black_update(self):
        self.surface.fill(THECOLORS["black"])
        #pygame.display.flip()

    def get_user_input(self):

        for event in pygame.event.get():
            if (event.type == pygame.QUIT): 
                return 'quit'
            elif (event.type == pygame.KEYDOWN):
                if (event.key == K_ESCAPE):
                    return 'quit'
                elif (event.key==K_1):            
                    return '1'           
                elif (event.key==K_2):                          
                    return '2'
                elif (event.key==K_3):                          
                    return '3'
                elif (event.key==K_4):                          
                    return '4'
                elif (event.key==K_5):                          
                    return '5'
                elif(event.key==K_a):
                    return 'a'
                elif(event.key==K_s):
                    return 's'
                elif(event.key==K_d):
                    return 'd'
                elif(event.key==K_f):
                    return 'f'
                elif(event.key==K_g):
                    return 'g'
                else:
                    return "Nothing set up for this key."
            
            elif (event.type == pygame.KEYUP):
                pass
            
            elif (event.type == pygame.MOUSEBUTTONDOWN):
                pass
            
            elif (event.type == pygame.MOUSEBUTTONUP):
                pass

class Car:
    def __init__(self):
        self.pos=[0,0]
        self.angle=0
        self.length=50
        self.width=25
        self.color=pygame.Color(200,200,200,100)
                
class World:
    def __init__(self):
        self.name='reverse_park'
        self.cars=[]
        self.agent_car=Car()
        self.agent_car.pos=[50,50]
        self.agent_car.color=pygame.Color(100,100,200,100)
        self.cars.append(self.agent_car)
        self.delta = 20
        self.limit_width = 240
        self.limit_length = 320
        
        self.close_right = 180
        self.close_left = 100
        self.close_under = 50
        self.close_above = 130
        
        self.correct_right = 145
        self.correct_left = 135
        self.correct_under = 85
        self.correct_above = 95
        self.correct_angle = 0.05
        self.t=0
        #world=World()
        for i in range(5):
            if i!=2:
                car=Car()
                car.pos=[i*70,90]
                self.cars.append(car)
        
        
                    
    def step(self,action):
        action_space = ['no_action', 'fll', 'fl', 'f', 'fr', 'frr', 'bll', 'bl', 'b', 'br', 'brr']
        action = action_space[action]        
        
        #if action!='no_action':
        #    print(action)
        delta= self.delta
        delta_a=0.02
        if action=='fll':
            self.agent_car.angle-=2*delta_a
        if action=='fl':
            self.agent_car.angle-=delta_a
        if action=='f':
            pass

        if action=='fr':
            self.agent_car.angle+=delta_a
        if action=='frr':
            self.agent_car.angle+=2*delta_a

        if action=='bll':
            delta*=-1
            self.agent_car.angle-=2*delta_a
        if action=='bl':
            delta*=-1
            self.agent_car.angle-=delta_a
        if action=='b':
            delta*=-1

        if action=='br':
            delta*=-1
            self.agent_car.angle+=2*delta_a
        if action=='brr':
            delta*=-1
            self.agent_car.angle+=delta_a


        if action!='no_action':

            x1=self.agent_car.pos[0]
            y1=self.agent_car.pos[1]

            self.agent_car.pos[0]=x1+delta*math.cos(self.agent_car.angle)
            self.agent_car.pos[1]=y1+delta*math.sin(self.agent_car.angle)

        while self.agent_car.angle>2*math.pi:
            self.agent_car.angle-=2*math.pi

        if self.agent_car.pos[0]>=self.correct_left and self.agent_car.pos[0]<=self.correct_right \
            and self.agent_car.pos[1]>=self.correct_under and self.agent_car.pos[1]<=self.correct_above \
            and self.agent_car.angle>=-1*self.correct_angle and self.agent_car.angle<=self.correct_angle:
            self.agent_car.color=pygame.Color(100,200,100,100)
            reward = 100
            terminal = True            

        elif self.agent_car.pos[0]>= self.limit_length or self.agent_car.pos[0]<=0 \
            or self.agent_car.pos[1]>= self.limit_width or self.agent_car.pos[1]<=0:
            reward = -10
            terminal = True

        elif self.agent_car.pos[0]>=self.close_left and self.agent_car.pos[0]<=self.close_right \
                    and self.agent_car.pos[1]>=self.close_under and self.agent_car.pos[1]<=self.close_above:
            self.agent_car.color=pygame.Color(100,0,100,100)
            reward = 5
            terminal = False
            
        else:
            self.agent_car.color=pygame.Color(100,100,200,100)
            reward = -1
            terminal = False

        self.t+=1 
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        #pygame.display.flip()
        return image_data, reward, terminal

    def draw(self,screen):

        pygame.draw.rect(screen,pygame.Color(50,50,50,50), pygame.Rect(0,0,640,480))
        pygame.draw.rect(screen,pygame.Color(100, 0, 100, 100), pygame.Rect(self.close_left, self.close_under, self.close_right-self.close_left+50, 25+self.close_above-self.close_under), 4)
        pygame.draw.rect(screen,pygame.Color(100 , 200, 100, 100), [self.correct_left, self.correct_under, self.correct_right-self.correct_left+50, 25+self.correct_above-self.correct_under], 4)


        
        for car in self.cars:
            color=car.color
            x1=car.pos[0]
            y1=car.pos[1]
            
            length=car.length
            width=car.width
            angle=car.angle

            x2=x1+length*math.cos(angle)
            y2=y1+length*math.sin(angle)

            x3=x2+width*math.sin(-1*angle)
            y3=y2+width*math.cos(angle)

            x4=x1+width*math.sin(-1*angle)
            y4=y1+width*math.cos(angle)
            pygame.draw.polygon(screen,color,[(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
    
    def reset(self):
        self.__init__()


def main():

    world=World()
    
    pygame.init()

    window_size = (640, 480)

    window=Window(window_size)
    screen=window.surface
    window.update_title(__file__)

    myclock=pygame.time.Clock()
        
    framerate_limit=20

    t=0.0
    finished=False
    
    while not finished:
        
        window.surface.fill(THECOLORS["black"])
        dt=float(myclock.tick(framerate_limit) * 1e-3)
        
        command=window.get_user_input()


        if command=='reset':
            world=World()
            world.draw(screen)
        
        elif (command=='quit'):
            finished = True
            
        elif (command!=None):
            pass

        if command=='space' or command==None:
            action=0
            world.step(action)

        if command=='1':
            action=1
            world.step(action)

        if command=='2':
            action=2
            world.step(action)

        if command=='3':
            action=3
            world.step(action)

        if command=='4':
            action=4
            world.step(action)

        if command=='5':
            action=5
            world.step(action)

        if command=='a':
            action=6
            world.step(action)

        if command=='s':
            action=7
            world.step(action)

        if command=='d':
            action=8
            world.step(action)

        if command=='f':
            action=9
            world.step(action)

        if command=='g':
            action=10
            world.step(action)
        
        world.draw(screen) 
        t+=dt
        pygame.display.flip()


if __name__=="__main__":
    main()