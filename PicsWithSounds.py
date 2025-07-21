#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:17:26 2025

@author: jan
"""
import scipy.io.wavfile as wav
import ClickTrainLibrary as cl
cl.soundPlayer=cl.pygameSoundHardware() # this will initialize pygame
import time
import pygame
from random import choice as randomChoice
from random import shuffle
from copy import deepcopy
import psyPhysConfig as config
from math import isnan
from glob import glob
from os import path
import numpy as np

# initialize the screen
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
imageSize=250 # height and width of images.
leftRightMargin=10

screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(WHITE)
#screen.fill(WHITE)
pygame.display.set_caption("Vocab Learning Task")

# Define the positions
positions = [
    (leftRightMargin,  HEIGHT // 2 - (imageSize/2)),  # Left position
    (WIDTH // 2 - (imageSize/2), HEIGHT // 2 - (imageSize/2)),  # Center position
    (WIDTH  - imageSize- leftRightMargin, HEIGHT // 2 - (imageSize/2))  # Right position
]
# positions = [
#     (WIDTH // 4 , HEIGHT // 2),  # Left position
#     (WIDTH // 2, HEIGHT // 2),  # Center position
#     (3 * WIDTH // 4, HEIGHT // 2)  # Right position
# ]
# availableStimTypes=[ path.basename(x) for x in glob('stimuli/pictures/*.jpeg')]        
# if len(availableStimTypes)<5:
#     raise Exception('Not enough image files for AV stimuli found in stimuli/pictures/')
# availableStimTypes=[ x[:-5] for x in availableStimTypes]
# availableStimTypes.remove('Fail')
# availableStimTypes.remove('Success')
availableStimTypes=['Apple',
 'Apricot',
 'Banana',
 'Cherry',
 'Fig',
 'Grape',
 'Grapefruit',
 'Kiwi',
 'Lemon',
 'Mango',
 'Peach',
 'Pineapple',
 'Plum',
 'Raspberry',
 'Strawberry',
 'Tomato',
 'Watermelon']

# self.availableStimTypes=availableStimTypes

cachedStimuli={}

def cacheStimulus(ID):
    newStim={}
    sound_file = 'stimuli/sounds/'+ ID + '.wav'
    # print("Caching WAV file:", sound_file)
    fs,signal=wav.read(sound_file)
    signal=signal/(2**15) # scale it to float in +/-1 range
    if len(signal.shape)==1:
        # mono signal. Make stereo
        signal=np.vstack((signal,signal)).T
    newStim["sounds"]=signal
    jpeg_file = 'stimuli/pictures/'+ ID + '.jpeg'
    # print("Caching JPEG file:", jpeg_file)
    img = pygame.image.load(jpeg_file)
    newStim["image"]= pygame.transform.scale(img, (imageSize, imageSize))
    cachedStimuli[ID]=newStim

def cachedImage(ID):
    if not ID in cachedStimuli.keys():
        cacheStimulus(ID)
    return cachedStimuli[ID]['image']

def cachedSounds(ID):
    return cachedStimuli[ID]['sounds']
    

# Default stimulus parameters

# picSound_defaultStimulusParams = {
#     'ID': 'banana',
#     'ABL (dB)': 80,
#     'imagePos': 1,
#     'Foil1_ID': "grape",
#     'Foil1_Pos': 2,
#     'Foil2_ID': "",
#     'Foil2_Pos': 0,
#     'loop': False
# }

picSound_defaultStimulusParams = {
    'ID': 'Apple',
    'imagePos': 1,
    'Foil1_ID': None,
    'Foil1_Pos': 2,
    'Foil2_ID': None,
    'Foil2_Pos': 0,
    'ABL (dB)': 80,
    'loop': False
}

anOrder=[0,1,2]

def myIsNan(x):
    if type(x)==str:
        return False
    return isnan(x)

class pictureAndSoundStimulus(cl.stimObject):
    """
    Audio-visual stimulus for vocab learning and testing. 
    self.stimParams['ID'] gives the name of the stimulus to present.
    The images and sound clips are kept in the dictionary "cachedStimuli".
    """
    def __init__(self, ID=None):
        super().__init__()
        self.stimParams = deepcopy(picSound_defaultStimulusParams)
        # self.readAvailableStimTypes()
        if ID:
            self.stimParams['ID'] = ID

    def selectRandomStimTypes(self, N=1, x1=None):
        if x1 is None:
            x1=randomChoice(availableStimTypes)
        x2=None
        x3=None
        if N>1:
            x2=randomChoice(availableStimTypes)
            while x2==x1:
                x2=randomChoice(availableStimTypes)
        if N>2:
            x3=randomChoice(availableStimTypes)
            while (x3==x1) or (x3==x2):
                x3=randomChoice(availableStimTypes)
        return [x1, x2, x3]

    def readyStimTypeSelection(self, selectedStim, selectedOrder=None):
        self.stimParams['ID']=selectedStim[0]
        self.stimParams['Foil1_ID']=selectedStim[1]
        self.stimParams['Foil2_ID']=selectedStim[2]
        if selectedOrder is None:
        # no order specified, so pick a random order
            shuffle(anOrder)
        selectedOrder = anOrder
        self.stimParams['imagePos']=selectedOrder[0]
        self.stimParams['Foil1_Pos']=selectedOrder[1]
        self.stimParams['Foil2_Pos']=selectedOrder[2]
        self.ready()
        return self.stimParams
            
    # def readAvailableStimTypes(self):            
    #     availableStimTypes=[ path.basename(x) for x in glob('stimuli/pictures/*.jpeg')]        
    #     if len(availableStimTypes)<5:
    #         raise Exception('Not enough image files for AV stimuli found in stimuli/pictures/')
    #     availableStimTypes=[ x[:-5] for x in availableStimTypes]
    #     availableStimTypes.remove('Fail')
    #     availableStimTypes.remove('Success')
    #     self.availableStimTypes=availableStimTypes

    def ready(self):
        if 'ID' not in self.stimParams or not self.stimParams['ID']:
            raise KeyError("Missing 'ID' in stimParams. Cannot load sound.")
        # load target from cache        
        self.image=cachedImage(self.stimParams['ID'])
        self.sounds=cachedSounds(self.stimParams['ID'])
        # load foils from cache
        if self.stimParams['Foil1_ID']:
            if not myIsNan(self.stimParams['Foil1_ID']):
                self.Foil1image=cachedImage(self.stimParams['Foil1_ID'])
        if self.stimParams['Foil2_ID']:
            if not myIsNan(self.stimParams['Foil2_ID']):
                self.Foil2image=cachedImage(self.stimParams['Foil2_ID'])
        super().ready()
        return self.isReady
            
    def show(self):
        position = self.stimParams['imagePos']
        if 0 <= position < len(positions):
            screen.blit(self.image, positions[position])
        if self.stimParams['Foil1_ID']:
            if not myIsNan(self.stimParams['Foil1_ID']):
                position = int(self.stimParams['Foil1_Pos'])
                if 0 <= position < len(positions):
                    screen.blit(self.Foil1image, positions[position])
        if self.stimParams['Foil2_ID']:
            if not myIsNan(self.stimParams['Foil2_ID']):
                position = int(self.stimParams['Foil2_Pos'])
                if 0 <= position < len(positions):
                    screen.blit(self.Foil2image, positions[position])
        pygame.display.flip()

    def hide(self):
        pygame.draw.rect(screen, WHITE, (0, positions[0][1], WIDTH, imageSize))
        pygame.display.flip()
        
    def correctResponse(self,aResponse):  
        if (aResponse=='LEFT') and (self.stimParams['imagePos']==0):
            return True
        if (aResponse=='MIDDLE') and (self.stimParams['imagePos']==1):
            return True
        if (aResponse=='RIGHT') and (self.stimParams['imagePos']==2):
            return True
        return False
    
    def statusChange(self,newStatus):
        # dblog(f"---> New status is {newStatus}")
        if not self.timeOutSound is None:
            if self.timeOutSound.playing():
                self.timeOutSound.stop()
        if newStatus=="start":
            self.ready()
            return
        if newStatus=="chooseNextTrial":
            # while self.stimulatorStatus()=='PLAYING':
            while self.playing():
                time.sleep(0.1) # wait for previous stimulus to complete
            self.hide()
            return
        if newStatus=="waitForStart":
            return
        if newStatus=="presentTrial":
            self.play()
            time.sleep(0.5)
            self.show()
            time.sleep(0.7)
            self.play()
            return
        if newStatus=="getResponse":
            return
        if newStatus=="reward":
            self.stop() # if the stimulus is still playing, stop it
            return
        if newStatus=="punish":
            self.stop() # if the stimulus is still playing, stop it
            if not self.timeOutSound is None: # if a timeout sound is defined, play it
                self.timeOutSound.ready()
                self.timeOutSound.play()
            return
        else:
            cl.dblog("stimulator module received unrecognized status: "+newStatus)
            return

#%% 
class responseDetector:

    def __init__(self,lickChannel,openTime):
        self.openTime=openTime
        
    def reward(self): # yet to be implemented
        pass
        
    def detectOn(self):
        # print('listening on GPIO pin {:d}'.format(self.lickChan))
        pass
        
    def detectOff(self):
        pass
    
    def display(self, color):
        pygame.draw.circle(screen, color , (self.x, self.y), 30)
        pygame.display.flip()            
     
                
#%%          
class detectors:

    def __init__(self):
        # set up instruction text box
        # if instruction =="":
        #     self.textsurface=None
        # else:
        #     pygame.font.init()
        #     myfont = pygame.font.SysFont('Comic Sans MS', 20)
        #     self.textsurface = myfont.render(instruction, False, (180, 180, 180))
        #     tsize=myfont.size(instruction)
        #     screen.blit(self.textsurface,((screenWidth-tsize[0])/2,200))
        # initialize array of detectors
        self.detectors=[]
        self.detectors.append(responseDetector(0,0.1))
        self.detectors.append(responseDetector(1,0.1))
        self.detectors.append(responseDetector(2,0.1))
        self.detectorTriggered=-1
        self.detectionCount=[0,0,0]
        pygame.display.flip()

    def statusChange(self,newStatus):
        if newStatus=="start":
            return
        if newStatus=="chooseNextTrial":
            return
        if newStatus=="waitForStart":
            self.flush() # we are getting ready for next stimulus. Clear out detector queue
            self.timeoutOff()
            # self.detectors[1].detectOn()
            return
        if newStatus=="presentTrial":
            self.reward_at(self.detectorTriggered)
            # self.detectors[0].detectOn()
            # self.detectors[2].detectOn()            
            self.flush() # we will be collecting responses soon. Clear out detector queue
            return
        if newStatus=="getResponse":
            return
        if newStatus=="reward":
            self.reward()
            return
        if newStatus=="punish":
            self.timeoutOn()
            time.sleep(config.timeoutDuration)
            return
        else:
            print("detectors module received unrecognized status: "+newStatus)
            return

    def responseDetected(self):
        self.detectorTriggered=-1
        responseCode='NONE'
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.detectorTriggered=100 # we use self.detectorTriggered 100 to signal abort
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.detectorTriggered= event.button-1
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.detectorTriggered=0
                if event.key == pygame.K_2:
                    self.detectorTriggered=1
                if event.key == pygame.K_3:
                    self.detectorTriggered=2
                if event.key == pygame.K_LEFT:
                    self.detectorTriggered=0
                if event.key == pygame.K_UP:
                    self.detectorTriggered=1
                if event.key == pygame.K_RIGHT:
                    self.detectorTriggered=2
                if event.key == pygame.K_q:
                    self.detectorTriggered=100
        if self.detectorTriggered > -1:
            if self.detectorTriggered == 0:
                responseCode='LEFT'
            if self.detectorTriggered == 1:
                responseCode='MIDDLE'
            if self.detectorTriggered == 2:
                responseCode='RIGHT'
            if self.detectorTriggered == 100:
                responseCode='QUIT'
            if self.detectorTriggered < 3:
                self.detectionCount[self.detectorTriggered]+=1
        return (responseCode,time.time())
    
    def flush(self):
        # flush any old response events that may be in the queue.
        response=self.responseDetected()
        while not response[0]=='NONE':
            response=self.responseDetected()

    def reward(self, numDrops = 0):
        if numDrops == 0:
            numDrops = config.rewardDrops
        print('Giving ', numDrops, " reward drops.")
        for rewards in range(numDrops): 
            self.reward_at(self.detectorTriggered)
            time.sleep(0.2)
             
    def timeoutOff(self):
        # resets detectors after timeout
        for aspout in self.detectors:
            aspout.detectOff()
        self.onTimeout=False

    def timeoutOn(self):
        # sets detectors into timeout mode
        # for aspout in self.detectors:
        #     aspout.display(timeoutColor)
        self.onTimeout=True
           
    def reward_at(self,thisSpout):
        if thisSpout < len(self.detectors):
            if thisSpout > -1:
                self.detectors[thisSpout].reward()
                
    def done(self):
        pygame.display.quit()


#%%
if __name__ == "__main__":
    
    stim=pictureAndSoundStimulus('banana')
    stim.ready()
    stim.play()
    # time.sleep(1)
    # stim.play()
    stim.show()
    time.sleep(1)
    stim.play()
    time.sleep(1)
    stim.hide()
    
      