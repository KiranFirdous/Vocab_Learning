#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:59:08 2025

runs spaced repetition visual auditory association learning task.

Todo: peach sound canto not working

@author: jan
"""
import numpy as np
from os import path as ospath
import psyPhysConfig as config

# clSource = '/home/colliculus/behaviourBoxes/software'
# if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
#     clSource = '/home/jan/Nextcloud/Documents/jan/behavbox' 
# if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
#     clSource = 'c:/Nextcloud/Documents/jan/behavbox' 
# if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
#     clSource = 'c:/users/colliculus'
# if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
#     clSource = 'c:/jan/behavbox'
# if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
#     clSource = 'd:/behavbox'
# if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
#     raise Exception('No valid path to ratCageProgramsV2 and 3 libraries')
# from sys import path as syspath
# syspath.append(config.clSource+'/ratCageProgramsV3')


spoutOpenTimes= [0.013, 0.010, 0.014]# <-- This needs to be calibrated !!! left-middle-right
config.rewardDrops= 3
config.spoutOpenTimes= spoutOpenTimes
config.timeoutDuration=5 # s  

#%% load stimulus module
import PicsWithSounds as sl
stim=sl.pictureAndSoundStimulus()

#% load detectors
detectors=sl.detectors()

#%% load data saving module
import dataHandlingVocab
# automatically create file name and check directory to save data in
dataHandler=dataHandlingVocab.dataHandler()
dataHandler.createDatafile(configFile='SpacedRepetition.ini',keyword='SpacedRep')
dataHandler.onlineFeedback=True

#%% set up scheduler
# import listScheduler
import spacedRepetitionScheduler

# mySchedule=listScheduler.schedule()
# mySchedule.readScheduleFromFile('VocabSchedulePhase1.csv')
scheduler=spacedRepetitionScheduler.vocabScheduler(detectors,stim,dataHandler)

#%%
print('\n**** Ready. Place the rat in the box\n\n')
try: 
    scheduler.start()    
except KeyboardInterrupt: 
    pass

#% tidy up modules
dataHandler.done()
detectors.done()
scheduler.done()
stim.done()


