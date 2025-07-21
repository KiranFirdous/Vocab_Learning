#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 14:56:52 2025

@author: jan
"""

import numpy as np
from math import isnan
import time
import psyPhysConfig as config
from random import choice as randomChoice
if not hasattr(config,'maxResponseTime'):
    config.maxResponseTime=15 # max interval between start spout lick and response in seconds
# import os
# clSource = '/home/colliculus/behaviourBoxes/software'
# if not os.path.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
#     clSource = '/home/pi'  # <-- qingjie: change this path to what works on the RPi
# if not os.path.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
#     clSource = 'c:/users/colliculus'
# if not os.path.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
#     clSource = 'c:/jan/behavbox'
# if not os.path.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
#     clSource = 'd:/behavbox'
# if not os.path.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
#     raise Exception('No valid path to ratCageProgramsV2 and 3 libraries')
# from sys import path as syspath
# from sys import stdout
# syspath.append(clSource+'/ratCageProgramsV3')
# syspath.append(clSource+'/ratCageProgramsV2')

import listScheduler2 as sc
import PicsWithSounds as sl
from glob import glob
successStim=sl.pictureAndSoundStimulus(ID='Success')
successStim.stimParams['Foil1_ID']='Success'
successStim.stimParams['Foil2_ID']='Success'
successStim.ready()
failStim=sl.pictureAndSoundStimulus(ID='Fail')
failStim.stimParams['Foil1_ID']='Fail'
failStim.stimParams['Foil2_ID']='Fail'
failStim.ready()
import os
import pandas as pd
from random import shuffle

order=[0,1,2]

def dblog(logstr):
    print(logstr)
    
def myIsNan(x):
    if type(x)==str:
        return False
    return isnan(x)    
    
def readCSVfiles(flist):
    """
    Reads multiple CSV files, concatenates them, and indexes by 'timeStamp'.
    
    Parameters:
    flist (list): List of CSV filenames to read
    
    Returns:
    pandas.DataFrame: Combined DataFrame indexed by 'timeStamp'
    """
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Read each file and append to the list
    for fname in flist:
        try:
            df = pd.read_csv(fname)
            dfs.append(df)
            dblog(f"Read {fname}")
        except Exception as e:
            dblog(f"Error reading {fname}: {str(e)}")
    
    # Concatenate all DataFrames if any were read
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Set 'timeStamp' as index
        if 'timeStamp' in combined_df.columns:
            combined_df.set_index('timeStamp', inplace=True)
            # Sort by index if it's a datetime
            try:
                combined_df.sort_index(inplace=True)
            except:
                pass
        else:
            dblog("Warning: 'timeStamp' column not found - returning with default index")
        
        return combined_df
    else:
        dblog("No valid data was read from any files")
        return pd.DataFrame()  # Return empty DataFrame if no files were read
    
def correctInARow(trialHistory, ID):
    """
    Work out most recent correct in a row score for stimulus with ID in trialHistory
    """
    if len(trialHistory) == 0:
        return 0
    # Filter and sort trials for the specific ID
    id_trials = trialHistory[trialHistory['ID'] == ID].sort_values('timeStamp', ascending=False)
    if len(id_trials)==0:
        # no record of trialling this ID, hence no correct responses so far
        return 0
    
    # Find the position of the first False (if any)
    false_positions = id_trials['correct'].eq(False).idxmax()
    
    # If no False values exist, return the count of all trials
    if not id_trials['correct'].eq(False).any():
        return len(id_trials)
    
    # Return the count of True values before the first False
    return id_trials.loc[:false_positions, 'correct'].sum()    



        

#%% 
class SRschedule: # the SR (spaced repetition) schedule object picks stimulus items for a given participant, 
                # taking into account what they need to review and what they still have to learn.
                
    def __init__(self, dataHandler):  
        self.stimVarParams= ['ID', 'Foil1_ID','Foil2_ID','imagePos', 'Foil1_Pos','Foil2_Pos']
        self.dataHandler=dataHandler
        self.readPreviousDataForParticipant()
        self.tripletToWorkOn=[]
        self.populateListsOfItemsToWorkOn()
        self.properties={} # this is a kind of working memory where we keep track of how many presentations / correct answers a given item has had in this session
        self.chooseCurrentItems()
        # self.targets=['Apple','','']
        # self.stimIndex=0
                
    def status(self, ID):
        """
        What is the current "status" of ID in trialHistory?
        Here status refers to how many foils the stimulus was presented with.
        """
        if len(self.trialHistory) == 0:
            return 0
        # Filter and sort trials for the specific ID
        id_trials = self.trialHistory[self.trialHistory['ID'] == ID].sort_values('timeStamp', ascending=False)
        if len(id_trials)==0:
            # no record of trialling this ID, hence status is zero
            return 0
        lastTrial=id_trials.iloc[0]
        stat=0
        if not myIsNan(lastTrial.Foil1_ID):
            stat=1
        if not myIsNan(lastTrial.Foil2_ID):
            stat=2
        return stat
    
    def retriveItemsToReview(self):
        oneHour=3600 # seconds
        hist=self.reviewHistory
        if len(hist) > 0:
            hist['dueDate']=hist.lastReviewedOn+(hist.reviewInHours-1)*oneHour
            hist['dueNow']=hist.dueDate < time.time()
            self.possibleReviewItems=list(hist[hist.dueNow==True].item)
        else:
            self.possibleReviewItems=[]

    def populateListsOfItemsToWorkOn(self):
        """
        Decide which items to train in this session. 
        Look through trial history for items that need reviewing.
        And pick a few new items. 
        """
        self.retriveItemsToReview()
        self.possibleNewItems=[ x for x in sl.availableStimTypes if x not in list(self.reviewHistory.item)]
        shuffle(self.possibleNewItems)
        numberOfNewItemsRemaining=len(self.possibleNewItems)
        if numberOfNewItemsRemaining < 10:
            dblog(f'### Note: only {numberOfNewItemsRemaining} possible new items to learn remaining!')
        newItemsToAdd=min(numberOfNewItemsRemaining,6)
        # self.ReviewItemsToWorkOn=self.prioritizeReviewItems()
        shuffle(self.possibleReviewItems)
        self.ReviewItemsToWorkOn=self.possibleReviewItems
        dblog(f'Selected items to review: {self.ReviewItemsToWorkOn}')
        self.NewItemsToWorkOn=self.possibleNewItems[:newItemsToAdd]
        dblog(f'Selected new items to work with: {self.NewItemsToWorkOn}\n')

    # def prioritizeReviewItems(self):
    #     """
    #     Prioritize items for review depending on "status":
    #         review items of status 0 (no foils on last attempt) are top priority
    #         review items of status 1 are next highest priority
    #         review items of status 2 are a priority only of they have a low % correct over the last few sessions or if they haven't been tested in a long time'

    #     Returns list of priority items to review
    #     """
    #     if len(self.possibleReviewItems)==0:
    #         return []
    #     status0items=[]
    #     status1items=[]
    #     oldItems=[]
    #     for idx, item in enumerate(self.possibleReviewItems):
    #         itemStatus=self.status(item)
    #         if itemStatus==0:
    #             dblog(f'scheduling item {item} for high priority review because it is of status 0.')
    #             status0items.append(item)
    #         elif itemStatus==1:
    #             dblog(f'scheduling item {item} for medium priority review because it is of status 1.')
    #             status1items.append(item)
    #         elif self.reviewOnGroundsOfAge(item):
    #             oldItems.append(item)
    #     return status0items+status1items+oldItems
        
    # def reviewOnGroundsOfAge(self,item):
    #     """
    #     XXX not yet implemented!
    #     """
    #     reviewThisItem=False
    #     if reviewThisItem:
    #          dblog(f'scheduling item {item} for low priority review because XXX.')
    #     return reviewThisItem

    def readPreviousDataForParticipant(self):
        datadir=os.path.dirname(self.dataHandler.filename)
        previousDataFiles=glob(datadir+f'/{self.dataHandler.participantID}_*.csv')
        dblog(f'found {len(previousDataFiles)} previous files for {self.dataHandler.participantID}')
        self.trialHistory=readCSVfiles(previousDataFiles)
        # create a separate review history to keep track of which items need to be reviewed when
        self.reviewHistoryFile=datadir+f'/review_history_{self.dataHandler.participantID}.csv'
        if os.path.exists(self.reviewHistoryFile):
            self.reviewHistory=pd.read_csv(self.reviewHistoryFile)
        else:
            self.reviewHistory=pd.DataFrame(columns=['item', 'reviewInHours', 'lastReviewedOn'])
        # we let the datahandler take care of saving the review history alongside the trial data
        self.dataHandler.reviewHistoryFile=self.reviewHistoryFile
        self.dataHandler.reviewHistory=self.reviewHistory
   
    def chooseCurrentItems(self):
        # the purpose of this function is to pick a triplet of stimuli to work with 
        # from review and new items.
        # First, check if we need to refresh our lists of items to work on
        if len(self.ReviewItemsToWorkOn) + len(self.NewItemsToWorkOn) == 0: 
            self.populateListsOfItemsToWorkOn()
        while len(self.tripletToWorkOn) < 3:
            # XXX at this point we may want to randomly choose between review and new items according to some 
            # weighted probablilities. For now we simply choose review items first.
            if len(self.ReviewItemsToWorkOn) > 0:
                nextItem=self.ReviewItemsToWorkOn.pop(0)
            else:
                nextItem=self.NewItemsToWorkOn.pop(0)
            # make sure properties are initiatlised
            if not nextItem in self.properties.keys():
                self.properties[nextItem]= {'status': self.status(nextItem),
                  'correctInARow' : correctInARow(self.trialHistory,nextItem),
                  'presentedInARow': 0}
            self.tripletToWorkOn.append(nextItem)
            
        dblog(f'\n*** next working on items "{self.tripletToWorkOn}".\n')
        
        self.currentItem=self.tripletToWorkOn[-1]
            
    
    def processLastTrial(self,lastTrial):
        """
        This function adds the last trial response to the history and 
        decides whether or how the currentItem needs to change based on recent responses.
        It also updates review history as needed.
        """
        # add the lastTrial to the trialHistory
        self.trialHistory=pd.concat([self.trialHistory, pd.DataFrame([lastTrial])],ignore_index=True)
        # make sure there is an entry for the presented item in the review history
        if (len(self.reviewHistory) == 0) or (sum(self.reviewHistory.item==self.currentItem)==0):
              self.reviewHistory.loc[len(self.reviewHistory)]={
                  'item':self.currentItem,
                  'reviewInHours':0,
                  'lastReviewedOn': time.time()}
        else:
             self.reviewHistory.loc[self.reviewHistory.item==self.currentItem,'lastReviewedOn']=time.time()
        # update parameters and change status of item or choose new item as required.
        dblog('--------------------------------------------------')
        self.properties[self.currentItem]['presentedInARow']+=1
        dblog(f'Item  {self.currentItem} has been seen {self.properties[self.currentItem]["presentedInARow"]} times.')
        if lastTrial['correct']:
            # because we had a correct response we randomly pick the next stimulus in the triplet
            shuffle(self.tripletToWorkOn)
            self.properties[self.currentItem]['correctInARow']+=1
            dblog(f"{self.properties[self.currentItem]['correctInARow']} correct responses in a row.")
            if self.properties[self.currentItem]['correctInARow'] > 2:
                # time to promote the current item to the next status
                if self.properties[self.currentItem]['status']<2:
                    self.properties[self.currentItem]['status']+=1
                    self.properties[self.currentItem]['correctInARow']=0
                    dblog(f'* Promoting {self.currentItem} to status {self.properties[self.currentItem]["status"]}')
                else:
                    # multiple correct at top status: we are done with this item for now
                    # dblog(f'Done training item {self.currentItem} for now')
                    self.weAreDoneWithcurrentItem()
        else: # incorrect response
            self.properties[self.currentItem]['correctInARow']=0
            dblog('incorrect response.')
            if self.properties[self.currentItem]['status']>0:
                self.properties[self.currentItem]['status']-=1
                dblog(f'Demoting {self.currentItem} to status {self.properties[self.currentItem]["status"]}')
            # any incorrect response means item should be reviewed ASAP
            self.reviewHistory.loc[self.reviewHistory.item==self.currentItem].reviewInHours=0
        dblog('')
        
    def weAreDoneWithcurrentItem(self):
        nextReview=\
           max(self.reviewHistory.loc[self.reviewHistory.item==self.currentItem].reviewInHours.iloc[0]*2, 24)
        self.reviewHistory.loc[self.reviewHistory.item==self.currentItem,'reviewInHours']=nextReview
        dblog(f'*** Item {self.currentItem} done. Scheduled for review in {nextReview} hours.')
        if self.currentItem in self.ReviewItemsToWorkOn:
            self.ReviewItemsToWorkOn.remove(self.currentItem)
        if self.currentItem in self.NewItemsToWorkOn:
            self.NewItemsToWorkOn.remove(self.currentItem)
        self.tripletToWorkOn.remove(self.currentItem)
        self.chooseCurrentItems()
  
    def paramNames(self):
        return self.stimVarParams

    def currentParams(self):
        return self.params
 
    def nextParams(self):
        """
        This function is called by the scheduler at the start of each trial.
        It returns the stimulus parameters to be presented in the upcoming trial. 
        The currentItem is initialized in __init__ and updated in processLastTrial based on previous responses.
        Here, this function only neds to compose the parameters for the currentItem, choosing appropriate foils depending on status
        and deciding the positioning (order) of item and foils
        """
        self.selectTargets()
        shuffle(order)
        self.params=self.targets+order
        return self.params
    
    def selectTargets(self):
        self.currentItem=self.tripletToWorkOn[-1]
        x2=None
        x3=None
        stat=self.properties[self.currentItem]['status']
        if stat>0:
            x2=randomChoice(sl.availableStimTypes)
            while x2==self.currentItem:
                x2=randomChoice(sl.availableStimTypes)
        if stat>1:
            x3=randomChoice(sl.availableStimTypes)
            while (x3==self.currentItem) or (x3==x2):
                x3=randomChoice(sl.availableStimTypes)
        self.targets=[self.currentItem, x2, x3]

    def finished(self):
        return False

#%%
""" 
adapt listScheduler to the fact that there is no start spout but there are three choice spouts:
    - waitForStart does not wait for detector
    - 
"""
class vocabScheduler(sc.scheduler):
    
    def __init__(self, detectors,stimulus,dataHandler):
        super().__init__(detectors,stimulus,dataHandler,SRschedule(dataHandler))
        self.absoluteTimestamps=True

    def dblog(self, logstr):
        dblog(logstr)
        
    def waitForStart(self):
        # successStim.hide()
        # listen to detectors and wait for a START signal
        self.broadCastStatus('waitForStart')
        # check if a minimum ISI is required and if it has elapsed
        if hasattr(self,'minimumISI'):
            timeElapsedSinceLastStart=time.time()-self.lastStart
            if timeElapsedSinceLastStart < self.minimumISI:
                self.dblog('Pausing for {:3.5} s for minimum ISI to elapse.'.format(self.minimumISI-timeElapsedSinceLastStart))
                time.sleep(self.minimumISI-timeElapsedSinceLastStart)
        failStim.hide()
        return self.presentTrial
    
    def handleCenterReward(self):
        pass # no center rewards for starting
        
    def chooseNextTrial(self):
        # XXX this needs to become a lot more sophisticated
        self.broadCastStatus('chooseNextTrial')
        self.nextParams=self.schedule.nextParams() 
        if (len(self.nextParams) == 0):
            return None # we are done, no more jobs to do
        else:
            self.getStimReadyForNextParam()
            return self.waitForStart        
        
    def punish(self):
        failStim.hide()
        failStim.show()
        failStim.play()
        self.dblog('Wrong response. Starting timeout then repeating trial')
        self.broadCastStatus('punish')
        time.sleep(0.5)
        return self.chooseNextTrial
        # return self.waitForStart # we will present same again, correction trial
        
    def reward(self):
        failStim.hide()
        successStim.show()
        successStim.play()
        self.dblog('Correct response. Giving reward then choosing next trial')
        self.broadCastStatus('reward')
        time.sleep(1)
        return self.chooseNextTrial
        
    def logCompletedStimPars(self,par):
        # at this point we are handed a copy of the just presented stimulus along with its response.
        # pass this to the schedule so that the response can be taken into account for the next stimulus choice.
        self.schedule.processLastTrial(par)
                
    
