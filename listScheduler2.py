# -*- coding: utf-8 -*-
"""
Created on August 6th 2018

@author: wschnupp

the "list scheduler" simply works through the list of stimuli
defined by stimVarParams and stimVarValues
for the number of times specified in stimListRepeats, 
shuffling the presentation order of the stimVarValues on each run

"""
import numpy as np
import time, sys
import psyPhysConfig as config
if not hasattr(config,'maxResponseTime'):
    config.maxResponseTime=15 # max interval between start spout lick and response in seconds
verbose=False
#%%

def myvstack(Y,newRow):
    if len(Y)==0:
        Y=np.array([newRow])
    else:
        if np.size(newRow)==0:
            return Y
        else:
            Y=np.vstack((Y,[newRow]))  
    return Y

def permute(A,B):
    Y=np.array([])
    A=np.array(A)
    B=np.array(B) 
    if A.size==0:
        return np.array([])
    for a in A:
        a=np.array(a)
        if B.size==0:
            Y=myvstack(Y,a)
        else:
            for b in B:
                b=np.array(b)
                newRow=np.append(a,b)
                Y=myvstack(Y,newRow)
    return Y
    
#%%
class schedule: # the schedule object creates and maintains a list of stim parameters to work through.
                #  by permuting the params given.
                # Alternatively the schedule can be read from a CSV file.
                
    def __init__(self, stimVarParams=[], loopScheduleWhenDone=False, shuffleOnLoop=False):  
        self.stimIndex=-1
        self.stimVarParams=stimVarParams
        self.loopScheduleWhenDone=loopScheduleWhenDone
        self.shuffleOnLoop=shuffleOnLoop
        # stimVarParams could be a file name for a CSV file instead of a list
        #  of variable paramaters.
        self.stimListRepeats=0
        if type(stimVarParams) == str:
            self.importFromCSV(stimVarParams)            
        
    def permute(self, valuesToPermute, stimListRepeats):       
        # build the stimVarValues by permuting the valuesToPermute lists
        self.stimListRepeats=stimListRepeats
        if type(valuesToPermute) == list:
            ndims=len(valuesToPermute)
        else:
            ndims=np.shape(valuesToPermute)[0]
        values=permute(valuesToPermute[0],[])
        for ii in range(ndims-1):
            values=permute(values,valuesToPermute[ii+1])
        # np.random.shuffle(values)
        self.stimVarValues=values
        for ii in range(stimListRepeats-1):
            # np.random.shuffle(values)
            self.stimVarValues= np.concatenate((self.stimVarValues,values))

    def importFromCSV(self, csvFileName):
        import pandas as pd
        tbl=pd.read_csv(csvFileName)
        self.stimVarParams=np.array(tbl.columns)
        self.stimVarValues=np.array(tbl)
        
    def repeatStimListNtimes(self, stimListRepeats):
        values=self.stimVarValues.copy()
        for ii in range(stimListRepeats-1):
            self.stimVarValues= np.concatenate((self.stimVarValues,values))        
            
    def shuffleStimList(self):
        #seedvalue = np.random.choice(10000,1)
        print('schedule shuffling stimVarValues.')
        np.random.seed(round(time.time()))
        np.random.shuffle(self.stimVarValues)

    def paramNames(self):
        return self.stimVarParams

    def currentParams(self):
        if not self.finished():
            return self.stimVarValues[max(self.stimIndex,0)]
        else:
            return []

    def nextParams(self):
        self.stimIndex+=1
        return self.currentParams()
                
    def finished(self):
        runThroughCompleted=self.stimIndex >= len(self.stimVarValues)
        if runThroughCompleted:
            if self.loopScheduleWhenDone:
            # if loopScheduleWhenDone is true then we are never finished.
            # Instead, when arriving at the end of the stim list, we shuffle and start over.
                if self.shuffleOnLoop:
                    self.shuffleStimList()
                self.stimIndex=0
                return False
        return runThroughCompleted
    
    def readScheduleFromFile(self,defaultName=''):
        #from tkinter import messagebox
        from tkinter import filedialog
        #import tkinter as tk
        filename = filedialog.askopenfilename(title = "Open Stim Params Table:",initialfile=defaultName,filetypes = (("CSV","*.csv"),("all files","*.*")))
        if filename=='':
            # opening canceled. 
            print('No valid file chosen.')
            return ''
        try:
            self.importFromCSV(filename)
            return filename
        except:
            print('Warning: Could not open file: '+filename)
            return ''
        
#%%
from io import TextIOWrapper

class scheduler:

    def __init__(self, detectors,stimulus,dataHandler,schedule):
        self.absoluteTimestamps=False
        self.detectors=detectors
        self.stimulus=stimulus
        self.dataHandler=dataHandler
        self.CSVheaderSaved=False
        self.schedule=schedule
        self.lastAction=time.time()
        self.lastStart=-1e6
        self.startTrialCallback = None # if assigned, this callback function is called just before a stimulus is triggered
        self.oldStatus=""
        
    def getReady(self):
        self.parametersProcessed=[]
        config.status=""
        self.broadCastStatus('start')
        self.nextJob=self.chooseNextTrial
        self.stimStarted=time.time()
        self.timeZero=time.time()
        
    def dblog(self, logstr):
        print(logstr)

    def start(self):
        self.getReady()
        self.processJobs()
        
    def processNextJob(self):
        # now keep working through jobs
        if self.nextJob != None:
            if config.status=="abort":
                self.dblog('Schedule aborted.')
                self.nextJob=None
            else:
                self.nextJob=self.nextJob()
        return self.nextJob

    def processJobs(self):
        # now keep working through jobs
        while self.nextJob != None:
            if config.status=="abort":
                self.dblog('Schedule aborted.')
                self.nextJob=None
            else:
                self.nextJob=self.nextJob()
            
    def didSucceed(self,returnCode):
        # status change functions of modules may return nothing or a boolean
        if returnCode is None:
            return True
        return returnCode
               
    def broadCastStatus(self,aStatus):
        # only broadcast new status if it has changed
        if aStatus==self.oldStatus:
            return
        self.oldStatus=aStatus
        if verbose:
            self.dblog('\n#### New status: '+aStatus)   
        sys.stdout.flush()
        if not self.detectors is None:
            if hasattr(self.detectors,'statusChange'):
                success=self.didSucceed(self.detectors.statusChange(aStatus))
                while not success:
                    time.sleep(0.1)
                    success=self.didSucceed(self.detectors.statusChange(aStatus))
        if hasattr(self.stimulus,'statusChange'):
            success=self.didSucceed(self.stimulus.statusChange(aStatus))
            while not success:
                time.sleep(0.1)
                success=self.didSucceed(self.stimulus.statusChange(aStatus))
        if hasattr(self.dataHandler,'statusChange'):
            success=self.didSucceed(self.dataHandler.statusChange(aStatus))
            while not success:
                time.sleep(0.1)
                success=self.didSucceed(self.dataHandler.statusChange(aStatus))    

    def chooseNextTrial(self):
        self.broadCastStatus('chooseNextTrial')
        self.nextParams=self.schedule.nextParams() 
        if (len(self.nextParams) == 0):
            return None # we are done, no more jobs to do
        else:
            self.getStimReadyForNextParam()
            return self.waitForStart
        
    def done(self):    
        pass

    def getStimReadyForNextParam(self):
        # get the stimulus ready for the chosen stimulus
        self.stimulus.setParams(list(self.schedule.paramNames()), self.nextParams)
        self.stimulus.ready()
        # for debugging, self.dblog info about the chosen stimulus
        if verbose:
            self.dblog('Next stimulus has parameters {}:{}'.format(self.schedule.paramNames(),self.nextParams))
            if self.stimulus.correctResponse("RIGHT"):
                side = 'RIGHT'
            else:
                side='LEFT'           
            if verbose:
                self.dblog(f'Correct response would be on the {side}\n')   
        # sys.stdout.flush()
    
    def getResponse(self):
        self.broadCastStatus('getResponse')
        # listen to detectors and wait for a response to a recently presented stimulus
        # if there are no detectors we just save the timestamp of when the stimulus started
        if self.detectors is None:
            #dataHandler can be an dataHandler object, a file handle (TextIOWrapper) or None.
            #Choose appropriate manner for saving stimulus parameter info
            par=self.stimulus.stimParams.copy()
            if self.absoluteTimestamps:
                par['timeStamp']=time.time()
            else:
                par['timeStamp']=config.currentStimTimestamp
            if not self.dataHandler is None:
                if type(self.dataHandler)==TextIOWrapper:
                    # if first stim, write stim param names to CSV file
                    outstr=''
                    # if self.schedule.stimIndex==0: 
                    if not self.CSVheaderSaved: 
                        for key in par.keys():
                            outstr+=key
                            outstr+=','
                        outstr=outstr[:-1].replace("'","")+'\n'
                        self.dataHandler.write(outstr)
                        self.CSVheaderSaved=True
                    # write current stimulus parameters
                    outstr=''
                    for key in par.keys():
                        outstr+=str(par[key])
                        outstr+=','
                    outstr=outstr[:-1].replace("'","").replace("[","").replace("]","").replace(".,",",")+'\n'
                    self.dataHandler.write(outstr)
                else:
                    self.dataHandler.saveTrial(par)
            self.logCompletedStimPars(par)
            return self.chooseNextTrial
        else:
            response=self.detectors.responseDetected() 
            return self.processResponse(response)
        
    def logCompletedStimPars(self,par):
        # self.dblog(f'\nChecking stim for clickIdx: {hasattr(self.stimulus,"clickIdx")}') 
        if hasattr(self.stimulus,'clickIdx'):
            par['clickIdx']=self.stimulus.clickIdx
        self.parametersProcessed.append(par)

    def punish(self):
        # give timeout signals, then repeat current stimulus as correction trial
        self.dblog('Wrong response. Starting timeout then repeating trial')
        sys.stdout.flush()
        self.broadCastStatus('punish')
        return self.waitForStart    
    
    def presentTrial(self):
        # present next stimulus
        # self.dblog('\n### presenting trial {} of {}'.format(self.schedule.stimIndex+1, len(self.schedule.stimVarValues)))
        self.stimStarted=time.time()
        config.currentStimTimestamp=self.stimStarted-self.timeZero
        config.currentStimParams=self.nextParams
        self.broadCastStatus('presentTrial')
        self.lastAction=time.time()
        self.lastStart=time.time()
        if not self.detectors is None:
            reward_coin=1
            if hasattr(config,'centreRewardsEvery1inN'):
                propCentreRewards=config.centreRewardsEvery1inN   
                if propCentreRewards > 1:
                    reward_coin = np.random.randint(low = 1, high = propCentreRewards)
            if reward_coin == 1:
                if hasattr(config,'delayedCenterRewardAt'):
                    if config.delayedCenterRewardAt > 0:
                        time.sleep(config.delayedCenterRewardAt)
                self.detectors.reward_at(1)
        return self.getResponse

    def processResponse(self, response):
        if response[0] in ['NONE','START']:
            # no response signal yet
            time.sleep(0.05)
            if (time.time()-self.lastAction) > config.maxResponseTime:
                self.dblog('Subject took too long to respond. Resetting trial.')
                sys.stdout.flush()
                self.lastAction=time.time()
                return self.waitForStart
            return self.getResponse

        # if we get this far we have a response. 
        self.stimulus.stop() # make sure stimulation ends after response
        nowStr=time.asctime(time.localtime(response[1]))
        self.dblog('Registered response {} at: {}'.format(response[0],nowStr))
        sys.stdout.flush()

        if response[0]=='ERROR':
            # detectors gave an invalid response. Reset trial.
            self.dblog('Invalid response. Resetting trial')
            sys.stdout.flush()
            self.lastAction=time.time()
            return self.waitForStart            
            
        if response[0]=='QUIT':
            # exit signal sent: return None as next job to do
            return None
        # at this point my response was neither none nor start nor quit, so it must be a judgment.
        # Save the trial. Build a dictionary with all the trial info
        par=self.stimulus.stimParams.copy()
        # Ask the stimulus whether the response was correct
        par['correct']=self.stimulus.correctResponse(response[0])
        par['response']=response[0]
        par['reactionTime']=response[1]-self.stimStarted
        if self.absoluteTimestamps:
            par['timeStamp']=response[1]
        else:
            par['timeStamp']=response[1]-self.timeZero
        self.dataHandler.saveTrial(par)
        self.logCompletedStimPars(par)
        
        # Give feedback
        if par['correct']:
            return self.reward
        else:
            return self.punish            
        
    def reward(self):
        # reward, then move on to next trial
        self.dblog('Correct response. Giving reward then choosing next trial')
        sys.stdout.flush()
        self.broadCastStatus('reward')
        return self.chooseNextTrial
        
    def waitForStart(self):
        # listen to detectors and wait for a START signal
        self.broadCastStatus('waitForStart')
        # check if a minimum ISI is required and if it has elapsed
        if hasattr(self,'minimumISI'):
            timeElapsedSinceLastStart=time.time()-self.lastStart
            if timeElapsedSinceLastStart < self.minimumISI:
                self.dblog('Pausing for {:3.5} s for minimum ISI to elapse.'.format(self.minimumISI-timeElapsedSinceLastStart))
                time.sleep(self.minimumISI-timeElapsedSinceLastStart)
        if self.detectors is None:
            # if there are no detectors to poll for start signal we just go
            return self.presentTrial
        response=self.detectors.responseDetected() 
        if response[0]=='NONE':
            # no start signal yet
            time.sleep(0.05)
            if (time.time()-self.lastAction) > 10:
                self.lastAction=time.time()
                nowStr=time.asctime(time.localtime())
                self.dblog(f'Still waiting for centre lick at {nowStr}')
                sys.stdout.flush()
            return self.waitForStart
        self.detectors.flush()
        if response[0]=='QUIT':
            # exit signal sent: return None as next job to do
            return None
        if response[0]=='START':
            if not self.startTrialCallback is None:
                self.startTrialCallback()
            return self.presentTrial
        else:
            # if we got here an inappropriate response was received.
            # Reset sensors and continue
            self.dblog('Received unexpected signal {}'.format(response[0]))
            sys.stdout.flush()
            return self.waitForStart


