# -*- coding: utf-8 -*-
"""
Created on Aug 2nd 2018

A binaural clicktrain library.

This has snowballed over the years to become a framework for all kinds of auditory stimulus generation.


@author: wschnupp
"""

# ready stimulus hardware
import numpy as np
import time
from sys import stdout
soundPlayer = None
import psyPhysConfig as config
config.verbose=True
import matplotlib.pyplot as plt
from scipy.signal.windows import hann as hanning
from scipy.signal import butter, lfilter
import random
import struct
import threading

def rms(x):
    return np.sqrt(x.dot(x)/x.size)

feedbackTraceLen=200


#%% helper functions

def semitonesToFactor(x):
    return 2**(x/12)

def factorToSemitones(x):
    return np.log2(x)*12

def octavesToFactor(x):
    return 2**x

def factorToOctaves(x):
    return np.log2(x)

def lin2dB(x, ref=1):
    if type(x) is list:
        x=np.array(x)
    return 20*np.log10(x/ref)

def dB2lin(x, ref=1):
    if type(x) is list:
        x=np.array(x)
    return np.power(10,x/20)*ref

def cosWin(x,riseFallN):
    # applies a cosine windows with a rise and fall time of riseFallN samples to signal x with sample rate FS
    wdw=hanning(2*riseFallN)
    if x.ndim==1:
        x[0:riseFallN]*=wdw[0:riseFallN]
        x[-riseFallN:]*=wdw[-riseFallN:]    
    else:
        for chan in range(x.shape[1]):
            x[0:riseFallN,chan]*=wdw[0:riseFallN]
            x[-riseFallN:,chan]*=wdw[-riseFallN:]
    return x

def linearRamp(x,riseFallN):
    # applies a cosine windows with a rise and fall time of riseFallN samples to signal x with sample rate FS
    wdw=np.arange(0,1,1/riseFallN)
    wdw2=np.arange(1,0,-1/riseFallN)
    wdw2-=wdw2[-1]
    if x.ndim==1:
        x[0:riseFallN]*=wdw[0:riseFallN]
        x[-riseFallN:]*=wdw2[-riseFallN:]    
    else:
        for chan in range(x.shape[1]):
            x[0:riseFallN,chan]*=wdw[0:riseFallN]
            x[-riseFallN:,chan]*=wdw2[-riseFallN:]
    return x

def applyEnvelope(signal, anEnv=None):
    if signal is None:
        return
    if anEnv is None:
       return signal # no envelope given. Nothing to do.
    if anEnv[0:3]=='rec':
        return signal # "rectangular" windows require no action
    if anEnv[0:3]=='han':
        wdw=hanning(signal.shape[0]).reshape((len(signal),1))
        if signal.shape[1] > 1:
            wdw=np.tile(wdw, signal.shape[1])
        signal=signal*wdw  
        return signal
    if anEnv[0:3]=='cos': # apply cosine window
        # rise-fall defaults to 5 ms but can be specified as a number in ms from
        # character 4 onwards
        try: 
            riseFall=float(anEnv[3:])
        except:
            riseFall=5    
        signal=cosWin(signal,soundPlayer.ms2samples(riseFall))  
        return signal
    if anEnv[0:3]=='lin': # apply linear window
        # rise-fall defaults to 5 ms but can be specified as a number in ms from
        # character 4 onwards
        try: 
            riseFall=float(anEnv[3:])
        except:
            riseFall=5    
        signal=linearRamp(signal,soundPlayer.ms2samples(riseFall))  
        return signal
        
def GRASmicVtodBSPL(V):
    # our GRAS micorphone has a sensitivity of 0.7e-6 V/Pa
    sens=0.7e-6
    return lin2dB(V/sens)
    
def golay(x):
    #% function [ga, gb] = golay(x);
    #%    generates a pair of golay codes [ga, gb]
    #%    of length 2^x
    #%
    #%    see Zhou, Green & Middlebrooks (1992) J.Acoust.Soc.Am. 92:1169-71 
    #%    for theory
    #%
    ga =np.array([1, 1])
    gb =np.array([1, -1])
    
    for idx in range(x):
      ha=np.hstack((ga,gb))
      hb=np.hstack((ga,-gb))
      ga=ha;
      gb=hb;
    
    return np.hstack((ga,np.array([0]))), np.hstack((gb,np.array([0])))

#def NBpluseByIFFT(fs=48000,samples=20,hp=1000,lp=10000):
#    # creates a narrmowband pulse by inverse fourier transform
#    nSamp=np.floor(samples/2)
#    nyq=fs/2
#    lowSamp=np.round(lp/nyq*nSamp)
#    hiSamp=np.round(hp/nyq*nSamp)

logMessageQueue=None    
def dblog(msg):
    if logMessageQueue is None:
        print(msg, flush=True)
    else:
        logMessageQueue.put(msg)
   
#%% define sound stimulus hardware objects
class soundHardware:
    def __init__(self, audioCalibFile=None):
        self.mysounds=None
        self.soundObject=None
        self.calibFilters=None
        if not audioCalibFile is None:
            self.loadCalibMatFile(audioCalibFile)
        # config.TDTcircuit = None
        
    def loadCalibMatFile(self,calibFileName):
        from scipy.io import loadmat 
        self.calibFileName=calibFileName
        calib=loadmat(calibFileName, simplify_cells=True)
        if calib['calibs']['sampleRate'] != self.sampleRate:
            raise Exception('Sample rate of calibration file ({} Hz) does not match that of output hardware ({} Hz)'.format(calib['calibs']['sampleRate'], self.sampleRate))            
        self.calibFilters=np.hstack((np.array(calib['calibs']['channel'][0]['filter'],ndmin=2).T,np.array(calib['calibs']['channel'][1]['filter'],ndmin=2).T))
            
    def setSoundBuffer(self,sounds, startAtSample=0, endAtSample=0):
        self.soundObject=sounds
        
    def play(self, loop=False):
        pass
        
    def stop(self):
        pass
        
    def reset(self):
        pass 
       
    def done(self):
        pass
        
    def samples2ms(self,x): 
        # converts number of samples to miliseconds
        return x/self.sampleRate*1000
    
    def ms2samples(self,x): 
        # converts miliseconds to nearest number of samples
        return np.int_(np.round(x*self.sampleRate/1000))        
    
class pyaudioHardware(soundHardware):
    def __init__(self, devIdx=0, triggerPulseOnChan3=False, audioCalibFile=None):
        import os 
        import pyaudio as _pyaudio
        self.pyaudio=_pyaudio
        self.paud=self.pyaudio.PyAudio()
        #self.pyg.init()
        #self.sampleRate=44100
        self.sampleRate=48000
        #self.sampleRate=int(48828/2)
        self.stream=None
        self.mysounds=None
        self.devIdx=devIdx # Device number will change depending on system config. Find device number via paud.get_device_info_by_index(). 
        # self.timeZero=time.time()
        soundHardware.__init__(self, audioCalibFile=audioCalibFile)
        if os.path.isfile('~/ratCagePrograms/vol.sh'):
            dblog('Making sure RPi volume is turned up full')
            os.system("~/ratCagePrograms/vol.sh 100");      # this makes sure the RPI volume is turned up full
        self.triggerPulseOnChan3 = triggerPulseOnChan3
        dev=self.paud.get_device_info_by_index(self.devIdx)
        api=self.paud.get_host_api_info_by_index(dev['hostApi'])
        dblog('Connected pyaudio hardware to physical device: '+ dev["name"]+ " with "+dev["maxOutputChannels"]+ 'output channels on api '+api['name'])
        if triggerPulseOnChan3:
            requiredOutputChannels=3
        else:
            requiredOutputChannels=2
        if dev['maxOutputChannels'] < requiredOutputChannels:
            dblog('Error: device has insufficient number of output channels: ', dev["maxOutputChannels"])
            
    def listAvailablePyaudioDevices(self):
        for i in range(self.paud.get_device_count()):
            dev=self.paud.get_device_info_by_index(i)
            api=self.paud.get_host_api_info_by_index(dev['hostApi'])
            dblog('Nr {}:\t{}\tout chans:{}\t api {}'.format(i,dev["name"],dev["maxOutputChannels"],api['name']))
        
    def setSoundBuffer(self,soundObject, startAtSample=0, endAtSample=0):
        if endAtSample<=0:
            # we take endAtSample < 1 to mean that we want to play to the end
            endAtSample=soundObject.numSamples()
        self.soundObject=soundObject
        if soundObject.ears=='both':
            self.mysounds=self.soundObject.sounds[startAtSample:endAtSample,:]
        else:
            self.mysounds=self.soundObject.sounds[startAtSample:endAtSample]
            if soundObject.ears=='left':
                self.mysounds=np.vstack((self.mysounds,np.zeros(self.mysounds.shape))).transpose()
            else:
                self.mysounds=np.vstack((np.zeros(self.mysounds.shape),self.mysounds)).transpose()
        if np.max(np.abs(self.mysounds)) > 1:
            dblog('*** pyaudioHardware - clip warning ')
        self.totalFrames=max(self.mysounds.shape)
        
        if self.triggerPulseOnChan3:
            tooShortBy=5020 - self.totalFrames
            if tooShortBy > 0:
                self.mysounds=np.vstack((self.mysounds,np.zeros((tooShortBy,2)) ))
                self.totalFrames=self.mysounds.shape[0]
            pulse=np.zeros(self.totalFrames)
            #pulse[0:10000]=1 
            pulse[0:5000]=1 # this is the positive end of a squarewave pulse
            self.mysounds=np.vstack((self.mysounds.transpose(),pulse)).transpose()
            self.numChans=3
        else:
            self.numChans=2
        self.nullFrame=()
        for ii in range(self.numChans):
            self.nullFrame+=(0.0,)
        # soundArr=self.mysounds.transpose().reshape(self.mysounds.size,1).squeeze()
        soundArr=self.mysounds.reshape(self.mysounds.size,1).squeeze()
        soundArr=soundArr.astype('float32').tolist()
        self.soundBytes=struct.pack('{}f'.format(len(soundArr)),*soundArr)
        self.framesServed=0
            
        
    def playing(self):
        if self.stream is None:
            return False
        else:
            return self.stream.is_active()
        
    def play(self, loop=False):
        if self.mysounds is None:
            # nothing to play
            return
        self.loop=loop
        # if loop:
        #     loopPar=-1 # this sets pygame to loop indefinitely until stopped
        # else:
        #     loopPar=0
        # startTime=time.time()-self.timeZero
        if not self.stream is None:
            self.stream.close()
        self.stream = self.paud.open(format=self.pyaudio.paFloat32, channels=self.numChans, output_device_index=self.devIdx, rate=self.sampleRate, start=True, output=True, stream_callback=self.serveAudioFrames)  #Set MOTU ASIO as device. Device index number found via paud.get_device_info_by_index()       

    def serveAudioFrames(self, in_data, frame_count, time_info, status):
        self.paStatus=status
        if not(status == self.pyaudio.paNoError):
            eMsg='';
            if status==self.pyaudio.paOutputOverflow:
                eMsg=' Output buffer overflow.'
            if status==self.pyaudio.paOutputUnderflow :
                eMsg=' Output buffer underflow.'
            if status==self.pyaudio.paPrimingOutput :
                eMsg=' Output priming, not yet playing.'
            dblog('ERROR: PyAudio raised error flag {} {}'.format(status,eMsg))
        if self.framesServed>=self.totalFrames:
            if self.loop:
                self.framesServed=0 # start over
            else:
                return (self.nullFrame, self.pyaudio.paComplete) # send stream termination code
        framesToServe=min(frame_count,self.totalFrames-self.framesServed)
        bytesToServe=4*framesToServe*self.numChans
        bytesStart=4*self.framesServed*self.numChans
        bytesOut= self.soundBytes[bytesStart:bytesStart+bytesToServe]
        self.framesServed+=framesToServe
        self.frame_count=frame_count
        if (self.framesServed < self.totalFrames) or self.loop:
            paFlag=self.pyaudio.paContinue
        else:
            paFlag=self.pyaudio.paComplete
        return (bytesOut, paFlag)
        
    def stop(self):
        if self.mysounds is None:
            # nothing to stop
            return
        self.stream.stop_stream()
        self.stream.close()

        
    def done(self):
        self.paud.terminate()
        
class pygameSoundHardware(soundHardware):
    def __init__(self, audioCalibFile=None):
        # config.TDTcircuit = None
        import pygame
        import os 
        self.pyg=pygame
        #self.pyg.init()
        #self.sampleRate=44100
        self.sampleRate=48000
        #self.sampleRate=int(48828/2)
        self.pyg.mixer.pre_init(self.sampleRate,-16,2,4096) #if jitter, change 256 to different value
        self.pyg.mixer.init()
        if not self.pyg.mixer.get_init()[0] == self.sampleRate:
            raise RuntimeError('Pygame failed to set the desired sample rate')
        self.snd=None
        self.outputChan=None
        # self.timeZero=time.time()
        soundHardware.__init__(self, audioCalibFile=audioCalibFile)
        if os.path.isfile('~/ratCagePrograms/vol.sh'):
            dblog('Making sure RPi volume is turned up full')
            os.system("~/ratCagePrograms/vol.sh 100");      # this makes sure the RPI volume is turned up full
        dblog("Initialized pygame sound hardware.")
        
    def setSoundBuffer(self,soundObject, startAtSample=0, endAtSample=0):
        if endAtSample<=0:
            # we take endAtSample < 1 to mean that we want to play to the end
            endAtSample=soundObject.numSamples()
        self.soundObject=soundObject
        # scale sounds to int16
        if soundObject.ears=='both':
            self.mysounds=self.soundObject.sounds[startAtSample:endAtSample,:]*2**14
        else:
            self.mysounds=self.soundObject.sounds[startAtSample:endAtSample]*2**14
            if soundObject.ears=='left':
                self.mysounds=np.vstack((self.mysounds,np.zeros(self.mysounds.shape))).transpose()
            else:
                self.mysounds=np.vstack((np.zeros(self.mysounds.shape),self.mysounds)).transpose()
        if np.max(np.abs(self.mysounds)) > 2**16:
            dblog('*** pygameSoundHardware - clip warning ')
        self.mysounds=np.ascontiguousarray(self.mysounds).astype('int16')
        # make Pygame Sound object
        self.snd = self.pyg.sndarray.make_sound(self.mysounds)
        
    def playing(self):
        if self.outputChan is None:
            return False
        else:
            return self.outputChan.get_busy()     
        
    def play(self, loop=False):
        if self.mysounds is []:
            # nothing to play
            return
        if loop:
            loopPar=-1 # this sets pygame to loop indefinitely until stopped
        else:
            loopPar=0
        # startTime=time.time()-self.timeZero
        self.outputChan=self.snd.play(loopPar)
#        if self.outputChan.get_busy():
#            dblog('pygame output started at {}.'.format(startTime))     
#            while self.outputChan.get_busy():
#                time.sleep(0.1)
#            dblog('pygame output ended by {}.'.format(time.time()-self.timeZero))     
#        else:
#            dblog('pygame output failed to start. Retrying()')
#            self.outputChan=self.snd.play(loopPar)
        
    def stop(self):
        if self.snd is None:
            # nothing to stop
            return
        self.snd.stop()
        
    def done(self):
        self.pyg.mixer.quit()
        
from array import array
        
class TDT_RZ6(soundHardware):
    
    def __init__(self, circuitFile=None, audioCalibFile=None):
        # from tdt import DSPProject, util
        # self.TDTproject = DSPProject()
        # from TDT_rpcox import RPcoX
        # config.TDTproject = self.TDTproject
        # if not 'RXdevice' in dir(config):
        #     config.RXdevice='RZ6'
        # print(f'Loading stimulus hardware of type {self.__class__.__name__}', flush=True)
        dblog(f'Loading stimulus hardware of type {self.__class__.__name__}')
        from comtypes import client
        self.RP = client.CreateObject('RPco.X')
        connectFun=self.RP.ConnectRZ6
        if 'RXdevice' in dir(config):
            if config.RXdevice=='RZ2':
                connectFun=self.RP.ConnectRZ2
        if connectFun('GB',1) == 0:
            dblog('ERROR: TDT ActiveX failed to connect to {config.RXdevice}')
            raise ImportError('TDT ActiveX failed to connect to RZ6')        
        config.RP=self.RP
        if circuitFile is None:
            circuitFile='BufferIO_50k.rcx'
        self.loadCircuitFile(circuitFile)
        # self.zBus=util.connect_zbus(interface='GB', address=None)
        from zBus import ZBUSx
        self.zBus = ZBUSx()
        if self.zBus.ConnectZBUS('GB') == 0:
            dblog('ERROR: TDT ActiveX failed to connect to ZBUS')
            raise ImportError('TDT ActiveX failed to connect to ZBUS')
        soundHardware.__init__(self, audioCalibFile=audioCalibFile)
        
    def loadCircuitFile(self,cFileName):
        self.circuitFile=cFileName
        dblog('Stimulation circuit loading RCX file {}'.format(self.circuitFile))

        if self.RP.LoadCOF(self.circuitFile) == 0:
            dblog('ERROR: TDT ActiveX failed to load circuit {}'.format(self.circuitFile))
            raise ImportError('TDT ActiveX failed to load circuit {}'.format(self.circuitFile))
        self.RP.Run()
        # self.TDTcircuit = self.TDTproject.load_circuit(cFileName, 'RZ6')
        # self.TDTcircuit.start()
        # config.TDTcircuit=self.TDTcircuit
        # self.RP=self.TDTcircuit._iface

        time.sleep(0.1)
        self.sampleRate=self.RP.GetSFreq()
        # self.sampleRate=self.TDTcircuit._iface.GetSFreq()
        # self.TDTcircuit.start()
        
    def setSoundBuffer(self,soundObject, startAtSample=0, endAtSample=0, nOffset=0):
        self.soundObject=soundObject
        if endAtSample<=0:
            # we take endAtSample < 1 to mean that we want to play to the end
            endAtSample=soundObject.numSamples()
        # dblog('loading sounds from {} to {}'.format(startAtSample, endAtSample))
        if soundObject.ears=='both':
            Lbuff= soundObject.sounds[startAtSample:endAtSample,0]
            Rbuff= soundObject.sounds[startAtSample:endAtSample,1]
        elif soundObject.ears=='left':
            Lbuff= soundObject.sounds[startAtSample:endAtSample]
            Rbuff= np.zeros(soundObject.numSamples())
        else:
            Rbuff= soundObject.sounds[startAtSample:endAtSample]
            Lbuff= np.zeros(soundObject.numSamples())
        if (self.RP.WriteTagV('Lbuff', nOffset, array('d',Lbuff)) ==0):
            raise Exception('TDT IO Error: WriteTagV to Lbuff failed.')
        if (self.RP.WriteTagV('Rbuff', nOffset, array('d',Rbuff)) ==0):
            raise Exception('TDT IO Error: WriteTagV to Rbuff failed.')
        self.RP.SetTagVal('StimSamples',endAtSample-startAtSample)
        # dblog(endAtSample-startAtSample)
        # dblog('length sndbuf is ',len(soundObject.sounds[startAtSample:endAtSample]))
        
    def playing(self):
        return self.RP.GetTagVal('running')
        
    def play(self, loop=False):
        if self.soundObject is None:
            # nothing to play
            return
        # XXX looping not yet implemented
        dblog('TDT sound player sending zBUS tigger A.')
        self.zBus.zBusTrigA(0,0,10)
        # self.TDTcircuit.trigger('A','pulse') 
        # dblog('Stimulus triggered via zBus A')
        stdout.flush()
        
    def stop(self):
        if self.soundObject is None:
            # nothing to stop
            return
        if self.playing():
            self.reset()
    
    def reset(self):
        self.RP.SoftTrg(9)
        # self.TDTcircuit.trigger(9,'pulse')   
            
    def done(self):
        self.RP.Halt()
        # self.TDTcircuit.stop()        
        
class TDT_RZ6_PlayRecord(TDT_RZ6):
    
    def __init__(self, circuitFile=None, audioCalibFile=None):
        if circuitFile is None:
            myCircuit='BufferIO_2_50k.rcx'
        else:
            myCircuit=circuitFile
        super().__init__( circuitFile=myCircuit , audioCalibFile=audioCalibFile)
        
    def readRecordings(self):
        # wait for the playback to stop before collecting recording buffers
        while self.playing():
            time.sleep(0.05)
        # transfer recordings from buffer
        bufSize=int(self.RP.GetTagVal('ADidx'))
        self.Lbuff=np.array(self.RP.ReadTagVEX('L_InBuff',0,bufSize,'F32','F32',1)[0]).astype('float32')
        if len(self.Lbuff)==0:
            raise Exception(f"RP.ReadTagVEX('L_InBuff',0,{bufSize},...) returned no data.")
        self.Rbuff=np.array(self.RP.ReadTagVEX('R_InBuff',0,bufSize,'F32','F32',1)[0]).astype('float32')
        if len(self.Rbuff)==0:
            raise Exception(f"RP.ReadTagVEX('R_InBuff',0,{bufSize},...) returned no data.")
            
        
        
class TDT_RZ_simulator(TDT_RZ6):
    # this is intended to be run on "Corpus" for simulations / debugging
    def __init__(self, circuitFile='TestCircuit.rcx', audioCalibFile=None):
        # if not 'RXdevice' in dir(config):
        config.RXdevice='RZ2'
        # print('#### Loading RZ simulator', flush=True)
        
        super().__init__(circuitFile=circuitFile)
        
    # def impedances(self):
    #     return 0,0 


class TDT_CI_Hardware(TDT_RZ6):
    
    def __init__(self, circuitFile=None, channelMap=None):
        if not 'RXdevice' in dir(config):
            config.RXdevice='RZ6'
        if circuitFile is None:
            if config.location=="Freiburg":            
                circuitFile='CIbufferIO_25k_FB.rcx'
            else:
                circuitFile='CIbufferIO_25k.rcx'     
        dblog('Loading {} for {} location'.format(circuitFile,config.location))

        super().__init__(circuitFile=circuitFile)
        # New feature 18/5/22 : there can be channel remapping in the circuit
        # Check if there is a channelMapValues tag. If so, load a map and set it.
        # self.chanMap=None
        # self.chanMapSize=self.TDTcircuit._iface.GetTagSize("channelMapValues")
        # # dblog('channel map size is ', chanMapTagSize)
        # if self.chanMapSize > 0:
        #     # need to make sure we have a channel map and load it.
        #     self.setChannelMap(channelMap) # XXX 
        
    # def setChannelMap(self, channelMap):
    #     if self.chanMapSize == 0:
    #         return # the hardware does not support a channel map
    #     if channelMap is None:
    #         # default map maps channels stright through, 1:1, 2:2, ...
    #         channelMap=tuple(np.arange(self.chanMapSize)+1)
    #     if len(channelMap) != self.chanMapSize:
    #         raise Exception('TDT_CI_Hardware.setChannelMap() given mapping of incorrect length')
    #     self.TDTcircuit._iface.WriteTagVEX('channelMapValues', tuple(channelMap), 0,10,'I32','I32',1)[0]
    #     self.TDTcircuit._iface.SetTagVal('channelMapSelect',0)

    def feedbackTraces(self):
        # dblog('Reading feedback traces ...')
        feedback1=self.RP.ReadTagV('Feedback1', 0,feedbackTraceLen)
        feedback2=self.RP.ReadTagV('Feedback2', 0,feedbackTraceLen)
        # dblog('... feedback traces read')
        return np.array(feedback1), np.array(feedback2)        

    def feedbackRMS(self):
        fTrace1, fTrace2 = self.feedbackTraces()
        return rms(fTrace1-fTrace1[0]), rms(fTrace2-fTrace2[0])
    
    def setSoundBuffer(self,soundObject, startAtSample=0, endAtSample=0):
        if np.max(soundObject.sounds)>3000:
            dblog('*** Warning, stimulus amplitudes exceed 3000uA. They will probably be limited by the hardware to prevent excessive stimulation')
        super().setSoundBuffer(soundObject, startAtSample, endAtSample)     
        
    def impedances(self):
        # calculates CI electrode impedances by comparing the RMS amplitudes in the 
        # first segment of the desired (current) waveform against that of the 
        # feedback voltage provided by the TDT circuit
        if self.soundObject is None:
            # nothing to compute
            return 0, 0
        try: 
            if self.soundObject.ears =='both':
                IRMS1=rms(self.soundObject.sounds[0:feedbackTraceLen,0])
                IRMS2=rms(self.soundObject.sounds[0:feedbackTraceLen,1])
                VRMS1, VRMS2 = self.feedbackRMS()
                return VRMS1*1e6/IRMS1, VRMS2*1e6/IRMS2
            else:
                if self.soundObject.ears =='left':
                    IRMS1=rms(self.soundObject.sounds[0:feedbackTraceLen])
                    VRMS1, VRMS2 = self.feedbackRMS()
                    return VRMS1*1e6/IRMS1, 0
                else:
                    IRMS2=rms(self.soundObject.sounds[0:feedbackTraceLen])
                    VRMS1, VRMS2 = self.feedbackRMS()
                    return 0, VRMS2*1e6/IRMS2
        except Exception as e:
            dblog('Impedance feedback failed: '+str(e))
            return 0, 0
        
defaultSoundPlayer=pygameSoundHardware
soundPlayer=None
#%% define the base class stimulus object
class stimObject:
    def __init__(self):
        global soundPlayer
        if soundPlayer is None:
            soundPlayer=defaultSoundPlayer()
        self.soundPlayer=soundPlayer
        self.isReady=False
        self.sounds=None # this should be given sound waveforms in a time by channel array
        self.ears='both' # this can be 'left','right' or 'both' for monaural or binaural stimulation respectively.
        self.lastPlay=0
        self.triggersSent=0
        self.repeatAfterNseconds=0
        self.levelRePeak=False # normally we set levels relative to RMS If you want to set them relative to abs max instead, set this to True
        self.timeOutSound=None
        self.loopTimer=None
        self.stimParams={}
        # self.reference= 20*10e-6# self.reference is used to set the stimulus' level.
        self.reference= 0.00002# self.reference is used to set the stimulus' level.
                        # by default we assume that the waverforms in "sounds" are to be specified in 
                        # Pascals,and that levels in dB would correspond to dB SPL, hence a reference pressure of 20*10-6 Pa
                        # However, if the stimuli are electircal stimuli, e.g. for CI work, the waveforms may be in microAmp, and the reference might be 100 uA.
            
    def calibrateWaveforms(self):
        if not hasattr(self.soundPlayer, "calibFilters"):
            return
        if self.soundPlayer.calibFilters is None:
            return
        if self.sounds is None:
            return
        # convolve the sounds with the calibration filters
        dblog('applying calibration filter')
        for chan in self.sounds.shape[1]:
            self.sounds[:,chan]=np.convolve(self.sounds[:,chan],self.soundPlayer.calibFilters[:,chan],mode='same')
        
    def append(self,stimList):
        # this allows us to concatenate multiple stimulus objects
        for stim in stimList:
            if self.sounds is None:
                self.sounds = stim.sounds
            else:
                if self.sounds.ndim==1:
                    self.sounds=np.append(self.sounds,stim.sounds)
                else:
                    self.sounds=np.vstack((self.sounds,stim.sounds))
        self.setDurationFromSoundBuffer()
                
    def setDurationFromSoundBuffer(self):
        self.stimParams['duration (s)']=self.soundPlayer.samples2ms(self.sounds.shape[0])/1000
        
    def numChannels(self):
        if self.sounds is None:
            return 0
        elif self.sounds.ndim<2:
            return 1
        else:
            return(self.sounds.shape[1])

    def superimpose(self,secondSig):
        if secondSig.sounds is None:
            return # nothing to do
        if self.sounds is None:
            self.sounds=secondSig.sounds.copy()
        else:
            minLen=np.min([self.sounds.shape[0],secondSig.sounds.shape[0]])
            self.sounds[0:minLen,:]+=secondSig.sounds[0:minLen,:]

    def setParams(self, paramName, paramValue):
        if (hasattr(paramName,'__iter__')):
            if (len(paramName)==1):
                if hasattr(paramValue, '__iter__'):
                    paramValue=paramValue[0]
                self.stimParams[paramName[0]]=paramValue
            else:
                if np.isscalar(paramValue):
                    paramValue=[paramValue]
                for ii in range(len(paramName)):
                    nextPar=paramValue[ii]
                    if type(paramValue[ii]) in [str, np.str_]:
                        try:
                           nextPar=float(paramValue[ii]) 
                        except:
                           pass    
                    self.stimParams[paramName[ii]]=nextPar
        elif (type(paramName) is dict):
            for parTag in paramName.keys():
                self.stimParams[parTag]=paramName[parTag]
        else:
            self.stimParams[paramName]=paramValue
        self.isReady=False
              
    def statusChange(self,newStatus):
        # dblog(f"---> New status is {newStatus}")
        if not self.timeOutSound is None:
            if self.timeOutSound.playing():
                self.timeOutSound.stop()
        if newStatus=="start":
            self.ready()
            if hasattr(self.soundPlayer,"circuitFile"):
                self.triggersSent=int(self.soundPlayer.RP.GetTagVal('TriggersReceived'))
            return
        if newStatus=="chooseNextTrial":
            # while self.stimulatorStatus()=='PLAYING':
            while self.playing():
                time.sleep(0.1) # wait for previous stimulus to complete
            return
        if newStatus=="waitForStart":
            return
        if newStatus=="presentTrial":
            # don't trigger while uploadInProgress as this may cause RZ2 to miss trigger
            if hasattr(config,"uploadInProgress"):
                if config.uploadInProgress:
                    dblog('Waiting for upload to finish before triggering stimulus.')
                while config.uploadInProgress:
                    pass
            self.play()
            if hasattr(self.soundPlayer,"circuitFile"):
                self.triggersSent+=1
                dblog(f'Stimulus module triggers sent: {int(self.triggersSent)}')
                # at this point, if we are running TDT CI hardware we 
                # may want to report on impedances 
    #            global soundPlayer
                tagtype=self.soundPlayer.RP.GetTagType('TriggersReceived')
                if  tagtype == 73:
                    tiggersReceived=int(self.soundPlayer.RP.GetTagVal('TriggersReceived'))
                    dblog(f'StimRCX trigger count {tiggersReceived}')
                    while tiggersReceived<self.triggersSent:
                        self.soundPlayer.zBus.zBusTrigA(0,0,15)
                        dblog(f'@@@@@ Last trigger not received. Resending.(N={tiggersReceived}) @@@@@')
                        tiggersReceived=self.soundPlayer.RP.GetTagVal('TriggersReceived')
                else:
                    dblog('RCX file does not count triggers (Tag TriggersReceived missing).'+
                          ' Consider updating.')
                tagtype=self.soundPlayer.RP.GetTagType('LastZBUSA_samples')
                # dblog(f"GetTagType('LastZBUSA_samples')' returned {tagtype}")
                if tagtype == 73:
                    triggerTimeSamples=int(self.soundPlayer.RP.GetTagVal('LastZBUSA_samples'))
                    self.stimParams['tSamples']=triggerTimeSamples
            #     time.sleep(0.05)
            if isinstance(self.soundPlayer, TDT_CI_Hardware):
                try: 
                    imps = self.soundPlayer.impedances()
                    dblog('Electrode impedance feedback:')
                    for ii, imp in enumerate(imps):
                        dblog('   Chan {} : {:.3f} kOhms'.format(ii+1, imp/1000))
                        paramName='imp_{}_ohm'.format(ii)
                        self.stimParams[paramName]=imp
                except:
                    dblog('Unable to collect electrode feedback. soundPlayer.impedances() failed.')
                    for ii in range(self.numChannels()):
                        paramName='imp_{}_ohm'.format(ii)
                        self.stimParams[paramName]='failed'
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
            dblog("stimulator module received unrecognized status: "+newStatus)
            return
            
    # def stimulatorStatus(self):
    #     if time.time()-self.lastPlay<self.stimParams['duration (s)']:
    #         return "PLAYING"
    #     if not self.isReady:
    #         self.ready()
    #     return("READY")
            
    def playing(self):
        # global soundPlayer
        # if soundPlayer is None:
        #     return False
        return (self.soundPlayer.soundObject==self) and self.soundPlayer.playing()

    def ready(self):
        # global soundPlayer
        # if soundPlayer is None:
        #     soundPlayer=defaultSoundPlayer()
        self.sampleRate=self.soundPlayer.sampleRate
        if 'ears' in self.stimParams.keys():
            self.ears=self.stimParams['ears']
            
        self.isReady=True
        return True
            
    def loopNtimes(self,N,ISI):
        # makes a new .sounds array from N repeats of the existing sounds array
        # sepratated by gaps that are ISI seconds long
        N=int(N)
        if self.sounds is None:
            return
        if N < 2:
            return
        silence = np.zeros((self.soundPlayer.ms2samples(ISI*1000),2))
        newSounds=self.sounds.copy()
        for ii in range(N-1):
            newSounds=np.vstack((newSounds,silence))
            newSounds=np.vstack((newSounds,self.sounds))
        self.sounds=newSounds       
        #self.stimParams['duration (s)']=self.soundPlayer.samples2ms(self.sounds.shape[0])/1000                     
        
    # def setChannelMap(self, amap=None):
    #     # if the stimulus parameters specify a channel mapping, send it to the player first. 
    #     if 'channelMap' in self.stimParams.keys():
    #         if not ("channelMap" in dir(self.soundPlayer)):
    #             raise Exception('mChanStimObject.play(): current sound player does not have a channel map')
    #         if amap is None:
    #             amap=self.stimParams['channelMap']
    #         newMap=[int(x)-1 for x in amap.split(" ")]
    #         if len(newMap) > self.soundPlayer.numOutputChannels:
    #             raise Exception('Channel map [{}] longer than the available number of channels {}.'.format(newMap,self.soundPlayer.numOutputChannels))
    #         for i, x in enumerate(newMap):
    #             self.soundPlayer.channelMap[i]=x 

    def play(self,startAtSample=0, endAtSample=0):
        try:
            if not self.isReady:
                self.ready()   
            if self.sounds is None:
                return
            # self.setChannelMap()
            self.soundPlayer.setSoundBuffer(self,startAtSample,endAtSample) 
            try: 
                doLoop=self.stimParams['loop']
            except:
                doLoop=False    
            self.lastPlay=time.time()
            self.soundPlayer.play(doLoop)  
            if self.repeatAfterNseconds > 0:
                self.loopTimer=threading.Timer(self.repeatAfterNseconds, self.timedReplay)
                self.loopTimer.start()
        except BaseException as e:
            dblog('ERROR: exception occuerd during stimulus play:'+str(e))
            raise(e)
        
    def timedReplay(self): # restarts the stimulus periodically through a timer
        self.soundPlayer.play()
        self.loopTimer=threading.Timer(self.repeatAfterNseconds, self.timedReplay)
        self.loopTimer.start()

        
    def numSamples(self):
        if self.sounds is None:
            return 0
        else:
            return self.sounds.shape[0]
        
    def getRMS(self):
        if self.sounds is None:
            return np.array([np.nan,np.nan])
        else:
            return np.sqrt(np.mean(np.square(self.sounds),axis=0))

    def getPeak(self):
        if self.sounds  is None:
            return np.array([np.nan,np.nan])
        else:
            return np.max(np.abs(self.sounds))        
        
    def getLeveldB(self): 
    # default assumption is that signal is in Pascal and that we work in dB SPL
    # hence a default reference of 20 micro Pascal
        if self.levelRePeak:
            rms=self.getPeak()
        else:
            rms=self.getRMS()
        if np.isnan(rms.any()):
            return np.array([np.nan,np.nan])
        else: 
            return 20*np.log10(rms/self.reference)
        
    def scale(self,a):
        if self.sounds is None:
            return
        if type(a) == int or type(a) == float or type(a)==np.float64:
            self.sounds=self.sounds*a
            return
        if self.ears=='both':
            try:
                # input probably two values, one for left, one for right channel. Scale each.
                self.sounds[:,0]=self.sounds[:,0]*a[0]
                self.sounds[:,1]=self.sounds[:,1]*a[1]
                return
            except:
                self.sounds=self.sounds*a
                return
        # if you got to this point scaling param a was not understood
        raise Exception("Could not scale by "+type(a))
                
    def applyEnvelope(self, anEnv=None):
        try:
           anEnv=self.stimParams['envelope'] 
        except:
           return # no envelope given. Nothing to do.
        if self.sounds is None:
            return # sounds not defined, nothing to do
        self.sounds = applyEnvelope(self.sounds, anEnv)

        # if anEnv is None:
        #     try:
        #        anEnv=self.stimParams['envelope'] 
        #     except:
        #        return # no envelope given. Nothing to do.
        # if anEnv[0:3]=='rec':
        #     return # "rectangular" windows require no action
        # if anEnv[0:3]=='han':
        #     wdw=hanning(self.sounds.shape[0])
        #     if self.sounds.shape[1] ==2:
        #         wdw=np.vstack((wdw,wdw)).transpose()
        #     self.sounds=self.sounds*wdw  
        #     return
        # if anEnv[0:3]=='cos': # apply cosine window
        #     # rise-fall defaults to 5 ms but can be specified as a number in ms from
        #     # character 4 onwards
        #     try: 
        #         riseFall=float(anEnv[3:])
        #     except:
        #         riseFall=5    
        #     #dblog('applying cosine window of length', riseFall)
        #     self.sounds=cosWin(self.sounds,soundPlayer.ms2samples(riseFall))  
        #     return
        
    def setLevel(self, ABL=None):
        if self.sounds is None:
            return
        if ABL is None:
            try:
               ABL=self.stimParams['ABL (dB)']
            except:
               return # no envelope given. Nothing to do.
        # first work out what the current ABL is
        if self.levelRePeak:
            currentRMS=self.getPeak()
        else:
            currentRMS=self.getRMS()
        currentRMS=np.mean(currentRMS)
        desiredRMS=dB2lin(ABL, self.reference)
        self.scale(float(desiredRMS/currentRMS))
        self.stimParams['ABL (dB)']=ABL
        self.calibrateWaveforms()
        
    def applyILD(self,anILD=None):
        if not self.ears=='both':
            return # not a binaural sound. Nothing to do
        if self.sounds is None:
            return
        if anILD is None:
            if not 'ILD (dB)' in self.stimParams:
                return # no ILD specified. Nothing to do
            else:
               anILD=self.stimParams['ILD (dB)'] 
            # if we get to this point the ILD has not been previsously specified
            # we assume current value is zero
            self.stimParams['ILD (dB)']=0
        if anILD==self.stimParams['ILD (dB)']:
            return # nothing to do, desired ILD alredy set
        changeILDby=anILD-self.stimParams['ILD (dB)'] # to set to new ILD we need to scale by difference between new and current 
        self.stimParams['ILD (dB)']=anILD
        self.scale( dB2lin([-changeILDby/2,changeILDby/2],1) ) # positive ILD means right ear louder
        
    def applyITD(self,anITD=None):
        if not self.ears=='both':
            return # not a binaural sound. Nothing to do
        if self.sounds is None:
            return
        if anITD is None:
            if not 'ITD (ms)' in self.stimParams:
                return # no ITD specified. Nothing to do
            else:
               anITD=self.stimParams['ITD (ms)'] 
            self.stimParams['ITD (ms)']=0
        #dblog('setting ITD to ',anITD)
        ITDsamples=self.soundPlayer.ms2samples(anITD-self.stimParams['ITD (ms)'])
        self.stimParams['ITD (ms)']=anITD
        if ITDsamples==0:
            return # nothing to do
        ITDpad=np.zeros(np.abs(ITDsamples))
        L=self.sounds[:,0]
        R=self.sounds[:,1]
        if anITD > 0: # positive ITDs mean right ear earlier
            L=np.hstack((ITDpad,L))
            R=np.hstack((R,ITDpad))
        else:
            L=np.hstack((L,ITDpad))
            R=np.hstack((ITDpad,R))
        self.sounds=np.vstack((L,R)).transpose()
        
    def collapseToMono(self):
        if self.sounds is []:
            return
        if not self.ears=='both':
            return # not a binaural sound. Nothing to do
        if len(self.sounds.shape) > 1:
            self.sounds=self.sounds[:,0]+self.sounds[:,1]
            self.ears=='left'
        
    def correctResponse(self,aResponse):
        # indicate whether aResponse is a correct response for the current stimulus
        # undefined for the base class. Needs to be overridden in derived stimuli if needed.
        return "UNDEFINED"
        
    def stop(self):
        if not self.loopTimer is None:
            self.loopTimer.cancel()
            self.loopTimer=None
        if self.playing():
            self.soundPlayer.stop()
        
    def plot(self, singleAxis=True):
        if not self.isReady:
            self.ready()   
        if self.sounds is []:
            return
        taxis=np.array(range(self.sounds.shape[0]))/self.sampleRate
        if singleAxis:
            if not (self.ears=='both'):
                if (self.ears=='right'):
                    myColor='red'
                else:
                    myColor='blue'
                lines=[plt.plot(taxis,self.sounds, color=myColor)]
            else:
                lines=[plt.plot(taxis,self.sounds)]
            plt.xlabel('time (s)')
            plt.ylabel('amplitude')
            return lines
        else:
            if (self.ears=='right') or (self.ears=='left'):
                Nchannels=1
            else:
                Nchannels=self.sounds.shape[1]
            lines=[]
            axes=[]
            for chan in range(Nchannels):
                if (self.ears=='right'):
                    myColor='red'
                    axes.append(plt.subplot(Nchannels,1,chan+1,myColor))
                else:
                    axes.append(plt.subplot(Nchannels,1,chan+1))
                lines.append(plt.plot(taxis,self.sounds[:,chan]))
                if chan < Nchannels-1:
                    plt.xticks([])
            plt.xlabel('time (s)')
            plt.ylabel('amplitude')
            return lines, axes

    
    def plotSpect(self, dBrange=80, xscale='linear'):
        if not self.isReady:
            self.ready()   
        if self.sounds is []:
            return
        spect=np.fft.fft(self.sounds, axis=0)
        spect=np.abs(spect)/spect.shape[0] # get amplitudes and scale by number of samples
        nyquist=self.soundPlayer.sampleRate/2
        halflen=int(spect.shape[0]/2)
        spect=spect[1:halflen,:]
        fstep=nyquist/halflen
        faxis=np.array(range(spect.shape[0]))*fstep
        lines=plt.plot(faxis,20*np.log10(spect/self.reference))
        plt.xlabel('frequency (Hz)') 
        plt.ylabel('level (dB)')
        plt.gca().set_xscale(xscale)
        ylim=plt.gca().get_ylim()
        yrange=ylim[1]-ylim[0]
        if yrange > dBrange:
            plt.gca().set_ylim((ylim[1]-dBrange, ylim[1]))
        return lines    
        
    def done(self):           
        pass  

#%% define a stimulus object for change detection tasks      
        # Change detection stimuli must have a method "timeReChange" specifying 
        # how much time has elapsed relative to the occurrence of the stimulus. (Negative if change is in future)
        # We assume that variable changeAfter indicates after how many s the stimulus changes
        
class simpleChangeDetectionStimulus(stimObject):
    def loadTokens(self):
        self.tokens=[]
        self.tokens.append(toneObject(1000))
        self.tokens[0].ready()
        self.tokens.append(toneObject(1500))
        self.tokens[1].ready()
        
    def __init__(self):
        super().__init__()
        self.stimParams={}
        self.tokens = None
        self.spacer=None
        self.ruleSeq=[0,0,0,0,0,0,0]
        self.violationSeq=[1,1,0,1,1,0]
        
    def ready(self):
        super().ready()
        if self.tokens is None:
            self.loadTokens()
        if self.spacer is None:
           self.spacer= silence(duration=0.1)
        # build stimulus by concatenating first the rule sequence, then the violation sequence
        self.sounds=None
        for nextToken in self.ruleSeq:
            self.append([self.tokens[nextToken], self.spacer])
        self.changeAfter=self.stimParams['duration (s)']
        for nextToken in self.violationSeq:
            self.append([self.tokens[nextToken], self.spacer])
        self.calibrateWaveforms()
        return self.isReady
        
    def timeReChange(self):
        return (time.time()-self.lastPlay) - self.changeAfter
    
#%% define a golaycode pair. 
# see Zhou, Green & Middlebrooks (1992) J.Acoust.Soc.Am. 92:1169-71 	for theory

golayDefaultStimulusParams= {
        # 'codeA': 1,
        'ABL (dB)': 90,
        'loop': False }

class golayPair(stimObject):
    def __init__(self, golayLen=8):
        super().__init__()
        self.stimParams=golayDefaultStimulusParams.copy()
        ga = np.array([1, 1])
        gb = np.array([1, -1])
        self.ears='left'
        for idx in range(golayLen):
          ga_=np.append(ga,gb)
          gb_=np.append(ga,-gb)
          ga=ga_
          gb=gb_
        self.ga=np.append(ga,np.zeros(ga.shape))
        self.gb=np.append(gb,np.zeros(gb.shape))
        
    def ready(self):
        # if self.stimParams['codeA']:
        #     self.sounds=self.ga
        # else:
        #     self.sounds=self.gb
        super().ready()       
        self.padding = round(25/1000*self.soundPlayer.sampleRate)
        tmp = np.append(np.zeros(self.padding),self.ga)
        self.sounds = np.append(tmp,self.gb)  
        self.setLevel()
        

#%% define a modulated click train
# This modulated click train is intended for change detection, ie 
# discriminate 0% modulation from some X% that is supra-threshold.
# Both amplitude and pulse rate can be modulated. Modulation depth 
# is given in percent. AM_pc=10 means that the PEAK AMPLITUDE alternates
# from 110% base level to 90% base level. Negative modulations simply flip the 
# phase (ie AM_pc=-10 means we alternate between 90 and 110% peak amplitude.
# Modulation in rectangular.
        

modDefaultStimulusParams= {
        'modCycle_s': 0.25,
        'baseRate_pps': 300,
        'baseLevel_dB': 50,
        'FM_pc': 30,
        'AM_pc': 10,
        'cycles' : 20,
        'timeoutFrequency':6000,
        'loop': False }

class ModulatedClickTrain(stimObject):

    def __init__(self, defaultParams=modDefaultStimulusParams, clickShape=np.array([0.7111 ,  -0.0476,   -0.4480,    0.1398,   -0.3584,    0.0183,   -0.0152])):
        super().__init__()
        self.stimParams=defaultParams.copy()
        self.stim1=lateClickTrainObject()
        self.stim2=lateClickTrainObject()
        self.stim1.clickShape=clickShape 
        self.stim2.clickShape=self.stim1.clickShape
        # self.timeOutSound=toneObject(freq=self.stimParams['timeoutFrequency'])
        # self.timeOutSound.stimParams['duration (s)']=config.timeoutDuration-0.1
        # self.timeOutSound.stimParams['ABL (dB)']=self.stimParams['baseLevel_dB']-5
        # self.timeOutSound.ready()                

    def ready(self):
        super().ready()
        self.stim1.reference=self.reference
        self.stim2.reference=self.reference
        self.stim1.levelRePeak=self.levelRePeak
        self.stim2.levelRePeak=self.levelRePeak
        # self.stim1.clickShape=self.clickShape
        # self.stim2.clickShape=self.clickShape
        self.stim1.ears=self.ears
        self.stim2.ears=self.ears
        # set params for subsidiary stimuli
        amp=self.stimParams['baseLevel_dB']
        modFactor1=1+self.stimParams['AM_pc']/100
        modFactor2=1-self.stimParams['AM_pc']/100
        self.stim1.stimParams['ABL (dB)']=amp+lin2dB(modFactor1)
        self.stim2.stimParams['ABL (dB)']=amp+lin2dB(modFactor2)
        
        rate=self.stimParams['baseRate_pps']
        modFactor1=1+self.stimParams['FM_pc']/100
        modFactor2=1-self.stimParams['FM_pc']/100
        self.stim1.stimParams['clickRate (Hz)']=rate*modFactor1
        self.stim2.stimParams['clickRate (Hz)']=rate*modFactor2
        
        self.stim1.stimParams['duration (s)']=self.stimParams['modCycle_s']/2
        self.stim2.stimParams['duration (s)']=self.stim1.stimParams['duration (s)']

        # ready constituent stimuli
        self.stim1.ready()
        self.stim2.ready()
        # print('stim 1 params')
        # print(self.stim1.stimParams)
        # print('stim 2 params')
        # print(self.stim2.stimParams)
        # now assemble the stimulus
        self.sounds=self.stim1.sounds.copy()
        self.append([self.stim2]) 
        for reps in range(1,self.stimParams['cycles']):
            self.append([self.stim1,self.stim2])

    def correctResponse(self,aResponse):
        # aResonse should be "LEFT" for 1st interval and "RIGHT for 2nd interval
        # print('evaluating response. Stim is:',self.stimParams )
        zeroMod=(np.abs(self.stimParams['FM_pc'])<0.00001) and (np.abs(self.stimParams['AM_pc'])<0.00001)
        # print('response: ',aResponse,',  params same: ',sameAvals)
        if aResponse=="LEFT":
            return zeroMod
        if aResponse=="RIGHT":
            return not zeroMod
        return False

class ModulatedClickTrain2(ModulatedClickTrain):

    def __init__(self, defaultParams=modDefaultStimulusParams, clickShape=np.array([0.7111 ,  -0.0476,   -0.4480,    0.1398,   -0.3584,    0.0183,   -0.0152])):
        super().__init__()
        self.stimParams=defaultParams.copy()
        self.stim1=lateClickTrainObject2()
        self.stim2=lateClickTrainObject2()
        self.stim1.clickShape=clickShape 
        self.stim2.clickShape=self.stim1.clickShape
        

#%% define a wav file stimulus object            
class silence(stimObject):
    def __init__(self,duration=0.2):
        super().__init__()
        self.stimParams={}
        self.stimParams['duration (s)']=duration
        silence.sampleRate=self.soundPlayer.sampleRate
    def ready(self):
        duration=self.stimParams['duration (s)']
        if self.ears=='both':
            self.sounds=np.zeros((int(self.soundPlayer.sampleRate*duration),2))
        else:
            self.sounds=np.zeros((int(self.soundPlayer.sampleRate*duration),))

    
#%% define a wav file stimulus object            
class wavFileObject(stimObject):
    def __init__(self, wavFileName=None):
        super().__init__()
        self.readWavFile(wavFileName)
    
    def readWavFile(self,wavFileName):    
        import scipy.io.wavfile as wav
        self.wavFile=wavFileName
        if wavFileName is None:
            self.stim=[]
        else:
            fs,signal=wav.read(wavFileName)
            signal=signal/(2**15) # scale it to float in +/-1 range
            self.sounds=signal
            # check whether sound player sample rate matches wav sample rate
            if not(self.soundPlayer.sampleRate == fs):
                dblog('************************\n'+
                  'Warning: wav file is at sample rate {}, hardware sample rate is {}'.format(fs,self.soundPlayer.sampleRate)+
                  '************************')
    
    def ready(self):
        super().ready()
        self.setLevel()
        self.isReady=True
        return self.isReady
   
#%% define tone stimulus object
toneDefaultStimulusParams= {
        'duration (s)': 0.2,
        'frequency (Hz)': 2000,
        'phase (rad)' : 0,
        'ABL (dB)': 50,
        'ITD (ms)': 0,
        'ILD (dB)': 0,
        'loop': False}

class toneObject(stimObject):
    def __init__(self, freq=1000):
        super().__init__()
        self.stimParams=toneDefaultStimulusParams.copy()
        self.stimParams['frequency (Hz)']=freq

    def makeWaveform(self):
        Nsamples=self.soundPlayer.ms2samples(self.stimParams['duration (s)']*1000) 
        t=np.linspace(0,self.stimParams['duration (s)'],Nsamples)
        x=np.cos(2*np.pi*self.stimParams['frequency (Hz)']*t+self.stimParams['phase (rad)']) 
        if self.ears=='both':
            self.sounds=np.vstack((x,x)).transpose()
        else:
            self.sounds=x
            
    def ready(self):
        super().ready()
        self.makeWaveform()
        self.applyEnvelope()
        self.setLevel()
        self.applyITD()
        self.applyILD()
        
        self.isReady=True
        return self.isReady
        
    def stimIsRight(self):
        # we decide on whether the stimulus "should be heard" on the right based on 
        #  ILD unless that is zero.
        isRight = (self.stimParams['ITD (ms)'] > 0)
        return isRight


        
#%% define noise stimulus object
noiseDefaultStimulusParams= {
        'duration (s)': 0.1,
        'ABL (dB)': 50,
        'ITD (ms)': 0,
        'ILD (dB)': 0,
        'loop': False}

class noiseObject(stimObject):
    def __init__(self):
        super().__init__()
        self.stimParams=noiseDefaultStimulusParams.copy()

    def makeWaveform(self):
        Nsamples=self.soundPlayer.ms2samples(self.stimParams['duration (s)']*1000) 
        #t=np.linspace(0,self.stimParams['duration (s)'],Nsamples)
        x=np.random.normal(0,1,Nsamples)
        if self.ears=='both':
            self.sounds=np.vstack((x,x)).transpose()
        else:
            self.sounds=x

    def ready(self):
        super().ready()
        self.makeWaveform()
        self.applyEnvelope()
        self.setLevel()
        self.applyITD()
        self.applyILD()
        
        self.isReady=True
        return self.isReady
        
    def stimIsRight(self):
        # we decide on whether the stimulus "should be heard" on the right based on 
        #  ILD unless that is zero.
        isRight = (self.stimParams['ITD (ms)'] > 0)
        return isRight  

frozenNoiseDefaultStimulusParams= {
        'duration (s)': 3,
        'ABL (dB)': 0,
        'ITD (ms)': 0,
        'ILD (dB)': 0,
        'leftSeed': 10,
        'rightSeed' : 20,
        'loop': False}

class frozenNoiseObject(noiseObject):    
    # makes gaussian noise from given seed values, so that identical noise can be reproduced.
    # different seeds can be specified for the left and right ears to get uncorrelated noise for left and right.

    def __init__(self):
        super().__init__()
        self.stimParams=frozenNoiseDefaultStimulusParams.copy()

    def makeWaveform(self):
        Nsamples=self.soundPlayer.ms2samples(self.stimParams['duration (s)']*1000) 
        if self.ears=='both':
            rng = np.random.default_rng(self.stimParams['leftSeed'])
            xL=rng.standard_normal(Nsamples)
            xL[-1]=0 # don't end sound on a non-zero DC value. 
            rng = np.random.default_rng(self.stimParams['rightSeed'])
            xR=rng.standard_normal(Nsamples)
            xR[-1]=0 # don't end sound on a non-zero DC value. 
            self.sounds=np.vstack((xL,xR)).transpose()
        elif self.ears=='left':
            rng = np.random.default_rng(self.stimParams['leftSeed'])
            xL=rng.standard_normal(Nsamples)
            xL[-1]=0 # don't end sound on a non-zero DC value. 
            self.sounds=xL
        elif self.ears=='right':
            rng = np.random.default_rng(self.stimParams['rightSeed'])
            xR=rng.standard_normal(Nsamples)
            xR[-1]=0 # don't end sound on a non-zero DC value. 
            self.sounds=xR
        else:
            raise Exception('frozenNoiseObject.makeWaveform does not know how to handle ears='+self.ears)
    
    
class pinkNoiseObject(noiseObject):
   
    def makeWaveform(self):
        """
        adapted from https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/generator.py
        Pink noise.
        :param N: Amount of samples.
        :param state: State of PRNG.
        :type state: :class:`np.random.RandomState`
        Pink noise has equal power in bands that are proportionally wide.
        Power density decreases with 3 dB per octave.
        """
        N=self.soundPlayer.ms2samples(self.stimParams['duration (s)']*1000) 
        #np.random.seed(0)
        uneven = N%2
        X = np.random.randn(N//2+1+uneven) + 1j * np.random.randn(N//2+1+uneven)
        S = np.sqrt(np.arange(len(X))+1.) # +1 to avoid divide by zero
        y = (np.fft.irfft(X/S)).real
        if uneven:
            y = y[:-1]
        if self.ears=='both':
            self.sounds=np.vstack((y,y)).transpose()
        else:
            self.sounds=y
    
#%% define a super flat base 2 noise stimulus object
        
def flatBase2NoiseSnip(N, randSeed=0):
    # uses IFFT method to make spectrally completely flat noise of length 2**N
    np.random.seed(randSeed)
    phases=np.random.uniform(0,2*np.pi,(2**(N-1))-1)
    spect=np.exp(1j*phases)
    spect=np.concatenate(([0], spect, [1], np.flipud(np.conj(spect))))
    noise=np.real(np.fft.ifft(spect))
    noise=noise/np.std(noise) # normalise to unit RMS
    return noise
    
flatBase2NoiseDefaultStimulusParams= {
        'N': 13,
        'randSeed': 1,
        'ABL (dB)': 50,
        'ITD (ms)': 0,
        'ILD (dB)': 0,        
        'loop': False}

class flatBase2Noise(stimObject):
    def __init__(self):
        super().__init__()
        self.stimParams=flatBase2NoiseDefaultStimulusParams.copy()

    def makeWaveform(self):
        x=flatBase2NoiseSnip(self.stimParams['N'],self.stimParams['randSeed'])
        self.stimParams['duration (s)']=self.soundPlayer.samples2ms(len(x))
        #t=np.linspace(0,self.stimParams['duration (s)'],Nsamples)
        if self.ears=='both':
            self.sounds=np.vstack((x,x)).transpose()
        else:
            self.sounds=x

    def ready(self):
        super().ready()
        self.makeWaveform()
        self.applyEnvelope()
        self.setLevel()
        self.applyITD()
        self.applyILD()
        
        self.isReady=True
        return self.isReady
        
#%% 
HiJeeNoiseDefaultStimulusParams= {
        'N': 13,
        'ABL (dB)': 50,
        'ITD (ms)': 0,
        'ILD (dB)': 0, 
        'randSeed0': 520,
        'randSeed1': 6,
        'randSeed2': 6,
        'randSeed3': 6,
        'randSeed4': 6,
        'randSeed5': 6,
        'envelope' : 'cos50',
#        'risefall_ms' : 5,
        'loop': False}

class HiJeeNoise(stimObject):
    def __init__(self):
        super().__init__()
        self.stimParams=HiJeeNoiseDefaultStimulusParams.copy()
        
    def makeWaveform(self):
        # the waveform is made of ifft flat noise segments, 
        # using the randSeed parameters.
        # randSeed0 defines the head/tail,
        # the other randSeeds define the middle bits
        #
        # Make a head and tail, with twice the number of samples 
        # and the requisit envelope
        headAndTail=flatBase2NoiseSnip(self.stimParams['N']+1,self.stimParams['randSeed0'])
        headAndTail=applyEnvelope(headAndTail,self.stimParams['envelope'])
        # Make a body
        x=[]
        rSeedIdx=1
#        riseFallN=0
        while 'randSeed{}'.format(rSeedIdx) in self.stimParams.keys():
            seed = int(np.round(self.stimParams['randSeed{}'.format(rSeedIdx)]))
            #dblog('next seed is ', seed)
            y=flatBase2NoiseSnip(self.stimParams['N'],seed)
#            if riseFallN==0:
#                riseFallN=self.soundPlayer.ms2samples(self.stimParams['riseFall_ms'])
#            x=concatWithOverlap(x,y,riseFallN)
            x=np.concatenate((x,y))
            rSeedIdx+=1
        # attach head and tail to body, as well as a 100 ms silence at the beginning
        silence=np.zeros(int(self.soundPlayer.sampleRate*0.1))
        L=2**self.stimParams['N']
        x=np.concatenate((silence, headAndTail[0:L],x,headAndTail[-L:]))          
        # make stereo if needed
        if self.ears=='both':
            self.sounds=np.vstack((x,x)).transpose()
        else:
            self.sounds=x
        # calculate final duration
        self.stimParams['duration (s)']=self.soundPlayer.samples2ms(len(x))/1000

    def ready(self):
        super().ready()
        self.makeWaveform()
        self.applyEnvelope()
        self.setLevel()
        self.applyITD()
        self.applyILD()
        
        self.isReady=True
        return self.isReady
    
    def correctResponse(self,aResponse):
        # indicate whether aResponse is a correct response for the current stimulus
        # note that for a HiJeeDRC or HiJee noise, any response is incorrect if the 
        #  response occurs while the stimulus is still playing.
        # Otherwise, a correct response is right of the stimulus is repeating, i.e. if randSeed1 == randSeed2
        if self.playing():
            self.stimParams['responseTooEarly']=True
            return False
        else:
            self.stimParams['responseTooEarly']=False
            responseIsRight=(aResponse=='RIGHT')
            stimIsRepeated=(self.stimParams['randSeed1']==self.stimParams['randSeed2'])
            return (responseIsRight==stimIsRepeated)    
        
#%% define DRC stimulus object
DRCdefaultStimulusParams= {
        'chordDur (s)': 0.02,
        'minFreq (Hz)': 500,
        'maxFreq (Hz)': 20000,
        'numFreqSteps': 15, 
        'numChords': 12,
        'randSeed': 0,
        'ABL (dB)': 60,
        'dB range +/-' : 15,
        'ITD (ms)': 0,
        'ILD (dB)': 0,
        'riseFall_ms': 5,
        'loop': False}

class DynamicRandomChord(stimObject):
    def __init__(self):
        super().__init__()
        self.stimParams=DRCdefaultStimulusParams.copy()

    def makeWaveform(self):
        # make a list of frequencies
        minF=np.log10(self.stimParams['minFreq (Hz)'])
        maxF=np.log10(self.stimParams['maxFreq (Hz)'])
        riseFall=self.stimParams['riseFall_ms']
        self.freqs=np.logspace(minF,maxF, self.stimParams['numFreqSteps'])
        # make a set of tone pips
        Nsamples=self.soundPlayer.ms2samples(self.stimParams['chordDur (s)']*1000) 
        t=np.linspace(0,self.stimParams['chordDur (s)'],Nsamples)
        self.pips=np.zeros((self.stimParams['numFreqSteps'],len(t)))
        riseFallN=self.soundPlayer.ms2samples(riseFall) 
        # halfRiseFall=int(np.floor(riseFallN/2))
        cosEnv=cosWin(np.ones(t.shape),riseFallN)
        for ii in range(len(self.freqs)):                  
            self.pips[ii,:]=np.cos(2*np.pi*self.freqs[ii]*t) * cosEnv
        # make an array of stimulus amplitudes
        toneLevel=lin2dB(dB2lin(self.stimParams['ABL (dB)'])/self.stimParams['numFreqSteps'])
        np.random.seed(self.stimParams['randSeed'])
        self.pip_dB=np.random.uniform(toneLevel-self.stimParams['dB range +/-'], toneLevel+self.stimParams['dB range +/-'], \
                                      (self.stimParams['numFreqSteps'],self.stimParams['numChords']))
        pipRMS=dB2lin(self.pip_dB, self.reference)
        # build DRC, one chord at a time. We overlap subsequent chords at half a rise time.
        totalN = (Nsamples-riseFallN) * self.stimParams['numChords'] + riseFallN
        x=None
        for nC in range(pipRMS.shape[1]):
            chord=self.pips[0,:]*pipRMS[0,nC]
            for nF in range(1,pipRMS.shape[0]):
                chord+=self.pips[nF,:]*pipRMS[nF,nC]
            if x is None:
                x=np.zeros(totalN)
                x[0:Nsamples]+=chord
            else:
                start=nC*(Nsamples-riseFallN)
                x[start:start+Nsamples]+=chord
                
        if self.ears=='both':
            self.sounds=np.vstack((x,x)).transpose()
        else:
            self.sounds=x
        # calculate final duration
        self.stimParams['duration (s)']=self.soundPlayer.samples2ms(len(x))/1000    

    def ready(self):
        super().ready()
        self.makeWaveform()
        self.calibrateWaveforms()
        self.applyITD()
        self.applyILD()
        
        self.isReady=True
        return self.isReady   

#%% 
        
def concatWithOverlap(x,y,Noverlap):
    Nx=len(x)
    if Nx==0:
        return y
    
    z=np.concatenate((x,y[Noverlap:]))
    z[Nx-Noverlap:Nx]+=y[0:Noverlap]
    return z
#%%
HiJeeDRCdefaultStimulusParams= {
        'chordDur (s)': 0.02,
        'minFreq (Hz)': 500,
        'maxFreq (Hz)': 20000,
        'numFreqSteps': 15, 
        'numChords': 12,
        'ABL (dB)': 60,
        'dB range +/-' : 20,
        'ITD (ms)': 0,
        'ILD (dB)': 0, 
        'randSeed0': 0,
#        'randSeed1': 10,
#        'randSeed2': 11,
#        'randSeed3': 12,
#        'randSeed4': 13,
#        'randSeed5': 14,
        'randSeed1': 1,
        'randSeed2': 1,
        'randSeed3': 1,
        'randSeed4': 1,
        'randSeed5': 1,
        'riseFall_ms': 5,
        'loop': False}

class HiJeeDRC(stimObject):
    def __init__(self):
        self.stimParams=HiJeeDRCdefaultStimulusParams.copy()
        
    def makeWaveform(self):
        # the waveform is made of DRC segments, 
        # using the randSeed parameters.
        x=[]
        rSeedIdx=0
        riseFallN=0
        while 'randSeed{}'.format(rSeedIdx) in self.stimParams.keys():
            seed = int(np.round(self.stimParams['randSeed{}'.format(rSeedIdx)]))
            #dblog('next seed is ', seed)
            y=DynamicRandomChord()
            y.ears='left'
            y.stimParams=self.stimParams.copy()
            y.stimParams['randSeed']=seed
            y.ready()
            if riseFallN==0:
                riseFallN=self.soundPlayer.ms2samples(y.stimParams['riseFall_ms'])
            #x=np.concatenate((x,y.sounds))
            x=concatWithOverlap(x,y.sounds,riseFallN)
            rSeedIdx+=1
        # attach a tail to body, as well as a 100 ms silence at the beginning
        seed = int(np.round(self.stimParams['randSeed0']))
        #dblog('next seed is ', seed)
        y=DynamicRandomChord()
        y.ears='left'
        y.stimParams=self.stimParams.copy()
        y.stimParams['randSeed']=seed
        y.ready()
        x=concatWithOverlap(x,y.sounds,riseFallN)
        silence=np.zeros(int(self.soundPlayer.sampleRate*0.1))
        x=np.concatenate((silence, x))          
        # make stereo if needed
        if self.ears=='both':
            self.sounds=np.vstack((x,x)).transpose()
        else:
            self.sounds=x
        # calculate final duration
        self.stimParams['duration (s)']=self.soundPlayer.samples2ms(len(x))/1000

    def ready(self):
        super().ready()
        self.makeWaveform()
        self.applyITD()
        self.applyILD()
        
        self.isReady=True
        return self.isReady     
    
    def correctResponse(self,aResponse):
        # indicate whether aResponse is a correct response for the current stimulus
        # note that for a HiJeeDRC or HiJee noise, any response is incorrect if the 
        #  response occurs while the stimulus is still playing.
        # Otherwise, a correct response is right of the stimulus is repeating, i.e. if randSeed1 == randSeed2
        if self.playing():
            self.stimParams['responseTooEarly']=True
            return False
        else:
            self.stimParams['responseTooEarly']=False
            responseIsRight=(aResponse=='RIGHT')
            stimIsRepeated=(self.stimParams['randSeed1']==self.stimParams['randSeed2'])
            return (responseIsRight==stimIsRepeated)    
    
#%% define a general purpose 2 interval object composed of two stimulus objects played one after the other

class twoIntervalObject(stimObject):
    """ Todo:
        - silence at the mo always stereo. Should depend on s1
        
        """
    def __init__(self, stim1, stim2, silentInterval=0.5):
        super().__init__()
        self.stim1=stim1
        self.stim2=stim2
        self.sampleRate=stim1.sampleRate
        self.silentInterval=silentInterval
        self.stimParams={}
        self.copyStimParamsFromSubstim()
        self.silence=silence(duration=silentInterval)
        
    def copyStimParamsFromSubstim(self):
        for key in self.stim1.stimParams.keys():
            newkey='1_'+key
            self.stimParams[newkey]=self.stim1.stimParams[key]
        for key in self.stim2.stimParams.keys():
            newkey='2_'+key
            self.stimParams[newkey]=self.stim2.stimParams[key]
            
    def copyStimParamsToSubstim(self):
        for key in self.stimParams.keys():
            if key[:3]=='1_':
                self.stim1.stimParams[key[2:]]=self.stimParams[key]
            if key[:3]=='2_':
                self.stim2.stimParams[key[2:]]=self.stimParams[key]
        
    def ready(self):
        self.copyStimParamsToSubstim()
        self.stim1.ready()
        self.stim2.ready()
        if not self.silence.stimParams['duration (s)']==self.silentInterval:
            self.silence=silence(duration=self.silentInterval)
        self.sounds=np.concatenate((self.stim1.sounds, self.silence.sounds, self.stim2.sounds))
        self.isReady=True
    

#%% define the base clickTrain stimulus object
clickTrainDefaultStimulusParams= {
        'duration (s)': 0.2,
        'clickRate (Hz)': 500,
        'loop': False}
            
class clickTrainObject(stimObject):
    def __init__(self, defaultParams=clickTrainDefaultStimulusParams):
        self.clickShape=np.array([1,0])                 
        super().__init__()   
        self.stimParams=defaultParams.copy()

    def makeWaveform(self):
        ICIms=1000/self.stimParams['clickRate (Hz)']
        ICIsamples=self.soundPlayer.ms2samples(ICIms)
        _len=len(self.clickShape)
        if _len > ICIsamples:
            print('WARNING: Click duration is longer than click interval.')
        nClicks=int(np.round((self.stimParams['duration (s)']*1000/ICIms)+1))
        stimSamples=ICIsamples*nClicks
        # try: # optional phase parameter in radians can shiftt the starting phase. 
        #     phaseOffset=self.stimParams['phase (rad)']/2/np.pi
        # except:
        #     phaseOffset=0
        # clickTimes=((np.array(range(nClicks))+phaseOffset)*ICIms)
        self.clickIdx=np.array(range(nClicks))*ICIsamples
        if self.ears=='both':
            self.sounds=np.zeros( (stimSamples,2) )
            for chan in range(2):
                for clk in range(len(self.clickIdx)):
                    self.sounds[self.clickIdx[clk]:self.clickIdx[clk]+_len,chan]+=self.clickShape
        else:
            self.sounds=np.zeros( (stimSamples) )
            for clk in range(len(self.clickIdx)):
                self.sounds[self.clickIdx[clk]:self.clickIdx[clk]+_len]+=self.clickShape
                
    def ready(self):
        super().ready()
        self.makeWaveform()
        #self.stimParams['duration (s)']=(nClicks+1)*ICIms/1000
        #self.sounds=(self.sounds*2**14).astype('int16')
        self.applyEnvelope()
        self.setLevel()     
        self.isReady=True
        return self.isReady   
    
    def correctResponse(self,aResponse):
        # indicate whether aResponse is a correct response for the current stimulus
        # undefined for the base class. Needs to be overridden in derived stimuli if needed.
        return aResponse == self.ears.upper()
    

class lateClickTrainObject(clickTrainObject):
    # a click train where the clicks appear at the end, not the beginning, of each period

    def makeWaveform(self):
        ICIms=1000/self.stimParams['clickRate (Hz)']
        ICIsamples=self.soundPlayer.ms2samples(ICIms)
        _len=len(self.clickShape)
        if _len > ICIsamples:
            print('WARNING: Click duration is longer than click interval.')
        nClicks=int(np.round((self.stimParams['duration (s)']*1000/ICIms)+1))
        stimSamples=ICIsamples*nClicks
        # try: # optional phase parameter in radians can shiftt the starting phase. 
        #     phaseOffset=self.stimParams['phase (rad)']/2/np.pi
        # except:
        #     phaseOffset=0
        # clickTimes=((np.array(range(nClicks))+phaseOffset)*ICIms)
        self.clickIdx=np.array(range(nClicks))*ICIsamples
        self.clickIdx+=(ICIsamples-_len) #<= this distinguishes this clicktrain from teh default one
        if self.ears=='both':
            self.sounds=np.zeros( (stimSamples,2) )
            for chan in range(2):
                for clk in range(len(self.clickIdx)):
                    self.sounds[self.clickIdx[clk]:self.clickIdx[clk]+_len,chan]+=self.clickShape
        else:
            self.sounds=np.zeros( (stimSamples) )
            for clk in range(len(self.clickIdx)):
                self.sounds[self.clickIdx[clk]:self.clickIdx[clk]+_len]+=self.clickShape

class lateClickTrainObject2(clickTrainObject):
    # a click train where the clicks appear at the end, not the beginning, of each period
    # like lateClickTrainObject, but under- rather than over-estimates click train
    def makeWaveform(self):
        ICIms=1000/self.stimParams['clickRate (Hz)']
        ICIsamples=self.soundPlayer.ms2samples(ICIms)
        _len=len(self.clickShape)
        if _len > ICIsamples:
            print('WARNING: Click duration is longer than click interval.')
        nClicks=int(np.floor((self.stimParams['duration (s)']*1000/ICIms)))
        stimSamples=ICIsamples*nClicks
        # try: # optional phase parameter in radians can shiftt the starting phase. 
        #     phaseOffset=self.stimParams['phase (rad)']/2/np.pi
        # except:
        #     phaseOffset=0
        # clickTimes=((np.array(range(nClicks))+phaseOffset)*ICIms)
        self.clickIdx=np.array(range(nClicks))*ICIsamples
        self.clickIdx+=(ICIsamples-_len) #<= this distinguishes this clicktrain from teh default one
        if self.ears=='both':
            self.sounds=np.zeros( (stimSamples,2) )
            for chan in range(2):
                for clk in range(len(self.clickIdx)):
                    self.sounds[self.clickIdx[clk]:self.clickIdx[clk]+_len,chan]+=self.clickShape
        else:
            self.sounds=np.zeros( (stimSamples) )
            for clk in range(len(self.clickIdx)):
                self.sounds[self.clickIdx[clk]:self.clickIdx[clk]+_len]+=self.clickShape

#%% define the base drewBeepTsh stimulus object
drewBeepTshDefaultStimulusParams= {
        'toneDur (s)': 0.2,
        'toneFreq (Hz)': 1100,
        'noiseDur (s)': 0.2,
        'noiseRanSeed' : 12345,
        'silenceDur (s)': 0.8,
        'ABL (dB)': 60,
        'loop': False}
            
class drewBeepTshObject(stimObject):
    def __init__(self):
        super().__init__()                    
        self.stimParams=drewBeepTshDefaultStimulusParams.copy()

    def makeWaveform(self):
        # make tone
        self.sounds=None
        myTone=toneObject()
        myTone.stimParams['frequency (Hz)']=self.stimParams['toneFreq (Hz)']
        myTone.stimParams['duration (s)']=self.stimParams['toneDur (s)']
        myTone.stimParams['ABL (dB)']=self.stimParams['ABL (dB)']
        myTone.stimParams['envelope']='cos5'
        myTone.ready()
        # make silences
        gap=silence(self.stimParams['silenceDur (s)'])
        after=silence(1)
        # make noise
        myNoise=pinkNoiseObject()
        myNoise.stimParams['duration (s)']=self.stimParams['noiseDur (s)']
        myNoise.stimParams['ABL (dB)']=self.stimParams['ABL (dB)']
        np.random.seed(self.stimParams['noiseRanSeed'])
        myNoise.ready()
        self.append([myTone,gap,myNoise,after])
        self.stimParams['duration (s)']=self.soundPlayer.samples2ms(self.sounds.shape[0])/1000
#        if self.ears=='both':
#            self.sounds=np.vstack((x,x)).transpose()
#        else:
#            self.sounds=x
                
    def ready(self):
        super().ready()
        self.makeWaveform()
        #self.stimParams['duration (s)']=(nClicks+1)*ICIms/1000
        #self.sounds=(self.sounds*2**14).astype('int16')
        self.isReady=True
        return self.isReady    

#%% artificial vowel 
        
# by default, let's make an /e/ like sound
artificialVowelDefaultStimulusParams= {
        'duration (s)': 0.1,
        'clickRate (Hz)': 150,
        'ABL (dB)': 60,
        'Formant1 (Hz)': 800,
        'Formant2 (Hz)': 1200,
        'envelope': 'cos 20',
        'loop': False}
        
class artificialVowel(clickTrainObject):
    def __init__(self, clickRate=0):
        clickTrainObject.__init__(self)
        self.clickShape=np.array([1,0]) # this could be replaced with a sexier glottal pulse shape
        self.stimParams=artificialVowelDefaultStimulusParams.copy()
        if clickRate != 0: 
            self.stimParams['clickRate (Hz)']=clickRate

    def makeWaveform(self):
        # we start off by making a click train and we then pass that clicktrain through 
        # 2nd order Butterworth (biquad) filters to add "formants"
        super().makeWaveform() # the clicktrain bit we inherit
        formant=1
        while 'Formant{} (Hz)'.format(formant) in self.stimParams:
            cf=self.stimParams['Formant{} (Hz)'.format(formant)]/(self.soundPlayer.sampleRate/2)
            b,a = butter(2,[cf*0.9, cf*1.1],btype='band')
            self.sounds=lfilter(b,a,self.sounds,axis=0)
            formant+=1

#%% artificial vowel pitch diff: stimulus for pitch discrimination.
# Present N stimuli in a row shich differ in pitch be deltaF0.
# Subject will be asked whether they sound same or different
            
            
        
# by default, let's make an /e/ like sound
artificialVowelPitchDiffDefaultParams= artificialVowelDefaultStimulusParams.copy()
artificialVowelPitchDiffDefaultParams['deltaF0semitones']=0.1
artificialVowelPitchDiffDefaultParams['vowelSequenceN']=6

class artificialVowelPitchDiff(stimObject):
    def __init__(self):
        super().__init__()
        self.stimParams=artificialVowelPitchDiffDefaultParams.copy()

    def ready(self):
        # Decide what pitches to use. Then concatenate vowels alternating in pitch between the 2 chosen values
        super().ready()
        halfPitchStep=self.stimParams['deltaF0semitones']/2 
        vowel1=artificialVowel(self.stimParams['clickRate (Hz)']*semitonesToFactor(halfPitchStep))
        vowel2=artificialVowel(self.stimParams['clickRate (Hz)']*semitonesToFactor(-halfPitchStep))
        vowel1.ready()
        vowel2.ready()
        self.sounds=None
        for Nvowel in range(self.stimParams['vowelSequenceN']):
            if Nvowel % 2 == 0:
                self.append([vowel1])
            else:
                self.append([vowel2])
        self.setLevel()
        self.isReady=True
        return self.isReady       

    def correctResponse(self,aResponse):
        # This is a 2 AFC stimulus. The correct response is 1 if deltaF0semitones is greater than zero
        if self.stimParams['deltaF0semitones']==0:
            correctChoice=1
        else:
            correctChoice=2      
        return ( aResponse == correctChoice )                 
        
#stim=artificialVowel(130)
#stim.ready()
#plt.clf()
#stim.plot()
#%% Single interval double vowel stimulus
doubleVowelStimDefaultParams= {
        'vowel dur (s)': 0.1,
        'F0_1': 150,
        'F0_2': 200,
        'deltaF0semitones' : 2,
        'ABL (dB)': 60,
        'formant1_1': 500,
        'formant1_2': 2400,
        'formant2_1': 800,
        'formant2_2': 1200,
        'envelope': 'cos 20',
        'loop': False}
        
class doubleVowelStim(stimObject):
    def __init__(self, pitches=[]):
        super().__init__()
        self.stimParams=doubleVowelStimDefaultParams.copy()
        # optionally we can specify pitches in the constructor as a list of four values
        if len(pitches) > 0:
            self.stimParams['F0_1']=pitches[0]
            self.stimParams['deltaF0semitones']=pitches[2]

    def correctResponse(self,aResponse):
        # This is a 1 IFC stimulus. The correct response is 1 if pitch Difference is zero
        if self.stimParams['deltaF0semitones']==0:
            correctChoice=1
        else:
            correctChoice=2
        return ( aResponse == correctChoice ) 

    def ready(self):
        super().ready()
        artificialVowelDefaultStimulusParams['Formant1 (Hz)']=self.stimParams['formant1_1']
        artificialVowelDefaultStimulusParams['Formant2 (Hz)']=self.stimParams['formant1_2']
        artificialVowelDefaultStimulusParams['duration (s)']=self.stimParams['vowel dur (s)']
        artificialVowelDefaultStimulusParams['envelope']=self.stimParams['envelope']
        # this is a 2afc stimulus.
        # Every time ready() is called we randomly choose which of the two sounds should 
        # have a pitch difference given by deltaF0factor
        scaleFactor=semitonesToFactor(self.stimParams['deltaF0semitones'])
        if random.random()>0.5:
            self.stimParams['F0_2']=round(self.stimParams['F0_1']*scaleFactor,3)
        else:
            self.stimParams['F0_2']=round(self.stimParams['F0_1']/scaleFactor,3)

        sound1=artificialVowel(self.stimParams['F0_1'])
        sound1.ready()
        sound2=artificialVowel(self.stimParams['F0_2'])
        # random phase offset between the vowels
        self.stimParams['phaseDiff_1']=random.random()*np.pi*2
        sound2.stimParams['phase (rad)']=self.stimParams['phaseDiff_1']
        sound2.ready()
        sound1.superimpose(sound2)
        self.sounds=sound1.sounds
        self.setLevel()
        self.stimParams['duration (s)']=sound1.stimParams['duration (s)']
        self.isReady=True
        return self.isReady       



#%% Two interval double vowel stimulus
Two_I_doubleVowelStimDefaultParams= {
        'vowel dur (s)': 0.3,
        'silent gap (s)': 0.5,
        'F0_1': 150,
        'F0_2': 150,
        'F0_3': 149,
        'F0_4': 150,
        'deltaF0semitones' : 2,
        'ABL (dB)': 60,
        'formant1_1': 300,
        'formant1_2': 3000,
        'formant2_1': 800,
        'formant2_2': 1200,
        'envelope': 'cos 20',
        'loop': False}
        
class Two_I_doubleVowelStim(stimObject):
    def __init__(self, pitches=[]):
        super().__init__()
        self.stimParams=Two_I_doubleVowelStimDefaultParams.copy()
        # optionally we can specify pitches in the constructor as a list of four values
        if len(pitches) > 0:
            self.stimParams['F0_1']=pitches[0]
            self.stimParams['F0_3']=pitches[1]
            self.stimParams['deltaF0semitones']=pitches[2]

    def correctResponse(self,aResponse):
        # This is a 2 IFC stimulus. The correct response is 1 if pitchDifferenceOnFirstSyllable is True
        if self.pitchDifferenceOnFirstSyllable:
            correctChoice=1
        else:
            correctChoice=2
        return ( aResponse == correctChoice ) 

    def ready(self):
        super().ready()
        artificialVowelDefaultStimulusParams['Formant1 (Hz)']=self.stimParams['formant1_1']
        artificialVowelDefaultStimulusParams['Formant2 (Hz)']=self.stimParams['formant1_2']
        artificialVowelDefaultStimulusParams['duration (s)']=self.stimParams['vowel dur (s)']
        artificialVowelDefaultStimulusParams['envelope']=self.stimParams['envelope']
        # this is a 2afc stimulus.
        # Every time ready() is called we randomly choose which of the two sounds should 
        # have a pitch difference given by deltaF0factor
        scaleFactor=semitonesToFactor(self.stimParams['deltaF0semitones'])
        self.pitchDifferenceOnFirstSyllable=(random.random()>0.5)
        if self.pitchDifferenceOnFirstSyllable:
            if random.random()>0.5:
                self.stimParams['F0_2']=round(self.stimParams['F0_1']*scaleFactor,3)
            else:
                self.stimParams['F0_2']=round(self.stimParams['F0_1']/scaleFactor,3)
            self.stimParams['F0_4']=self.stimParams['F0_3']
        else:
            if random.random()>0.5:
                self.stimParams['F0_4']=round(self.stimParams['F0_3']*scaleFactor,3)
            else:
                self.stimParams['F0_4']=round(self.stimParams['F0_3']/scaleFactor,3)
            self.stimParams['F0_2']=self.stimParams['F0_1']

        sound1=artificialVowel(self.stimParams['F0_1'])
        sound1.ready()
        sound3=artificialVowel(self.stimParams['F0_3'])
        # random phase offset between the vowels
        self.stimParams['phaseDiff_1']=random.random()*np.pi*2
        sound3.stimParams['phase (rad)']=self.stimParams['phaseDiff_1']
        sound3.ready()
        artificialVowelDefaultStimulusParams['Formant1 (Hz)']=self.stimParams['formant2_1']
        artificialVowelDefaultStimulusParams['Formant2 (Hz)']=self.stimParams['formant2_2']
        sound2=artificialVowel(self.stimParams['F0_2'])
        sound2.ready()
        sound4=artificialVowel(self.stimParams['F0_4'])
        self.stimParams['phaseDiff_2']=random.random()*np.pi*2
        sound4.stimParams['phase (rad)']=self.stimParams['phaseDiff_2']
        sound4.ready()
        sound1.superimpose(sound2)
        sound3.superimpose(sound4)
        sound3.setLevel()
        gap=silence(self.stimParams['silent gap (s)'])
        self.sounds=None
        self.append([sound1, gap, sound3])
        self.setLevel()
        self.isReady=True
        return self.isReady       

#%%        
#self=doubleVowelStim([random.random()*200+150,random.random()*200+150,0.5])
##%
#self.ready()
#plt.clf()
#self.plot()
##%
#self.play()
#%% ITD_ILD_clicktrain object
ITD_ILD_TrainDefaultStimulusParams= {
        'duration (s)': 0.2,
        'clickRate (Hz)': 500,
        'ABL (dB)': 90,
        'ITD (ms)': 0,
        'ILD (dB)': 0}
        
class ITD_ILD_clicktrain(clickTrainObject):

    def __init__(self):
        clickTrainObject.__init__(self)
        self.stimParams=ITD_ILD_TrainDefaultStimulusParams.copy()

    def ready(self):
        super().ready()
        self.applyITD()
        self.applyILD() 
        if 'Nloop' in self.stimParams.keys():
            if self.stimParams['Nloop'] > 1:
                if not ('loopInterval' in self.stimParams.keys()):
                    self.stimParams['loopInterval']=self.stimParams['duration (s)']
            self.loopNtimes(self.stimParams['Nloop'],self.stimParams['loopInterval'])        
        
        self.isReady=True
        return self.isReady
        
    def stimIsRight(self):
        # we decide on whether the stimulus "should be heard" on the right based on 
        #  ILD unless that is zero.
        isRight = (self.stimParams['ITD (ms)'] > 0)
        return isRight

    def correctResponse(self,aResponse):
        # indicate whether aResponse is a correct response for the current stimulus
        # note that responses for ENV stimuli are always deemed correct if the
        # response points in the direction of EITHER the envelope OR the fine structure
        correct=False # assume the worst
        responseIsRight=(aResponse=='RIGHT')
        if self.stimParams['ITD (ms)']==0 :
            paramToCheck='ILD (dB)'
        else:
            paramToCheck='ITD (ms)'
        
        if responseIsRight:
            if (self.stimParams[paramToCheck] > 0) :
                correct = True
        else:
            if (self.stimParams[paramToCheck] < 0) :
                correct = True
        return correct    
    
#%% ITD_ILD_clicktrain object
ITD_ILD_ENV_TrainDefaultStimulusParams= {
        'duration (s)': 0.01,
        'clickRate (Hz)': 900,
        'ABL (dB)': 50,
        'ITD (ms)': 0.16,
        'env ITD (ms)': -0.16,
        'envelope': 'hanning',
        'ILD (dB)': 0,
        'Nloop': 1}
        
class ITD_ILD_ENV_clicktrain(clickTrainObject):
# Click train with separate envelope and fine structure ITD
    def __init__(self):
        clickTrainObject.__init__(self)
        self.stimParams=ITD_ILD_ENV_TrainDefaultStimulusParams.copy()
        self.clickShape=np.array([1,0])
        self.Lenv=None
        self.Renv=None
        
#    def scale(self,a):
#        super().scale(a)
#        if type(a) == int or type(a) == float or type(a)==np.float64:
#            a=[a,a]
#        if not self.Lenv is None:
#            self.Lenv*=a[0]
#        if not self.Renv is None:
#            self.Renv*=a[1]       

    def applyEnvelope(self, anEnv=None):
        # this differs from the inherited version in that the envelope
        # functions themselves are shifted by stimParams['env ITD (ms)']
        if not self.ears=='both':
            raise Exception("ITD_ILD click train cannot be monaural")
        if self.sounds is None:
            return
        # get the appropriate envelope function
        if anEnv is None:
            anEnv=self.stimParams['envelope'] 
        ITDsamples=self.soundPlayer.ms2samples(self.stimParams['env ITD (ms)'])
        Nsamples=self.sounds.shape[0]-np.abs(ITDsamples)
        env=np.ones(Nsamples) 
        if anEnv[0:3]=='rec':
            pass 
        if anEnv[0:3]=='han':
            env=hanning(env.shape[0])
        if anEnv[0:3]=='cos': # apply 5 ms rise fall cosine window
            env=cosWin(env,5) 
        if anEnv[0:3]=='SAM': # apply sinusoidal amplitude modulation                        
            # make a sinusoidal envelope
            t=np.linspace(0,self.stimParams['duration (s)'],Nsamples)
            x=np.cos(2*np.pi*self.stimParams['modFreq (Hz)']*t)
            env=(x)*self.stimParams['modDepth']+1
            
        if ITDsamples==0:
            self.Lenv=env
            self.Renv=env
        else:
            ITDpad=np.zeros(np.abs(ITDsamples))
            if self.stimParams['env ITD (ms)'] > 0: # positive ITDs mean right ear earlier
                self.Lenv=np.hstack((ITDpad,env))
                self.Renv=np.hstack((env,ITDpad))
            else:
                self.Lenv=np.hstack((env,ITDpad))
                self.Renv=np.hstack((ITDpad,env))
        self.sounds[:,0]*=self.Lenv
        self.sounds[:,1]*=self.Renv
    
    def ready(self):
        super().ready()
        self.makeWaveform()
        self.applyITD()
        self.applyEnvelope()
        self.setLevel()     
        self.applyILD()     
        if self.stimParams['Nloop'] > 1:
            self.loopNtimes(self.stimParams['Nloop'],self.stimParams['loopInterval'])        
        self.isReady=True
        return self.isReady
    
    def correctResponse(self,aResponse):
        # indicate whether aResponse is a correct response for the current stimulus
        # note that responses for ENV stimuli are always deemed correct if the
        # response points in the direction of EITHER the envelope OR the fine structure
        correct=False # assume the worst
        responseIsRight=(aResponse=='RIGHT')
        if responseIsRight:
            if (self.stimParams['ITD (ms)'] > 0) or (self.stimParams['env ITD (ms)'] > 0):
                correct = True
        else:
            if (self.stimParams['ITD (ms)'] < 0) or (self.stimParams['env ITD (ms)'] < 0):
                correct = True
        return correct
      
    def plot(self): 
        if not self.isReady:
            self.ready()   
        if self.sounds is []:
            return
        taxis=np.array(range(self.sounds.shape[0]))/self.soundPlayer.sampleRate
        plt.plot(taxis,self.sounds[:,0],'b')
        plt.plot(taxis,self.sounds[:,1],'r')
        taxis=np.array(range(self.Renv.shape[0]))/self.soundPlayer.sampleRate
        envScalar=dB2lin(self.stimParams['ABL (dB)'], self.reference)
        # plt.plot(taxis,self.Lenv*envScalar,'b', alpha=0.2, linewidth=0.7)
        # plt.plot(taxis,self.Renv*envScalar,'r', alpha=0.2, linewidth=0.7)
        plt.plot(taxis,self.Lenv*envScalar,'b:', alpha=0.7, linewidth=1)
        plt.plot(taxis,self.Renv*envScalar,'r:', alpha=0.7, linewidth=1)
        plt.xlabel('time (s)')
        plt.ylabel('amplitude')

#%% ITD_ILD_SAM_clicktrain object. Click trains with sinusoidal amplitude modulation
#These are the stimuli for the 2018 GRF grant. In the proposal they are described as follows:
#    Our CI pulse trains will be sinusoidally amplitude modulated starting 
#    with cosine phase at a fixed AM rate of 6.66 Hz. That AM rate will put
#    2 cycles over the duration of our 300 ms long stimulus bursts, and it 
#    lies within the range of AM frequencies which have been shown to be
#    important for speech processing (Leong et al. 2017). During the 
#    AM training (stage 3) the animals will be trained to discriminate 0% 
#    modulation depth (unmodulated) from 50% modulation depth bursts. Human
#    studies (McKay and Henshall 2010) suggest that this should be perceptually
#    an easy task. Constant amplitude (0% modulation), cues the rat to collect
#    a reward on the left, while a modulated amplitude envelope cues to the right.
#    As the animals learn the task, the modulation depth will be adaptively 
#    decreased from its original 50% value. The RMS intensity of the stimuli 
#    will vary randomly from trial to trial to preclude the use of possible 
#    perceived loudness cues.
#Binaural Cues
#As in real-world listening tasks, where binaural spatial cues can provide
# valuable additional information, our CI rats will also receive binaural
# cues to facilitate learning the correct stimulus-reward associations. Thus,
# in their initial stage 3 training, ILD+ animals will always receive 0% AM 
# stimuli with a -3 dB ILD and a -0.16 ms ITD (where negative values mean the 
# left ear is louder or leading in time), while the 50% AM stimuli will have 
# +3 dB ILD and +0.16 ms ITD. Pilot experiments have shown that these 
# ILD and ITD values are very easily lateralized by CI rats. 
# Bear in mind that electrical hearing has a much smaller dynamic range 
# than acoustic hearing due to the absence of non-linear outer hair cell 
# amplification, so a +/- 3 dB electrical ILD is large. We know from pilot
# experiments that these large and congruent binaural cues will be easily 
# learned in a matter of days. Meanwhile, the ITD- cohort will also 
# receive +/- 3 dB ILDs which are congruent with the AM, but the ITDs
# of their stimuli will be random, chosen uniformly from a +/- 0.16 ms
# range. ITD- animals must therefore learn to “attend” to AM and ILD
# only and “ignore” (i.e. become insensitive to) ITD to perform the task.  

ITD_ILD_SAM_TrainDefaultStimulusParams= {
        'duration (s)': 0.6,
        'clickRate (Hz)': 900,
        'ABL (dB)': 10,
        'ITD (ms)': 0.12,
        'env ITD (ms)': 0.12,
        'modFreq (Hz)': 6.66,
        'modDepth': 1,
        'ILD (dB)': 3,
        'Nloop': 1,
        'loopInterval': 0.5}
        
class ITD_ILD_SAM_clicktrain(ITD_ILD_ENV_clicktrain):
# Click train with separate envelope and fine structure ITD
    def __init__(self):
        clickTrainObject.__init__(self)
        self.stimParams=ITD_ILD_SAM_TrainDefaultStimulusParams.copy()
        self.clickShape=np.array([1,0,-1,0])
        
    def applyEnvelope(self, anEnv=None):
        if self.stimParams['modDepth']==0:
            return # no envelope given. Nothing to do.
        if self.sounds is None:
            return
        super().applyEnvelope(anEnv='SAM')
    #     # make a sinusoidal envelope
    #     Nsamples=self.sounds.shape[0] 
    #     t=np.linspace(0,self.stimParams['duration (s)'],Nsamples)
    #     x=np.cos(2*np.pi*self.stimParams['modFreq (Hz)']*t)
    #     wdw=(x)*self.stimParams['modDepth']+1
    #     if self.sounds.shape[1] ==2:
    #         wdw=np.vstack((wdw,wdw)).transpose()
    #     self.sounds=self.sounds*wdw  
        
      
    def plot(self):
        ITD_ILD_clicktrain.plot(self)
        
    def correctResponse(self,aResponse):
        # all responses are correct for probe trials
        if  'rewardType' in self.stimParams.keys():
            if self.stimParams['rewardType'][0:4].lower() == 'prob':
                dblog('-------THIS IS A PROBE TRIAL-----------')
                return True
        # for these SAM click trains, the correct response is RIGHT if modDepth > 0  
        responseIsRight=(aResponse=='RIGHT')
        modDepthAboveZero=(self.stimParams['modDepth']>0)
        # for "probe trials" the binaural cues may point in the opposite
        # direction of the AM, or they may conflict with each other. 
        # need to think more carefully as for ITD- animals this will not work to identify probeTrials
        return (responseIsRight==modDepthAboveZero)
    
#%% class narrowband clicktrain

class BandPassNoise:
    
    def __init__(self, sample_rate, duration):
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples = int(self.sample_rate * self.duration)
        return

    def fftnoise(self, f):
        f = np.array(f, dtype='complex')
        Np = (len(f) - 1) // 2
        phases = np.random.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1:Np+1] *= phases
        f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
        return np.fft.ifft(f).real
    
    def band_limited_noise(self, min_freq, max_freq):
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        freqs = np.abs(np.fft.fftfreq(self.samples, 1/self.sample_rate))
        f = np.zeros(self.samples)
        idx = np.where(np.logical_and(freqs>=self.min_freq, freqs<=self.max_freq))[0]
        f[idx] = 1
        return self.fftnoise(f)
        
ITD_ILD_NarrowbandDefaultParams= {
    'duration (s)': 0.2,
    'clickRate (Hz)': 100,
    'ABL (dB)': 0,
    'ITD (ms)': 0,
    'ILD (dB)': 0,
    'minFreq (Hz)': 500,
    'maxFreq (Hz)': 2000,
    'clickDur (ms)': 5,
    'loop': False}
        
class ITD_ILD_narrowbandTrain(ITD_ILD_clicktrain):

    def __init__(self):
        clickTrainObject.__init__(self)
        self.stimParams=ITD_ILD_NarrowbandDefaultParams.copy()

    def  setClickShape(self):
        # Cecilia, put code to geenrate short noise snipet and return it as XXX in here
        self.clickShape= BandPassNoise(self.soundPlayer.sampleRate,self.stimParams['clickDur (ms)']/1000).band_limited_noise(self.stimParams['minFreq (Hz)'], self.stimParams['maxFreq (Hz)'])

#%% ITD_ILD_FM_clicktrain object
# changes frequency of clicktrain to do frequency modulated sweep
ITD_ILD_FM_TrainDefaultStimulusParams= {
        'duration (s)': 0.2,
        'clickRate (Hz)': 500,
        'ABL (dB)': 0,
        'ITD (ms)': 0,
        'ILD (dB)': 0,
        'octaves/s': 5,
        'loop': False}
        
class ITD_ILD_FM_clicktrain(ITD_ILD_clicktrain):

    def __init__(self):
        ITD_ILD_clicktrain.__init__(self)
        self.stimParams=ITD_ILD_FM_TrainDefaultStimulusParams.copy()

    def ready(self):
        super().ready()
        ABL=self.stimParams['ABL (dB)']
        # because this is an FM click train, the 
        # inter click intervals need to change dynamically
        ICI=1/self.stimParams['clickRate (Hz)']
        #octavesChange=self.stimParams['octaves/s']*self.stimParams['duration (s)']
        octavesPerICI=self.stimParams['octaves/s']*ICI
        increment=2**(-octavesPerICI)
        dur=self.stimParams['duration (s)']
        clickTimes=[0]
        ICI*=increment
        while (clickTimes[-1]) < dur:
            clickTimes+=[clickTimes[-1]+ICI]
            octavesPerICI=self.stimParams['octaves/s']*ICI
            increment=2**(-octavesPerICI)
            ICI*=increment
        clickTimes=np.array(clickTimes)*1000 # convert and scale to ms
        ITDvals=self.stimParams['ITD (ms)']
        self.RclickIdx=self.soundPlayer.ms2samples(clickTimes-ITDvals/2)
        self.LclickIdx=self.soundPlayer.ms2samples(clickTimes+ITDvals/2) 
        self.ITDsMs=self.soundPlayer.samples2ms(self.LclickIdx[0]-self.RclickIdx[0])
        self.stimParams['ITD (ms)']=self.ITDsMs
        
        amp = 10**( float(ABL)/20)
    
        self.sounds=np.zeros( (self.soundPlayer.ms2samples(clickTimes[-1]+5),2) )

        _len=len(self.clickShape)
        for clk in range(len(self.LclickIdx)):
            self.sounds[self.LclickIdx[clk]:self.LclickIdx[clk]+_len,0]=amp*self.clickShape
            self.sounds[self.RclickIdx[clk]:self.RclickIdx[clk]+_len,1]=amp*self.clickShape

        #self.sounds=(self.sounds*2**14).astype('int16')
        self.isReady=True        
        return self.isReady
    
#%% temporal weighting function click train object        
#  for measuring ITD temporal weighting functions a la Stecker
TWF_ITD_defaultStimulusParams= {
            'numClicks': 8,
            'clickRate (Hz)' : 300,
            'jitter (ms)': 0.1,
            'offset (ms)': 0.0,
            'randomSeed' : 364,
            'ABL (dB)' : 50,
            'Nloop' : 3,
            'loopInterval': 0.05}

TWF_ITD_alternativeDefaultParams= {
            'numClicks': 4,
            'clickRate (Hz)' : 300,
            'ABL (dB)' : 50,
            'preSilence (s)': 0.1,
            'ITD 0' : 0.3,
            'ITD 1' : 0.3,
            'ITD 2' : 0.3,
            'ITD 3' : 0.3,
            'Nloop' : 1,
            'loopInterval': 0.05}
            
class ITD_TWFclicktrain(stimObject):

    def __init__(self, stimParams=TWF_ITD_defaultStimulusParams):
        super().__init__()
        self.stimParams=stimParams.copy()
        self.clickShape=np.array([1,0])
        # define a timeout sound as negative feedback thing
        # self.timeOutSound=ITD_ILD_FM_clicktrain()
        # self.timeOutSound.stimParams=ITD_ILD_FM_TrainDefaultStimulusParams
        # self.timeOutSound.clickShape=self.clickShape
        # self.timeOutSound.stimParams["loop"]=True
        # self.timeOutSound.ready()

    def ready(self):
        super().ready()
        if not self.ears=='both':
            dblog("### warning: ITD_TWFclicktrain.ears must be 'both'. They cannot be monaural. Set to both.")
            self.ears='both'     
        nClicks=self.stimParams['numClicks']
        ICIms=1000/self.stimParams['clickRate (Hz)']
        clickTimes=np.array(range(nClicks))*ICIms
        if 'preSilence (s)' in self.stimParams:
            clickTimes+=self.stimParams['preSilence (s)']*1000
        clickTimes=self.soundPlayer.ms2samples(clickTimes)
        # the stimuli can EITHER be specified using a randomSeed, jitter and offset
        # OR the ITD values can be specified explicitly
        if 'randomSeed' in self.stimParams:
            nClicks=self.stimParams['numClicks']
            np.random.seed(np.int(np.round(self.stimParams['randomSeed'])))
            jitterRange=self.stimParams['jitter (ms)']
            ITDvals=np.random.uniform(-jitterRange,jitterRange,nClicks)
            ITDvals=ITDvals+self.stimParams['offset (ms)']
        else:
            ITDvals=[]
            nClicks=0
            while 'ITD {}'.format(nClicks) in self.stimParams:
                ITDvals.append(self.stimParams['ITD {}'.format(nClicks)])
                nClicks+=1
            
        self.clickTimes=self.soundPlayer.ms2samples(np.array(range(nClicks))*ICIms)
        self.LclickIdx=self.clickTimes.copy()
        self.RclickIdx=self.clickTimes.copy()

        self.applyITDvals(ITDvals)
        self.setLevel()

        if self.stimParams['Nloop'] > 1:
            self.loopNtimes(self.stimParams['Nloop'],self.stimParams['loopInterval'])
        self.isReady=True
        
    def applyITDvals(self, ITDvals):
        if type(ITDvals) == list:
            ITDvals=np.array(ITDvals)
        ITDvals=self.soundPlayer.ms2samples(ITDvals)
        #ITDvals
        for ii in range(len(ITDvals)):
            if ITDvals[ii]>0:
                self.LclickIdx[ii] += ITDvals[ii]
            else:
                self.RclickIdx[ii] -=ITDvals[ii]
        self.ITDsMs=self.soundPlayer.samples2ms(self.LclickIdx-self.RclickIdx)
        for ii in range(len(self.ITDsMs)):
            self.stimParams['ITD {}'.format(ii)]=self.ITDsMs[ii]
        
        _len=len(self.clickShape)    
        self.sounds=np.zeros( (self.clickTimes[-1]+self.clickTimes[1]+_len+1,2) )
        self.stimParams['duration (s)']=self.soundPlayer.samples2ms(self.sounds.shape[0])/1000
        #dblog('click shape length is ',_len)
        for clk in range(len(self.LclickIdx)):
            self.sounds[self.LclickIdx[clk]:self.LclickIdx[clk]+_len,0]=self.clickShape
            self.sounds[self.RclickIdx[clk]:self.RclickIdx[clk]+_len,1]=self.clickShape

    def correctResponse(self,aResponse):
        # indicate whether aResponse is a correct response for the current stimulus
        # note that responses for TWF stimuli are always deemed correct if the
        # jitter is larger than the offset and the stimulus is therefore 
        # potentially ambiguous
        if self.stimParams['jitter (ms)'] > abs(self.stimParams['offset (ms)']):
            return True
        responseIsRight=(aResponse=='RIGHT')
        return (responseIsRight==self.stimIsRight())

    def stimIsRight(self):
        # we decide on whether the stimulus "should be heard" on the right based on 
        #  ILD unless that is zero. For cue trading experiments that may have to change
        isRight = (self.stimParams['offset (ms)'] > 0)
        return isRight
    
    
#%% temporal weighting function click train object        
#  for measuring ILD temporal weighting functions a la Stecker
TWF_ILD_defaultStimulusParams= {
            'numClicks': 8,
            'clickRate (Hz)' : 300,
            'jitter (dB)': 2,
            'offset (dB)': 0.0,
            'randomSeed' : 364,
            'ABL (dB)' : 50,
            'Nloop' : 3,
            'loopInterval': 0.05}

TWF_ILD_alternativeDefaultParams= {
            'numClicks': 4,
            'clickRate (Hz)' : 300,
            'ABL (dB)' : 50,
            'preSilence (s)': 0.1,
            'ILD 0' : -6,
            'ILD 1' : 6,
            'ILD 2' : -6,
            'ILD 3' : 6,
            'Nloop' : 0,
            'loopInterval': 0.05}
            
class ILD_TWFclicktrain(stimObject):

    def __init__(self, stimParams=TWF_ILD_defaultStimulusParams):
        super().__init__()
        self.stimParams=stimParams.copy()
        # define a timeout sound as negative feedback thing
        self.clickShape=np.array([1,0])
        self.timeOutSound=ITD_ILD_FM_clicktrain()
        self.timeOutSound.stimParams=ITD_ILD_FM_TrainDefaultStimulusParams
        self.timeOutSound.clickShape=self.clickShape
        self.timeOutSound.stimParams["loop"]=True
        self.timeOutSound.ready()

    def ready(self):
        super().ready()
        if not self.ears=='both':
            dblog("### warning: ILD_TWFclicktrain.ears must be 'both'. They cannot be monaural. Set to both.")
            self.ears='both'     
        nClicks=self.stimParams['numClicks']
        ICIms=1000/self.stimParams['clickRate (Hz)']
        clickTimes=np.array(range(nClicks))*ICIms
        if 'preSilence (s)' in self.stimParams:
            clickTimes+=self.stimParams['preSilence (s)']*1000
        clickTimes=self.soundPlayer.ms2samples(clickTimes)
        # the stimuli can EITHER be specified using a randomSeed, jitter and offset
        # OR the ILD values can be specified explicitly
        if 'randomSeed' in self.stimParams:
            nClicks=self.stimParams['numClicks']
            np.random.seed(int(np.round(self.stimParams['randomSeed'])))
            jitterRange=self.stimParams['jitter (dB)']
            ILDvals=np.random.uniform(-jitterRange,jitterRange,nClicks)
            ILDvals=ILDvals+self.stimParams['offset (dB)']
            # copy the ILDs of the individual clicks to stimParams so that a datahandler will save them
            for ii in range(len(ILDvals)):
                self.stimParams['ILD {}'.format(ii)]=ILDvals[ii]
        else:
            ILDvals=[]
            nClicks=0
            while 'ILD {}'.format(nClicks) in self.stimParams:
                ILDvals.append(self.stimParams['ILD {}'.format(nClicks)])
                nClicks+=1
            ILDvals=np.array(ILDvals)

        scaleFactors=dB2lin(ILDvals/2)
        # copy the ILDs of the individual clicks to stimParams so that a datahandler will save them
        
        _len=len(self.clickShape)    
        self.sounds=np.zeros( (clickTimes[-1]+_len+5,2) )
        self.stimParams['duration (s)']=self.soundPlayer.samples2ms(self.sounds.shape[0])/1000
        #dblog('click shape length is ',_len)
        for clk in range(len(clickTimes)):
            # make clicks, scaling the left ear down and the right ear up by the appropriate scale factor
            self.sounds[clickTimes[clk]:clickTimes[clk]+_len,0]=self.clickShape/scaleFactors[clk]
            self.sounds[clickTimes[clk]:clickTimes[clk]+_len,1]=self.clickShape*scaleFactors[clk]

        self.setLevel()

        if self.stimParams['Nloop'] > 1:
            self.loopNtimes(self.stimParams['Nloop'],self.stimParams['loopInterval'])
        self.isReady=True

    def correctResponse(self,aResponse):
        # indicate whether aResponse is a correct response for the current stimulus
        # note that responses for TWF stimuli are always deemed correct if the
        # jitter is larger than the offset and the stimulus is therefore 
        # potentially ambiguous
        if self.stimParams['jitter (dB)'] > abs(self.stimParams['offset (dB)']):
            return True
        responseIsRight=(aResponse=='RIGHT')
        return (responseIsRight==self.stimIsRight())

    def stimIsRight(self):
        # we decide on whether the stimulus "should be heard" on the right based on 
        #  ILD unless that is zero. For cue trading experiments that may have to change
        isRight = (self.stimParams['offset (dB)'] > 0)
        return isRight
    

#%% blinking LED timeout stimulus for CI setup

class BlinkingLEDstimForCISetup:

    def __init__(self):
        self.timerObject=threading.Timer
        self.timer=None
        self.interval=0.1        
        self.isPlaying=False

    def ready(self):
        #self.stimParams['duration (s)']=config.timeoutDuration
        pass
        
    def playing(self):
        return self.isPlaying  
    
    def blink(self):
        self.soundPlayer.RP.SetTagVal('LickDetectOn',1) 
        time.sleep(0.1)
        self.soundPlayer.RP.SetTagVal('LickDetectOn',0)
        self.timer=self.timerObject(self.interval,self.blink) 
        self.timer.start()         

    def play(self):
        self.isPlaying=True
        self.timer=self.timerObject(self.interval,self.blink) 
        self.timer.start()
#        start=time.time()
#        while (time.time()-start) < self.stimParams['duration (s)']:
#            # the CI behaviour box has an LED fitted in parallel to the 
#            # lick detector power supply.
#            # We make it blink by switching the lick detectors on and off.
#             
#            time.sleep(0.1)        
    
    def stop(self):
        if self.timer is None:
            return # nothing to stop
        self.timer.cancel()
        self.isPlaying=False

    def correctResponse(self,aResponse):
        return True

    def stimIsRight(self):
        return True
        
#%%
    
def stimulatorDone():
    global soundPlayer
    if not soundPlayer is None:
        # default to Pygame hardware
        soundPlayer.done()
