#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 18:00:18 2021

@author: jan
"""

from os import path as ospath
clSource = '/home/colliculus/behaviourBoxes/software'
if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
    clSource = '/home/jan/Nextcloud/Documents/jan/behavbox' 
if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
    clSource = 'c:/Nextcloud/Documents/jan/behavbox' 
if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
    clSource = 'c:/Users/jan/Nextcloud/Documents/jan/behavbox' 
if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
    clSource = 'c:/users/colliculus'
if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
    clSource = 'c:/jan/behavbox'
if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
    clSource = 'd:/behavbox'
if not ospath.exists(clSource+'/ratCageProgramsV3/ClickTrainLibrary.py'):
    raise Exception('No valid path to ratCageProgramsV2 and 3 libraries')
from sys import path as syspath
syspath.append(clSource+'/ratCageProgramsV3')


location='Freiburg'
TDTcircuit=None
RXdevice="RZ6"
