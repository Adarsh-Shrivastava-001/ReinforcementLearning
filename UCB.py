#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:11:49 2019

@author: adarsh
"""

import numpy as np
import math

def data_gen(n_classes, prob):
    d=np.zeros((n_classes, 10000))
    for i in range(n_classes):
        d[i]=np.random.choice([0,1], size=10000, p=[1-prob[i], prob[i]])
    return d



def UCB(truth):
    n_classes=len(truth)
    test_cases=len(truth[0])
    init_trials=50
    expect=[1/len(truth)]*n_classes
    conf=[0]*n_classes
    n_step=[init_trials]*n_classes
    step=0
    rewards=0
    
    
    for i in range(n_classes):
        expect[i]=np.sum(truth[i][step:step+init_trials])/init_trials
        conf[i]=math.sqrt(math.log(2*init_trials*(i+1))/init_trials)
        rewards=rewards+np.sum(truth[i][step:step+init_trials])
        step=step+init_trials
    print()
        
    upper_bound=[expect[i]+conf[i] for i in range(n_classes)]
    
    
    while step<test_cases:
        action=np.argmax(upper_bound)
        if truth[action][step]==1:
            expect[action]=(expect[action]*n_step[action]/(n_step[action]+1))+(1/(n_step[action]+1))
            conf[action]=math.sqrt(math.log(2*step)/n_step[action])
            upper_bound[action]=expect[action]+conf[action]
            rewards=rewards+1
        else:
            expect[action]=(expect[action]*n_step[action]/(n_step[action]+1))
            conf[action]=math.sqrt(math.log(2*step)/n_step[action])
            upper_bound[action]=expect[action]+conf[action]
        
        n_step[action]+=1
        step=step+1

            
        
        
    step=0
    ran_rewards=0
    while step<test_cases:
        action=np.random.choice([0,1,2,3,4])
        if truth[action][step]==1:
            expect[action]=(expect[action]*n_step[action]/(n_step[action]+1))+(1/(n_step[action]+1))
            ran_rewards=ran_rewards+1
        else:
            expect[action]=(expect[action]*n_step[action]/(n_step[action]+1))
                    
        step=step+1
        
    
    
    print(n_step)
    print(expect)
    print(conf)
    print(upper_bound)
    print(rewards)
    print(ran_rewards)
    
        
        
    
    
    