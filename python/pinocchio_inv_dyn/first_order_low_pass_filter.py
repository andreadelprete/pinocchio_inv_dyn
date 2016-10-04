# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:44:07 2015

@author: adelpret
"""

from math import pi
import numpy as np
import matplotlib.pyplot as plt

class FirstOrderLowPassFilter(object):
    Ts = 0.001; # sampling period
    fc = 1;     # cut frequency
    tau = 0;    # time constant
    
    y_old = []; # previous output
    x_old = []; # previous input
    
    def __init__(self, sampling_period, cut_frequency, y0):
        self.Ts = sampling_period;
        self.fc = cut_frequency;
        self.tau = 1.0/(2.0*pi*self.fc);
        self.y_old = np.copy(y0);
        self.x_old = np.zeros(len(y0));
        
    def filter_data(self, x):
        y = (self.Ts*x + self.Ts*self.x_old - (self.Ts-2.0*self.tau)*self.y_old)/(2.0*self.tau+self.Ts);
        self.x_old = x;
        self.y_old = y;
        return y;
        
        
if __name__=='__main__':
    FC = 20;
    DT = 0.002;
    T = 50000;
    OFFSET = 50;
    lpf = FirstOrderLowPassFilter(DT, FC, np.array([OFFSET]));
    x = OFFSET + np.random.normal(0.0, 1.0, T);
    x_filt = np.zeros(T);
    for i in range(T):
        x_filt[i] = lpf.filter_data(x[i]);
    
    std_filt = np.std(x_filt);
    print "Standard deviation of noise after filtering at %f Hz is %f" % (FC,std_filt);
    print "This means that its amplitude has been reduced by a factor of %f" % (1.0/std_filt);
    plt.plot(x, 'b');
    plt.plot(x_filt, 'r');
    plt.show();
    