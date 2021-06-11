#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from typing import Tuple

import numpy as np
import spectrum
from scipy.io import wavfile


class Utils:

    def __init__(self, ar_model_order):

        self._ar_model_order = ar_model_order

        self.list_of_computable_things = ['signal_frame','windowed_signal_frame','var_lpc','lpc']

        self.results = {}

    def _reset(self):

        """
        Resets the cache.

        """

        for e in self.list_of_computable_things:

            self.results[e] = None

    def set_frm(self,signal_frame):

        """
        Sets the frame of the signal - this is than used to compute the all signal representations

        :param signal_frame: signal frame, ndim = 1
        :type signal_frame: numpy vector of floats

        :returns: nothing

        """

        self._reset()

        self._frame = signal_frame.shape[0]

        self._window = np.ones((self._frame,))

        self.results['signal_frame'] = signal_frame 


    def window(self):

        """
        Gets the signal windowing function.

        :returns: the windowing function

        :rtype: numpy array with dimension 1 x frame

        """

        return self._window

    def fr(self):

        """
        Gets frame

        :returns: the frame
        :rtype: numpy vector of floats

        """

        if self.results['signal_frame'] is None:

            print("Frame not given (emtpy vector), program exits ...")

            sys.exit(1)

        else:

            return self.results['signal_frame']


    def fr_win(self):

        """
        Gets windowed frame

        :returns: the windowed frame
        :rtype: numpy vector of floats

        """

        if self.results['windowed_signal_frame'] is None:

            self.results['windowed_signal_frame'] = self.fr() * self.window()

        else:

            pass

        return self.results['windowed_signal_frame']

    def lpc(self):

        """
        Gets LPC coefficients

        :return: LPC with the leading 1

        """

        if self.results['lpc'] is None:

            _lpc, self.results['var_lpc'], k = spectrum.aryule(self.fr_win(), self._ar_model_order)

            self.results['lpc'] = np.concatenate((np.array([1]),_lpc))

        else:

            pass

        return self.results['lpc']

    def var_lpc(self):

        """
        Gets variance of the short term residual spectrum

        :return: short term residual variance

        """

        if self.results['var_lpc'] is None:

            self.results['lpc'], self.results['var_lpc'], k = spectrum.aryule(self.fr_win(), self._ar_order)

        else:

            pass

        return self.results['var_lpc']

def load_wav(file_ : str) -> Tuple[np.ndarray,int]:

    if not os.path.exists(file_):
        raise ValueError(f"The file {file_} does not exist.")

    sr, s = wavfile.read(file_)

    s = s/2.**15
    

    np.random.seed(1000) # causes the dither is same on each run

    s += np.random.randn(*s.shape)*1.e-6 # add dither to improve numerical behaviour

    return s, int(sr)

def save_wav(signal : np.ndarray, sampling_rate : int, file_name : str):

    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError:
            print(f'Can not create the directory {os.path.dirname(file_name)}')
        else:
            print(f'Path {os.path.dirname(file_name)} succesfully created.')

    wavfile.write(file_name, sampling_rate, np.int16(signal*2**15))


def clear_output_directory():

    if os.path.exists('../data/output/'): 
        if len(os.listdir('../data/output/')) > 0:
            shutil.rmtree('../data/output') 
    else: 
        try:
            os.makedirs('../data/output/')
        except OSError:
            print("Creation of ../data/output/ failed")
        else:
            print("Successfully created the directory ../data/output/")
