#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

            self.results['windowed_signal_frame'][:self.frame] = self.fr() * self.window()

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

def load_wave(wavfile, tsr = 16000.):

    """Function reading in the audio file. Handling many audio files types
    (all provided by the sndfile C library, which have to be installed independently)

    :param wavfile: path to the audio file
    :type wavfile: basestring
    :param tsr: sampling rate in Hz, if the audio file sr differs resampling is performed
    :type tsr: float/int

    :return:
                 * **r** (`numpy vector of floats`) audio file samples
                 * **tsr** (`float/int`) sampling rate

    """



    # load wave to process

    f = scikits.audiolab.Sndfile(wavfile, "r")

    if f.channels > 1:

        print("No multichanel data supported, program exits ...")

        sys.exit(0)

    sr = f.samplerate

    nf = f.nframes

    r = f.read_frames(nf, dtype=np.float32)

    if sr != tsr:

        r = resample(r,tsr/float(sr),'sinc_best')

    return r, tsr

def save_wave(wavfile, smpls, sr = 16000, format='wav'):

    """Function writes audiosamples to file

    :param wavfile: file name
    :type wavfile: basestring
    :param smpls: the samples
    :type smpls: numpy vector of floats
    :param sr: sampling rate in Hz
    :type sr: float/int
    :param format: format of the output file, wav is the Microsoft WAV
    :type format: basestring

    :returns: nothing

    """

    format = scikits.audiolab.Format(format)

    f = scikits.audiolab.Sndfile(wavfile, 'w', format, 1, sr)

    f.write_frames(smpls)

    f.close()

        
