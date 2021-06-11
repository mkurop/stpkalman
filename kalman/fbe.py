
"""FBE module provides several utilities and signal parametrization methods. An implementation of the auditory model by
Van De Par is also included.
"""

# -*- coding: utf-8 -*_

__author__ = 'marcin'

import sys

import spectrum

import random

import scikits.audiolab

import numpy as np

import matplotlib.pyplot as plt

#import fls

import copy

import time

#import enmc

from scipy.spatial import cKDTree

import scipy.special

from scikits.samplerate import resample

# from MfccKaldi import MfccKaldi

class sin2cos2:

    """
    Class for computing signal windowing function with sin(x)^2 and cos(x)^2 tails.

    :param frame: the frame length

    :param overlap: the size of the overlaping part of the window (the length of the tails on both sides)

    :type frame: int

    :type overlap: int

    :returns: nothing

    """

    def __init__(self, frame = 512, overlap = 50):

        self._win = np.zeros((frame,))

        self._frame = frame

        self._overlap = overlap

        self._compute_window()

    def _compute_window(self):

        for i in range(self._overlap):

            self._win[i] = np.sin(2*np.pi/(4*(self._overlap+2))*(i+1))**2

        for i in range(self._overlap,self._frame-self._overlap):

            self._win[i] = 1

        for i in range(self._frame-self._overlap,self._frame):

            self._win[i] = np.cos(2*np.pi/(4*(self._overlap+2))*(i-self._frame+self._overlap+1))**2

    def window(self):

        """
        Method returning the vector of window's values.



        :return: the window

        :rtype: numpy array of length frame

        """

        return self._win

class fbe:

    """
    Versatile class computing various speech signal representations, mostly based on AR modelling and Mel Frequency
    Filterbanks. Also includes an auditory model methods

    :param frame_zero_adding: required length of the sequence after zero adding operation, defaults to None, which indicates no zero adding
    :type frame_zero_adding: int

    :param frame: frame length in samples
    :type frame: int

    :param sr: sampling frequency in Hz
    :type sr: float

    :param preem_alfa: the preemphasis coefficient
    :type preem_alfa: float

    :param freq_range: frequency range in which the mel frequency filterbanks should be computed
    :type freq_range: two elemements vector of floats

    :param filts_num: number of mel frequency triangular filters in the filterbank
    :type filts_num: int

    :param num_gfs: number of gammatone filters
    :type num_gfs: int

    :param spl_of_max_amplitude: Sound Pressure Level in dB for maximal amplitude signal
    :type spl_of_max_amplitude: float

    :param window: the windowing function
    :type window: numpy vector of floats, defaults to None, which causes using of rectangular window

    :param ar_order: the AR model order
    :type ar_order: int

    :param cepstral_lifter: the cepstral lifter in MFCC computation
    :type cepstral_lifter: int

    :param num_ceps: number of cepstra
    :type num_ceps: int

    :returns: nothing

    .. note:: PSD is abbreviation for power spectral density in the whole documentation. AR is abbreviation for
                autoregressive in the whole documentation.

    """

    def __init__(self, frame_zero_adding=None, frame=512, sr=16000, preem_alfa=0.95, overlap=0,
                 freq_range=[20., 8000.], filts_num=23, num_gfs=70, spl_of_max_amplitude=88,
                 window=None, ar_order=16, cepstral_lifter=22, num_ceps=13):
        if overlap==0 or overlap > frame/2:
            overlap = frame/2

        if window is None:

            window = np.ones((frame,))

        if frame != len(window):

            print("ERROR in fbe, frame and window lengths do not match, program exits ...")

            sys.exit(1)

        self.sr = sr  # sampling frequency in Hz

        self.frame = frame  # number of samples in the frame

        self.num_ceps = num_ceps

        if not frame_zero_adding is None:

            self._nfft = frame_zero_adding  # fft length, sets the self._nfft atribute

        else:

            self._nfft = frame

        self.preem_alfa = preem_alfa  # preemphasis coefficient

        self.freq_range = freq_range  # frequency range in Hz

        self.filts_num = filts_num  # number of triangular filterbank channels

        self._num_gfs = num_gfs # number of gammatone channels

        self.K = int(self._nfft / 2.) + 1  # length of the unique part of the FFT

        self.f_min = 0

        self.f_max = float(sr) / 2.

        self.f_low = self.freq_range[0]

        self.f_high = self.freq_range[1]

        # matrices

        self._tfb = self._tfb()  # compute the mel-frequency triangular filterbank, sets the H atribute

        # print self._tfb[10,:]
        #
        # raw_input()

        self._pinv_tfb = self._pinv_tfb()

        self._wgh_mat = self._wgh_mat()

        self._inv_wgh_mat = self._inv_wgh_mat()

        # window

        self._window = window

        self._ar_order = ar_order

        # setting up the van de Par auditory model

        #reference pressure 20uP

        self.p_ref = 20.e-6

        if self.f_high > self.sr/2.:

            print("WARNING in fbe, max freq exceeds the half of the smpl_rate, setting to sr/2")

            self.f_high = self.sr/2.

        self.spl_of_max_amplitude = spl_of_max_amplitude

        self.max_pressure = self.spl2pres(spl_of_max_amplitude)

        self.max_am = 1 # amplitude corresponding to the max pressure

        #compute the gammatone filterbanks

        self._comp_gammatone()

        #computation of the effective duration

        self.L = np.min([frame/(0.3*self.sr),1])

        #compute the threshold in quiet transfer function

        self._comp_threshold_in_quiet()

        #calibrate the Par model

        self._calibrate()

        # compute cepstral lifter

        L = cepstral_lifter
        N = num_ceps

        self.cepstral_lifter =  1+0.5*L*np.sin(np.pi*np.asarray(range(N))/float(L))

        # dct matrix

        self.dctmat = np.zeros((self.num_ceps,self.filts_num))

        for i in xrange(self.num_ceps):

            for j in xrange(self.filts_num):

                self.dctmat[i,j] = np.sqrt(2./self.filts_num) * np.cos(np.pi*i/self.filts_num*(j+.5))

        self.lst_elm = ['fr','frwin','fft','mag','ang','psd','senmatpsd','senpsd','lpc','var_lpc','armag','arpsd','fbe',\
                        'fbekaldi','arfbe','wgh','arwgh','sfbe','sarfbe','smag','spsd','sarmag','sarpsd','sfbewgh',\
                        'smagwgh','spsdwgh','senmatspsdwgh','senspsdwgh','sarfbewgh','sarpsdwgh','psdspl']

        self.results = {}

        self._reset()

    def get_frame_len(self):

        """Returns the frame length in samples

        :returns: the frame length

        :rtype: int

        """

        return self.frame

    def get_tfb(self):

        """Gets the triangular mel frequency filterbank.

        :returns: the filter matrix containing in each row a single filter

        :rtype: numpy array with filts_num rows

        """

        return self._tfb

    def get_wgh(self):

        """Gets the weighting matrix, which is a square of the product of pseudo inverses of the Jacobian of the linear
        magnitude spectrum filter banks transform.


        :returns: the weighting matrix

        :rtype: numpy array with dimension filts_num x filts_num

        """
        return self._wgh_mat

    def get_inv_wgh(self):

        """
        Gets pseudo inverse of the weighting matrix.

        :returns: the pseudo inverse of the weighting matrix

        :rtype: numpy array with dimension filts_num x filts_num


        """

        return self._inv_wgh_mat

    def get_pinv_tfb(self):

        """
        Gets the pseudoinverse of the filterbanks matrix.

        :returns: the pseudo inverse of the weighting matrix

        :rtype: numpy array with dimension filts_num x filts_num


        """

        return self._pinv_tfb

    def window(self):

        """
        Gets the signal windowing function.

        :returns: the windowing function

        :rtype: numpy array with dimension 1 x frame

        """

        return self._window

    def _tfb(self):
        """
        Computes the mel frequency triangular filterbank.

        """

        # filter cutoff frequencies (Hz) for all filters, size 1x(M+2)

        aux = np.linspace(0, self.filts_num + 1, self.filts_num + 2)

        c = self._mel2hz(
            (self._hz2mel(self.f_low) + aux * (self._hz2mel(self.f_high) - self._hz2mel(self.f_low)) / float(self.filts_num + 1)))

        f = np.linspace(self.f_min, self.f_max, self.K)

        H = np.zeros((self.filts_num, self.K))

        for m in xrange(self.filts_num):

            a = list(f >= c[m])

            b = list(f <= c[m + 1])

            k = np.array([a[i] and b[i] for i in xrange(len(f))])

            H[m, k] = (f[k] - c[m]) / (c[m + 1] - c[m])

            a = list(f >= c[m + 1])

            b = list(f <= c[m + 2])

            k = np.array([a[i] and b[i] for i in xrange(len(f))])

            H[m, k] = (c[m + 2] - f[k]) / (c[m + 2] - c[m + 1])

        return H

    def _proj_symmat_pd(self,A,rcond):

        A = .5*(A.T+A)

        w, v = np.linalg.eigh(A)

        w[w<rcond*np.max(w)] = rcond*np.max(w)

        f = (np.sqrt(w) * v).T

        f = f.T.dot(f)

        return f

    def _wgh_mat(self):

        """
        The weighting matrix.

        """

        W = np.dot(self._pinv_tfb.T, self._pinv_tfb)

        W = self._proj_symmat_pd(W,10.e-3)

        # w, v = np.linalg.eigh(W)
        #
        # f = (np.sqrt(w) * v).T  # <- that's it

        f = np.linalg.cholesky(W).T

        return f

    def _inv_wgh_mat(self):

        """
        The inverse of the weighting matrix

        """

        return np.linalg.pinv(self._wgh_mat)

    def _pinv_tfb(self):

        """
        The pseudoinverse of the mel frequency triangular filter banks matrix

        """

        return np.linalg.pinv(self._tfb)

    def _nextpow2(self, i):

        """
        The next nearest power of 2.

        """

        n = 1

        while n < i: n *= 2

        self._nfft = n

    def _hz2mel(self, hz):

        """
        Hertz to mel frequency scale

        """

        mel = 1127. * np.log(1 + hz / 700.)

        return mel

    def _mel2hz(self, mel):

        """
        Mel to frequency scale

        """

        hz = 700. * np.exp(mel / 1127.) - 700.

        return hz

    def _calibrate(self):

        "calibration procedure"

        #calibration of the Ca

        aspl = self.threshold_in_quiet(1000.)

        a = self.spl2am(aspl)

        s = self.sinus(1000.,a)

        S = self.psd_calib(s)

        g1kHz = self.L*np.sum(self._SM(S))

        A53 = self.spl2am(53.)

        A70 = self.spl2am(70.)

        fm = 1000.

        m = self.sinus(fm,A70)

        M = self.psd_calib(m)

        f = 1000.

        s = self.sinus(f,A53)

        S = self.psd_calib(s)

        # compute Mis

        Mi = self._SM(M)

        # compute Sis

        Si = self._SM(S)

        # bisection

        Cs_min = 1.e-6

        Cs_max = 1.e6

        s_min = np.sign(1./Cs_min - self.L * sum(Si/(Mi+Cs_min*g1kHz)))

        while Cs_max - Cs_min > 1e-4:

            C_cand = .5*(Cs_max + Cs_min)

            F = 1/C_cand - self.L * sum(Si/(Mi+C_cand*g1kHz))

            if np.sign(F) ==  s_min:

                Cs_min = C_cand

            else:

                Cs_max = C_cand

        self.Cs = C_cand

        self.Ca = self.Cs*g1kHz

    def idx2freq(self,i):

        """Converts frequency index to frequency in Hz

        :param i: frequency index

        :type i: int

        :return: frequency in Hz

        :rtype: float

        """

        f = float(i)/self._nfft*self.sr

        return f

    def freq2idx(self,f):

        """Converts frequency in Hz to the frequency index

        :param f: frequency in Herz
        :type f: float

        :return: frequency index
        :rtype: int


        """

        idx = int(np.round(f*float(self._nfft)/self.sr))

        return idx

    def threshold_in_quiet(self,f):
        """
        Terhard formula for the threshold in quiet.

        :param f: frequency in Hz
        :type f: float

        :return: threshod in quiet for the frequency f
        :rtype: float

        """

        A = 3.64*(f/1000.)**-0.8-6.5*np.exp(-0.6*(f/1000.-3.3)**2.)+(10**-3.)*(f/1000.)**4.

        return A

    def erbbnd(self, f):

        """
        ERB for the given frequency

        :param f: frequency in Hz
        :type f: float

        :return: ERB bandwith in Hz
        :rtype: float

        """

        bnd = 6.23 * 10**-6. * f**2. + 93.39 * 10**-3.*f+28.52

        return bnd

    def freq2erb(self, f):
        """
        Converts frequency to ERB

        :param f: frequency in Hz
        :type f: float
        :return: point on the ERB scale
        :rtype: float
        """

        erb = 11.17268*np.log(1+46.06538*f/(f+14678.49))

        return erb

    def erb2freq(self, erbs):
        """
        Converts ERB to frequency

        :param erbs: point on the erb scale
        :type erbs: float
        :return: corresponding frequency in Hz
        :rtype: float
        """

        f = 676170.4/(47.06538-np.exp(0.08950404*erbs))-14678.49

        return f

    def _gammatone(self, f, f0):
        """
        Computes the formula for the gammatone filterbank

        :param f: frequency in Hz
        :type f: float
        :param f0: central frequency in Hz
        :type f0: float
        :return: gamma filterbank
        :rtype: float
        """

        ERB = self.erbbnd(f0)

        gamma = np.abs(1/(1+1j*(f-f0)/(1.019*ERB))**4.)

        return gamma

    def _comp_gammatone(self):

        """Computation of the gammatone filterbank

        :returns: nothing

        """

        min_erb = self.freq2erb(self.f_low)

        max_erb = self.freq2erb(self.f_high)

        aux = np.linspace(min_erb,max_erb,self._num_gfs)

        self.gtfb2 = np.zeros((self._num_gfs,self.K))

        for j in xrange(int(round(self.K))):

            f = self.idx2freq(j)

            for i, erb in enumerate(aux):

                f0 = self.erb2freq(erb)

                self.gtfb2[i,j] = self._gammatone(f,f0)**2.

    def _comp_threshold_in_quiet(self):

        # comp of the threshold in quiet

        H = np.zeros((self.K,))

        for i in range(1,int(self.K)):

            H[i] = self.threshold_in_quiet(self.idx2freq(i))

        H[0] = self.spl_of_max_amplitude

        # threshold in quiet - linear scale

        Htq = self.spl2am(H)

        # inver

        self.Hom2 = np.exp((np.log(1)-np.log(Htq))*2)

    def pres2spl(self,p):

        """Pressure in Pascal into dB SPL

        :param p: presure in Pascal
        :type p: float

        :return: dB SPL relative to p0 = 20uP
        :rtype: float

        """

        aux = 20.*np.log10(p/self.p_ref)

        return aux

    def spl2pres(self,spl):
        """dB SPL into Pascals

        :param spl: SPL in dB
        :type spl: float

        :return: pressure in Pascals
        :rtype: float

        """

        p = np.exp(np.log(self.p_ref) +spl/20.*np.log(10))

        return p

    def spl2am(self,spl):

        """SPL in dB into linear amplitude (1 amplitude corresponds to self.max_pressure)

        :param spl: SPL in dB
        :type spl: float

        :return: linear amplitude
        :rtype: float

        """

        a = np.exp(np.log(self.p_ref) +spl/20.*np.log(10)-np.log(self.max_pressure))

        return a

    def am2pres(self,a):
        """Amplitude in the range 0..1 converted to pressure in Pascal

        :param a: amplitude
        :type a: float

        :return: pressure
        :rtype: float

        """

        p = a * self.max_pressure

        return p

    def am2spl(self,a):

        """Amplitude in the range 0..1 converted to dB SPL

        :param a: amplitude
        :type a: float

        :return: SPL in dB
        :rtype: float

        """

        p = self.am2pres(float(a))

        s = self.pres2spl(float(p))

        return s

    def sinus(self,freq,am):
        """
        Generate sine wave based on given sampling frequency, amplitude in range (0..1) with the number of samples equal to the
        frame length

        :param freq: frequency in Hz
        :type freq: float

        :param am: amplitude
        :type am: float

        :return: sine wave
        :rtype: numpy floats vector

        """

        r = np.linspace(0,float(self.frame)/float(self.sr)-1/float(self.sr),self.frame)

        s = am*np.sin(2.*np.pi*freq*r)

        return s

    def psd_calib(self,s):

        """
        Compute the psd of the given signal (if unitary fourier transform desired scale output by 1/sqrt(self._nfft))

        :param s: the signal
        :type s: numpy vector of floats

        :return: the psd
        :rtype: numpy vector of floats
        """

        v = np.zeros((self._nfft,))

        v[:len(s)] = s*np.hanning(len(s))

        p = np.abs(np.fft.fft(v))**2

        p[0] *= 0.5

        p[self.K-1] *= 0.5

        return p[:self.K]

    def _SM(self,psd):

        """
        formulas for M and S are equivalent
        :param psd:
        :return:
        """

        S = np.dot(self.Hom2*self.gtfb2,psd)

        return S

    def sens_mat(self,M):
        """
        Diagonal elements of the, diagonal, sensitivity matrix

        :param M: PSD masking signal as computed using self.psd()
        :type M: numpy vector of floats

        :return: the diagonal of the auditory model sensitivity matrix
        :rtype: numpy vector of floats

        """

        Mi = np.array(self.SM(M)+self.Ca)

        h = np.zeros((self._nfft,))

        min_idx = self.freq2idx(self.f_low)

        max_idx = self.freq2idx(self.f_high)

        for fi in xrange(min_idx,max_idx):

            h[fi] = np.sum(self.Cs*self.L*self.Hom2[fi]*self.gtfb2[:,fi]/Mi)

        return h[:self.K]

    def mskg_thrsh_mag(self,M):

        """Masking threshod in the Fourier magnitude domain

        :param M: PSD masking signal as computed using self.psd()
        :type M: numpy vector of floats

        :return: masking threshold in the magnitude domain
        :rtype: numpy vector of floats

        """

        mt = np.sqrt( 1/np.maximum(self.sens_mat(M),1.e-3))

        return mt

    def mskg_thrsh_psd(self,M):

        """Masking threshod in the Fourier PSD domain

        :param M: PSD masking signal as computed using self.psd()
        :type M: numpy vector of floats

        :return: masking threshold in the PSD domain
        :rtype: numpy vector of floats

        """

        mt = 1 / np.maximum(self.sens_mat(M), 1.e-3)

        return mt

    def _reset(self):

        """
        Resets the cache.

        """

        for e in self.lst_elm:

            self.results[e] = None

    def set_frm(self,fr):

        """
        Sets the frame of the signal - this is than used to compute the all signal representations

        :param fr: signal frame
        :type fr: numpy vector of floats

        :returns: nothing

        """

        self._reset()

        self.results['fr'] = fr

    def set_wgh(self,wgh):

        """
        Set compact spectrum

        :param wgh: the compact specturm with filt_num elements
        :type wgh: numpy vector of floats

        :returns: nothing

        """

        self._reset()

        self.results['wgh'] = wgh

    def set_arwgh(self,arwgh):

        """
        Set AR compact spectrum

        :param arwgh: the compact autoregresive specturm with filt_num elements
        :type arwgh: numpy vector of floats

        :returns: nothing

        """

        self._reset()

        self.results['arwgh'] = arwgh

    def set_fbe(self,fbe):

        """
        Set filterbank energies

        :param fbe: the filter bank energies (vector with filt_num elements)
        :type fbe: numpy vector of floats

        :returns: nothing

        """

        self._reset()

        self.results['fbe'] = fbe


    def set_mag(self,mag):

        """Set magnitude spectrum

        :param mag: the magnitude spectrum
        :type mag: numpy vector of floats

        :returns: nothing


        """

        self._reset()

        self.results['mag'] = mag

    def set_psd(self,psd):

        """
        Set power density spectrum

        :param psd: the power density spectrum
        :type psd: numpy vector of floats

        :returns: nothing

        """

        self._reset()

        self.results['psd'] = psd

    def fr(self):

        """
        Gets frame

        :returns: the frame
        :rtype: numpy vector of floats

        """

        if self.results['fr'] is None:

            print("Frame not given (emtpy vector), program exits ...")

            sys.exit(1)

        else:

            return self.results['fr']

    def fr_win(self):

        """
        Gets windowed frame

        :returns: the windowed frame
        :rtype: numpy vector of floats

        """

        if self.results['frwin'] is None:

            self.results['frwin'] = np.zeros((self._nfft,))

            self.results['frwin'][:self.frame] = self.fr() * self.window()

        else:

            pass

        return self.results['frwin']

    def fft(self):

        """
        Gets FFT

        :returns: the fft of the, possibly zero added, signal frame
        :rtype: numpy vector of complex floats

        """

        if self.results['fft'] is None:

            self.results['fft'] = np.fft.fft(self.fr_win())

        else:

            pass

        return self.results['fft']

    def mag(self):

        """
        Gets magnitude spectrum

        :returns: the magnitude of the, possibly zero added, signal frame
        :rtype: numpy vector of floats

        """

        if self.results['mag'] is None:

            self.results['mag'] = np.abs(self.fft())[ : self.K]

        else:

            pass

        return self.results['mag']

    def ang(self):

        """
        Gets angular spectrum.

        :returns: the angular spectrum of the, possibly zero added, signal frame
        :rtype: numpy vector of floats

        """

        if self.results['ang'] is None:

            self.results['ang'] = np.angle( self.fft() )

        else:

            pass

        return self.results['ang']

    def psd(self):

        """
        Gets power density spectrum

        :returns: the PSD of the, possibly zero added, signal frame
        :rtype: numpy vector of floats


        """

        if self.results['psd'] is None:

            self.results['psd'] = self.mag()**2.

        else:

            pass

        return self.results['psd']

    def magspl(self):

        """Computes the magnitude spectrum in dB SPL

        :return: magnitude spectrum in dB SPL
        :rtype: numpy vector of floats

        """

        if self.results['psdspl'] is None:

            p = self.mag()

            self.results['psdspl'] = []

            for _ in p:

                _ /= (.5*self._nfft)

                self.results['psdspl'].append(self.am2spl(_))

            self.results['psdspl'] = np.asarray(self.results['psdspl'])

        else:

            pass

        return self.results['psdspl']

    def senpsd(self): # a kind of perceptual power spectrum

        """Perceptual PSD

        :return: perceptual PSD
        :rtype: numpy vector of floats

        """

        if self.results['senpsd'] is None:

            self.results['senpsd'] = self.sens_mat(self.psd())*self.psd()

        else:

            pass

        return self.results['senpsd']

    def lpc(self):

        """
        Gets LPC coefficients

        :return: LPC with the leading 1

        """

        if self.results['lpc'] is None:

            _lpc, self.results['var_lpc'], k = spectrum.aryule(self.fr_win(), self._ar_order)

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

    def set_ar(self,a,var):

        """Sets the AR coefficients"""

        self._reset()

        self.results['var_lpc'] = var

        self.results['lpc'] = a

    def armag(self):

        """
        Gets AR magnitude spectrum

        :return: AR magnitude spectrum
        :rtype: numpy vector of floats of length _nfft/2+1

        """

        if self.results['armag'] is None:

            p = len(self.lpc())-1

            aux = np.concatenate([self.lpc(),np.zeros((self._nfft-p-1,))],axis=0)

            fftaux = np.abs(np.fft.fft(aux))

            std = np.sqrt(self.var_lpc()*self._nfft)

            self.results['armag'] = np.real(std/fftaux[ : self.K])

        else:

            pass

        return self.results['armag']

    def arpsd(self):

        """
        Gets AR power density spectrum

        :return: the AR PSD

        """


        if self.results['arpsd'] is None:

            self.results['arpsd'] = self.armag() ** 2.

        else:

            pass

        return self.results['arpsd']

    def fbe(self):

        """
        Gets filter banks outputs based on magnitude spectrum

        :return: filter bank filtered magnitude spectrum
        :rtype: numpy vector of floats of length filt_num

        """

        if self.results['fbe'] is None:

            self.results['fbe'] = np.dot(self.get_tfb(),self.mag())

        else:

            pass

        return self.results['fbe']

    def fbekaldi(self):

        """
        Gets filter banks outputs based on power density spectrum (this is how Kaldi computes MFCCs)

        :return: filter bank filtered PSD
        :rtype: numpy vector of floats of length filt_num

        """

        if self.results['fbekaldi'] is None:

            self.results['fbekaldi'] = np.dot(self.get_tfb(),self.psd())

        else:

            pass

        return self.results['fbekaldi']

    def sfbe2mfcc(self):

        """
        Converts smoothed filter banks energies to MFCC coefficients

        :return: MFCC coefficients
        :rtype: numpy vector of floats (size num_cep)

        """

        fbe = self.sfbe()

        logfbe = np.log(fbe)

        mfcc = np.dot(self.dctmat,logfbe) #scipy.fftpack.dct(logfbe,n=self.num_ceps,norm='ortho')

        # mfcc = scipy.fftpack.dct(logfbe,n=self.num_ceps,norm='ortho')

        # liftering

        cmfcc = self.cepstral_lifter*mfcc

        return cmfcc

    def fbe2mfcc(self):

        """
        Converts filter banks energies to MFCC coefficients

        :return: MFCC coefficients
        :rtype: numpy vector of floats (size num_cep)

        """

        fbe = self.fbe()

        logfbe = np.log(fbe)

        mfcc = np.dot(self.dctmat,logfbe) #scipy.fftpack.dct(logfbe,n=self.num_ceps,norm='ortho')

        # here comes liftering

        cmfcc = self.cepstral_lifter*mfcc

        return cmfcc

    def _fbekaldi2kaldimfcc(self): # not equivalent - another solution sought

        fbe = self.fbekaldi()

        logfbe = np.log(fbe)

        mfcc = scipy.fftpack.dct(logfbe,n=self.num_ceps,norm='ortho')

        cmfcc = self.cepstral_lifter*mfcc

        return cmfcc

    def arfbe(self):

        """
        AR magnitude spectrum to filter banks energies

        :return: filter bank filtered AR magnitude spectrum
        :rtype: numpy vector of floats (size num_filt)

        """

        if self.results['arfbe'] is None:

            self.results['arfbe'] = np.dot(self.get_tfb(),self.armag())

        else:

            pass

        return self.results['arfbe']

    def wgh(self):
        """
        Weighted filter bank energies

        :return: the magnitude compact spectrum
        :rtype: numpy vector of floats (size num_filt)

        """

        if self.results['wgh'] is None:

            self.results['wgh'] = np.dot(self.get_wgh(),self.fbe())

        else:

            pass

        return self.results['wgh']

    def arwgh(self):
        """
        AR weighted filter bank energies

        :return: the AR magnitude compact spectrum
        :rtype: numpy vector of floats (size num_filt)

        """

        if self.results['arwgh'] is None:

            self.results['arwgh']  = np.real(np.dot(self.get_wgh(),self.arfbe()))

        else:

            pass

        return self.results['arwgh']

    def smag(self):
        """
        Smoothed magnitude spectrum

        :return: magnitude spectrum computed from filter bank energies
        :rtype: numpy vector of floats (size _nfft/2+1)

        """

        if self.results['smag'] is None:

            self.results['smag'] = np.dot(self.get_pinv_tfb(), self.fbe())

        else:

            pass

        return self.results['smag']

    def spsd(self):

        """
        Smoothed power density spectrum

        :return: PSD computed from filter bank energies
        :rtype: numpy vector of floats(size _nfft/2+1)

        """

        if self.results['spsd'] is None:

            self.results['spsd'] = self.smag()**2.

        else:

            pass

        return self.results['spsd']

    def senspsd(self):

        """
        Perceptual smoothed PSD

        :returns: perceputal smoothed (from fbe) PSD
        :rtype: numpy vector of floats

        """

        if self.results['senspsd'] is None:

             self.results['senspsd'] = self.sens_mat(self.spsd())*self.spsd()

        else:

            pass

        return self.results['senspsd']

    def sarmag(self):

        """
        Smoothed AR magnitude spectrum

        :return: smoothed (from arfbe) AR magnitude spectrum (size _nfft/2+1)
        :rtype: numpy vector of floats

        """

        if self.results['sarmag'] is None:

            self.results['sarmag'] = np.dot(self.get_pinv_tfb(), self.arfbe())

        else:

            pass

        return self.results['sarmag']

    def sarpsd(self):

        """
        Smoothed AR PSD

        :return: smoothed (from arfbe) AR PSD (size _nfft/2+1)
        :rtype: numpy vector of floats

        """

        if self.results['sarpsd'] is None:

            self.results['sarpsd']  = self.sarmag() ** 2.

        else:

            pass

        return self.results['sarpsd']

    def preemphasis(self, signal):

        """Perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :type signal: numpy vector of floats
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
        :type coeff: float
        :returns: the filtered signal.
        :rtype: numpy vector of floats
        """

        return np.asarray(np.append(signal[0], signal[1:] - self.preem_alfa * signal[:-1]))

    def sfbe(self):

        """
        Smoothed filter bank energies

        :return: smoothed, from compact spectrum, filter bank energies
        :rtype: numpy vector of floats (size num_filt)

        """

        if self.results['sfbe'] is None:

            self.results['sfbe'] = np.dot(self.get_inv_wgh(), self.wgh())

        else:

            pass

        return self.results['sfbe']

    def sarfbe(self):

        """
        Smoothed AR filter bank energies

        :return smoothed, from compact AR spectrum, filter bank energies
        :rtype: numpy vector of floats (size num_filt)

        """

        if self.results['sarfbe'] is None:

            self.results['sarfbe'] = np.dot(self.get_inv_wgh(), self.arwgh())

        else:

            pass

        return self.results['sarfbe']

    def smagwgh(self):

        """
        Smoothed magnitude spectrum

        :return: computed from compact spectum magnitude spectrum
        :rtype: numpy vector of floats (size _nfft/2+1)

        """

        if self.results['smagwgh'] is None:

            self.results['smagwgh'] = np.dot(self.get_pinv_tfb(), self.sfbe())

        else:

            pass

        return self.results['smagwgh']

    def sarmagwgh(self):

        """
        Smoothed AR magnitude spectrum

        :return: computed from AR compact spectrum magnitude spectrum
        :rtype: numpy vector of floats (size _nfft/2+1)

        """

        if self.results['sarmagwgh'] is None:

            self.results['sarmagwgh'] = np.dot(self.get_pinv_tfb(), self.sarfbe())

        else:

            pass

        return self.results['sarmagwgh']

    def spsdwgh(self):

        """
        Smoothed PSD

        :return: computed from compact spectrum PSD
        :rtype: numpy vector of floats (size _nfft/2+1)

        """

        if self.results['spsdwgh'] is None:

            self.results['spsdwgh'] = self.smagwgh() ** 2.

        else:

            pass

        return self.results['spsdwgh']

    def senspsdwgh(self):
        """A kind of smoothed power perceptual power spectrum

        :return: smoothed perceptual PSD
        :rtype: numpy vector of floats (size _nfft/2+1)

        """

        if self.results['senspsdwgh'] is None:

            self.results['senspsdwgh'] = np.zeros((self.K,))

            self.results['senspsdwgh'][self.freq2idx(self.f_low):self.freq2idx(self.f_high)] = self.sens_mat(self.spsdwgh())[self.freq2idx(self.f_low):self.freq2idx(self.f_high)]*self.spsdwgh()[self.freq2idx(self.f_low):self.freq2idx(self.f_high)]

        else:

            pass

        return self.results['senspsdwgh']

    def sarpsdwgh(self):

        """
        Smoothed AR PSD

        :return: PSD computed from AR compact spectra
        :rtype: numpy vector of floats (size _nfft/2+1)

        """

        if self.results['sarpsdwgh'] is None:

            self.results['sarpsdwgh'] = self.sarmagwgh() ** 2.

        else:

            pass

        return self.results['sarpsdwgh']

    def senmatspsdwgh(self):

        """
        Sensitivity matrix for smoothed PSD

        :return: sensitivity for smoothed PSD
        :rtype: numpy vector of floats (size _nfft/2+1)

        """

        if self.results['senmatspsdwgh'] is None:

            self.results['senmatspsdwgh'] = self.sens_mat(self.spsdwgh())

        else:

            pass

        return self.results['senmatspsdwgh']

    def psd2wgh(self,psd):

        """
        PSD -> weighted compact spectrum

        :param psd: the PSD
        :type psd: numpy vector of floats (size _nfft/2+1)

        :return: compact spectrum based on PSD
        :rtype: numpy vector of floats (size num_filt)

        """

        mag = np.sqrt(psd)

        fbe = np.dot(self.get_tfb(),mag)

        return np.dot(self.get_wgh(),fbe)

    def psd2ar(self,psd,LpcOrder):
        """

        Converting PSD into LPC coefficients and excitation variance

        :param psd: left half of the PSD
        :type psd: numpy vector of floats (size _nfft/2+1)
        :param LpcOrder: AR model order
        :type LpcOrder: int

        :return: * (`vector of floats`) direct form AR coeff. with leading 1
                 * (`float`) the variance of the short term residual
        """

        D = len(psd)

        B = np.concatenate([psd,psd[D-2:0:-1]])

        xc = np.real(np.fft.ifft(B))

        xc = xc[:LpcOrder+1]

        a, var, k = spectrum.LEVINSON(xc)

        a = np.concatenate([[1],a])

        var = var/(2*D-2)

        return a, var

    def getkaldi_mel_bank(self, vtln_warp=1.0):
        """
        Extracts mel bank from kaldi mfcc class.

        :param vtln_warp: vocal tract length normalization
        :type vtln_warp: float, int

        :return: Mel banks computed by Kaldi
        :rtype: numpy 2D array of 32 bits floats
        """
        e, u = self.kmfcc.getKaldiMelBankSizes(vtln_warp)
        print(e, u)
        ar = np.zeros(u*e, dtype=np.float32)
        self.kfcc.getKaldiMelBankArray(vtln_warp, ar, u, e)
        ar = np.reshape(ar, [e,u])
        return ar

    def kaldi_mfcc_compute(self, wave, vtln_warp=1.0, scale=1):
        """
        Compute wave using kaldi compute mfcc method.

        :param wave: frame to compute
        :type wave: list of floats, numpy 1D array of floats

        :param vtln_warp: vocal tract length normalization
        :type wave: float, int

        :param scale: should wave be multiplied by 2**15?
        :type scale: boolean, int

        :return: array mfcc computed by Kaldi
        :rtype: numpy 2D array of 32 bits floats
        """
        r, c = self.kmfcc.getOutputArraySizes(wave.size)
        out = np.zeros(r*c, dtype=np.float32)

        if(scale):
            wave = np.multiply(wave, 2**15)

        self.kmfcc.kaldiCompute(wave, vtln_warp, out)
        if r>1:
            out = np.reshape(out, [r, c])

        return out

    def get_kaldi_center_frequencies(self, vtln_warp=1.0):
        """
        Extracts center frequencies vector from kaldi mfcc.

        :param vtln_warp: vocal tract length normalization
        :type vtln_warp: float, int

        :return: center frequencies for mel bank.
        :rtype: numpy 1D array of 32 bits floats
        """
        out = np.zeros(self.filts_num, dtype=np.float32)

        self.kmffc.getCenterFrequency(out, vtln_warp)
        return out



def synth(mag_enh, angular_r,an_win):

    """
    Signal synthesis based on magnitude and angular spectra

    :param mag_enh: enhanced speech magnitude spectrum
    :type psd_enh: numpy vector of floats
    :param angular_r: angular noisy signal frame spectrum
    :type angular_r: numpy vector of floats
    :param an_win: windowing function
    :type an_win: numpy vector of floats

    :return: time domain enhanced signal frame (windowed)
    :rtype: numpy vector of floats

    """

    # X = np.sqrt( psd_enh )

    X = mag_enh

    X[-1] = 0

    enhDft = np.concatenate( (X, X[-2:0:-1]) ) * np.exp( 1j * angular_r )

    an_win = np.sqrt( an_win )

    enh_fs = an_win*np.real( np.fft.ifft( enhDft ) )

    return enh_fs

def enhance_mag( mag_r, psd_n, psd_s):

    """
    The Wiener filter in frequency domain

    :param mag_r: noisy, unprocessed signal frame magnitude spectrum
    :type psd_r: numpy vector of floats
    :param psd_n: noise, estimated noise frame PSD
    :type psd_n: numpy vector of floats
    :param psd_s: speech, estimated speech frame PSD
    :type psd_s: numpy vector of floats

    :return: enhanced speech PSD
    :rtype: numpy vector of floats

    """

    psd_r_smag = psd_s + psd_n

    mag_enh = np.maximum(psd_s, 1.e-6) / np.maximum(psd_r_smag, 1.e-4) * mag_r

    mag_enh = np.maximum(mag_enh,0.001*mag_r)

    mag_enh[-1] = 0 # filter out the most high frequency peak

    return mag_enh

def enhance( mag_r, psd_n, psd_s, angular_r, an_win ):

    """
    The Wiener filter returning the time frequency signal frame

    :param mag_r: noisy, unprocessed signal frame magnitude spectrum
    :type mag_r: numpy vector of floats
    :param psd_n: noise, estimated noise frame PSD
    :type psd_n: numpy vector of floats
    :param psd_s: speech, estimated speech frame PSD
    :type psd_s: numpy vector of floats
    :param angular_r: angular noisy signal frame spectrum
    :type angular_r: numpy vector of floats
    :param an_win: windowing function
    :type an_win: numpy vector of floats

    :return: time domain enhanced signal frame (windowed)
    :rtype: numpy vector of floats

    """

    mag_enh = enhance_mag(mag_r, psd_n, psd_s)

    enh_fs = synth(mag_enh,angular_r,an_win)

    return enh_fs #, psd_enh

def enhance1 ( mag_r, psd_n, psd_s, angular_r, an_win):
    """
    Perceptual enhancement (SETAP p. 245)

    :param psd_r: noisy, unprocessed signal frame PSD
    :type psd_r: numpy vector of floats
    :param psd_n: noise, estimated noise frame PSD
    :type psd_n: numpy vector of floats
    :param psd_s: speech, estimated speech frame PSD
    :type psd_s: numpy vector of floats
    :param angular_r: angular noisy signal frame spectrum
    :type angular_r: numpy vector of floats
    :param an_win: windowing function
    :type an_win: numpy vector of floats

    :return: time domain enhanced signal frame (windowed)
    :rtype: numpy vector of floats

    """

    e = psd_s/np.maximum(psd_n,1.e-4)+1.e-6

    g = mag_r**2/np.maximum(psd_n,1.e-4)+1.e-6

    v = e*g/(1+e)

    aux = np.maximum(.5*v,1.e-2)

    if np.sum(np.isnan(aux)) > 0:

        print('nan found')

        raw_input()

    gain = np.sqrt(v)/(np.sqrt(np.pi)*g)*1/scipy.special.i0e(aux)

    if np.sum(np.isnan(gain)) > 0:

        print (v)

        print (np.sqrt(v))

        print (gain)

        print (scipy.special.iv(0,aux))

        print ('nan found')

        raw_input()

    # plt.plot(gain)
    #
    # plt.show()

    mag_enh = gain*mag_r

    enh_fs = synth(mag_enh,angular_r,an_win)

    return enh_fs #, psd_enh

def enhance2(psd_r, psd_n, mt, angular_r, an_win, ro=0.05): # note that instead of psd_s we pass the masking treshold (mt)

    """
    Perceptual Wiener filter

    :param psd_r: PSD of the noisy signal frame
    :type psd_r: numpy vector of floats
    :param psd_n: PSD of noise (may be approximate, smoothed etc.)
    :type psd_n: numpy vector of floats
    :param mt: PSD masking threshold
    :type mt: numpy vector of floats
    :param angular_r: angular noisy signal frame spectrum
    :type angular_r: numpy vector of floats
    :param an_win: windowing function
    :type an_win: numpy vector of floats
    :param ro: (0.-1.) larger causes less signal modification
    :type ro: float

    :return: time domain enhanced signal frame (windowed)
    :rtype: numpy vector of floats

    """

    gain = np.minimum(np.sqrt(mt/psd_n)+ro,1)

    psd_enh = gain*psd_r

    enh_fs = synth(psd_enh,angular_r,an_win)

    return enh_fs #, psd_enh


def load_wave(wavfile, tsr = 16000.):

    """Function reading in the audio file. Handling many audio files types
    (all provided by the sndfile C library)

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

def main():
    # load speech database

    filts_num = 40

    tsr = 16000

    N = 1000

    k = 64000

    # swgh = np.load('../densefbe/features_train_speech_'+str(tsr)+'_'+str(filts_num)+'.npy')[:N]
    #
    # pmu_lst = []
    #
    # for mv in swgh[:N]: # nmdl.mu
    #
    #     for sv in swgh[:N]:
    #
    #         pmu = mv + sv
    #
    #         pmu_lst.append(np.float16(pmu))
    #
    # swghp = np.array(pmu_lst)
    #
    # st = time.time()
    # treep = cKDTree(swghp)
    # en = time.time()
    #
    #
    # st = time.time()
    # tree = cKDTree(np.float16(swgh))
    # en = time.time()
    #
    # print "constructing tree in", en-st
    #
    # raw_input()

    # declare window

    frame = 512

    overlap = 256

    s2c2 = sin2cos2(frame=frame, overlap=overlap)

    step = frame - overlap

    rfbe = fbe(window=np.hanning(frame), freq_range=[100, 7800], filts_num=filts_num, spl_of_max_amplitude=40)

    # W = np.dot(rfbe.get_wgh().T,rfbe.get_wgh())
    #
    # np.savetxt('W.npy',W)
    #
    # raw_input()
    #
    # tm = fls.timit()
    #
    # ns = fls.noises()
    #
    # s_fle_lst = tm.get_files_list()
    #
    # td = tm.get_test_files_list()
    #
    # n_fle_lst = ns.get_files_list()
    #
    # s, sr = load_wave( td['wav'][0] )
    #
    # n, sr = load_wave(td['wav'][1]) #mest.gen_wgn(s)
    #
    # r = s[:min(len(s),len(n))] + n[:min(len(s),len(n))]

    s, sr = load_wave('/home/szymon/Music/sounds_and_speech_databases/speech_tascam_cln_lvl20/speech_high_01.wav')

    n, sr = load_wave('/home/szymon/Music/sounds_and_speech_databases/noise_tascam_lvl20/street_lvl20.wav', tsr=tsr)

    testspeach, spr = load_wave('/home/szymon/Downloads/SA1.wav')

    # swgh = np.load('../densefbe/features_train_speech_sph_'+str(tsr)+'_'+str(filts_num)+'.npy')

    # print "loading finished"

    # aux = random.randint(0,len(n)-len(s))
    #
    # print aux
    #
    # raw_input()

    aux = 1549031

    n = n[aux:aux + len(s)]

    r = s + n

    e = np.zeros_like(r)

    start = 0

    place = []

    rfbe.shouldIScale(testspeach)  # for now it only print (minimum, maximum) and its scaled copy... not evry helpful...

    if len(r) < len(testspeach):
        fin = len(r)
    else:
        fin = len(testspeach)
        print
        "ok"

    iterator = 2

    while start + frame <= fin:

        # print start,

        # noisy speech

        fr = r[start:start + frame]

        """ kaldi mfcc tests:
            1: obtaining center frequencies:    """
        if not start:
            # freq =
            pass

        """ 2: obtaining mel banks:             """

        """ 3: computing frame(repeatidily):    """
        testfr = testspeach[start:start + frame]

        kcomputed = rfbe.KaldiCompute(testfr, 1)

        print
        iterator, kcomputed
        iterator += 1

        """ fin """

        rfbe.set_frm(fr)

        rwgh = rfbe.wgh()

        rpsd = rfbe.psd()

        ang = rfbe.ang()

        # mfcc = rfbe.sfbe2mfcc()

        # print mfcc

        # raw_input()

        # d, i_p = treep.query(rwgh,k=k)

        # pset = set([])
        # plst = []

        # for _ in i_p:
        #     pset.add((_%N,_/N))
        #     plst.append([_%N,_/N])

        # plt.plot(np.log(rpsd))
        # plt.plot(np.log(rspsd),'g')
        # plt.show()

        # speech

        fs = s[start:start + frame]

        rfbe.set_frm(fs)

        sspsd = rfbe.spsd()

        mt = rfbe.mskg_thrsh_psd(sspsd)

        # plt.plot(np.log(sspsd[rfbe.freq2idx(rfbe.f_low):rfbe.freq2idx(rfbe.f_high)]))
        # plt.plot(np.log(1./np.maximum(sm,1.e-3)[rfbe.freq2idx(rfbe.f_low):rfbe.freq2idx(rfbe.f_high)]))
        #
        # plt.show()

        # swgh_tmp = rfbe.wgh()

        # d, i_s = tree.query(swgh_tmp)

        # rfbe.set_wgh(np.float16(swgh[i_s]))

        # sspsd = rfbe.spsdwgh()

        # plt.plot(np.log(spsd))
        # plt.plot(np.log(sspsd),'g')
        # plt.show()

        # noise

        fn = n[start:start + frame]

        rfbe.set_frm(fn)

        # nwgh = rfbe.wgh()

        # d, i_n = tree.query(nwgh)

        # if (i_s,i_n) in pset:

        # for l, _ in enumerate(plst):

        # if [i_s,i_n] == _ or [i_n,i_s] == _:

        # place.append(l)

        # break

        # rfbe.set_wgh(np.float16(swgh[i_n]))

        nspsd = rfbe.spsd()

        # nspsd = rfbe.spsdwgh()

        # plt.plot(np.log(npsd))
        # plt.plot(np.log(nspsd),'g')
        # plt.show()

        # enhancement



        fe = enhance2(rpsd, nspsd, mt, ang, np.hanning(512), ro=0.001)

        # fe = enhance1(rpsd, nspsd, sspsd, ang, np.hanning(512))

        e[start: start + frame] += fe

        start += step

    save_wave('nsy.wav', r)

    save_wave('cln.wav', s)

    save_wave('enh_str.wav', e)

    # print "SNR in", enmc.snr_c(r,s)
    # print "SNR out", enmc.snr_c(e,s)
    # print len(place), len(r)/step
    # print place
    # print np.mean(place)


if __name__ == "__main__":

    f = fbe(frame_zero_adding=16*8192,frame=16*8192,filts_num=21, num_gfs=21)

    f._comp_gammatone()

    # print f.gtfb2

    H = f.get_tfb()

    # print H

    # import matplotlib.pyplot as pl
    #
    # pl.plot(f.gtfb2)
    # pl.show()

    # print H.shape

    # print f.gtfb2.shape

    np.savetxt('/home/marcin/Downloads/tfb.txt', H)
    np.savetxt('/home/marcin/Downloads/gtfb.txt', f.gtfb2)
