import numpy as np
import os
import sys

#  sys.path.append(".")
#  import utils as fbe
import kalman.utils as fbe
from numba import int32, float32, deferred_type
from numba.experimental import jitclass
  
spec_error_covariance_matrix = [
    ('p', int32),
    ('q', int32),
    ('mqq', float32[:,:]),
    ('mqp', float32[:,:]),
    ('mpq', float32[:,:]),
    ('mpp', float32[:,:]),
    ('oqq', int32[:]),
    ('oqp', int32[:]),
    ('opq', int32[:]),
    ('opp', int32[:]),
]


@jitclass(spec_error_covariance_matrix)
class ErrorCovarianceMatrix:

    def __init__(self,p,q):

        self.p = p

        self.q = q

        # initialize matrices

        self.mqq = np.zeros((q,q), dtype=np.float32)
        self.mqp = np.zeros((q,p), dtype=np.float32)
        self.mpq = np.zeros((p,q), dtype=np.float32)
        self.mpp = np.zeros((p,p), dtype=np.float32)

        self.oqq = np.asarray([0,0], dtype=np.int32)
        self.oqp = np.asarray([0,0], dtype=np.int32)
        self.opq = np.asarray([0,0], dtype=np.int32)
        self.opp = np.asarray([0,0], dtype=np.int32)

    def getqq(self,i,j):

        return self.mqq[(self.oqq[0]+i) % self.q,(self.oqq[1]+j) % self.q]

    def getqp(self, i, j):

        return self.mqp[(self.oqp[0] + i) % self.q, (self.oqp[1] + j) % self.p]

    def getpq(self, i, j):

        return self.mpq[(self.opq[0]+i) % self.p, (self.opq[1] + j) % self.q]

    def getpp(self, i, j):

        return self.mpp[(self.opp[0]+i) % self.p, (self.opp[1]+j) % self.p]

    def setqq(self,i,j,v):

        self.mqq[(self.oqq[0] + i) % self.q, (self.oqq[1] + j) % self.q] = v

    def setqp(self, i, j, v):

        self.mqp[(self.oqp[0] + i) % self.q, (self.oqp[1] + j) % self.p] = v

    def setpq(self, i, j, v):

        self.mpq[(self.opq[0] + i) % self.p, (self.opq[1] + j) % self.q] = v

    def setpp(self, i, j, v):

        self.mpp[(self.opp[0] + i) % self.p, (self.opp[1] + j) % self.p] = v

    def setC(self,C):

        self.mqq = np.float32(C[:self.q,:self.q])
        self.mqp = np.float32(C[:self.q,self.q:])
        self.mpq = np.float32(C[self.q:,:self.q])
        self.mpp = np.float32(C[self.q:,self.q:])

    def get(self):

        C = np.zeros((self.q+self.p,self.q+self.p), dtype=np.float32)

        for i in range(self.q):
            for j in range(self.q):
                C[i,j] = self.getqq(i,j)

        for i in range(self.q):
            for j in range(self.p):
                C[i,self.q+j] = self.getqp(i,j)

        for i in range(self.p):
            for j in range(self.q):
                C[self.q+i,j] = self.getpq(i,j)

        for i in range(self.p):
            for j in range(self.p):
                C[self.q+i,self.q+j] = self.getpp(i,j)

        return C

    def FCT(self, asp, ans):

        # qq

        for i in range(self.q):

            accu = 0

            for j in range(self.q):

                accu -= ans[j]*self.getqq(i,j)

            self.setqq(i, self.q, accu)

        self.oqq[1] = (self.oqq[1] + 1) % self.q

        # qp

        for i in range(self.p):

            accu = 0

            for j in range(self.q):

                accu -= ans[j] * self.getpq(i, j)

            self.setpq(i, self.q, accu)

        self.opq[1] = (self.opq[1] + 1) % self.q

        # pq

        for i in range(self.q):

            accu = 0

            for j in range(self.p):

                accu -= asp[j] * self.getqp(i, j)

            self.setqp(i, self.p, accu)

        self.oqp[1] = (self.oqp[1] + 1) % self.p

        # pp

        for i in range(self.p):

            accu = 0

            for j in range(self.p):

                accu -= asp[j] * self.getpp(i, j)

            self.setpp(i, self.p, accu)

        self.opp[1] = (self.opp[1] + 1) % self.p


    def FC(self, asp, ans):

        # qq

        for i in range(self.q):

            accu = 0

            for j in range(self.q):

                accu -= ans[j]*self.getqq(j,i)

            self.setqq(self.q,i,accu)

        self.oqq[0] = (self.oqq[0]+1)%self.q

        # qp

        for i in range(self.p):

            accu = 0

            for j in range(self.q):

                accu -= ans[j] * self.getqp(j, i)

            self.setqp(self.q, i, accu)

        self.oqp[0] = (self.oqp[0] + 1) % self.q

        # pq

        for i in range(self.q):

            accu = 0

            for j in range(self.p):

                accu -= asp[j] * self.getpq(j, i)

            self.setpq(self.p, i, accu)

        self.opq[0] = (self.opq[0] + 1) % self.p

        # pp

        for i in range(self.p):

            accu = 0

            for j in range(self.p):

                accu -= asp[j] * self.getpp(j, i)

            self.setpp(self.p, i, accu)

        self.opp[0] = (self.opp[0] + 1) % self.p

    def predictor(self,asp, ans, sig_s, sig_n):

        self.FC(asp,ans)
        self.FCT(asp,ans)
        qq = self.getqq(self.q-1,self.q-1) + sig_n*sig_n
        pp = self.getpp(self.p-1,self.p-1) + sig_s*sig_s
        self.setqq(self.q-1,self.q-1,qq)
        self.setpp(self.p-1,self.p-1,pp)

    def gain(self):

        g = np.zeros((self.p+self.q,), np.float32)

        for i in range(self.q):

            g[i] = self.getqq(i,self.q-1) + self.getqp(i,self.p-1)

        for i in range(self.p):

            g[self.q+i] = self.getpq(i, self.q-1) + self.getpp(i, self.p-1)

        g /= (self.getqq(self.q-1,self.q-1)+self.getqp(self.q-1,self.p-1)+self.getpp(self.p-1,self.p-1) + self.getpq(self.p-1,self.q-1))

        return g

    def update(self,gain):

        auxqq = np.zeros((self.q,self.q), dtype=np.float32)
        auxqp = np.zeros((self.q,self.p), dtype=np.float32)
        auxpq = np.zeros((self.p,self.q), dtype=np.float32)
        auxpp = np.zeros((self.p,self.p), dtype=np.float32)

        for i in range(self.q):

            for j in range(self.q):

                v = self.getqq(i,j) - gain[i]*(self.getqq(self.q-1,j)+self.getpq(self.p-1,j))

                auxqq[i,j] = v

        for i in range(self.q):

            for j in range(self.p):

                v = self.getqp(i,j) - gain[i]*(self.getqp(self.q-1,j) + self.getpp(self.p-1,j))

                auxqp[i,j] = v

        for i in range(self.p):

            for j in range(self.q):

                v = self.getpq(i,j) - gain[self.q+i]*(self.getpq(self.p-1,j)+self.getqq(self.q-1,j))

                auxpq[i,j] = v

        for i in range(self.p):

            for j in range(self.p):

                v = self.getpp(i,j) - gain[self.q+i]*(self.getpp(self.p-1,j) + self.getqp(self.q-1,j))

                auxpp[i,j] = v

        """================"""

        for i in range(self.q):

            for j in range(self.q):

                self.setqq(i,j,auxqq[i,j])

        for i in range(self.q):

            for j in range(self.p):

                self.setqp(i,j,auxqp[i,j])

        for i in range(self.p):

            for j in range(self.q):

                self.setpq(i,j,auxpq[i,j])

        for i in range(self.p):

            for j in range(self.p):

                self.setpp(i,j,auxpp[i,j])

spec_state_vector = [
    ('xq', float32[:]),
    ('xp', float32[:]),
    ('p', int32),
    ('q', int32),
    ('op', int32),
    ('oq', int32),
]

@jitclass(spec_state_vector)
class StateVector:

    def __init__(self,p,q):

        self.xq = np.zeros((q,), dtype=np.float32)
        self.xp = np.zeros((p,), dtype=np.float32)

        self.q = q
        self.p = p

        self.oq = 0
        self.op = 0

    def getq(self,i):

        return self.xq[(self.oq+i)%self.q]

    def getp(self,i):

        return self.xp[(self.op+i)%self.p]

    def setq(self,i,v):

        self.xq[(self.oq+i)%self.q] = v

    def setp(self,i,v):

        self.xp[(self.op+i)%self.p] = v

    def setX(self,x):

        self.xq = x[:self.q]

        self.xp = x[self.q:]

    def predictor(self,asp,ans):

        accu = 0

        for i in range(self.q):

            accu -= self.getq(i)*ans[i]

        self.setq(self.q, accu)

        self.oq = (self.oq+1)%self.q

        accu = 0

        for i in range(self.p):

            accu -= self.getp(i)*asp[i]

        self.setp(self.p,accu)

        self.op = (self.op+1)%self.p

    def update(self,gain,rt):

        aux = self.getq(self.q-1)+self.getp(self.p-1)

        aux = rt-aux

        ga = gain*aux

        for i in range(self.q):

            v = self.getq(i)+ga[i]

            self.setq(i,v)

        for i in range(self.p):

            v = self.getp(i) + ga[self.q+i]

            self.setp(i,v)

    def get(self):

        x = np.zeros((self.p+self.q,), dtype=np.float32)

        for i in range(self.q):

            x[i] = self.getq(i)

        for i in range(self.p):

            x[self.q+i] = self.getp(i)

        return x

error_covariance_matrix_type = deferred_type()
error_covariance_matrix_type.define(ErrorCovarianceMatrix.class_type.instance_type)

state_vector_type = deferred_type()
state_vector_type.define(StateVector.class_type.instance_type)

spec_stp_kalman = [
    ('p', int32),
    ('q', int32),
    ('ecm', error_covariance_matrix_type),
    ('sv', state_vector_type),
]

@jitclass(spec_stp_kalman)
class STPKalman:

    def __init__(self, p=10, q=1):

        self.p = p
        self.q = q

        self.ecm = ErrorCovarianceMatrix(p,q)
        self.sv = StateVector(p,q)

    def step(self,rt,asp,ans,sig_s,sig_n):
        """
        :param asp: STP speech parameters, the analyzing polynomial: 1+asp[p-1]*z**-1+...+asp[0]*z**-p
        :param ans: STP noise parameters, analyzing polynomia analogously
        :return: filtered state vector, first q elements for noise and next p elements for speech
        """

        asp = asp.astype(np.float32)
        ans = ans.astype(np.float32)
        sig_s = np.float32(sig_s)
        sig_n = np.float32(sig_n)

        self.sv.predictor(asp,ans)
        self.ecm.predictor(asp,ans,sig_s,sig_n)
        G = self.ecm.gain()
        self.sv.update(G,rt)
        self.ecm.update(G)

        return self.sv.getp(0)

SAMPLING_RATE = 16000 # sampling rate of the project in Hz

if __name__ == "__main__":

    fbe.clear_output_directory()

    s, sr = fbe.load_wav("../data/input/speech.wav")
    n, sr = fbe.load_wav("../data/input/noise.wav")

    n = n[:len(s)]

    SNR = 10. # signal to noise ratio for the noisy file

    n *= (np.linalg.norm(s)**2/np.linalg.norm(n)**2/10**(SNR/10.))**.5

    r = s + n # mix speech and noise

    start = 0
    frame = 320 # length of the signal frame

    p=16 # speech autoregressive model order
    q=1 # noise autoregressive model order

    fbes = fbe.Utils(p)
    fben = fbe.Utils(q)

    kl = STPKalman(p,q)

    e = np.zeros_like(r)

    while start+frame <= len(r):

        fs = s[start:start+frame]
        fn = n[start:start+frame]
        fr = r[start:start+frame]

        fbes.set_frm(fs)
        asp = fbes.lpc()[1:]
        sig_s = fbes.var_lpc()**.5
        fben.set_frm(fn)
        ans = fben.lpc()[1:]
        sig_n = fben.var_lpc()**.5

        for i in range(frame):

            e[start + i] = kl.step(fr[i],asp[::-1],ans[::-1],sig_s,sig_n)

        start += frame
    
    fbe.save_wav(r,SAMPLING_RATE,"../data/output/noisy.wav")
    fbe.save_wav(e,SAMPLING_RATE,"../data/output/output.wav")



