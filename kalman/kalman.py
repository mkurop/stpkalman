
fbemodule = '../densefbe/fbe.py'
celpmodule = '../nsereduction/decoder.py'
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.expanduser(fbemodule)))
sys.path.append(os.path.dirname(os.path.expanduser(celpmodule)))

import fbe
import decoder

class ErrorCovarianceMatrix:

    def __init__(self,p,q):

        self.p = p

        self.q = q

        # initialize matrices

        self.mqq = np.zeros((q,q))
        self.mqp = np.zeros((q,p))
        self.mpq = np.zeros((p,q))
        self.mpp = np.zeros((p,p))

        self.oqq = [0,0]
        self.oqp = [0,0]
        self.opq = [0,0]
        self.opp = [0,0]

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

        self.mqq = C[:self.q,:self.q]
        self.mqp = C[:self.q,self.q:]
        self.mpq = C[self.q:,:self.q]
        self.mpp = C[self.q:,self.q:]

    def get(self):

        C = np.zeros((self.q+self.p,self.q+self.p))

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

        g = np.zeros((self.p+self.q,1))

        for i in range(self.q):

            g[i] = self.getqq(i,self.q-1) + self.getqp(i,self.p-1)

        for i in range(self.p):

            g[self.q+i] = self.getpq(i, self.q-1) + self.getpp(i, self.p-1)

        g /= (self.getqq(self.q-1,self.q-1)+self.getqp(self.q-1,self.p-1)+self.getpp(self.p-1,self.p-1) + self.getpq(self.p-1,self.q-1))

        return g

    def update(self,gain):

        auxqq = np.zeros((self.q,self.q))
        auxqp = np.zeros((self.q,self.p))
        auxpq = np.zeros((self.p,self.q))
        auxpp = np.zeros((self.p,self.p))

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

class StateVector:

    def __init__(self,p,q):

        self.xq = np.zeros((q,1))
        self.xp = np.zeros((p,1))

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

        x = np.zeros((self.p+self.q,1))

        for i in range(self.q):

            x[i,0] = self.getq(i)

        for i in range(self.p):

            x[self.q+i,0] = self.getp(i)

        return x

class STPKalman:

    def __init__(self, p=10, q=1):

        self.p = p
        self.q = q

        self.ecm = ErrorCovarianceMatrix(p,q)
        self.sv = StateVector(p,q)

    def step(self,rt,asp,ans,sig_s,sig_n):
        """
        :param asp: STP speech parameters, the analyzing polynomial: 1+asp[0]*z**-1+...+asp[p-1]*z**-p
        :param ans: STP noise parameters, analyzing polynomia analogously
        :return: filtered state vector, first q elements for noise and next p elements for speech
        """

        self.sv.predictor(asp,ans)
        self.ecm.predictor(asp,ans,sig_s,sig_n)
        G = self.ecm.gain()
        self.sv.update(G,rt)
        self.ecm.update(G)

        return self.sv.getp(0)

if __name__ == "__main__":

    Fs = np.zeros((3,3))
    Fs[:2,1:] = np.eye(2,2)
    ans = np.random.randn(3)
    Fs[2,:] = -ans

    Fn = np.zeros((3,3))
    Fn[:2, 1:] = np.eye(2,2)
    asp = np.random.randn(3)
    Fn[2, :] = -asp

    F = np.zeros((6,6))
    F[:3,:3] = Fs
    F[3:,3:] = Fn

    Q = np.zeros((6,6))

    Q[2,2] = 0.1**2

    Q[5,5] = 0.2**2

    K = np.zeros((6,1))

    K[2,0] = 1
    K[5,0] = 1

    C = np.random.randn(6,6)

    ecm = ErrorCovarianceMatrix(3, 3)

    ecm.setC(C)

    CX = F.dot(C).dot(F.T) + Q

    ecm.predictor(asp, ans, 0.2, 0.1)

    CC = ecm.get()

    print "after predictor"

    print CX-CC

    G = CX.dot(K)/(K.T.dot(CX).dot(K))

    GG = ecm.gain()

    print "gain"

    print G-GG

    CX = CX-G.dot(K.T).dot(CX)

    ecm.update(GG)

    print "after update"

    print CX-ecm.get()

    x = np.random.randn(6,1)

    sv = StateVector(3, 3)

    sv.setX(x)





    Fx = F.dot(x)

    sv.predictor(asp, ans)

    print "x after predictor"
    print Fx-sv.get()
    sv.update(GG, 1.23)
    print "x after update"
    print Fx+G*(1.23-Fx[2,0]-Fx[5,0])-sv.get()

    raw_input()












    s, sr = fbe.load_wave("/home/data/morgenstern/speech_database/michal_sound_mono_without_instr_2_74.wav",tsr=8000)
    n, sr = fbe.load_wave("/home/data/morgenstern/Noise_Recordings/cafeteria_babble.wav",tsr=8000)

    n = n[:len(s)]

    SNR = 10.

    n *= (np.linalg.norm(s)**2/np.linalg.norm(n)**2/10**(SNR/10.))**.5

    r = s + n

    print "SNR: ", 10*np.log10(np.linalg.norm(s)**2/np.linalg.norm( n*(np.linalg.norm(s)**2/np.linalg.norm(n)**2/10**(SNR/10.))**.5)**2)

    start = 0
    frame = 200

    p=10
    q=4

    fbes = fbe.fbe(frame=frame,ar_order=p,sr=8000)
    fben = fbe.fbe(frame=frame,ar_order=q,sr=8000)

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

    fbe.save_wave("test_stp.wav",e,sr=8000)



