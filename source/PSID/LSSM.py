""" Omid Sani, Shanechi Lab, University of Southern California, 2020 """
"An LSSM object for keeping parameters, filtering, etc"

import numpy as np
from scipy import linalg
from .sim import drawRandomPoles

def generate_random_eigenvalues(count):
    """Generates complex conjugate pairs of eigen values with a uniform distribution in the unit circle"""
    # eigvals = 0.95 * np.exp(1j * np.pi/8 * np.array([-1, +1]))
    eigvals = drawRandomPoles(count)
    return eigvals

def dict_get_either(d, fieldNames, defaultVal = None):
    for f in fieldNames:
        if f in d:
            return d[f]
    return defaultVal

def genRandomGaussianNoise(N, Q, m=None):
    Q2 = np.atleast_2d(Q)
    dim = Q2.shape[0]
    if m is None:
        m = np.zeros((dim, 1))
    
    D, V = linalg.eig(Q2)
    if np.any(D < 0):
        raise("Cov matrix is not PSD!")
    QShaping = np.real(np.matmul(V, np.sqrt(np.diag(D))))
    w = np.matmul(np.random.randn(N, dim), QShaping.T)
    return w, QShaping
    
class LSSM:
    def __init__(self, output_dim = None, state_dim = None, input_dim = None, params = None, randomizationSettings = None):
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.input_dim = input_dim
        if params == None:
            self.randomize(randomizationSettings)
        else:
            self.setParams(params)
    
    def setParams(self, params = {}):

        A = dict_get_either(params, ['A', 'a'])
        A = np.atleast_2d(A)
        
        C = dict_get_either(params, ['C', 'c'])
        C = np.atleast_2d(C)
        
        self.A = A
        self.state_dim = self.A.shape[0]
        if C.shape[1] != self.state_dim:
            C = C.T
        self.C = C
        self.output_dim = self.C.shape[0]
        
        B = dict_get_either(params, ['B', 'b'], None)
        D = dict_get_either(params, ['D', 'd'], None)
        if isinstance(B, np.ndarray):
            B = np.atleast_2d(B)
            self.input_dim = B.shape[1]
        elif isinstance(D, np.ndarray):
            D = np.atleast_2d(D)
            self.input_dim = D.shape[1]
        else:
            self.input_dim = 0
        if B is None:
            B = np.zeros((self.state_dim, self.input_dim))
        B = np.atleast_2d(B)
        self.B = B
        if D is None:
            D = np.zeros((self.output_dim, self.input_dim))
        D = np.atleast_2d(D)
        self.D = D
        

        if 'q' in params or 'Q' in params:  # Stochastic form with QRS provided
            Q = dict_get_either(params, ['Q', 'q'], None)
            R = dict_get_either(params, ['R', 'r'], None)
            S = dict_get_either(params, ['S', 's'], None)
            Q = np.atleast_2d(Q)
            R = np.atleast_2d(R)

            self.Q = Q
            self.R = R
            if S is None:
                S = np.zeros((self.state_dim, self.output_dim))
            S = np.atleast_2d(S)
            if S.shape[0] != self.state_dim:
                S = S.T
            self.S = S
        elif 'k' in params or 'K' in params:
            self.Q = None
            self.R = None
            self.S = None
            self.K = dict_get_either(params, ['K', 'k'], None)
            self.innovCov = dict_get_either(params, ['innovCov'], None)
            
        self.update_secondary_params()

        for f, v in params.items(): # Add any remaining params (e.g. Cz)
            if not hasattr(self, f) and not hasattr(self, f.upper()) and \
                f not in set(['sig', 'L0', 'P']):
                setattr(self, f, v)
        
    
    def randomize(self, randomizationSettings = None):
        if randomizationSettings is None:
            randomizationSettings = dict()
        
        isOk = False
        while not isOk:
            self.eigenvalues = generate_random_eigenvalues(self.state_dim)
            self.A, ev = linalg.cdf2rdf(self.eigenvalues, np.eye(self.state_dim))
            self.C = np.random.randn(self.output_dim, self.state_dim)
            
            tmp = np.random.randn(self.output_dim + self.state_dim, self.output_dim + self.state_dim)
            QRS = tmp @ np.transpose(tmp)
            self.Q = QRS[:self.state_dim, :self.state_dim]
            self.S = QRS[:self.state_dim, self.state_dim:]
            self.R = QRS[self.state_dim:, self.state_dim:]
            if 'ySNR' in randomizationSettings and randomizationSettings['ySNR'] is not None:
                try:
                    self.update_secondary_params()
                    ySNR = np.diag((self.C @ self.XCov @ self.C.T) / self.R)
                    CRowScale = np.sqrt(randomizationSettings['ySNR'] / ySNR)
                    self.C = np.diag(CRowScale) @ self.C
                except:
                    continue
                    pass

            try:
                self.update_secondary_params()
                A_KC_Eigs = linalg.eig(self.A_KC)[0]
                if np.any(np.abs(A_KC_Eigs)>1):
                    isOk = False
                    break

                isOk = True
            except:
                pass
    
    def update_secondary_params(self):
        if self.Q is not None: # Given QRS
            self.XCov = linalg.solve_discrete_lyapunov(self.A, self.Q)
            self.G = self.A @ self.XCov @ self.C.T + self.S
            self.YCov = self.C @ self.XCov @ self.C.T + self.R
            self.YCov = (self.YCov + self.YCov.T)/2

            self.Pp = linalg.solve_discrete_are(self.A.T, self.C.T, self.Q, self.R, s=self.S) # Solves Katayama eq. 5.42a
            self.innovCov = self.C @ self.Pp @ self.C.T + self.R
            innovCovInv = np.linalg.inv( self.innovCov )
            self.K = (self.A @ self.Pp @ self.C.T + self.S) @ innovCovInv
            self.Kf = self.Pp @ self.C.T @ innovCovInv
            self.Kv = self.S @ innovCovInv
            self.A_KC = self.A - self.K @ self.C

            self.P2 = self.XCov - self.Pp # (should give the solvric solution) Proof: Katayama Theorem 5.3 and A.3 in pvo book
        elif self.K is not None: # Given K
            self.XCov = None
            self.G = None
            self.YCov = None
        
            self.Pp = None
            self.Kf = None
            self.Kv = None
            self.A_KC = self.A - self.K @ self.C
            self.P2 = None
    
    def isStable(self):
        return np.all(np.abs(self.eigenvalues) < 1)
    
    def generateRealizationWithQRS(self, N, x0=None, u0=None, w0=None, u=None):
        QRS = np.block([[self.Q,self.S], [self.S.T,self.R]])
        wv, self.QRSShaping = genRandomGaussianNoise(N, QRS)
        w = wv[:, :self.state_dim]
        v = wv[:, self.state_dim:]
        if x0 == None:
            x0 = np.zeros((self.state_dim, 1))
        if w0 == None:
            w0 = np.zeros((self.state_dim, 1))
        X = np.empty((N, self.state_dim))
        Y = np.empty((N, self.output_dim))
        for i in range(N):
            if i == 0:
                Xt_1 = x0
                Wt_1 = w0
            else:
                Xt_1 = X[i-1, :].T
                Wt_1 = w[i-1, :].T
            X[i, :] = (self.A @ Xt_1 + Wt_1).T
            Y[i, :] = (self.C @ X[i, :].T + v[i, :].T).T

        return Y, X
    
    def generateRealizationWithKF(self, N, x0=None, **kwargs):
        e, innovShaping = genRandomGaussianNoise(N, self.innovCov)
        if x0 == None:
            x0 = np.zeros((self.state_dim, 1))
        X = np.empty((N, self.state_dim))
        Y = np.empty((N, self.output_dim))
        Xp = x0
        for i in range(N):
            ek = e[i, :][:, np.newaxis]
            yk = self.C @ Xp + ek
            X[i, :] = np.squeeze(Xp)
            Y[i, :] = np.squeeze(yk)
            Xp = self.A_KC @ Xp + self.K @ yk
            # Xp = self.A @ Xp + self.K @ ek

        return Y, X

    def generateRealization(self, N, **kwargs):
        if self.R is not None: 
            Y, X = self.generateRealizationWithQRS(N, **kwargs)
        else:
            Y, X = self.generateRealizationWithKF(N, **kwargs)
        return Y, X

    def kalman(self, Y, x0=None, P0=None, u=None):
        N = Y.shape[0]
        allXp = np.empty((N, self.state_dim))  # X(i|i-1)
        allX = np.empty((N, self.state_dim))
        if x0 == None:
            x0 = np.zeros((self.state_dim, 1))
        if P0 == None:
            P0 = np.zeros((self.state_dim, self.state_dim))
        Xp = x0
        for i in range(N):
            allXp[i, :] = np.transpose(Xp) # X(i|i-1)
            zi = Y[i, :][:, np.newaxis] - self.C @ Xp # Innovation Z(i)
            
            if self.Kf is not None:  # Otherwise cannot do filtering
                X = Xp + self.Kf @ zi # X(i|i)
                allX[i, :] = np.transpose(X)

            newXp = self.A @ Xp
            newXp += self.K @ zi

            Xp = newXp
        return allXp, allX
    
    def predict(self, Y):
        allXp = self.kalman(Y)[0]
        allYp = np.transpose(self.C @ allXp.T)
        if hasattr(self, 'Cz') and self.Cz is not None:
            allZp = np.transpose(self.Cz @ allXp.T)
        else:
            allZp = None
        return allZp, allYp, allXp

    def applySimTransform(self, E):
        EInv = np.linalg.inv(E)
        
        ALikeFields = {'A'}
        for f in ALikeFields:
            if hasattr(self, f):
                val = getattr(self, f)
                if val is not None and val.shape[0] == E.shape[1] and val.shape[0] == val.shape[1]:
                    setattr(self, f, E @ val @ EInv) # newA = E * A * EInv

        BLikeFields = {'B', 'S', 'G', 'K', 'Kf', 'Kv'}
        for f in BLikeFields:
            if hasattr(self, f):
                val = getattr(self, f)
                if val is not None and val.shape[0] == E.shape[1]:
                    setattr(self, f, E @ val) # newB = E * B

        CLikeFields = {'C', 'Cz'}
        for f in CLikeFields:
            if hasattr(self, f):
                val = getattr(self, f)
                if val is not None and val.shape[1] == EInv.shape[0]:
                    setattr(self, f, val @ EInv) # newC = C * EInv
        
        QLikeFields = {'Q', 'P', 'Pp', 'P2', 'XCov'}
        for f in QLikeFields:
            if hasattr(self, f):
                val = getattr(self, f)
                if val is not None and val.shape[0] == E.shape[1] and val.shape[0] == val.shape[1]:
                    setattr(self, f, E @ val @ E.T) # newA = E * A * E'

        
        self.update_secondary_params()
    
    def makeSimilarTo(self, s2):
        N = 100000
        Y, X = s2.generateRealization(N)
        xPredTrg, xFiltTrg = s2.kalman(Y)
        xPredSrc, xFiltSrc = self.kalman(Y)

        E = np.transpose(np.linalg.pinv(xPredSrc) @ xPredTrg)
        self.applySimTransform(E)
        return E

    def makeCanonical(self):
        J, EInv = linalg.schur(self.A, output='real')
        E = np.linalg.inv(EInv)
        self.applySimTransform(E)
        return E

