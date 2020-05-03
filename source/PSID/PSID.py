""" Omid Sani, Shanechi Lab, University of Southern California, 2020 """
"Example for using the PSID algorithm"

import numpy as np
from scipy import linalg
from . import LSSM

def projOrth(A, B):
    """
    Projects A onto B. A and B must be wide matrices with dim x samples.
    Returns:
    1) AHat: projection of A onto B
    2) W: The matrix that gives AHat when it is right multiplied by B
    """
    if B is not None:
        W = A @ B.T @ np.linalg.pinv(B @ B.T) # or: A / B = A * B.' * pinv(B * B.')
        AHat = W @ B  # or: A * B.' * pinv(B * B.') * B
    else:
        W = np.zeros( (A.shape[0], B.shape[1]) )
        AHat = np.zeros( A.shape )
    return (AHat, W)

def blkhankskip(Y, i, j = None, s = 0):
    ny, N = Y.shape[0], Y.shape[1]
    if j == None: 
        j = N - i + 1
    H = np.empty((ny * i, j))
    for r in range(i):
        H[slice(r*ny, r*ny + ny), :] = Y[:, slice(s+r, s+r+j)]
    return H

def PSID(Y, Z=None, nx=None, n1=0, i=None, WS=dict(), fitCzViaKF=True):
    """
    PSID PSID: Preferential Subspace Identification Algorithm
    Identifies a linear stochastic model for a signal y, while prioritizing
    the latent states that are predictive of another signal z. The model is
    as follows:
    [x1(k+1); x2(k+1)] = [A11 0; A21 A22] * [x1(k); x2(k)] + w(k)
                  y(k) =      [Cy1   Cy2] * [x1(k); x2(k)] + v(k)
                  z(k) =      [Cz1     0] * [x1(k); x2(k)] + e(k)
    x(k) = [x1(k); x2(k)] => Latent state time series
    x1(k) => Latent states related to z ( the pair (A11, Cz1) is observable )
    x2(k) => Latent states unrelated to z 
    Given training time series from y(k) and z(k), the dimension of x(k) 
    (i.e. nx), and the dimension of x1(k) (i.e. n1), the algorithm finds 
    all model parameters and noise statistics:
        - A  : [A11 0; A21 A22]
        - Cy : [Cy1   Cy2]
        - Cz : [Cz1     0]
        - Q  : Cov( w(k), w(k) )
        - R  : Cov( v(k), v(k) )
        - S  : Cov( w(k), v(k) )
    as well as the following model characteristics/parameters: 
        - G  : Cov( x(k+1), y(k) )
        - YCov: Cov( y(k), y(k) )
        - K: steady state stationary Kalman filter for estimating x from y
        - innovCov: covariance of innovation for the Kalman filter
        - P: covariance of Kalman predicted state error
        - xPCov: covariance of Kalman predicted state itself
        - xCov: covariance of the latent state
    
    Inputs:
        - (1) Y: Inputs signal 1 (e.g. neural signal). 
                Must be ny x T:
                [y(1), y(2), y(3), ..., y(T)]
        - (2) Z: Inputs signal 2, to be studied using y (e.g. behavior). 
                Must be nz x T:
                [z(1), z(2), z(3), ..., z(T)]
        - (3) nx: the total number of latent states in the stochastic model
        - (4) n1: number of latent states to extract in the first stage.
        - (5) i: the number of block-rows (i.e. future and past horizon) 
        - (6) WS: the WS output from a previous call using the exact 
                same data. If calling PSID repeatedly with the same data 
                and horizon, several computationally costly steps can be 
                reused from before. Otherwise will be discarded.
        - (7) fitCzViaKF (default: True): if true (preferred option), 
                refits Cz more accurately using a KF after all other 
                paramters are learned
    Outputs:
        - (1) idSys: an LSSM object with the system parameters for 
                the identified system. Will have the following 
                attributes (defined above), and some more attributes 
                and methods:    
                'A', 'Cy', 'Cz', 'Q', 'R', 'S'
                'G', 'YCov', 'K', 'innovCov', 'P', 'xPCov', 'xCov' 
        - (2) WS (optional): dictionary to provide to later calls of PSID
                on the same data (see input (6) for more details)
    Usage example:
        idSys = PSID(Y, Z, nx, n1, i);
        [idSys, WS] = PSID(Y, Z, nx, n1, i, WS);
        idSysSID = PSID(Y, Z, nx, 0, i); % Set n1=0 for SID
    """
    ny, Ndat = Y.shape[0], Y.shape[1]
    N = Ndat-2*i
    if Z is not None:
        nz, NdatZ = Z.shape[0], Z.shape[1]
    else:
        nz = 0

    if 'N' in WS and WS['N'] == N and 'i' in WS and WS['i'] == i and \
        'YShape' in WS and WS['YShape'] == Y.shape and \
        'ZShape' in WS and WS['ZShape'] == Z.shape and \
        'Y1' in WS and WS['Y1'] == Y[0,0] and \
        (nz == 0 or ('Z1' in WS and WS['Z1'] == Z[0,0])):
        # Have WS from previous call with the same data
        pass
    else:
        WS = {
            'N': N,
            'i': i,
            'YShape': Y.shape,
            'Y1': Y[0,0]
        }
        if nz > 0:
            WS['ZShape'] = Z.shape
            WS['Z1'] = Z[0,0]

    if 'Yp' not in WS or WS['Yp'] is None:
        WS['Yp'] = blkhankskip(Y, i, N)
        WS['Yii'] = blkhankskip(Y, 1, N, i)
        if nz > 0:
            WS['Zii'] = blkhankskip(Z, 1, N, i)
    
    if n1 > nx:
        n1 = nx  # Max possible n1 value

    # Stage 1
    if n1 > 0 and nz > 0:
        if 'ZHat_U' not in WS or WS['ZHat_U'] is None:
            Zf = blkhankskip(Z, i, N, i)
            WS['ZHat'] = projOrth(Zf, WS['Yp'])[0] # Zf @ WS['Yp'].T @ np.linalg.pinv(WS['Yp'] @ WS['Yp'].T) @ WS['Yp']  # Eq. (10)
            Yp_Plus = np.concatenate((WS['Yp'], WS['Yii']))
            Zf_Minus = Zf[nz:, :]
            WS['ZHatMinus'] = projOrth(Zf_Minus, Yp_Plus)[0] # Zf_Minus @ Yp_Plus.T @ np.linalg.pinv(Yp_Plus @ Yp_Plus.T) @ Yp_Plus  # Eq. (11)            

            # Take SVD of ZHat
            WS['ZHat_U'], WS['ZHat_S'], ZHat_V = np.linalg.svd(WS['ZHat'], full_matrices=False)     # Eq. (12)

        Sz = np.diag(WS['ZHat_S'][:n1])        # Eq. (12)
        Uz = WS['ZHat_U'][:  , :n1]            # Eq. (12)

        Oz = Uz @ Sz**(1/2)                    # Eq. (13)
        Oz_Minus = Oz[:-nz, :]                 # Eq. (15)

        Xk = np.linalg.pinv(Oz) @ WS['ZHat'];                    # Eq. (14)
        Xk_Plus1 = np.linalg.pinv(Oz_Minus) @ WS['ZHatMinus'];   # Eq. (16)
    else:
        n1 = 0
        Xk = np.empty([0, N])
        Xk_Plus1 = np.empty([0, N])
    
    # Stage 2
    n2 = nx - n1     
    if n2 > 0:
        if 'YHat_U' not in WS or WS['YHat_U'] is None or \
            'n1' not in WS or WS['n1'] != n1:
            WS['n1'] = n1
            
            Yf = blkhankskip(Y, i, N, i)
            Yf_Minus = Yf[ny:, :]

            if n1 > 0: # Have already extracted some states, so remove the already predicted part of Yf
                # Remove the already predicted part of future y
                Oy1 = projOrth(Yf, Xk)[1] # Yf @ Xk.T @ np.linalg.pinv(Xk @ Xk.T)  # Eq. (18) - Find the y observability matrix for Xk
                Yf = Yf - Oy1 @ Xk                           # Eq. (19)

                Oy1_Minus = Oy1[:-ny, :]                     # Eq. (20)
                Yf_Minus = Yf_Minus - Oy1_Minus @ Xk_Plus1   # Eq. (21)
            
            WS['YHat'] = projOrth(Yf, WS['Yp'])[0] # Yf @ WS['Yp'].T @ np.linalg.pinv(WS['Yp'] @ WS['Yp'].T) @ WS['Yp']
            Yp_Plus = np.concatenate((WS['Yp'], WS['Yii']))
            WS['YHatMinus'] = projOrth(Yf_Minus, Yp_Plus)[0] # Yf_Minus @ Yp_Plus.T @ np.linalg.pinv(Yp_Plus @ Yp_Plus.T) @ Yp_Plus  # Eq. (23)

            # Take SVD of YHat
            WS['YHat_U'], WS['YHat_S'], YHat_V = np.linalg.svd(WS['YHat'], full_matrices=False)     # Eq. (24)
    
        S2 = np.diag(WS['YHat_S'][:n2])              # Eq. (24)
        U2 = WS['YHat_U'][:  , :n2]                  # Eq. (24)
    
        Oy = U2 @ S2**(1/2)                          # Eq. (25) - VODM Book p86, Step 4
        Oy_Minus = Oy[:-ny, :]                       # Eq. (27) - VODM Book p86, Step 4

        Xk2 = np.linalg.pinv(Oy) @ WS['YHat'];                    # Eq. (26) - VODM Book p86, Step 5
        Xk2_Plus1 = np.linalg.pinv(Oy_Minus) @ WS['YHatMinus'];   # Eq. (28) - VODM Book p86, Step 5

        Xk = np.concatenate((Xk, Xk2))                            # Eq. (29)
        Xk_Plus1 = np.concatenate((Xk_Plus1, Xk2_Plus1))          # Eq. (29)

    
    # Parameter identification
    if n1 > 0:
        # A associated with the z-related states
        A = projOrth( Xk_Plus1[:n1, :], Xk[:n1, :] )[1]              # Eq. (17)
    else:
        A = np.empty([0, 0])
    
    if n2 > 0:
        A23 = projOrth(Xk_Plus1[n1:, :], Xk)[1] # Xk_Plus1[n1:, :] @ Xk.T @ np.linalg.pinv(Xk @ Xk.T)   # Eq. (30)
        if n1 > 0:
            A10 = np.concatenate((A, np.zeros([n1,n2])), axis=1)
            A = np.concatenate((A10, A23))   # Eq. (31)
        else:
            A = A23

    w = Xk_Plus1 -  A @ Xk                         # Eq. (34)

    if nz > 0:
        Cz = projOrth(WS['Zii'], Xk)[1] # WS['Zii'] @ Xk.T @ np.linalg.pinv(Xk @ Xk.T)    # Eq. (33)
    else:
        Cz = np.empty([0, nx])

    Cy = projOrth(WS['Yii'], Xk)[1] # WS['Yii'] @ Xk.T @ np.linalg.pinv(Xk @ Xk.T)    # Eq. (32) - VODM Book p86, Step 6
    v  = WS['Yii'] - Cy @ Xk                             # Eq. (34)

    # Compute noise covariances
    Q = (w @ w.T)/N                                # Eq. (35)
    S = (w @ v.T)/N                                # Eq. (35)
    R = (v @ v.T)/N                                # Eq. (35)

    Q = (Q + Q.T)/2      # Make precisely symmetric
    R = (R + R.T)/2      # Make precisely symmetric

    s = LSSM.LSSM(params = {
        'A': A,
        'C': Cy,
        'Q': Q,
        'R': R,
        'S': S
    })
    if fitCzViaKF and nz > 0:
        xHat = s.kalman(Y.T)[0]
        Cz = projOrth(Z, xHat.T)[1]
    s.Cz = Cz
    
    return s