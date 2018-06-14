# Module to calculate relative positions and radial velocities for
# binary system
import scipy
f = scipy.MachAr()
machep = f.eps

def eccan(Ecc, M, Tol = 1.0e-8, Nmax = 50):
    """Calculate eccentric anomaly using Newton-Raphson process."""
    if M < Tol: return M
    x = Ecc * scipy.sin(M) / (1 - Ecc * scipy.cos(M))
    Eo = M + x * (1-x*x/2.)
    Diff = 1
    Flag = 0
    i = 0
    while (Diff > Tol):
        En = Eo + (M  + Ecc * scipy.sin(Eo) - Eo) / (1 - Ecc * scipy.cos(Eo))
        Diff = abs((En - Eo) / Eo)
        Eo = En
        i += 1
        if i >= Nmax:
            if Flag ==1:
                print Ecc, M
                print 'Eccan did not converge'
                return M
            Flag = 1
            i = 0
            Eo = M 
            Diff = 1
    return En

def phase(JD, P, T0 = 0.0):
    """Phase-fold array of dates; result in range [0:1]."""
    Phase = ((JD-T0) % P) / P
# ensure > 0
    Phase[Phase <= 0.] += 1
    return Phase

def truean(JD, P, T0 = 0, Ecc = 0):
    """Calculate true anomaly for array of dates."""
    Phase = phase(JD, P, T0) # phases
    M = 2 * scipy.pi * Phase # mean anomaly
    if Ecc <= machep:
        return M
    eccanV = scipy.vectorize(eccan)
    E = eccanV(Ecc, M) % (2 * scipy.pi)  # eccentric anomaly
    cosE = scipy.cos(E)
    cosNu = (cosE - Ecc) / (1 - Ecc * cosE)
    Nu = scipy.arccos(cosNu) # true anomaly
    Nu = scipy.select([E <= scipy.pi, scipy.ones(len(Nu))], \
                          [Nu, 2 * scipy.pi - Nu]) # E>pi cases
    return Nu

def truedist(Nu, a, Ecc = 0):
    """True distance from true anomaly, orbital distance & eccentricity."""
    return a * (1 - Ecc**2) / (1 + Ecc * scipy.cos(Nu))

def orbitcoord(JD, P, T0 = 0, Ecc = 0, a = 1):
    """Coordinates in orbital plane. X is towards observer."""
    Nu = truean(JD, P, T0, Ecc)
    r = truedist(Nu, a, Ecc)
    X = r * scipy.cos(Nu)
    Y = r * scipy.sin(Nu)
    return X, Y, Nu

def skycoord(JD, P, T0 = 0, Ecc = 0, a = 1, \
                 incl = scipy.pi/2, Omega = 0, omega = 0):
    """Coordinates in plane of sky. y is North."""
    X, Y, Nu = orbitcoord(JD, P, T0, Ecc, a)
    cosi = scipy.cos(incl)
    sini = scipy.sin(incl)
    cosO = scipy.cos(Omega)
    sinO = scipy.sin(Omega)
    coso = scipy.cos(omega)
    sino = scipy.sin(omega)
    cosxX = - cosi * sinO * sino + cosO * coso
    cosxY = - cosi * sinO * coso - cosO * sino
    cosyX = cosi * cosO * sino + sinO * coso
    cosyY = cosi * cosO * coso - sinO * sino
    x = X * cosxX + Y * cosxY
    y = X * cosyX + Y * cosyY
    z = scipy.sqrt(X**2+Y**2) * sini * scipy.sin(omega+Nu)
    return x, y, z

def radvel(JD, P, K, T0 = 0, V0 = 0, Ecc = 0, omega = 0):
    """Radial velocity (user-defined semi-amplitude)"""
    Nu = truean(JD, P, T0, Ecc)
    Vr = V0 + K * (scipy.cos(omega + Nu) + Ecc * scipy.cos(omega))
    # if (K < 0): Vr[:] = -999
    return Vr

def rv_K(P, f):
    """Calculate RV semi-amplitude for a given mass function f."""
    return (2 * scipy.pi * 6.67e-11 / (P*86400)**2)**(1/3.) * f

def msini(a_AU, K_min=1, dur_yr = 10, M_Msun = 1):
    """Calculate the minimum detectable mass (in Jupiter masses) as a
    function of semi-major axis (in AU) and for a given minimum
    semi-amplitude (in m/s), stellar mass (in Solar masses) and survey
    duration (in years)."""
    a = a_AU * 1.49e11
    print a_AU
    dur = 365.0 * 3600.0 * 24.0 * dur_yr
    print dur_yr
    M = M_Msun * 1.989e30
    print M_Msun
    G = 6.67e-11
    P = 2 * scipy.pi * (a**3 / (G * M))**(1/2.)
    P_yr = P / (365.0 * 3600.0 * 24.0)
    print P_yr
    mlim = K_min * (M**2 * P / (2 * scipy.pi * G))**(1/3.)
    if scipy.size(mlim) > 1:
        mlim[P>dur] = scipy.nan
    else:
        if P > dur: mlim = scipy.nan
    m_mjup = mlim / 1.89e27
    print m_mjup
    return m_mjup

def getT0(P, Ttr, omega = 0, Ecc = 0):
    """Compute time of periastron passage from time of transit centre
    and other orbital elements."""    
    nu = scipy.pi/2 - omega
    cosnu = scipy.cos(nu)
    cosE = (Ecc + cosnu) / (1 + Ecc * cosnu)
    E = scipy.arccos(cosE)
    if (nu > -scipy.pi) and (nu < 0.0): E = -E  #mcquillan:1/4/10
    M = E - Ecc * scipy.sin(E)
    T0 = Ttr - M * P / (2 * scipy.pi)
    return T0

def getTtr(P, T0, omega = 0, Ecc = 0):
    """Compute time of transit centre from time of periastron passage
    and other orbital elements."""    
    nu = scipy.pi/2 - omega
    cosnu = scipy.cos(nu)
    cosE = (Ecc + cosnu) / (1 + Ecc * cosnu)
    E = scipy.arccos(cosE)
    if (nu > -scipy.pi) and (nu < 0.0): E = -E  #mcquillan:1/4/10
    M = E - Ecc * scipy.sin(E)
    Ttr = T0 + M * P / (2 * scipy.pi)
    return Ttr

