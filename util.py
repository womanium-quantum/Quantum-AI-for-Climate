"""
Created by Ben Kroul, in 2024

Defines useful utility functions and constants. Run printModule(util) after importing to see full list of imports.
- CVALS: object of physics constants
- printModule
- timeIt
- binarySearch
- linearInterpolate
- uFormat: PDG-style formatting of numbers with uncertainty, or just to significant figures
- RSquared, NRMSE
- FuncWLabels & FuncAdder: objects to fit composite functions for signal modelling. 
"""
import numpy as np
from glob import glob
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
#import matplotlib.lines as mlines
#import plotly.graph_objects as go

import sys, time, os

from scipy import special  # for voigt function
from scipy.optimize import curve_fit


# -- CONSTANTS -- #
DATADIR      = "/Users/benkroul/Documents/Physics/Data/"
SAVEDIR      = "/Users/benkroul/Documents/Physics/plots/"
SAVEEXT      = ".png"
FIGSIZE      = (10,6)
TICKSPERTICK = 5
FUNCTYPE     = type(sum)

class justADictionary():
    def __init__(self, my_name):
        self.name = my_name
        self.c    = 2.99792458 # 1e8   m/s speed of lgiht
        self.h    = 6.62607015 # 1e-34 J/s Plancks constant,
        self.kB   = 1.380649   # 1e-23 J/K Boltzmanns constant, 
        self.e    = 1.60217663 # 1e-19 C electron charge in coulombs
        self.a    = 6.02214076 # 1e23  /mol avogadros number
        self.Rinf = 10973731.56816  # /m rydberg constant
        self.G    = 0.0 # m^3/kg/s^2 Gravitational constant
        self.neutron_proton_mass_ratio = 1.00137842     # m_n / m_p
        self.proton_electron_mass_ratio = 1836.15267343 # m_p / m_e
        self.wien = 2.89777 # 1e-3  m*K  peak_lambda = wien_const / temp
    
    def __str__(self):
        return self.name

CVALS = justADictionary("Useful Physics constants, indexed in class for easy access")

# IBM's colorblind-friendly colors
#           |   Red  |   Blue  |  Orange |  Purple | Yellow  |   Green |   Teal  | Grey
hexcolors = ['DC267F', '648FFF', 'FE6100', '785EF0', 'FFB000', '009E73', '3DDBD9', '808080']
mpl.rcParams['axes.prop_cycle'] = cycler('color', [mpl.colors.to_rgba('#' + c) for c in hexcolors])

def savefig(title):
    plt.savefig(SAVEDIR + title + SAVEEXT, bbox_inches='tight')


# -- GENERAL FUNCTIONS -- #
def printModule(module):
    """print a module AFTER IMPORTING IT"""
    print("all imports:")
    numListedPerLine = 3; i = 0
    imported_stuff = dir(module)
    lst = [] # list of tuples of thing, type
    types = []
    for name in imported_stuff:
        if not name.startswith('__'):  # ignore the default namespace variables
            typ = str(type(eval(name))).split("'")[1]
            lst.append((name,typ))
            if typ not in types:
                types.append(typ)
    for typ in types:
        rowstart = '  '+typ+'(s): '
        i = 0; row = rowstart
        for id in lst:
            if id[1] != typ: continue
            i += 1
            row += id[0] +', '
            if not i % numListedPerLine:
                print(row[:-2])
                row = rowstart
        if len(row) > len(rowstart):
            print(row[:-2])
        i += numListedPerLine

def timeIt(func):
    """@ timeIt: Wrapper to print run time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.clock_gettime_ns(0)
        res = func(*args, **kwargs)
        end_time = time.clock_gettime_ns(0)
        diff = (end_time - start_time) * 10**(-9)
        print(func.__name__, 'ran in %.6fs' % diff)
        return res
    return wrapper

def binarySearch(X_val, X: list|tuple|np.ndarray, decreasing=False):
    """
    For sorted X, returns index i such that X[:i] < X_val, X[i:] >= X_val
     - if decreasing,returns i such that    X[:i] > X_val, X[i:] <= X_val
    """
    l = 0; r = len(X) - 1
    #print(f"searching for {X_val}, negative={negative}")
    m = (l + r) // 2
    while r > l:  # common binary search
        #print(f"{l}:{r} is {X[l:r+1]}, middle {X[m]}")
        if X[m] == X_val:  # repeat elements of X_val in array
            break
        if decreasing: # left is always larger than right
            if X[m] > X_val:
                l = m + 1
            else:
                r = m - 1
        else:        # right is always larger than left
            if X[m] < X_val:
                l = m + 1
            else:
                r = m - 1
        m = (l + r) // 2
    if r < l:
        return l
    if m + 1 < len(X):  # make sure we are always on right side of X_val
        if X[m] < X_val and not decreasing:
            return m + 1
        if X[m] > X_val and decreasing:
            return m + 1
    if X[m] == X_val:  # repeat elements of X_val in array
        if decreasing:
            while m > 0 and X[m - 1] == X_val:
                m -= 1
        elif not decreasing:
            while m + 1 < len(X) and X[m + 1] == X_val:
                m += 1
    return m

# linear interpolate 1D with sorted X
def linearInterpolate(x,X,Y):
    """example: 2D linear interpolate by adding interpolations from both
    - """
    i = binarySearch(x,X)
    if i == 0: i += 1  # lowest ting but we interpolate backwards
    m = (Y[i]-Y[i-1])/(X[i]-X[i-1])
    b = Y[i] - m*X[i]
    return m*x + b


# - ---- -STATS FUNCTIONS

def RSquared(y, model_y):
    """R^2 correlation coefficient of data"""
    n = len(y)
    # get mean
    SSR = SST = sm = 0
    for i in range(n):
        sm += y[i]
    mean_y = sm / n
    for i in range(n):
        SSR += (y[i] - model_y[i])**2
        SST += (y[i] - mean_y)**2
    return 1 - (SSR / SST)

def NRMSE(y, model_y, normalize=True):
    """Root mean squared error, can be normalized by range of data"""
    n = len(y)
    sm = 0; min_y = y[0]; max_y = y[0]
    for i in range(n):
        if y[i] < min_y: min_y = y[i]
        if y[i] > max_y: max_y = y[i]
        sm += (y[i] - model_y[i])**2
    y_range = max_y - min_y
    val = np.sqrt(sm/n)
    if normalize: 
        val = val / y_range
    return val

# ----- TEXT MANIPULATION

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# method to return the string form of an integer (0th, 1st, 2nd, 3rd, 4th...)
Ith = lambda i: str(i) + ("th" if (abs(i) % 100 in (11,12,13)) else ["th","st","nd","rd","th","th","th","th","th","th"][abs(i) % 10])

def arrFromString(data, col_separator = '\t', row_separator = '\n'):
    """ Return numpy array from string
    - great for pasting Notion tables into np array """
    x = []; L = 0
    for row in data.split(row_separator):
        if len(row):  # ignore any empty rows
            entries = row.split(col_separator)
            newL = len(entries)
            if L != 0 and newL != L:
                print(f"Rows have different lengths {L} and {newL}:")
                print(x)
                print(entries)
            L = newL
            x.extend(entries)
    return np.reshape(np.array(x,dtype='float64'),(-1,L))

def uFormat(number, uncertainty=0, figs = 4, shift = 0, FormatDecimals = False):
    """
    Returns "num_rounded(with_sgnfcnt_dgts_ofuncrtnty)", formatted to 10^shift
      or number rounded to figs significant figures if uncertainty = 0
    According to section 5.3 of "https://pdg.lbl.gov/2011/reviews/rpp2011-rev-rpp-intro.pdf"

    Arguments:
    - float number:      the value
    - float uncertainty: the absolute uncertainty (stddev) in the value
       - if zero, will just format number to figs significant figures (see figs)
    - int figs: when uncertainty = 0, format number to degree of sig figs instead
       - if zero, will simply return number as string
    - int shift:  optionally, shift the resultant number to a higher/lower digit expression
       - i.e. if number is in Hz and you want a string in GHz, specify shift = 9
               likewise for going from MHz to Hz, specify shift = -6
    - bool FormatDecimals:  for a number 0.00X < 1e-2, option to express in "X.XXe-D" format
             for conciseness. doesnt work in math mode because '-' is taken as minus sign
    """
    num = str(number); err = str(uncertainty)
    
    sigFigsMode = not uncertainty    # UNCERTAINTY ZERO: IN SIG FIGS MODE
    if sigFigsMode and not figs: # nothing to format
        return num
    
    negative = False  # add back negative later
    if num[0] == '-':
        num = num[1:]
        negative = True
    if err[0] == '-':  # stddev is symmetric ab number
        err = err[1:]
    
    # ni = NUM DIGITS to the RIGHT of DECIMAL
    # 0.00001234=1.234e-4 has ni = 8, 4 digs after decimal and 4 sig figs
    # 1234 w/ ni=5 corresponds to 0.01234
    ni = ei = 0  
    if 'e' in num:
        ff = num.split('e')
        num = ff[0]
        ni = -int(ff[1])
    if 'e' in err:
        ff = err.split('e')
        err = ff[0]
        ei = -int(ff[1])

    if not num[0].isdigit():
        print(f"uFormat: {num} isn't a number")
        return num
    if not err[0].isdigit():
        err = '?'

    # comb through error, get three most significant figs
    foundSig = False; decimal = False
    topThree = ""; numFound = 0
    jErr = ""
    for ch in err:
        if decimal:
            ei += 1
        if not foundSig and ch == '0': # dont care ab leading zeroes
            continue  
        if ch == '.':
            decimal = True
            continue
        jErr += ch
        if numFound >= 3:  # get place only to three sigfigs
            ei -= 1
            continue
        foundSig = True
        topThree += ch
        numFound += 1
    
    foundSig = False; decimal = False
    jNum = ""
    for ch in num:
        if decimal:
            ni += 1
        if not foundSig and ch == '0': # dont care ab leading zeroes
            continue
        if ch == '.':
            decimal = True
            continue
        jNum += ch
        foundSig = True
    if len(jNum) == 0:  # our number is literally zero!
        return '0'
    
    # round error correctly according to PDG
    if len(topThree) == 3:
        nTop = int(topThree)
        if nTop < 355: # 123 -> (12.)
            Err = int(topThree[:2])
            if int(topThree[2]) >= 5:
                Err += 1
            ei -= 1
        elif nTop > 949: # 950 -> (10..)
            Err = 10
            ei -= 2
        else:  # 355 -> (4..)
            Err = int(topThree[0])
            if int(topThree[1]) >= 5:
                Err += 1
            ei -= 2
        Err = str(Err)
    else:
        Err = topThree

    n = len(jNum); m = len(Err)
    nBefore = ni - n  #; print(num, jNum, n, ni, nBefore)
    eBefore = ei - m  #; print(err, Err, m, ei, eBefore)
    if nBefore > eBefore:  # uncertainty is a magnitude larger than number, still format number
        if not sigFigsMode:
            print(f'Uncrtnty: {uncertainty} IS MAGNITUDE(S) > THAN Numba: {number}')
        Err = '?'
    if sigFigsMode or nBefore > eBefore:
        ei = nBefore + figs

    # round number to error
    d = ni - ei 
    if ni == ei: 
        Num = jNum[:n-d]
    elif d > 0:  # error has smaller digits than number = round number
        Num = int(jNum[:n-d])
        if int(jNum[n-d]) >= 5:
            Num += 1
        Num = str(Num)
    else:  # error << num
        Num = jNum
        if ei < m + ni:
            Err = Err[n+d-1]
        else:
            Err = '0'
    if ni >= ei: ni = ei  # indicate number has been rounded

    n = len(Num)
    # if were at <= e-3 == 0.009, save formatting space by removing decimal zeroes
    extraDigs = 0
    if not shift and FormatDecimals and (ni-n) >= 2:
        shift -= ni - n + 1
        extraDigs = ni - n + 1
    
    # shift digits up/down by round argument
    ni += shift
    end = ''

    # there are digits to the right of decimal and we dont 
    # care about exact sig figs (ex. cut zeroes from 0.02000)
    if ni > 0 and not sigFigsMode:
        while Num[-1] == '0':
            if len(Num) == 1: break
            Num = Num[:-1]
            ni -= 1
            n -= 1
    
    if ni >= n:   # place decimal before any digits
        Num = '0.' + "0"*(ni-n) + Num
    elif ni > 0:  # place decimal in-between digits
        Num = Num[:n-ni] + '.' + Num[n-ni:]
    elif ni < 0:  # add non-significant zeroes after number
        end = 'e'+str(-ni)
    if extraDigs:  # format removed decimal zeroes
        end = 'e'+str(-extraDigs)
    
    if negative: Num = '-' + Num  # add back negative
    if not sigFigsMode:
        end = '(' + Err + ')' + end
    return Num + end

# ----- SIGNAL MODELS ------ #
def ScaledGaussian(X, m, s, a):    # statistical/thermodynamic normal distribution
    return abs(a)*np.exp(-(X-m)**2/(2*s*s))

def ScaledLorentzian(X, m, s, a):  # physics signal function
    return abs(a)/((2*(X - m)/s)**2 + 1)

def ScaledVoigt(X, m, sg, sl, a):  # most robust signal modeler
    sg = abs(sg); sl = abs(sl)
    z = (X - m + sl*1j)/(sg*np.sqrt(2))
    peak = special.wofz(sl*1j / (sg*np.sqrt(2)) ).real
    if peak == 0 or peak == np.NaN:
        return np.zeros(len(X))
    if peak == np.inf:
        return np.ones(len(X))*np.inf
    return a*special.wofz(z).real / peak

def planckFreqSpectrum(X,T,A): # in frequency
    return A*(X**3)*CVALS.h*2e-32 / (CVALS.c*CVALS.c*(np.exp(X*CVALS.h*1e-5/(CVALS.kB*T)) - 1))

def planckWavelengthSpectrum(X,T,A):
    A0 = 1e6/(4.09567405227*T**5)  # normalize spectrum, set peak to 1
    return A*A0*CVALS.h*CVALS.c*CVALS.c*2e27 / (X**5*(np.exp(CVALS.h*CVALS.c*1e6/(X*CVALS.kB*T)) - 1))

def Polynomial(X, *coeffs):  # general corrective polynomial? usually just constant
    Y = np.ones(len(X))*coeffs[0]
    for i in range(1,len(coeffs)):
        Y += coeffs[i] * X**i
    return Y

def scaledPowerFunc(X,p,w):
    return w*(X**p)

def scaledExponentialFunc(X,exp,w):
    return w*(exp**X)

def linearStep(X, m1, b1, m2, b2):
    cutoff = (b2-b1)/(m1-m2)
    cut_idx = binarySearch(X, cutoff)
    return np.concatenate((X[:cut_idx]*m1+b1,X[cut_idx:]*m2+b2))

# ---- WRAP MODELS W/ PARAM NAMES/LABELS -----
class FuncWLabels():
    """Wrap a function with coefficient labels, excluding the first arg"""
    def __init__(self, func, lbls):
        if func.__code__.co_argcount - 1 != len(lbls):
            print(f"function {func.__name__} has {func.__code__.co_argcount - 1} coeffs \
                  and you provided {len(lbls)} labels.")
        self.func = func
        self.lbls = lbls
    
    def __str__(self):
        return self.func.__name__
    
    def __len__(self):
        return len(self.lbls)

Gaussian = FuncWLabels(ScaledGaussian,[r"\mu_{",r"\sigma_{","a_{"])
Lorentzian = FuncWLabels(ScaledLorentzian,[r"\mu_{",r"\sigma_{","a_{"])
Voigt = FuncWLabels(ScaledVoigt, [r"\mu_{",r"\sigma_{g",r"\sigma_{\ell","a_{"])
WavelengthBBR = FuncWLabels(planckWavelengthSpectrum, [r"T_{",r"A_{"])
FrequencyBBR = FuncWLabels(planckFreqSpectrum, [r"T_{",r"A_{"])
ScaledPower = FuncWLabels(scaledPowerFunc, ["p_{","w_{"])
ScaledExp   = FuncWLabels(scaledExponentialFunc, ["exp_{","w_{"])
LinearStep  = FuncWLabels(linearStep, ["m1_{","b1_{","m2_{","b2_{"])

# func can be single component funcs or a list of component funcs
# will make ncopies of each func or list of component funcs
# FUNC = sum of funcs and their components
class FuncAdder():
    """
    Makes a composite function from a list of FuncWLabels
     - FuncWLabels or list of FuncWLabels, functions stored w their coeff labels
       - self.funcs: list of functions
       - self.lbls:  list of all labels, each ending with an open "{" for labeling
       - self.nargs: number of arguments for each function (not including X)
     - ncopies of all functions to add together, or list of ncopies per function
       - self.ncopies:  int OR list of ints
       - if 0, function is "frozen" and added once but IS NOT FIT LATER
       - only works if "frozen" functions are all placed before "unfrozen" ones
     - initialize coefficients - FULL coeff list given to composite function, including frozies
       - self.coeffs: list of all coeffs
     - name given to the funcAdder (composite function name)
       - self.name
     - addPoly to add a polynomial function up to the nth polynomial offset
       - addPoly = 1 will add y-intercept, 2 will add parabolic func, etc.
    """
    def __init__(self, funcWLabels, ncopies = 1, coeffs = [], name = "funcAdder",
                 addPoly = 0) -> None:
        self.name = self.__name__ = name
        if not isinstance(ncopies, list): ncopies = [ncopies]
        self.ncopies = ncopies
        self.coeffs = coeffs
        self.covar  = []
        self.set_funcs(funcWLabels, self.ncopies, addPoly)

    def __str__(self) -> str:
        return self.printCoeffs()
    
    def set_funcs(self, funcWLabels, ncopies, addPoly = 0) -> None:
        """Compile self.funcs, self.nargs, and self.lbls from funcWLabels and ncopies
        - funcWLabels = list of funcWLabels
        - ncopies = list of ncopies to be made for each function, can be shorter than funcWLabels
        - addPoly = k: add polynomial of degree k
        """
        self.funcs = []; self.nargs = []; self.lbls = []
        self.frozen_idx = 0  # every coeff before this will not be fit
        if not isinstance(funcWLabels, list):      # list of functions
            if isinstance(funcWLabels, FuncWLabels):
                funcWLabels = [funcWLabels]
            elif isinstance(funcWLabels, str):     # u can name your function as a string?
                funcWLabels = [exec(funcWLabels)]
                if not isinstance(funcWLabels[0], FuncWLabels):
                    sys.exit("this should be a FuncWLabels type",funcWLabels,type(funcWLabels))
            elif addPoly:
                funcWLabels = []
                print("assuming just polynomial of degree",addPoly)
        elif not isinstance(funcWLabels[0], FuncWLabels):
            if isinstance(funcWLabels[0], str): # of function strs?
                funcWLabels = [exec(func_str) for func_str in funcWLabels]
            else:
                sys.exit("what have u done... what is this--", funcWLabels, type(funcWLabels))
        # iterate thru functions, ncopies, and populate relevant fields
        ncopy_idx = 0; ncopy = ncopies[ncopy_idx] # iterate thru ncopies
        frozen = ncopy == 0
        for funcWLbl in funcWLabels: # for all functions
            f = funcWLbl.func
            lbls = funcWLbl.lbls
            if not frozen and ncopy == 0:
                print(f"Please place frozen functions at the beginning. \
                        proceeding with ncopy=1 for {f.__name__}")
            if frozen and ncopy:
                frozen = False
            if not ncopy:
                ncopy = 1
            self.funcs.extend([f]*ncopy)  # duplicate ncopies times
            nargs = f.__code__.co_argcount - 1
            if nargs != len(lbls):
                print("NOT OK",self.name,f,lbls)
            self.nargs.extend([nargs]*ncopy)
            if frozen:
                self.frozen_idx += nargs
            if ncopy == 1:
                self.lbls.extend([lbl+'}' for lbl in lbls])
            else:
                for i in range(1,ncopy+1):
                    self.lbls.extend([lbl+str(i)+'}' for lbl in lbls])
            if ncopy_idx + 1 < len(ncopies): # iterate ncopy
                ncopy_idx += 1
                ncopy = ncopies[ncopy_idx]
        # add an intercept for addPoly = 1, up to any arbitrary polynomial
        if addPoly > 0: 
            self.funcs.append(Polynomial)
            self.nargs.append(addPoly)
            if addPoly == 1:
                self.lbls.append("C")
            elif addPoly == 2:
                self.lbls.extend(["b","m"])
            else:
                self.lbls.extend(ALPHABET[:addPoly][::-1])

    # return the individual Ys of all funcs
    def indiv_Ys(self, X, coeffs = None, alsoFuncName = False):
        n = len(self.funcs)
        outputs = []
        coeff_index = 0
        if not coeffs:  # no coeffs specified, go with the saved ones
            coeffs = self.coeffs
        if len(coeffs) < len(self.coeffs): # self.fit w/ frozen coeffs
            coeffs = self.coeffs[:self.frozen_idx] + list(coeffs)
        for i in range(n):
            f = self.funcs[i]
            nargs = self.nargs[i]
            cs = coeffs[coeff_index: coeff_index + nargs]
            coeff_index += nargs
            if alsoFuncName:
                outputs.append((f(X, *cs), f.__name__))
            else:
                outputs.append(f(X, *cs))
        return outputs

    def predict(self, X, *coeffs):
        """Use model to return Y_predict from X"""
        return np.sum(self.indiv_Ys(X, coeffs), axis=0)
    
    def fit(self, X, Y, initial_coeffs = None):
        """fits model over X, Y, returning Y_predict"""
        if not initial_coeffs:
            initial_coeffs = self.coeffs
        if len(initial_coeffs) + self.frozen_idx > len(self.coeffs):
            initial_coeffs = initial_coeffs[self.frozen_idx:]
        try:
            self.coeffs[self.frozen_idx:], self.covar = curve_fit(self.predict, X, Y, p0=initial_coeffs)
        except RuntimeError:
            print("ERROR: curve_fit could not predict given the model")
            print('\t',', '.join(["%d:%s" % (self.nargs[i],self.funcs[i].__name__) for i in range(len(self.funcs))]))
            print('\tcoeffs:',self.formatCoeffs(range(0,len(self.coeffs)),1))
        return self.predict(X, *self.coeffs)

    def formatCoeffs(self, indices, printType=2):
        """uses .lbls and uFormat to return list of correctly formated coefficients for single or list of indices
        - printType = 
          - 0: return copy-able list
          - 1: format coeffs to sig figs w/ labels
          - 2: format coeffs to uncertainty w/ labels"""
        if type(indices) == int:  # single coefficient index given
            indices = [indices]
        if printType == 2:
            lst = []
            for j in indices:
                if j < self.frozen_idx:  # frozen idxs aren't fit, and thus have no covariance
                    cvar = 0
                else:
                    cvar = np.sqrt(self.covar[j-self.frozen_idx,j-self.frozen_idx])
                lst.append('$'+self.lbls[j]+'='+uFormat(self.coeffs[j], cvar)+'$')
            return lst
        elif printType == 1: 
            return ['$'+self.lbls[j]+'='+uFormat(self.coeffs[j],0,figs=4)+'$' for j in indices]
        else:
            return [str(self.coeffs[j]) for j in indices]

    def printCoeffs(self, printType = 2, nCoeffsPerLine = 2, onlyMus=False) -> str:
        ''' Returns string of model coefficients formatted for printing or plotting
        - printType:
          - 0: print coeffs in copy-able format as list of coeffs, w/ lines separated by function
          - 1: format coeffs for plotting with their labels
          - 2: format coeffs with labels AND uncertainties for plotting
        - nCoeffsPerLine = max num coefficients for plotting formatting
        - onlyMus: only return 'mean' coefficients (for gaussian, voigt, lorentzian)
        '''
        if not len(self.coeffs):
            return self.name+" has no coeffs to print"
        if not len(self.covar) and printType == 2:  # no covars to format
            printType = 1
        n = len(self.coeffs); i = 0; allCoeffs = []
        func = allCoeffs.extend if printType else lambda x: allCoeffs.append(', '.join(x))
        for fi in range(len(self.funcs)):
            nargs = self.nargs[fi]
            fname = self.funcs[fi].__name__
            thing = ''
            if onlyMus:
                if fname == "Gaussian" or fname == "Lorentzian" or fname == "Voigt":
                    thing = self.formatCoeffs(i, printType)
            else:
                thing = self.formatCoeffs(range(i, i+nargs), printType)
            func(thing)
            i += nargs
        # join extra non-function labels, if they exist
        if i + 1 < n:
            func(self.formatCoeffs(range(i, n), printType))
        if printType:
            i = 0; n = len(allCoeffs); lines = []
            while i < n:
                lines.append(', '.join(allCoeffs[i: i + nCoeffsPerLine]))
                i += nCoeffsPerLine
            return '\n'.join(lines)
        return '['+"\n".join(allCoeffs)+']'

    def plotComposite(self, X, plot_name = "", coeffs = [], opacity=1., labelPeaks = False, 
                      Axis = None, plot_size = FIGSIZE, showFig = False):
        """
        Plot all the individual component functions that add up to composite model 
        - plot_name is self-explanatory
        - coeffs: optionally specify coeffs for composite func, else will use self.coeffs
        - opacity: of the component lines
        - labelPeaks: will add peaks of any signal functions to the legend
        - Axis: specify existing pyplot axis to plot onto. if None, will make a new plot
        - plot_size: size of plot for making a new plot
        - showFig: plt.show() to show figure after making new plot
        """
        if not len(coeffs):
            coeffs = self.coeffs
        Y_add = self.predict(X, *coeffs)

        if not Axis:
            plt.figure("COMPOSITION_OF_FUNCADD", figsize=plot_size)
            ax = plt.gca()
        else:
            ax = Axis
        ax.plot(X, Y_add, label='Composite Fit', alpha=1)
        coeff_index = 0
        for i in range(len(self.funcs)):  # plot components individually
            f = self.funcs[i]
            nargs = self.nargs[i]
            cs = coeffs[coeff_index: coeff_index + nargs]
            if labelPeaks and f.__name__ != "Polynomial":  # add peaks into plot
                #print(f.__name__)
                if "planck" == f.__name__[:6]: # black body spectra
                    mu = 1e6*CVALS.wien / cs[0]  # wien displacement law for peak
                else:
                    mu = cs[0]
                y_mu = f(mu, *cs)
                ax.plot(X, f(X, *cs), label = "(%.1f,%.4f" % (mu, y_mu) + ')', linestyle='dashed', color=f"C{i}", alpha=opacity)
                ax.scatter(mu, y_mu, marker='o', color=f"C{i}", alpha=opacity, s=5)
            else:
                ax.plot(X, f(X, *cs), linestyle='dashed', color=f"C{i}", alpha=opacity)
            coeff_index += nargs
        
        if not Axis:
            ticksPerTick = 5
            ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(ticksPerTick))
            ax.yaxis.set_minor_locator(mtick.AutoMinorLocator(ticksPerTick))
            plt.grid(which="both",color='#E5E5E5')
            plt.title("Composite Plot of " + plot_name + ", " + self.name)
            plt.legend(loc='best')
            print('made',plot_name.replace(' ','_') + '_' + self.name.replace(' ','_')+ '_composite_plot.png')
            plt.savefig(plot_name.replace(' ','_') + '_'+self.name.replace(' ','_')+'_composite_plot.png', bbox_inches='tight')
            if showFig:
                plt.show()
            plt.clf()
    
# EXAMPLE data array generation
make_example_data = False
if make_example_data:
    maxchars = max([len(fname) for fname in glob(dir+'/*')])
    data_entry = np.dtype([('name', f'U{maxchars}'),('data', 'O'),('numpts','i4')])
    data = np.array([], dtype=data_entry)
    # get data from file
    for fname in glob(dir+'/*'):
        arr = np.loadtxt(fname)
        entry = np.array([(fname, arr, len(arr))], dtype = data_entry)
        data = np.append(data, entry)

# ---- PLOTTING SCRIPTS ----- #
def plotRaw(arr: np.ndarray | tuple[np.ndarray], title: str, axes_titles: str | tuple[str], saveName=None, Axis=None, lines = [], axes_ranges = []):
    """Plots 2D array or tuple of arrays. Specify plt.ax with Axis.
    - axes_ranges: set (xlim, ylim) for tuple axis limitations
    - lines: list of X-positions for drawing vertical lines"""
    X = arr[0] if isinstance(arr, tuple) else arr[:,0]
    Y = arr[1] if isinstance(arr, tuple) else arr[:,1]
    if not isinstance(axes_titles, tuple):  # default to x-label
        axes_titles = (axes_titles, r"Wavelength $\lambda$, nm")
    fig, ax = (Axis, plt.gcf()) if Axis else plt.subplots(figsize=FIGSIZE)
    if len(lines):
        for line_x in lines:
            ax.plot([line_x,line_x],[min(Y),max(Y)+0.002], linewidth=1,color='blue')
    ax.plot(X, Y, label='data',color='C0',zorder=2)
    ax.xaxis.set_minor_locator(mtick.AutoMinorLocator(TICKSPERTICK))
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator(TICKSPERTICK))
    ax.set(xlabel = axes_titles[0])
    if axes_titles[0][0] == 'C':
        ax.set(ylabel = axes_titles[1])
    if axes_ranges:  # set ranges on axes from list of two tuples
        if not axes_ranges[0]:
            axes_ranges[0] = ax.get_xlim()
        if not axes_ranges[1]:
            axes_ranges[1] = ax.get_ylim()
        ax.set(xlim = axes_ranges[0], ylim = axes_ranges[1])
    if Axis:
        ax.set(title=title)
        return
    if saveName:
        fig.savefig(SAVEDIR+saveName+".pdf", bbox_inches="tight")
        print("Saved figure to "+SAVEDIR+saveName+".pdf")
    plt.legend()
    plt.show()

#print("da master physics/CS folder - good luck code monkey")
if __name__ == "__main__":
    # behold my pride and joy - uFormat
    while True:
        t = input("Enter space-separated arguments to uFormat in the order of \n\
                  number, uncertainty, sig_figs=4, shift=0, FormatDecimals=T/F)\n:")
        if not t:
            break 
        args = t.rstrip().split(' ')
        i = 0; cs = []
        while i < min(len(args),6):
            if i < 2:
                cs.append(float(args[i]))
            elif i < 4:
                cs.append(int(args[i]))
            else:
                cs.append('t' in args[i].lower())
            i += 1
        print(uFormat(*cs))
            