def feature_extraction_jio(frame, frame_size, frame_shift, fs, nfft):

    import math
    import numpy as np
    from python_speech_features import logfbank
    import lpAnalysis
    
    
    frame_size_seconds = frame_size/fs
    frame_shift_seconds = frame_shift/fs
    
    feature_fbank = logfbank(frame, samplerate=fs, winlen=frame_size_seconds, winstep=frame_shift_seconds, nfilt=21, nfft=nfft)

    feature_energy = math.sqrt(np.mean(frame*frame))
    feature_energy = np.reshape(feature_energy,(-1,1))

    _, pitch, _, formants = lpAnalysis( frame, fs )
    pitch = np.reshape(pitch, (1,-1))
    if math.isnan(pitch):
        pitch = fs/2
    pitch = np.reshape(pitch, (1,-1))

    feature = np.hstack([feature_fbank, feature_energy, pitch, formants])
    
    return feature




def lpAnalysis( s, fs, p=12 ):

    '''
    For the given signal estimate:
        LP Coefficients of the given signal using Burg's method,
        LP Residual
        Pitch using autocorrelation of LP Residual
        LP Spectrum
        Formants

    INPUT:
    s     =   signal upon which lp analysis is to be performed
    fs    =   sampling frequency of the signal (preferably 8kHz)
    p     =   lp order
    

    OUTPUT:
    lpres   =   LP Residual
    pitch   =   Pitch value in milli-seconds
    spectrum =  LP Spectrum
    formants    =   Formants in the spectrum (if any)

    '''

    import librosa, scipy, numpy
    import calculateF0once

    if not s.flags['F_CONTIGUOUS']:  # to avoid memomy management based error
        s = numpy.asfortranarray(s)

    s = s*numpy.hamming(len(s)) # this step is necessary to destroy any discontinuites present at beginning/end of frame

    lp_coeff = librosa.lpc(s, p) # returns lp filter denom polynomial
    

    lp_res_org = scipy.signal.lfilter(lp_coeff, [1], s) # lp residual
    lp_res = lp_res_org/max(abs(lp_res_org))

    # Calculate pitch
    pitch = calculateF0once(lp_res, fs)


    # Calculate Formats
    frequency, spectrum = scipy.signal.freqz(1, lp_coeff)
    spectrum = abs(spectrum) # frequency spectrum
    spectrum = 20*numpy.log10(spectrum) # this is an optional step. This will only highlight the peaks in spectrum
    frequency = frequency*fs/(2*numpy.pi) # convert frequency to Hertz

    peaks, _ = scipy.signal.find_peaks(spectrum) # indices of peaks
    formants = frequency[peaks] # value is in Hertz

    return lp_res, pitch, spectrum, formants



def calculateF0once( 
    data, 
    fs, 
    Fmin = 50,
    Fmax = 3000,
    voicingThreshold = 0.3,
    applyWindow = False
):
    '''
    calculates the fundamental frequency of a given signal.

    In this analysis the signal is treated as a monolithic data block, so this function, albeit being faster in execution, is only useful for stationary data. See calculateF0() for calculation of the time-varying fundemental frequency.

    Parameters
    data	a numpy array or a list if floats
    fs	sampling frequency [Hz]
    Fmin	lowest possible fundamental frequency [Hz]
    Fmax	highest possible fundamental frequency [Hz]
    voicingThreshold	threshold of the maximum in the autocorrelation function - similar to Praat's "Voicing threshold" parameter
    applyWindow	if True, a Hann window is applied to the FFT data during analysis
    
    Returns
    the estimated fundamental frequency [Hz], or 0 if none is found.
    '''

    import copy, numpy

    LOOKUP_TABLE_HANN = 4

    dataTmp = copy.deepcopy(data)
    
    # apply window
    if applyWindow:
        fftWindow = createLookupTable(len(dataTmp), LOOKUP_TABLE_HANN)
        dataTmp *= fftWindow
    
    # autocorrelation
    result = numpy.correlate(dataTmp, dataTmp, mode = 'full')
    r = result[result.size//2:] / float(len(data))
    
    # find peak in AC
    freq = numpy.nan
    try:
        xOfMax, valMax = findArrayMaximum(r,
            int(round(float(fs) / Fmax)),
            int(round(float(fs) / Fmin)))
        valMax /= max(r)
        freq = float(fs) / xOfMax
    except Exception as e:
        pass
    return freq
    

def createLookupTable(size, type = 3):
    '''
    creates a lookup table covering the range of [0..1]

    Parameters
    size	number of data values that are distributed over the range [0..1] the type of the lookup table. To date these types are supported:
    LOOKUP_TABLE_NONE: a rectangular window
    LOOKUP_TABLE_SINE: a sine function
    LOOKUP_TABLE_COSINE: a cosine function
    LOOKUP_TABLE_HAMMING: a Hamming window
    LOOKUP_TABLE_HANN: a Hann window
    '''

    import numpy, math


    LOOKUP_TABLE_NONE = 0
    LOOKUP_TABLE_SINE = 1
    LOOKUP_TABLE_COSINE = 2
    LOOKUP_TABLE_HAMMING = 3
    LOOKUP_TABLE_HANN = 4


    data = numpy.zeros(size)
    for i in range(size):
        xrel = float(i) / float(size)
        if type == LOOKUP_TABLE_NONE:
            tmp = 1
        elif type == LOOKUP_TABLE_SINE:
            tmp = math.sin (xrel * math.pi * 2)
        elif type == LOOKUP_TABLE_COSINE:
            tmp = math.cos (xrel * math.pi * 2)
        elif type == LOOKUP_TABLE_HAMMING:
            tmp = 0.54 - 0.46 * math.cos(2 * math.pi * xrel)
        elif type == LOOKUP_TABLE_HANN:
            tmp = 0.5 - 0.5 * math.cos(2 * math.pi * xrel)
        #elif type == LOOKUP_TABLE_GAUSSIAN:
        #   // y = exp(1) .^ ( - ((x-size./2).*pi ./ (size ./ 2)) .^ 2 ./ 2);
        #   tmp = pow((double)exp(1.0), (double)(( - pow ((double)(((FLOAT)x-table_size / 2.0) * math.pi / (table_size / 2.0)) , (double)2.0)) / 2.0));
        else:
            raise Exception('type ' + str(type) + ' not recognized')
        data[i] = tmp
    return data

def findArrayMaximum(
        data, 
        offsetLeft = 0, 
        offsetRight = -1, # if -1, the array size will be used
        doInterpolate = True, # increase accuracy by performing a 
                              # parabolic interpolation
):
    '''
    Parameters
    data	a numpy array
    offsetLeft	the index position at which analysis will commence
    offsetRight	the terminating index position. if -1, the array size will be used
    doInterpolate	if True: increase accuracy by performing a parabolic interpolation within the results
    
    Returns
    a list containing the index and the value of the maximum
    '''

    
    objType = type(data).__name__.strip()
    if objType != "ndarray":
        raise Exception('data argument is no instance of numpy.array')
    size = len(data)
    if (size < 1):
        raise Exception('data array is empty')
    xOfMax = -1
    valMax = min(data)
    if offsetRight == -1:
        offsetRight = size
    for i in range(offsetLeft + 1, offsetRight - 1):
        if data[i] >= data[i-1] and data[i] >= data[i + 1]:
            if data[i] > valMax:
                valMax = data[i]
                xOfMax = i
    if doInterpolate:
        if xOfMax > 0 and xOfMax < size - 1:
            # use parabolic interpolation to increase accuracty of result
            alpha = data[xOfMax - 1]
            beta = data[xOfMax]
            gamma = data[xOfMax + 1]
            xTmp = (alpha - gamma) / (alpha - beta * 2 + gamma) / 2.0
            xOfMax = xTmp + xOfMax
            valMax = interpolateParabolic(alpha, beta, gamma, xTmp)
    if xOfMax == -1:
        raise Exception("no maximum found")
    return [xOfMax, valMax]
    


def interpolateParabolic(
        alpha, 
        beta, 
        gamma, 
        x # relative position of read offset [-1..1]
):
    '''
    parabolic interpolation between three equally spaced values

    Parameters
    alpha	first value
    beta	second value
    gamma	third value
    x	relative position of read offset [-1..1]
    
    Returns
    the interpolated value
    '''

    import numpy, math

    if (x == 0): return beta
    
    #we want all numbers above zero ...
    offset = alpha;
    if (beta < offset): offset = beta
    if (gamma < offset): offset = gamma
    offset = math.fabs(offset) + 1
    
    alpha += offset;
    beta += offset;
    gamma += offset;
    
    a = b = c = 0;
    a = (alpha - 2.0 * beta + gamma) / 2.0
    if (a == 0):
        if (x > 1):
            return interpolateLinear(beta, gamma, x) - offset
        else:
            return interpolateLinear(alpha, beta, x + 1) - offset
    else:
        c = (alpha - gamma) / (4.0 * alpha)
        b = beta - a * c * c
        return (a * (x - c) * (x - c) + b) - offset



def interpolateLinear(
        y1, #
        y2, #
        x # weighting [0..1]. 0 would be 100 % y1, 1 would be 100 % y2
):
    '''
    simple linear interpolation between two variables

    Parameters
    y1	
    y2	
    x	weighting [0..1]: 0 would be 100 % y1, 1 would be 100 % y2
    
    Returns
    the interpolated value
    '''
    return y1 * (1.0 - x) + y2 * x

