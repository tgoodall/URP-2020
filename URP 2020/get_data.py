from define_func import read_from_file, dftAnal, peakDetection, peakInterp, sineTracking, cleaningSineTracks
from scipy.signal import get_window
import math
import numpy as np
import pyfftw
import math
import os


# Creates Data Array!
def data(file, window, M, t, N, Ns, freqDevOffset, freqDevSlope, maxnSines, minSineDur):
    

    # read audio chunks from audio file
    sample_rate, audio_data = read_from_file(file)

    
    H = N//4 #hop size for analysis and synthesis
    w = get_window(window, M)
    hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
    hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
    audio_data = np.append(np.zeros(hM2),audio_data)                          # add zeros at beginning to center first window at sample 0
    audio_data = np.append(audio_data,np.zeros(hM2))                          # add zeros at the end to analyze last sample
    #hNs = Ns//2                                             # half of synthesis FFT size
    #pin = max(hNs, hM1)                                     # init sound pointer in middle of anal window       
    pin = hM1                                     # init sound pointer in middle of anal window
    #pend = audio_data.size - max(hNs, hM1)                           # last sample to start a frame
    pend = audio_data.size - hM1                           # last sample to start a frame
    w = w / sum(w)                                          # normalize analysis window
    #hM1 = (w.size+1)//2                                     # half analysis window size by rounding
    #hM2 = w.size//2                                         # half analysis window size by floor

    numfft = math.floor((int(audio_data.shape[0])+(H/2))/H)
    ind = 0
    tol = 1e-14                                                 # threshold used to compute phase
    tfreq=np.array([])

    while pin<pend:                                         # while input sound pointer is within sound 
        # Roughly DFT Anal code:
        x1 = audio_data[pin-hM1:pin+hM2]                               # select frame
        mX, pX = dftAnal(x1, w, N)                        # compute dft
        ploc = peakDetection(mX, t)                        # detect locations of peaks
        iploc, ipmag, ipphase = peakInterp(mX, pX, ploc)   # refine peak values by interpolation
        ipfreq = sample_rate*iploc/float(N)                            # convert peak locations to Hertz
        tfreq, tmag, tphase = sineTracking(ipfreq, ipmag, ipphase, tfreq, freqDevOffset, freqDevSlope)
        tfreq = np.resize(tfreq, min(maxnSines, tfreq.size))  # limit number of tracks to maxnSines
        #put amp val and freq vals in first 300 bins
        tfreq = np.resize(tfreq, min(maxnSines, tfreq.size))  # limit number of tracks to maxnSines
        tmag = np.resize(tmag, min(maxnSines, tmag.size))     # limit number of tracks to maxnSines
        tphase = np.resize(tphase, min(maxnSines, tphase.size)) # limit number of tracks to maxnSines
        jtfreq = np.zeros(maxnSines)                          # temporary output array
        jtmag = np.zeros(maxnSines)                           # temporary output array
        jtphase = np.zeros(maxnSines)                         # temporary output array   
        jtfreq[:tfreq.size]=tfreq                             # save track frequencies to temporary array
        jtmag[:tmag.size]=tmag                                # save track magnitudes to temporary array
        jtphase[:tphase.size]=tphase                          # save track magnitudes to temporary array
        if pin == hM1:                                        # if first frame initialize output sine tracks
            xtfreq = jtfreq 
            xtmag = jtmag
            xtphase = jtphase
            #before sine tracking
            mags = iploc[:maxnSines]
            freaks = ipfreq[:maxnSines]
            #ffts
            ffts = mX
        else:                                                 # rest of frames append values to sine tracks
            xtfreq = np.vstack((xtfreq, jtfreq))
            xtmag = np.vstack((xtmag, jtmag))
            xtphase = np.vstack((xtphase, jtphase))
            #before sine tracking
            mags = np.vstack((mags, ipmag[:maxnSines]))
            freaks = np.vstack((freaks, ipfreq[:maxnSines]))
            #ffts
            ffts = np.vstack((ffts,mX))
        pin += H
    xtfreq = cleaningSineTracks(xtfreq, round(sample_rate*minSineDur/H))
    return sample_rate, freaks, ffts, mags
    