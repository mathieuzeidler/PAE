import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from scipy.signal import wiener
from math_func import localMaxOP2, localMinOP2, localMaxPos, divideMaximums, divideMinimums, windowedSmoothing, finiteDiffDiscrete, savgolSmoothing
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def apply_gaussian_filter(M_SIGNAL, signal_name, sigma, k):

    # Apply Gaussian filter to ART signal
    if M_SIGNAL.size > 0 and signal_name == "M_ART" :
        filtered_M_SIGNAL = gaussian_filter1d(M_SIGNAL, sigma=sigma, mode='reflect')
        #filtered_M_SIGNAL = medfilt(M_SIGNAL, kernel_size=5
        #filtered_M_SIGNAL = wiener(M_SIGNAL, mysize=5, noise=0.5)
         
        # Display the filtered ART signal
        plt.figure(k)
        plt.plot(filtered_M_SIGNAL[1000000:1002001], label='Filtered ART')
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART (Filtered)")
        plt.title("Filtered SNUADC/ART Signal")
        plt.legend()
        k += 1

        # Display the original and filtered signals
        plt.figure(k)
        plt.plot(M_SIGNAL[1000000:1002001], label='Original ART', alpha=0.5)
        plt.plot(filtered_M_SIGNAL[1000000:1002001], label='Filtered ART', linewidth=2)
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART")
        plt.title("Original vs. Filtered SNUADC/ART Signal")
        plt.legend()
        k += 1

        plt.figure(k)
        smoothedM_Signal = windowedSmoothing(M_SIGNAL[1000000:1002001],75,0.1,4)
        dxM_Signal = finiteDiffDiscrete(smoothedM_Signal)
        critM_Signal = localMaxPos(dxM_Signal,0.5e-2)
        locMaxM_Art = localMaxOP2(smoothedM_Signal)
        locMinM_Art = localMinOP2(smoothedM_Signal)
        plt.plot(smoothedM_Signal,label="WINDOWED FILTER ART/PLETH")
        #plt.vlines(critM_Art,30,98,colors='lightcoral')
        plt.vlines(locMaxM_Art,30,98,colors='green')
        plt.vlines(locMinM_Art,30,98,colors='yellow')
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART (Filtered)")
        plt.legend()
        k += 1
        
        # Display the original and filtered signals
        plt.figure(k)
        plt.plot(M_SIGNAL[1000000:1002001], label='Original ART', alpha=0.5)
        plt.plot(smoothedM_Signal, label='Filtered ART', linewidth=2)
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART")
        plt.title("Original vs. Filtered SNUADC/ART Signal")
        plt.legend()
        k += 1
        
        # plt.figure(k)
        # smoothedM_Art2 = gaussian_filter1d(smoothedM_Art, sigma=6, mode='reflect')
        # dxM_Art = finiteDiffDiscrete(smoothedM_Art)
        # locMaxM_Art2 = localMaxOP2(smoothedM_Art2)
        # locMinM_Art2 = localMinOP2(smoothedM_Art2)
        # plt.plot(smoothedM_Art2,label="WINDOWED FILTER ART2")
        # #plt.vlines(critM_Art,30,98,colors='lightcoral')
        # plt.vlines(locMaxM_Art2,30,98,colors='green')
        # plt.vlines(locMinM_Art2,30,98,colors='yellow')
        # plt.xlabel("t")
        # plt.ylabel("SNUADC/ART (Filtered2)")
        # plt.legend()
        # k += 1
        
        
        plt.figure(k)
        plt.plot(smoothedM_Signal[0:300],label="WINDOWED FILTER ART")
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART (Filtered)")
        plt.legend()
        k += 1
        
        plt.figure(k)
        plt.plot(smoothedM_Signal,label="WINDOWED FILTER ART")
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART (Filtered)")
        plt.legend()
        k += 1
        
        plt.figure(k)
        plt.plot(dxM_Signal,label="DX WINDOWED FILTER ART")
        plt.plot(finiteDiffDiscrete(dxM_Signal),label="DXDX WINDOWED FILTER ART")
        plt.plot()
        plt.plot
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART (Derivative)")
        plt.legend()
        k += 1
        
        plt.figure(k)
        plt.plot(finiteDiffDiscrete(dxM_Signal),label="DXDX WINDOWED FILTER ART")
        plt.plot()
        plt.plot
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART (Derivative)")
        plt.legend()
        k += 1
        
        plt.figure(k)
        corr = np.correlate(smoothedM_Signal, smoothedM_Signal,mode='full')
        plt.plot(corr, label='correlation', alpha=0.5)
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART")
        plt.title("Original vs. Filtered SNUADC/ART Signal")
        plt.legend()
        k += 1
        
        plt.figure(k)
        meanSIGNAL = np.mean(M_SIGNAL)
        plt.plot(filtered_M_SIGNAL, label='correlation', alpha=0.5)
        plt.hlines(meanSIGNAL,0,6e6)
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART")
        plt.title("Original vs. Filtered SNUADC/ART Signal")
        plt.legend()
        k += 1
        
        #plt.figure(k)
        #meanSIGNAL = np.mean(M_SIGNAL)
        #plt.plot(filtered_M_SIGNAL[2000000], label='correlation', alpha=0.5)
        #plt.hlines(meanSIGNAL,0,6e6)
        #plt.xlabel("t")
        #plt.ylabel("SNUADC/ART")
        #plt.title("Original vs. Filtered SNUADC/ART Signal")
        #plt.legend()
        #k += 1
        smoothed5= savgolSmoothing(M_SIGNAL[1000000:1002001],100,3,sigma)
        locAbsM = divideMaximums(smoothed5,locMaxM_Art[0])
        locAbsm = divideMinimums(smoothed5,locMinM_Art[0])
        #plt.figure(k)
        #meanSIGNAL= np.mean(M_SIGNAL)
        #plt.plot(smoothedM_Signal, label='smoothed')
        #plt.vlines(locAbsM[0,:],color='red',ymin=60,ymax=95)
        #plt.vlines(locAbsM[1,:],color='green',ymin=60,ymax=95)
        #plt.vlines(locAbsm[0,:],color='yellow',ymin=60,ymax=95)
        #plt.vlines(locAbsm[1,:],color='blue',ymin=60,ymax=95)
        #plt.xlabel("t")
        #plt.ylabel("SNUADC/ART")
        #plt.title("Original vs. Filtered SNUADC/ART Signal")
        #plt.legend()
        #k += 1

        plt.figure(k)
        smoothed2 = savgolSmoothing(M_SIGNAL[1000000:1002001],40,3,sigma)
        plt.plot(M_SIGNAL[1000000:1002001], label='p', alpha=0.5)
        plt.plot(smoothed2, label='Filtered ART', linewidth=2)
        plt.xlabel("t")
        plt.ylabel("MIRAR AHORA")
        plt.title("Original vs. Filtered SNUADC/ART Signal")
        plt.legend()
        k += 1

        plt.figure(k)
        smoothed3 = gaussian_filter1d(smoothed2, sigma=sigma, mode='reflect')
        plt.plot(M_SIGNAL[1000000:1002001], label='orig', alpha=0.5)
        plt.plot(smoothed2, label='smoothed-savgol', linewidth=2, color='red')
        plt.plot(smoothed3, label='smoothed-gauss(savgol)',color='green')
        plt.plot(smoothedM_Signal, label='smoothed-gauss(savgol)',color='black', alpha=0.5)
        k += 1

        plt.figure(k)
        smoothed4= savgolSmoothing(M_SIGNAL[1000000:1002001],8,3,sigma)
        smoothed5= savgolSmoothing(M_SIGNAL[1000000:1002001],100,3,sigma)
        plt.plot(M_SIGNAL[1000000:1002001], label='orig', alpha=0.5)
        plt.plot(smoothed4, label='smoothed-savgol', linewidth=2, color='red')
        plt.plot(smoothed5, label='smoothed-gauss(savgol)',color='green')
        plt.plot(smoothedM_Signal, label='smoothed-gauss(savgol)',color='black', alpha=0.5)
        plt.title("EEEEEEE")
        k += 1

        plt.figure(k)
        dx5 = finiteDiffDiscrete(smoothed5)
        plt.plot(dx5, label='smoothed-savgol', linewidth=2, color='red')
        plt.title("EEEEEEE")
        k += 1

    # Apply Gaussian filter to PLETH signal
    if M_SIGNAL.size > 0 and signal_name == "M_PLETH" :
        filtered_M_SIGNAL = gaussian_filter1d(M_SIGNAL, sigma=sigma, mode='reflect')
        #filtered_M_SIGNAL = medfilt(M_SIGNAL, kernel_size=5)
        #filtered_M_SIGNAL = wiener(M_SIGNAL, mysize=5, noise=0.5)
        
        # Display the filtered ART signal
        plt.figure(k)
        plt.plot(filtered_M_SIGNAL[1000000:1002001], label='Filtered PLETH')
        plt.xlabel("t")
        plt.ylabel("SNUADC/PLETH (Filtered)")
        plt.title("Filtered SNUADC/PLETH Signal")
        plt.legend()
        k += 1

        # Display the original and filtered signals
        plt.figure(k)
        plt.plot(M_SIGNAL[1000000:1002001], label='Original PLETH', alpha=0.5)
        plt.plot(filtered_M_SIGNAL[1000000:1002001], label='Filtered PLETH', linewidth=2)
        plt.xlabel("t")
        plt.ylabel("SNUADC/PLETH")
        plt.title("Original vs. Filtered SNUADC/PLETH Signal")
        plt.legend()
        k += 1

        plt.figure(k)
        smoothedM_Signal = windowedSmoothing(M_SIGNAL[1000000:1002001],75,0.1,4)
        smoothed2 = savgolSmoothing(M_SIGNAL[1000000:1002001],40,3,sigma)
        dxM_Signal = finiteDiffDiscrete(smoothed2)
        critM_Signal = localMaxPos(dxM_Signal,0.5e-2)
        locMaxM_Art = localMaxOP2(smoothed2)
        locMinM_Art = localMinOP2(smoothed2)
        plt.plot(smoothed2,label="WINDOWED FILTER ART/PLETH")
        #plt.vlines(critM_Art,30,98,colors='lightcoral')
        plt.vlines(locMaxM_Art,30,98,colors='green')
        plt.vlines(locMinM_Art,30,98,colors='yellow')
        plt.xlabel("t")
        plt.ylabel("SNUADC/PLETH (Filtered)")
        plt.legend()
        k += 1
        
        # Display the original and filtered signals
        plt.figure(k)
        plt.plot(M_SIGNAL[1000000:1002001], label='Original PLETH', alpha=0.5)
        plt.plot(smoothedM_Signal, label='Filtered PLETH', linewidth=2)
        plt.xlabel("t")
        plt.ylabel("SNUADC/PLETH")
        plt.title("Original vs. Filtered SNUADC/PLETH Signal")
        plt.legend()
        k += 1
        
        # plt.figure(k)
        # smoothedM_Art2 = gaussian_filter1d(smoothedM_Art, sigma=6, mode='reflect')
        # dxM_Art = finiteDiffDiscrete(smoothedM_Art)
        # locMaxM_Art2 = localMaxOP2(smoothedM_Art2)
        # locMinM_Art2 = localMinOP2(smoothedM_Art2)
        # plt.plot(smoothedM_Art2,label="WINDOWED FILTER ART2")
        # #plt.vlines(critM_Art,30,98,colors='lightcoral')
        # plt.vlines(locMaxM_Art2,30,98,colors='green')
        # plt.vlines(locMinM_Art2,30,98,colors='yellow')
        # plt.xlabel("t")
        # plt.ylabel("SNUADC/ART (Filtered2)")
        # plt.legend()
        # k += 1
        
        
        plt.figure(k)
        plt.plot(smoothedM_Signal[0:300],label="WINDOWED FILTER PLETH")
        plt.xlabel("t")
        plt.ylabel("SNUADC/PLETH (Filtered)")
        plt.legend()
        k += 1
        
        plt.figure(k)
        plt.plot(smoothedM_Signal,label="WINDOWED FILTER PLETH")
        plt.xlabel("t")
        plt.ylabel("SNUADC/PLETH (Filtered)")
        plt.legend()
        k += 1
        
        plt.figure(k)
        plt.plot(dxM_Signal,label="DX WINDOWED FILTER PLETH")
        plt.plot(finiteDiffDiscrete(dxM_Signal),label="DXDX WINDOWED FILTER PLETH")
        plt.plot()
        plt.plot
        plt.xlabel("t")
        plt.ylabel("SNUADC/PLETH (Derivative)")
        plt.legend()
        k += 1
        
        plt.figure(k)
        plt.plot(finiteDiffDiscrete(dxM_Signal),label="DXDX WINDOWED FILTER PLETH")
        plt.plot()
        plt.plot
        plt.xlabel("t")
        plt.ylabel("SNUADC/PLETH (Derivative)")
        plt.legend()
        k += 1
        
        plt.figure(k)
        corr = np.correlate(smoothedM_Signal, smoothedM_Signal,mode='full')
        plt.plot(corr, label='correlation', alpha=0.5)
        plt.xlabel("t")
        plt.ylabel("SNUADC/PLETH")
        plt.title("Original vs. Filtered SNUADC/PLETH Signal")
        plt.legend()
        k += 1
        
        plt.figure(k)
        meanSIGNAL = np.mean(M_SIGNAL)
        plt.plot(filtered_M_SIGNAL, label='correlation', alpha=0.5)
        plt.hlines(meanSIGNAL,0,6e6)
        plt.xlabel("t")
        plt.ylabel("SNUADC/PLETH")
        plt.title("Original vs. Filtered SNUADC/PLETH Signal")
        plt.legend()
        k += 1
        
        #plt.figure(k)
        #meanSIGNAL = np.mean(M_SIGNAL)
        #plt.plot(filtered_M_SIGNAL[2000000], label='correlation', alpha=0.5)
        #plt.hlines(meanSIGNAL,0,6e6)
        #plt.xlabel("t")
        #plt.ylabel("SNUADC/PLETH")
        #plt.title("Original vs. Filtered SNUADC/PLETH Signal")
        #plt.legend()
        #k += 1
        
        
        smoothed5= savgolSmoothing(M_SIGNAL[1000000:1002001],100,3,sigma)
        locAbsM = divideMaximums(smoothed5,locMaxM_Art[0])
        locAbsm = divideMinimums(smoothed5,locMinM_Art[0])
        #plt.figure(k)
        #meanSIGNAL= np.mean(M_SIGNAL)
        #plt.plot(smoothedM_Signal, label='smoothed')
        #plt.vlines(locAbsM[0,:],color='red',ymin=60,ymax=95)
        #plt.vlines(locAbsM[1,:],color='green',ymin=60,ymax=95)
        #plt.vlines(locAbsm[0,:],color='yellow',ymin=60,ymax=95)
        #plt.vlines(locAbsm[1,:],color='blue',ymin=60,ymax=95)
        #plt.xlabel("t")
        #plt.ylabel("SNUADC/PLETH")
        #plt.title("Original vs. Filtered SNUADC/PLETH Signal")
        #plt.legend()
        #k += 1

        plt.figure(k)
        smoothed2 = savgolSmoothing(M_SIGNAL[1000000:1002001],40,3,sigma)
        plt.plot(M_SIGNAL[1000000:1002001], label='p', alpha=0.5)
        plt.plot(smoothed2, label='Filtered PLETH', linewidth=2)
        plt.xlabel("t")
        plt.ylabel("MIRAR AHORA")
        plt.title("Original vs. Filtered SNUADC/PLETH Signal")
        plt.legend()
        k += 1

        plt.figure(k)
        smoothed3 = gaussian_filter1d(smoothed2, sigma=sigma, mode='reflect')
        plt.plot(M_SIGNAL[1000000:1002001], label='orig', alpha=0.5)
        plt.plot(smoothed2, label='smoothed-savgol', linewidth=2, color='red')
        plt.plot(smoothed3, label='smoothed-gauss(savgol)',color='green')
        plt.plot(smoothedM_Signal, label='smoothed-gauss(savgol)',color='black', alpha=0.5)
        k += 1

        plt.figure(k)
        smoothed4= savgolSmoothing(M_SIGNAL[1000000:1002001],8,3,sigma)
        smoothed5= savgolSmoothing(M_SIGNAL[1000000:1002001],100,3,sigma)
        plt.plot(M_SIGNAL[1000000:1002001], label='orig', alpha=0.5)
        plt.plot(smoothed4, label='smoothed-savgol', linewidth=2, color='red')
        plt.plot(smoothed5, label='smoothed-gauss(savgol)',color='green')
        plt.plot(smoothedM_Signal, label='smoothed-gauss(savgol)',color='black', alpha=0.5)
        plt.title("EEEEEEE")
        k += 1

        plt.figure(k)
        dx5 = finiteDiffDiscrete(smoothed5)
        plt.plot(dx5, label='smoothed-savgol', linewidth=2, color='red')
        plt.title("EEEEEEE")
        k += 1

    return filtered_M_SIGNAL,k, locAbsM, locAbsm, smoothedM_Signal