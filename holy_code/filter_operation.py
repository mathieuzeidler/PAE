import numpy as np
from scipy.ndimage import gaussian_filter1d
from math_func import localMaxOP2, localMinOP2, localMaxPos, divideMaximums, divideMinimums, windowedSmoothing, finiteDiffDiscrete
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def apply_gaussian_filter(M_ART, sigma, k):

    # Apply Gaussian filter to ART signal
    if M_ART.size > 0:
        filtered_M_ART = gaussian_filter1d(M_ART, sigma=sigma, mode='reflect')

        # Display the filtered ART signal
        plt.figure(k)
        plt.plot(filtered_M_ART[1000000:1002001], label='Filtered ART')
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART (Filtered)")
        plt.title("Filtered SNUADC/ART Signal")
        plt.legend()
        k += 1

        # Display the original and filtered signals
        plt.figure(k)
        plt.plot(M_ART[1000000:1002001], label='Original ART', alpha=0.5)
        plt.plot(filtered_M_ART[1000000:1002001], label='Filtered ART', linewidth=2)
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART")
        plt.title("Original vs. Filtered SNUADC/ART Signal")
        plt.legend()
        k += 1

        plt.figure(k)
        smoothedM_Art = windowedSmoothing(M_ART[1000000:1002001],75,0.1,4)
        dxM_Art = finiteDiffDiscrete(smoothedM_Art)
        critM_Art = localMaxPos(dxM_Art,0.5e-2)
        locMaxM_Art = localMaxOP2(smoothedM_Art)
        locMinM_Art = localMinOP2(smoothedM_Art)
        plt.plot(smoothedM_Art,label="WINDOWED FILTER ART")
        #plt.vlines(critM_Art,30,98,colors='lightcoral')
        plt.vlines(locMaxM_Art,30,98,colors='green')
        plt.vlines(locMinM_Art,30,98,colors='yellow')
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART (Filtered)")
        plt.legend()
        k += 1
        
        # Display the original and filtered signals
        plt.figure(k)
        plt.plot(M_ART[1000000:1002001], label='Original ART', alpha=0.5)
        plt.plot(smoothedM_Art, label='Filtered ART', linewidth=2)
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
        plt.plot(smoothedM_Art[0:300],label="WINDOWED FILTER ART")
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART (Filtered)")
        plt.legend()
        k += 1
        
        plt.figure(k)
        plt.plot(smoothedM_Art,label="WINDOWED FILTER ART")
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART (Filtered)")
        plt.legend()
        k += 1
        
        plt.figure(k)
        plt.plot(dxM_Art,label="DX WINDOWED FILTER ART")
        plt.plot(finiteDiffDiscrete(dxM_Art),label="DXDX WINDOWED FILTER ART")
        plt.plot()
        plt.plot
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART (Derivative)")
        plt.legend()
        k += 1
        
        plt.figure(k)
        plt.plot(finiteDiffDiscrete(dxM_Art),label="DXDX WINDOWED FILTER ART")
        plt.plot()
        plt.plot
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART (Derivative)")
        plt.legend()
        k += 1
        
        plt.figure(k)
        corr = np.correlate(smoothedM_Art, smoothedM_Art,mode='full')
        plt.plot(corr, label='correlation', alpha=0.5)
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART")
        plt.title("Original vs. Filtered SNUADC/ART Signal")
        plt.legend()
        k += 1
        
        plt.figure(k)
        meanART = np.mean(M_ART)
        plt.plot(filtered_M_ART, label='correlation', alpha=0.5)
        plt.hlines(meanART,0,6e6)
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART")
        plt.title("Original vs. Filtered SNUADC/ART Signal")
        plt.legend()
        k += 1
        
        plt.figure(k)
        meanART = np.mean(M_ART)
        plt.plot(filtered_M_ART[2000000], label='correlation', alpha=0.5)
        plt.hlines(meanART,0,6e6)
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART")
        plt.title("Original vs. Filtered SNUADC/ART Signal")
        plt.legend()
        k += 1
        
        locAbsM = divideMaximums(smoothedM_Art,locMaxM_Art[0])
        locAbsm = divideMinimums(smoothedM_Art,locMinM_Art[0])
        plt.figure(k)
        meanART = np.mean(M_ART)
        plt.plot(smoothedM_Art, label='smoothed')
        plt.vlines(locAbsM[0,:],color='red',ymin=60,ymax=95)
        plt.vlines(locAbsM[1,:],color='green',ymin=60,ymax=95)
        plt.vlines(locAbsm[0,:],color='yellow',ymin=60,ymax=95)
        plt.vlines(locAbsm[1,:],color='blue',ymin=60,ymax=95)
        plt.xlabel("t")
        plt.ylabel("SNUADC/ART")
        plt.title("Original vs. Filtered SNUADC/ART Signal")
        plt.legend()
        k += 1
            
        # Update M_ART with the filtered values
        M_ART = filtered_M_ART

    return M_ART,k