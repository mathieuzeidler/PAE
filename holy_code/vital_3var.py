import vitaldb as vd
import matplotlib.pyplot as plt
import numpy as np



#VARIABLES WE WANT TO KEEP
list_var = ['SNUADC/PLETH', 'SNUADC/ART', 'SNUADC/PLETH_SPO2']

#CONVERT THE .VITAL FILE TO A .VITAL FILE WITH ONLY 3 VARIABLES
vf = vd.VitalFile("1.vital", list_var)
vf.to_vital('3var.vital')

#NOW WE HAVE A .VITAL FILE WITH ONLY 3 VARIABLES
#WE CAN NOW READ THE FILE
vf = vd.VitalFile("3var.vital", list_var)

#FIRST WE WANT TO SEE THE ARRAYS OF THE VARIABLES
#ELIMINATE THE NAN VALUES
list_var1 = vf.get_track_samples('SNUADC/PLETH', 1/100)
list_var1 = [x for x in list_var1 if not np.isnan(x)]
list_var2 = vf.get_track_samples('SNUADC/ART', 1/100)
list_var2 = [x for x in list_var2 if not np.isnan(x)]

#print(list_var1)  
#print(list_var2)  
#WE CAN NOW PLOT THE DATA

samples_pleth = vf.to_numpy('SNUADC/PLETH', 1/100)
samples_art = vf.to_numpy('SNUADC/ART', 1/100)
#samples_spo2 = vf.to_numpy('SNUADC/PLETH_SPO2', 1/100)
plt.figure(figsize=(20, 10))

# Primer gráfico
plt.subplot(2, 1, 1)  # 2 filas, 1 columna, primer gráfico
plt.plot(list_var1, label='SNUADC/PLETH')
plt.legend()

# Segundo gráfico
plt.subplot(2, 1, 2)  # 2 filas, 1 columna, segundo gráfico
plt.plot(list_var2, label='SNUADC/ART')
plt.legend()

plt.show()