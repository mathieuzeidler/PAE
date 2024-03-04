import vitaldb

import matplotlib.pyplot as plt

track_names = ['SNUADC/ART']

vf = vitaldb.VitalFile(1, track_names)

samples = vf.to_numpy(track_names, 1/100)

print(samples)

#vitaldb.VitalFile.to_wav("VitalDB_data/VitalDB_data/NewData", track_names, srate=None)

plt.figure(figsize=(20, 5))

plt.plot(samples[:, 0])

plt.show()


