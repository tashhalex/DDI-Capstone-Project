import glob
import os
print(glob.glob('data/raw_tles/*.csv'))

print('Current Working Director:', os.getcwd())
print('Contents of data/raw_tles/:', os.listdir('data/raw_tles'))