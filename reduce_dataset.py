import skimage.measure as skm
import numpy as np 

F = np.load('Thumos14reduced-UntrimmedNets-JOINTFeatures.npy')
print('loaded...')
features = []
for f in F:
	features.append(np.clip(skm.block_reduce(np.array(f), (4,1), np.max), -10.0, 10.0))
	print(len(features))
np.save('Thumos14reduced-UNT-JOINTFeatures.npy', features)
