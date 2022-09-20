from array import array
import numpy as np

def stack_ragged(array_list, axis=0):
    lengths = [np.shape(a)[axis] for a in array_list]
    idx = np.cumsum(lengths[:-1])
    stacked = np.concatenate(array_list, axis=axis)
    return stacked, idx

def save_stacked_array(fname, array_list, axis=0):
    stacked, idx = stack_ragged(array_list, axis=axis)
    np.savez(fname, stacked_array=stacked, stacked_index=idx)

def load_stacked_arrays(fname, axis=0):
    npzfile = np.load(fname)
    idx = npzfile['stacked_index']
    stacked = npzfile['stacked_array']
    return np.split(stacked, idx, axis=axis)

a = np.array([1, 2, 3])
b = np.array([1,2,3,4,5])
c = np.array([1,2])

save_stacked_array('arrs.npz', [a,b,c], axis=0)
array_list = load_stacked_arrays('arrs.npz', axis=0)
print(array_list)