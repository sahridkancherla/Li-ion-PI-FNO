import numpy as np
data = np.load('spm_data_v2.npz')
print('c_sn_train shape:', data['c_sn_train'].shape)  # expect (800, 20, 500)
print('c_sn_val shape:', data['c_sn_val'].shape)        # expect (100, 20, 500)
print('c_sn_test shape:', data['c_sn_test'].shape)       # expect (100, 20, 500)

I_train = data['I_train']
t_end_train = data['t_end_train']

n = I_train.shape[0]
n_charge, n_rest = 0, 0
for i in range(n):
    I_sample = I_train[i, 0, :]
    if (I_sample < -1e-6).any():
        n_charge += 1
    if (np.abs(I_sample) < 1e-3).any():
        n_rest += 1

print(f'Out of {n} training samples:')
print(f'  samples with charging current: {n_charge} ({100*n_charge/n:.1f}%)')
print(f'  samples with rest periods: {n_rest} ({100*n_rest/n:.1f}%)')
print()
print('t_end range:', t_end_train.min(), '-', t_end_train.max(), 'seconds')
print('any NaN in c_sn?', np.isnan(data['c_sn_train']).any())
print('any NaN in c_sp?', np.isnan(data['c_sp_train']).any())