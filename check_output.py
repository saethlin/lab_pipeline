import numpy as np

kimock = np.loadtxt('state_09.txt')
xiang = np.loadtxt('lycrt-example/model/state_09.txt')

kimock_metric = (kimock[:,0] * kimock[:, 2]).sum() / kimock[:, 2].sum()
xiang_metric = (xiang[:,0] * xiang[:, 2]).sum() / xiang[:, 2].sum()

print('Kimock:')
print(kimock_metric)
print(kimock.shape)

print('Ma')
print(xiang_metric)
print(xiang.shape)

