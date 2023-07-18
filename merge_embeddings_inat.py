import numpy as np

representation_list = []

idx = 0

for i in range(10):
    rep_file = np.load('embeddings/representations_inat_{}.npz'.format(i))
    for j in range(1000):
        reps = rep_file['arr_{}'.format(j)]
        representation_list.append(reps)
        print(idx)
        idx += 1

np.savez('embeddings/representations_inat.npz', *representation_list)
