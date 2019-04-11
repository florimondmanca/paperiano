import pickle
import numpy as np

labels = pickle.load(file=open('test_labels.pickle', 'rb'))
results = pickle.load(file=open('test_results.pickle', 'rb'))

count = 0
count_error = 0

for i in range(len(labels)):
    lab = labels[i]
    res = results[i]
    for j in range(len(lab)):
        lab_cat = np.argmax(lab[j])
        res_cat = np.argmax(res[j])
        if lab_cat != res_cat:
            count_error += 1
        count += 1

print('nb error :', count_error)
print('nb images :', count)
print('error rate:', count_error/count)
