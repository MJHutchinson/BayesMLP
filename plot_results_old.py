import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_immediate_files

log_dir = './results'
data_set = 'MNIST'

results_dir = f'{log_dir}/{data_set}'

files = get_immediate_files(results_dir)

fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax2.set_xlabel('First epoch accuracy')
ax2.set_ylabel('Best accuracy')

for file in files:

    results = pickle.load(open(f'{results_dir}/{file}', 'rb'))

    for key in results.keys():
        result = results[key]
        ax1.plot(result['accuracies'])

    initial_accs = []
    best_accs = []
    hiddens = []

    for key in results.keys():
        result = results[key]
        initial_accs.append(result['accuracies'][0])
        best_accs.append(max(result['accuracies']))
        hiddens.append(key)

    ax2.scatter(initial_accs, best_accs)
    ax2.plot(np.unique(initial_accs), np.poly1d(np.polyfit(initial_accs, best_accs, 1))(np.unique(initial_accs)))

ax2.legend(files)

plt.show()



