from tensorboard.backend.event_processing import event_accumulator
from pprint import pprint
from sys import argv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(argv) != 2:
        print('Usage: plot.py <log_file>')
        exit(1)

    ea = event_accumulator.EventAccumulator(argv[1])

    ea.Reload()  # loads events from file

    for serie_name in ea.Tags()['scalars']:
        serie = [e.value for e in ea.Scalars(serie_name)]
        
        plt.plot(serie, label=serie_name)
        plt.legend()
        plt.show()
