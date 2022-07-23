from tensorboard.backend.event_processing import event_accumulator
from pprint import pp, pprint
from sys import argv
import matplotlib.pyplot as plt
import glob

if __name__ == '__main__':
    if len(argv) != 2:
        print('Usage: plot.py <log_dir>')
        exit(1)

    
    logs = glob.glob(argv[1] + '/**/events.out.*', recursive=True)

    eas = [event_accumulator.EventAccumulator(l) for l in logs]
    names = [l.replace('\\','/').split('/')[-2] for l in logs]

    for ea in eas:
        ea.Reload()

    #pprint(eas[0].Tags()['scalars'])

    for serie_name in ['avg_cost', 'actor_loss', 'nll', 'grad_norm', 'val_avg_reward']:
        plt.figure(dpi=100)

        for i, ea in enumerate(eas):    
            serie = [e.value for e in ea.Scalars(serie_name)]
            plt.plot(serie, label=names[i])

        plt.legend()
        plt.title(serie_name)
        plt.show()
