import numpy as np
import matplotlib.pyplot as plt

def plot_training_results(path: str) -> None:
    f = np.load(f'{path}/evaluations.npz')

    mean = np.mean(f["results"], 1)
    max_ind = np.argmax(mean)

    plt.figure()
    plt.plot(f["timesteps"], mean)
    plt.plot(f["timesteps"][max_ind],mean[max_ind],'r*')
    plt.xlabel("Total number of steps taken")
    plt.ylabel("Mean return over %d evaluation episode" % len(f["results"][0]))
    plt.title("Training results")
    plt.show()