import matplotlib.pyplot as plt
from random import randrange


def plotImage(arr,**kwargs ):   # Placeholder values if 'igrid' and 'par.mdist' are not available:
    plt.figure(num=randrange(400),figsize=(6, 4.5))
    plt.imshow(arr, aspect='auto', origin='lower', cmap='viridis', extent=kwargs.get('grid', None))
    plt.colorbar(label="Pixel Values")
    plt.xlabel(kwargs.get('xlabel', "x pixel"))
    pts = kwargs.get('points', None)
    if pts is not None:
        for pt in pts:
            plt.plot(pt[0], pt[1], 'ro', markersize=5)
    plt.title(kwargs.get('title', ''))
    plt.ylabel(kwargs.get('title', "y pixel"))
    # Plotting a central cross: fig = plt.figure()

    if kwargs.get('cross', None) is not None:
        #fig = plt.figure() 
        #ax = fig.add_subplot(1, 1, 1)
        plt.axhline(color='black', lw=0.5)
        plt.axvline(color='black', lw=0.5)
        #ax.spines['left'].set_position('center')
        #ax.spines['bottom'].set_position('center')
    
    save = kwargs.get('save', None)
    if kwargs.get('save', None) is not None:
        plt.savefig(save, format="pdf", bbox_inches="tight")

    plt.show()
