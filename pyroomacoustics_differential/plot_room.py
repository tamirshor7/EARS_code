import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pdb

def plot2DRoom(room, mic_arrays, sources, img_order=0, freq=None, figsize=None, no_axis=False, marker_size=10, 
               room_plot_out=None, **kwargs):
    ''' Plots the room with its walls, microphones, sources and images '''
    try:
        import matplotlib
        from matplotlib.patches import Circle, Wedge, Polygon
        from matplotlib.collections import PatchCollection
        import matplotlib.pyplot as plt
        import numpy as onp
    except ImportError:
        import warnings
        warnings.warn('Matplotlib is required for plotting')
        return

    plt.clf()
    
    if (room['dim'] != 2):
        print('This function (currenty) does not support 3D rooms.')

    fig = plt.figure(figsize=figsize)

    if no_axis is True:
        ax = fig.add_axes([0, 0, 1, 1], aspect='equal', **kwargs)
        ax.axis('off')
        rect = fig.patch
        rect.set_facecolor('gray')
        rect.set_alpha(0.15)
    else:
        ax = fig.add_subplot(111, aspect='equal', **kwargs)

    # draw room
    polygons = [Polygon(room['corners'].T, True)]
    p = PatchCollection(polygons, cmap=matplotlib.cm.jet,
            facecolor=onp.array([1, 1, 1]), edgecolor=onp.array([0, 0, 0]))
    ax.add_collection(p)
    

    # draw the microphones
    # if (len(mic_arrays) is not 0):
    if (len(mic_arrays) != 0):
        for i, mic in enumerate(mic_arrays['R'].T):
            if i==0:
                # just add label for legend
                # ax.scatter(mic[0], mic[1],
                #         marker=f'${i}$', linewidth=0.5, s=marker_size, c='k', label='microphone')
                ax.scatter(mic[0], mic[1],
                         marker='o', linewidth=0.5, s=marker_size, c='k', label='microphone')
            else:
                # ax.scatter(mic[0], mic[1],
                #         marker=f'${i}$', linewidth=0.5, s=marker_size, c='k') # marker='$'+str(i)+'$'
                ax.scatter(mic[0], mic[1],
                         marker='o', linewidth=0.5, s=marker_size, c='k', label='microphone')

    # define some markers for different sources and colormap for damping
    #markers = ['x', 'x', 'x', 'x']
    # cmap = plt.get_cmap('YlGnBu')
    # draw the scatter of images

    # TODO!: I commented out since source did not have 'orders' attribute
    # for i, source in enumerate(sources):
    #     # draw source with its images
    #     for ord in range(int(max(source['orders']))):
    #         ord_images_poses = onp.where(source['orders']==ord)[0]
    #         if ord==0:
    #             label = 'direct'
    #         else:
    #             label = f'order {ord}'
    #         ax.scatter(source['images'].T[ord_images_poses,0], source['images'].T[ord_images_poses, 1],
    #             marker= 'x', s=marker_size, linewidth=0.7, label=label)



    ax.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    if room_plot_out is None: # show on screen
        plt.show()
    else: # save to file in path room_plot_out
        plt.savefig(room_plot_out)
    
    plt.clf() # clear fig after plotting to screen or file
    plt.close(fig)
    # return fig, ax
