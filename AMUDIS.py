
import matplotlib.pyplot as plt


def alignment_plot(profile, config):
    """ Plot the alignment lines pixels along the channel axis defined in profile.

    """

    align = config['pixels']

    # extract pixels of alignment
    pixels = align + config['channel_pixel_adj']

    plt.figure(figsize=(12, 9), dpi=300)

    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4)
    ax1.plot(profile, '-*')
    ax1.axis([0, 1060, 0, profile.max() + 20])
    for xc in pixels:
        plt.axvline(x=xc, color='r')
    plt.xlabel('pixels')
    plt.ylabel('counts')
    plt.title('Channel alignment')

    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
    # First section
    ax2.plot(profile, '-*')
    ax2.axis([0, 200, 0, profile.max() + 20])
    for xc in pixels:
        plt.axvline(x=xc, color='r')
    plt.xlabel('pixels')
    plt.ylabel('counts')
    plt.title('Initial section')

    ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
    # final section
    ax3.plot(profile, '-*')
    ax3.axis([400, 600, 0, profile.max() + 20])
    for xc in pixels:
        plt.axvline(x=xc, color='r')
    plt.xlabel('pixels')
    plt.ylabel('counts')
    plt.title('Final section')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
