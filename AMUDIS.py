###############################################################################
# -----------------------------------
#          AMUDIS FUNCTIONS
# ------------------------------------
# Functions used for the calibration, data analysis and data procccessing
# can be found in this library
#
# Author: Mario Tobar F.
# E-mail: mario.tobar.foster@gmail.com
# History: [20190418] First version
# Status: UNDER DEVELOPMENT
#
###############################################################################

import datetime
import glob
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import os

#######################################################################
#  INSTRUMENT CHARACTERIZATION
#######################################################################


def alignment_export(filename, config):
    """Export alignment file from .dat to nc, using the alignment files structu
    re for IDL AMUDIS script

    parameters:
        config: configuration dictionary contains variables:
            'cwd': current working directory
            'SR': Spectral Range (UV, VIS or NIR)
    """

    # load alignment file
    align = np.genfromtxt(config['cwd'] + '/config/{}.dat'.format(filename),
                          delimiter='',
                          skip_header=1)

    # save alignment data to file
    with nc.Dataset('config/{}.nc'.format(filename), 'w', format="NETCDF4") as ds:
        ds.createDimension('channel', 145)
        ds.createDimension('info', 3)
        ds.createVariable('data', 'f4', ('channel', 'info'), fill_value=np.nan)

        # definitions
        ds.Conventions = 'CF-1.6'
        ds.title = 'AMUDIS Alignment {}'.format(config['SR'])
        ds.institution = 'Solar Radiation and Remote Sensing, IMuK, University of Hannover, Germany'
        ds.history = '[] File created'.format(datetime.datetime.strftime(datetime.date.today(), format='%Y%m%d'))

        ds['data'][:] = align[:, 1:4]

        ds['data'].Columns = ('azimuth', 'zenith', 'pixel')
        ds['data'].units = ('degree', 'degree', 'px')

        print(ds)
    print("File {}.dat exported".format(filename))


def alignment_plot(profile, config):
    """ Plot the alignment lines pixels along the channel axis defined in profile.

    """
    # extract pixels of alignment
    pixels = config['pixels'] + config['channel_pixel_adj']

    plt.figure(figsize=(12, 9), dpi=300)

    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4)
    ax1.plot(profile, '-*')
    ax1.axis([0, 540, 0, profile.max() + 20])
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
    ax3.axis([400, 540, 0, profile.max() + 20])
    for xc in pixels:
        plt.axvline(x=xc, color='r')
    plt.xlabel('pixels')
    plt.ylabel('counts')
    plt.title('Final section')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


def amudis_pattern(cam="VIS"):
    """Create a plot of the AMUDIS Pattern defined in the cam parameter(UV,
    VIS or NIR) """
    # create configuration parameters
    config = {'cwd': os.getcwd(),
              'SR': cam}

    file = glob.glob(config['cwd'] + '/config/*{}*.nc'.format(config['SR']))
    print(file)

    if len(file) > 1:
        print("Only one alignment file per camera is necesary")
    else:
        pass

    # load alignment file
    with nc.Dataset(file[0], 'r') as da:
        align = da['data'][:]
        dead_fibre = np.argwhere(align[:, 2].mask is True)

    config['dead_fibres'] = dead_fibre
    config['channel_pixel_adj'] = 0
    config['pixels'] = align[:, 2].astype(int)
    config['skymap'] = np.array([align[:, 0], align[:, 1]])

    # Plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(12, 12))
    azim = np.radians(config['skymap'][0, :])
    zen = config['skymap'][1, :]

    # Designed FOV position
    plt.scatter(azim, zen, s=1500, c='k', alpha=0.5, label='AMUDIS {} Pattern'.format(config['SR']))
    ax.grid(False)
    ax.set_yticklabels([]) # eliminate radius values
    ax.set_theta_zero_location("N")  # Set the direction of polar plot
    ax.set_theta_direction(1)  # Set the increase direction on azimuth angles
        # (-1 to clockwise, 1 counterclockwise)
    plt.ylim(0, 90)
    plt.xticks(size=14)

    lines, labels = plt.thetagrids(range(0,360,90), ('0º N', '90º E', '180º S','270º W'))
    ax.tick_params(axis='x', pad=20)
    plt.legend(fontsize=14, bbox_to_anchor=(1.1, 1.1))

    for i in np.arange(1, 146):
        ax.annotate(i, (azim[i - 1], zen[i - 1]), ha='center', va='center', fontsize=13, fontweight='bold', color='white')

    plt.savefig('AMUDIS_{}_pattern.png'.format(config['SR']), dpi=300)
    plt.close()
