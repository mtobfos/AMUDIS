# Library for the FOV data analysis of AMUDIS entrance optic measured with
# the Robot Setup
#
# Author: Mario Tobar F.
# E-mail: mario.tobar.foster@gmail.com
# History: [20190418] First version
# Status: UNDER DEVELOPMENT

import datetime
import glob
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import scipy.interpolate
import scipy.signal

###############################################################################
#            FOV using robot
###############################################################################


def data_arrange(section, config):
    """Filter repeated positions and save measurement data into a netCDF file
    parameters:
        section: number of the scanned section(1, 2, 3)
    """
    # %% Read data from raw files and save into netcdf4 file
    # ---------------------------

    offset_table = 270 # deviation between table and north(Only when there is
    # not aligned with the north) IF aligned use 0.
    delta_az = {'1': 22.5,
               '2': 22.5,
               '3': 15}

    step_az = {'1': 45,
                '2': 45,
                '3': 30}

    centr_zn = {'1': 18,
                '2': 48,
                '3': 75}

    delta_zn = {'1': 18,
                '2': 12,
                '3': 15}
    # ---------------------------

    rng = int(360 / step_az[str(section)])  # total of measurements in a section
#
    for meas in np.arange(1, rng + 1):

        config['name'] = '{:d}m{:d}'.format(section, meas)

        files = sorted(glob.glob(config['cwd'] + '/data/Raw/{}/*.nc'.format(config['name'])))
        print(meas, section)
        print(config['name'])

        # Read positions from txt file
        path = config['cwd'] + '/positions/'
        file = glob.glob(path + '{}/positions.txt'.format(config['name']))
        raw = np.genfromtxt(file[0], delimiter='::', usecols=5) # raw positions

        position = np.zeros([int(len(raw) / 2), 2])

        for i in np.arange(int(len(raw) / 2)):
            angle = ((meas - 1) * step_az[str(section)]) + raw[2*i] + offset_table # azimuth (section 1 is 180 in the rotating table)
            if angle > 360:
                position[i] = angle - 360, raw[2*i + 1]
            else:
                position[i] = angle, raw[2*i + 1]  # azimuth, zenith

        config['positions'] = position
        print(position)

        indexs = list()

        # filter positions in the scanned range
        for i in np.arange(len(config['positions'])):
            azim = config['positions'][i, 0]
            zen = config['positions'][i, 1]
            section = str(section)

            centr_az = ((meas - 1) * step_az[section] + (360 - offset_table))

            if centr_az >= 360:
                centr_az -= 360
            else:
                pass

            if azim >= 360:
                azim -= 360
            else:
                pass

            # compare position fibre and scanned area
            if (azim > centr_az - delta_az[section]) & (azim < centr_az + delta_az[section]) & \
            (zen > centr_zn[section] - delta_zn[section]) & (zen < centr_zn[section] + delta_zn[section]):
                indexs.append(i)

        print(len(indexs))

        section = int(section)
        #----------------------

        # load data and save in compressed file
        data_raw = np.zeros((len(indexs), len(config['pixels']), 1024))
        # correct nan values in pixels files
        pixels = config['pixels'].astype(np.int)
        pixels[pixels < 0] = 0

        for k, m in enumerate(indexs):
            with nc.Dataset(files[m], 'r') as dt:
                image = np.reshape(dt['Data'][:], (int(dt.getncattr('xDim')), int(dt.getncattr('yDim'))))
                data_raw[k] = image[pixels, :]
                data_raw[:, config['dead_fibres'], :] = np.nan

        # define channel coordinates
        channels = np.arange(1, data_raw.shape[1] + 1, dtype=np.int16)
        # define wavelength coordinates
        wavelengths = np.arange(1, 1025, dtype=np.int16)
        # define position coordinates
        positions = np.arange(len(indexs))

        with nc.Dataset(config['cwd'] + '/data/arranged/{:d}m{:02d}.nc'.format(section, meas), mode='w', format='NETCDF4') as dt:
            dt.createDimension('channel', len(channels))
            dt.createDimension('wavelength', len(wavelengths))
            dt.createDimension('position', len(positions))
            dt.createDimension('angles', 2)

            dt.createVariable('position', 'f4', ('position', 'angles'))
            dt.createVariable('data', 'f4', ('position', 'channel', 'wavelength'), fill_value=np.nan, chunksizes=(100, len(channels), len(wavelengths)))

            # definitions
            dt.Conventions = 'CF-1.6'
            dt.title = 'AMUDIS FOV section {}'.format(config['name'])
            dt.institution = 'Solar Radiation and Remote Sensing, IMuK, University of Hannover, Germany'
            dt.history = '[{}] File created'.format(datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S"))

            dt['data'][:] = data_raw
            dt['position'][:] = position[indexs]

    print('completed')


def FOV_Dataset(config):
    """Append all measurements into a single file without repeated positions
    parameters:
        config['SR']: Section muss be specified by (UV, VIS or NIR)
    """

    # load alignment file
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
    config['skymap'] = np.concatenate((align[:, 0], align[:, 1]), axis=0)

    print('Reading arranged datasets')
    files = sorted(glob.glob(config['cwd'] + '/data/arranged/*.nc'))

    with nc.Dataset(files[1], mode='r') as dt:
        channels = dt.dimensions['channel'].size
        wavelengths = dt.dimensions['wavelength'].size

    # create file to save dataset
    with nc.Dataset(config['cwd'] + '/data/VIS.nc', mode='w', format='NETCDF4') as dt:
        dt.createDimension('channel', channels)
        dt.createDimension('wavelength', wavelengths)
        dt.createDimension('position', None)
        dt.createDimension('angles', 2)

        dt.createVariable('position', 'f4', ('position', 'angles'))
        dt.createVariable('data', 'f4', ('position', 'channel', 'wavelength'), chunksizes=(100, 145, 1024))

        # definitions
        dt.Conventions = 'CF-1.6'
        dt.title = 'AMUDIS FOV VISIBLE DATASET'
        dt.institution = 'Solar Radiation and Remote Sensing, IMuK, University of Hannover, Germany'
        dt.history = '[20190412] File created'


    with nc.Dataset(config['cwd'] + '/data/VIS.nc', mode='r+', format='NETCDF4') as dt:
        for i, file in enumerate(files):
            size = dt.dimensions['position'].size
            print(size)

            with nc.Dataset(file, 'r') as ds:
                size_new = ds.dimensions['position'].size
                ini, fin = size, size + size_new

                dt.variables['data'][ini:fin, :] = ds.variables['data'][:]
                dt.variables['position'][ini:fin, :] = ds.variables['position'][:]

            dt.sync() # free up memory RAM
            print('appending file number #', i + 1 , 'from ', len(files))

    print('Completed')


# def find_channel(meas=1, section='1', config=dict()):
#     """Return the indexes of the fibres which are in the scanned area during
#     the FOV measurement using the robot setup"""
#     delta_az = {'1': 35,
#                '2': 35,  #22.5
#                '3': 25}  #15
#
#     step_az = {'1': 45,
#                 '2': 45,
#                 '3': 30}
#
#     centr_zn = {'1': 18,
#                 '2': 48,
#                 '3': 75}
#
#     delta_zn = {'1': 22,
#                 '2': 22,
#                 '3': 25}
#
#     indexs = list()
#     # scan along channel values
#     for i in np.arange(len(config['alignment'])):
#         azim = config['alignment'][i, 0]
#         zen = config['alignment'][i, 1]
#
#         centr_az = ((meas - 1) * step_az[section])
#
#         if centr_az >= 360:
#             centr_az -= 360
#         else:
#             pass
#         # compare position fibre and scanned area
#         if (azim > centr_az - delta_az[section]) & (azim < centr_az + delta_az[section]) & \
#         (zen > centr_zn[section] - delta_zn[section]) & (zen < centr_zn[section] + delta_zn[section]):
#             indexs.append(i)
#
#     return indexs


# def fov_arrange(meas_range=np.arange(1, 9), section='1', config=dict()):
#     """Divide the fov from the section to individual ranges for the single
#     analysis"""
#
#     for j in meas_range:
#         meas = j
#         section = section
#         indx = find_channel(meas, section, config)
#
#         for i in indx:
#             if i in config['dead_fibres']:
#                 print('dead fibre', config['dead_fibres'])
#                 pass
#             else:
#                 # look for file in folder
#                 filename = config['cwd'] + '/single_fibre_data/{:03.0f}.nc'.format(i)
#                 files = sorted(glob.glob(config['cwd'] + '/single_fibre_data/*.nc'))
#
#                 attrs={'channel':'{:03.0f}'.format(i),
#                    'columns': ['azimuth', 'zenith'],
#                    'angular step':'1º',
#                    'ExposureTime':'80ms',
#                    'Average':'3'}
#
#                 # filter data from raw file
#                 with xr.open_dataset(config['cwd'] + '/arranged_meas_data/' + '{}m{:02d}.nc'.format(section, meas)) as raw:
#                     data = raw['radiance'].data[:, i, :]
#                     position = raw['positions'].data
#                 raw.close()
#
#                 if filename in files:
#                     # load data from file
#                     with xr.open_dataset(filename) as old:
#                         rad = old['radiance'].data
#                         pos = old['positions'].data
#                     old.close()
#
#                     rad_app = np.concatenate((rad, data), axis=0)
#                     pos_app = np.concatenate((pos, position), axis=0)
#
#                     ds = xr.Dataset({'radiance':(('position', 'wavelength'), rad_app), 'positions':(('position', 'columns'), pos_app)}, attrs=attrs)
#                     ds.to_netcdf(filename, mode='w', format='NETCDF4', engine='netcdf4')
#                 else:
#                     fov_data = xr.DataArray(data, dims=('position', 'wavelength'))
#                     fov_pos = xr.DataArray(position, attrs={'colums': ['azimuth', 'zenith']})
#                     ds = xr.Dataset({'radiance': fov_data, 'positions': fov_pos}, attrs=attrs)
#                     ds.to_netcdf(filename, mode='w', format='NETCDF4', engine='netcdf4')
#
#                 ds.close()


# def filter_single(azim, zen, radiance):
#     """Filter the data in the zenith direction to avoid repeated values in the
#     matrix"""
#     indexes = np.empty(0)
#     comp = 0
#     for i in np.arange(1, len(zen)):
#         if zen[i - 1] <= comp:
#             pass
#         else:
#             comp = zen[i - 1]
#         if zen[i] >= comp:
#             indexes = np.append(indexes, i)
#         else:
#             pass
#     indexes = indexes.astype(int)
#     # assign new values to arrays
#     azim = [azim[i] for i in indexes]
#     zen = [zen[i] for i in indexes]
#     radiance = [radiance[i] for i in indexes]
#
#     return azim, zen, radiance


def FOV_plot(azim, zen, radiance, config, type_plot='measured', add_points=False):
    """Plot the FOV measured in a contour plot. If type_plot is equal to
    measured, it uses the tricontourf functions.
    If type_plot is 'interpolated', uses the contourf function which need 2d
    arrays"""

    delta = config['delta']
    name = 'corrected_FOV'

    config['meas_zen'] = config['alignment'][config['channel'], 1]
    config['meas_azim'] = config['alignment'][config['channel'], 0]

    ylim_min = config['meas_zen'] - delta
    ylim_max = config['meas_zen'] + delta
    xlim_min = config['meas_azim'] - delta
    xlim_max = config['meas_azim'] + delta

    # parameters
    levels = 30
    cmap = 'rainbow'  # 'gray'#'nipy_spectral'

    fig = plt.figure()

    if type_plot == 'measured':
        plt.tricontourf(azim, zen, radiance, levels, cmap=cmap)

    if type_plot == 'interpolated':
        plt.contourf(azim, zen, radiance, levels, cmap=cmap)

    plt.ylim(ylim_min, ylim_max)
    plt.title('FOV AZ{} ZN{}'.format(config['meas_azim'], config['meas_zen']))
    plt.xlabel('azimuth angle[deg]', fontsize=12)
    plt.ylabel('zenith angle[deg]', fontsize=12)
    plt.gca().invert_yaxis()  # invert y axis

    # plot center of fiber
    plt.plot([-delta + config['meas_azim'], delta - 1 + config['meas_azim']],
             [config['meas_zen'], config['meas_zen']], '--r', label='Designed Centre')

    plt.plot([config['meas_azim'], config['meas_azim']], [ylim_min, ylim_max],
             '--r')
    plt.axis(ratio=1)
    plt.xticks(size=11)
    plt.yticks(size=11)
    plt.axis([-delta + config['meas_azim'],
              config['meas_azim'] + delta, ylim_max, ylim_min])

    # plot FOV peaks
    plt.plot([-delta + config['peaks'][0], delta - 1 + config['peaks'][0]],
             [config['peaks'][1], config['peaks'][1]], '--g', label='Real centre')

    plt.plot([config['peaks'][0], config['peaks'][0]],
             [-delta + config['peaks'][1], delta - 1 + config['peaks'][1]], '--g')

    plt.axes().set_aspect('equal', 'datalim')
    plt.axis([xlim_min, xlim_max, ylim_max, ylim_min])
    plt.colorbar()
    plt.legend()

    if add_points is True:
        # add measurement points
        plt.scatter(azim, zen, cmap=cmap, s=0.4, c='w')

    plt.show()

    return fig


def load_data(ds, config):
    """Load data from xarray Dataset with the parameters of config and
    return the azim, zen, radiance for the defined wavelength"""

    config['meas_zen'] = config['alignment'][config['channel'], 1]
    config['meas_azim'] = config['alignment'][config['channel'], 0]

    azim = ds.positions.data[:, 0]
    zen = ds.positions.data[:, 1]
    radiance = ds.radiance.data[:, config['wave_px']] / ds.radiance.data[:, config['wave_px']].max()
    # correct values of azimuth for the FOV plot and analysis for the fibres in
    # the zero azimuth positions

    if config['meas_azim'] == 0:
        azim = np.where(azim > 90, azim - 360, azim)
        print('azimuth corrected for angles near to zero')
    else:
        pass

    # angle viewed by the fiber correction
    azim = angle_correction(azim, 90, 12.5, config['meas_azim'])
    zen = angle_correction(zen, 90, 12.5, config['meas_zen'])

    # spatial correction on azimuth
    azim = ((azim - config['meas_azim']) * np.sin(np.radians(zen))) + config['meas_azim']

    return azim, zen, radiance


def find_centre_fov(azim_, zen_, radiance_):
    """ Find the FOV using the center of mass, 2d array"""

    radiance_pks = radiance_ / radiance_.max()
    for i in np.arange(len(radiance_)):
        for j in np.arange(len(radiance_)):
            if radiance_pks[i, j] < 0.5:
                radiance_pks[i, j] = 0
            else:
                radiance_pks[i, j] = radiance_pks[i, j]

    peak_azim = np.nansum(azim_ * radiance_pks) / np.nansum(radiance_pks)
    peak_zen = np.nansum(zen_ * radiance_pks) / np.nansum(radiance_pks)

    return peak_azim, peak_zen


def select_range(azim, zen, radiance, config):
    """ Return the values around the defined FOV. Input are the measured values
    """
    azim_meas = config['alignment'][config['channel'], 0]
    zen_meas = config['alignment'][config['channel'], 1]

    dat = np.column_stack((azim, zen, radiance))

    # select range values
    dat_1 = dat[(azim > azim_meas - config['delta']) & (
            azim < azim_meas + config['delta']) &
                (zen >  zen_meas - config['delta']) & (
                        zen <  zen_meas + config['delta'])]

    azim, zen, radiance = dat_1[:, 0], dat_1[:, 1], dat_1[:, 2]

    return azim, zen, radiance


def interpolate_data(azim, zen, radiance):
    """ Interpolate the measured data into a regular grid using scipy.interpol-
    ate.griddata()"""
    num_points = 100
    # Interpolation of data
    x = np.linspace(azim.min(), azim.max(), num=num_points)
    y = np.linspace(zen.min(), zen.max(), num=num_points)
    xx, yy = np.meshgrid(x, y)
    plane = scipy.interpolate.griddata((azim, zen), radiance, (xx, yy),
                                       method='cubic', fill_value=0)

    return xx, yy, plane


def select_curve(azim, zen, radiance, config):
    """ Select values for FOV curves from interpolated data"""
    num_points = 100
    x = np.linspace(azim.min(), azim.max(), num=num_points)
    y = np.linspace(zen.min(), zen.max(), num=num_points)

    xx, yy = np.meshgrid(x, y)
    plane = scipy.interpolate.griddata((azim, zen), radiance, (xx, yy),
                                       method='cubic', fill_value=0)

    interpolated_radiance = scipy.interpolate.interp2d(x, y, plane, kind='cubic')

    azimuth_pk = config['peaks'][0]
    zenith_pk = config['peaks'][1]

    # azimuth FOV
    if config['var'] == 'azimuth':
        curv_fov = interpolated_radiance(x, zenith_pk)
        data_peaks = np.column_stack((x, curv_fov))

    # zenith FOV
    if config['var'] == 'zenith':
        curv_fov = interpolated_radiance(azimuth_pk, y)
        data_peaks = np.column_stack((y, curv_fov))

    return data_peaks


def FOV(function, first, last, tol=0.01, value=0.5):
    """Calcule the FOV of a function with two minimum values and one maximum"""
    val = []
    num = int((last - first) / tol)

    for i in np.linspace(first, last, num=num):
        if (value - tol) <= function(i) <= (value + tol):
            val = np.append(val, i)
        else:
            pass

    # Separate range to determine the max and min value in the degree axis
    fov_min = []
    fov_max = []

    for i in range(len(val)):
        if val[i] < ((first + last) / 2):
            fov_min = np.append(fov_min, val[i])
        else:
            fov_max = np.append(fov_max, val[i])

    fov = np.mean(fov_max) - np.mean(fov_min)
    pos = [np.mean(fov_min), np.mean(fov_max)]

    return fov, pos


# FOV azimuth ::::::::::::::::::::::::::::::::::::::::::::::::::::
def FOV_curve(ds, config):

    config['meas_zen'] = config['alignment'][config['channel'], 1]
    config['meas_azim'] = config['alignment'][config['channel'], 0]

    try:
        azim, zen, radiance = load_data(ds, config)
        azim, zen, radiance = select_range(azim, zen, radiance, config)
        azim_, zen_, radiance_ = interpolate_data(azim, zen, radiance)
        config['peaks'] = find_centre_fov(azim_, zen_, radiance_)

        data_plot = select_curve(azim, zen, radiance, config)
        num_points = len(data_plot)

        # interpolate data
        rad_interp = scipy.interpolate.interp1d(data_plot[:, 0], data_plot[:, 1], kind='cubic')
        axis_new = sorted(np.linspace(data_plot[:, 0].min(), data_plot[:, 0].max(), num=num_points))

        if int((len(axis_new) / 4)) % 2 == 0:
            windows_len = int(len(axis_new) / 4) - 1
        else:
            windows_len = int(len(axis_new) / 4)

        # smooth the radiance curve
        print(windows_len)
        sm = scipy.signal.savgol_filter(rad_interp(tuple(axis_new)),
                                        window_length=windows_len, polyorder=2)
        sm_inter = scipy.interpolate.interp1d(axis_new, sm, kind='cubic')

        # find the values equal to 0.5
        fov_val, ind_f = FOV(sm_inter, axis_new[0], axis_new[-1], tol=0.01)

        # FOV maximum found
        max_fov = ((ind_f[0] + ind_f[1]) / 2)
        FOV_az = fov_val

        # Figure azimuth
        if config['var'] == 'azimuth':
            fibre_pos = config['meas_azim']
        if config['var'] == 'zenith':
            fibre_pos = config['meas_zen']

        plt.plot(data_plot[:, 0], data_plot[:, 1], '.-r', label=config['var'])
        plt.plot(axis_new, sm_inter(axis_new), label="smoothing")
        plt.ylabel('Radiance[1]')
        plt.xlabel('Angle[deg]')
        plt.xlim(fibre_pos - config['delta'], fibre_pos + config['delta'])
        plt.title("FOV along {} Axis".format(config['var']))
        plt.plot([ind_f[0], ind_f[1]], [0.5, 0.5], '-g')
        plt.legend()
        plt.show()

    except ValueError:
        print("Could not calculate {} FOV".format(config['var']))
        FOV_az = np.nan


def spectral_FOV(data, config, correc=False):
    """Spectral FOV from xarray dataset"""

    config['meas_zen'] = config['alignment'][config['channel'], 1]
    config['meas_azim'] = config['alignment'][config['channel'], 0]

    w_i = 100
    w_f = 1024
    st = config['step_waves']

    azim, zen, radiance = load_data(data, config)

    config['var'] = 'azimuth'
    fovs = np.zeros((int((w_f - w_i) / st) + 1, 4))
    cnt = 0
    azim_0 = azim
    zen_0 = zen

    for wave in np.arange(w_i, w_f, st):

        radiance_0 = data.radiance[:, wave] / data.radiance[:, wave].max()
        azim, zen, radiance = select_range(azim_0, zen_0, radiance_0, config)
        azim_, zen_, radiance_ = interpolate_data(azim, zen, radiance)
        config['peaks'] = find_centre_fov(azim_, zen_, radiance_)

        data_plot = select_curve(azim, zen, radiance, config)
        num_points = len(data_plot)

        # interpolate data
        rad_interp = scipy.interpolate.interp1d(data_plot[:, 0], data_plot[:, 1], kind='cubic')
        axis_new = sorted(np.linspace(data_plot[:, 0].min(), data_plot[:, 0].max(), num=num_points))

        if int((len(axis_new) / 2)) % 2 == 0:
            windows_len = int(len(axis_new) / 2) - 1
        else:
            windows_len = int(len(axis_new) / 2)

        # smooth the radiance curve
        sm = scipy.signal.savgol_filter(rad_interp(axis_new), window_length=windows_len, polyorder=5)
        sm_inter = scipy.interpolate.interp1d(axis_new, sm, kind='cubic')

        # find the values equal to 0.5
        fov_val, ind_f = FOV(sm_inter, axis_new[0], axis_new[-1], tol=0.01)

        # FOV maximum found
        max_fov = ((ind_f[0] + ind_f[1]) / 2)
        fovs[cnt, :2] = wave, fov_val

        fovs[cnt, 2:] = find_centre_fov(azim_, zen_, radiance_)

        cnt += 1

    return fovs


def angle_correction(ang, d, r, centre):
    """ Correct the angles viewed by the optical fibre when the movement is
    centred on the input optic. If the movement is centred on the optical fibre
    it is not needed """

    return ang + np.degrees(np.arctan((r * np.sin(np.radians(ang - centre)))
                                      / (d + r * (1 - np.cos(np.radians(ang - centre))))))


def plot_fov_spectral(dt):
    """Plot the FOV saved in the data file using the function spectral_FOV() and
    save_fov()"""

    dt = np.asfarray(dt)

    waves = 400 + (500 / 1024) * dt[:, 0]

    windows_len = int(len(dt[:, 1]) / 5)

    if windows_len % 2 == 0:
        windows_len = windows_len - 1
    else:
        windows_len = windows_len

    # smooth the FOV data
    sm = scipy.signal.savgol_filter(dt[:, 1], window_length=windows_len, polyorder=3, mode='nearest')
    sm_inter = scipy.interpolate.interp1d(waves, sm, kind='cubic')

    waves_new = np.arange(400, 900, 1000)
    plt.plot(waves, dt[:, 1], '.b', markersize=3)
    plt.title("Spectral FOV")
    plt.plot(waves, sm_inter(waves), '-r')
    plt.xlabel('wavelength[nm]')
    plt.ylabel('FOV[deg]')
    plt.show()

    plt.plot(waves, dt[:, 2], 'r*-', label='azimuth')
    plt.xlabel('wavelength[nm]')
    plt.ylabel('azimuth[º]')
    plt.legend()
    plt.show()

    plt.plot(waves, dt[:, 3], 'b*-', label='zenith')
    plt.xlabel('wavelength[nm]')
    plt.ylabel('zenith[º]')
    plt.legend()
    plt.show()


def save_fov(data, config):
    """Save the spectral FOV calculated with spectral_fov()"""

    with nc.Dataset('Results/{}.nc'.format(config['name']), 'w', format="NETCDF4") as fov_nc:
        fov_nc.createDimension('wavelength_pixel', None)
        fov_nc.createDimension('info', 4)
        fov_nc.createVariable('data', 'f4', ('wavelength_pixel', 'info'), fill_value=np.nan)

        fov_nc['data'][:] = data

        fov_nc.Columns = ('wavelength_pixel', 'fov_az', 'position_az', 'position_zn')
        fov_nc.PositionFibre = config['meas_azim'], config['meas_zen']

        print(fov_nc['data'])


def FOV_all(data, config, correc=False):
    """Plot all data together into a figure
    UNDER DEVELOPMENT!!!!!

    Parameters---------------------------------------
        data: xarray.Dataset with the arranged data

        """

    # Correct the data
    azim = data.positions[:, 0]
    zen = data.positions[:, 1]

    # angular correction
    if correc is True:
        azim = angle_correction(azim, d=90, r=12.5, centre=config['peaks'][0])
        zen = angle_correction(zen, d=90, r=12.5, centre=config['peaks'][1])

    azim, zen, radiance = load_data(data, config)
    azim, zen, radiance = select_range(azim, zen, radiance, config)

    # Define figure
    plt.figure(figsize=(12, 9))
    grid = plt.GridSpec(3, 4, wspace=0.6, hspace=0.5)

    # Plot FOV shape
    plt.subplot(grid[0:2, 0:2])
    FOV_plot(azim, zen, radiance, config, type_plot='measured', add_points=True)

    #os.makedirs(os.path.dirname(cwd + '/results/'), exist_ok=True)

    # FOV azimuth ::::::::::::::::::::::::::::::::::::::::::::::::::::

    print('Peak of the FOV in:', config['peaks'])

    try:
        config['var'] = 'azimuth'
        curve = select_curve(azim_, zen_, radiance_, config)
        plt.subplot(grid[2, 0:2])
        FOV_curve(azim, zen, radiance, config)
    except:
        pass

    try:
        config['var'] = 'zenith'
        curve = select(azim_, zen_, radiance_, config)
        plt.subplot(grid[2, 2:4])
        FOV_curve(azim, zen, radiance, config)
    except:
        pass
    # :::::::::::::: Spectral FOV::::::::::::::::::::::

    config['var'] = 'azimuth'
    config['wave'] = 500

    dt = spectral_FOV(data, config, correc=correc)

    # delete nan values in array
    dt = dt[~np.isnan(dt[:, 1])]

    np.savetxt('{name}spectral_fov_{index:d}.txt'.format(name=config['name'],
                                                          index=config['channel']),
               X=dt, delimiter=',', header="Wavelength Index, FOV")

    waves = 400 + (500 / 1024) * dt[:, 0]

    config['wave'] = 500

    config['peaks'] = find_centre_fov(azim_, zen_, radiance_)

    plt.subplot(grid[1, 2:])
    plt.plot(waves, dt[:, 1], '.b', markersize=3)
    plt.title("Spectral FOV")

    if len(dt[:, 1]) % 2 == 0:
        windows_len = len(dt[:, 1]) - 1
    else:
        windows_len = len(dt[:, 1])

    # smooth the FOV data
    sm = scipy.signal.savgol_filter(dt[:, 1], window_length=windows_len, polyorder=5, mode='nearest')
    sm_inter = scipy.interpolate.interp1d(waves, sm, kind='cubic')

    plt.plot(waves, sm_inter(waves), '-r')
    plt.xlabel('wavelength[nm]')
    plt.ylabel('FOV[deg]')

    # Adding text information
    plt.subplot(grid[:1, 2:])

    plt.text(0.5, 1, "Information", transform=plt.gca().transAxes, fontsize=15,
            verticalalignment='top', horizontalalignment='center')

    text = "Peak = {azim:.2f}º, {zen:.2f}º\n"\
    "FOV azimuth = {FOV_az:.2f}º\n"\
    "FOV zenith = {FOV_zn:.2f}º".format(azim=config['peaks'][0], zen=config['peaks'][1],
                                   FOV_az=dt[:, 1][10], FOV_zn=dt[:, 1][10])
    plt.text(0.1, 0.80, text, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='left')

    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())

    plt.savefig('fig_{name}_new.png'.format(name=config['name']), dpi=300)
