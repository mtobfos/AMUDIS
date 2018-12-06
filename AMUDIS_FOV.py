
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import scipy.interpolate


def FOV_plot(azim, zen, radiance, config, type_plot='measured', add_points=False):
    """Plot the FOV measured in a contour plot. If type_plot is equal to measured, it uses the tricontourf functions.
    If type_plot is 'interpolated', uses the contourf function which need 2d arrays"""

    delta = config['delta']
    name = 'corrected_FOV'

    ylim_min = config['meas_zen'] - delta
    ylim_max = config['meas_zen'] + delta
    xlim_min = config['meas_azim'] - delta
    xlim_max = config['meas_azim'] + delta

    # parameters
    levels = 30
    cmap = 'rainbow'  # 'gray'#'nipy_spectral'

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
    plt.axis(ratio=1)
    plt.xticks(size=11)
    plt.yticks(size=11)
    plt.axis([xlim_min, xlim_max, ylim_max, ylim_min])
    plt.colorbar()

    if add_points is True:
        # add measurement points
        plt.scatter(azim, zen, cmap=cmap, s=0.4, c='w')


def find_centre_fov(azim_, zen_, radiance_):
    """ Find the FOV using the center of mass, 2d array"""

    radiance_pks = radiance_
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
    dat = np.column_stack((azim, zen, radiance))

    # select range values
    dat_1 = dat[(azim > config['meas_azim'] - config['delta']) & (
            azim < config['meas_azim'] + config['delta']) &
                (zen > config['meas_zen'] - config['delta']) & (
                        zen < config['meas_zen'] + config['delta'])]

    azim, zen, radiance = dat_1[:, 0], dat_1[:, 1], dat_1[:, 2]

    return azim, zen, radiance


def interpolate_fov(azim, zen, radiance):
    """ Interpolate the measured data into a regular grid using scipy.interpol-
    ate.griddata()"""
    num_points = 100
    # Interpolation of data  #
    x = np.linspace(azim.min(), azim.max(), num=num_points)
    y = np.linspace(zen.min(), zen.max(), num=num_points)
    xx, yy = np.meshgrid(x, y)
    plane = scipy.interpolate.griddata((azim, zen), radiance, (xx, yy),
                                       method='cubic', fill_value=0)

    return xx, yy, plane

def select(azim_, zen_, radiance_, config):
    """ Select values for FOV curves from interpolated data"""
    # peak_azim, peak_zen

    # find the values
    tol = config['delta'] / 100

    # azimuth FOV
    if config['var'] == 'azimuth':
        axis = zen_
        axis_t = config['peaks'][1]

        index = np.where((axis >= axis_t - tol) &
                         (axis <= axis_t + tol))

        # eliminate nan values
        x = azim_[index]
        y = radiance_[index]
        par = np.column_stack((x, y))
        par = par[~np.isnan(y)]
        data_peaks = np.asfarray((par[:, 0], par[:, 1]))

    # zenith FOV
    if config['var'] == 'zenith':
        axis = azim_
        axis_t = config['peaks'][0]

        index = np.where((axis > axis_t - tol) &
                         (axis <= axis_t + tol))

        # eliminate nan values
        x = zen_[index]
        y = radiance_[index]
        par = np.column_stack((x, y))
        par = par[~np.isnan(y)]
        data_peaks = np.asfarray((par[:, 0], par[:, 1]))

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
def FOV_curve(azim, zen, radiance, config):
    try:
        azim_, zen_, radiance_ = interpolate_fov(azim, zen, radiance)

        data_plot = select(azim_, zen_, radiance_, config)

        num_points = len(data_plot[0])

        # interpolate data
        rad_interp = scipy.interpolate.interp1d(data_plot[0], data_plot[1], kind='cubic')
        axis_new = sorted(np.linspace(data_plot[0][0], data_plot[0][-1], num=num_points))

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
        FOV_az = fov_val

        # Figure azimuth
        if config['var'] == 'azimuth':
            fibre_pos = config['meas_azim']
        if config['var'] == 'zenith':
            fibre_pos = config['meas_zen']

        plt.plot(data_plot[0], data_plot[1], '.-r', label=config['var'])
        plt.plot(axis_new, sm_inter(axis_new), label="smoothing")
        plt.ylabel('Radiance[1]')
        plt.xlabel('Angle[deg]')
        plt.xlim(fibre_pos - config['delta'], fibre_pos + config['delta'])
        plt.title("FOV along {} Axis".format(config['var']))
        plt.plot([ind_f[0], ind_f[1]], [0.5, 0.5], '-g')
        plt.legend()

    except ValueError:
        print("Could not calculate {} FOV".format(config['var']))
        FOV_az = np.nan


def spectral_FOV(data, config, correc=False):
    """Spectral FOV"""
    w_i = 0
    w_f = 1024
    st = config['step_waves']

    azim = data.positions[:, 0]
    zen = data.positions[:, 1]

    azim = ((azim - config['meas_azim']) * np.sin(np.radians(zen))) + config['meas_azim']

    # angular correction
    if correc is True:
        azim = angle_correction(azim, d=90, r=12.5, centre=config['peaks'][0])
        zen = angle_correction(zen, d=90, r=12.5, centre=config['peaks'][1])

    config['var'] = 'azimuth'
    fovs = np.zeros((int((w_f - w_i) / st) + 1, 4))
    cnt = 0
    azim_0 = azim
    zen_0 = zen

    for wave in np.arange(w_i, w_f, st):

        radiance_0 = data.radiance[:, config['channel'], wave] / data.radiance[:, config['channel'], wave].max()
        azim, zen, radiance = select_range(azim_0, zen_0, radiance_0, config)
        azim_, zen_, radiance_ = interpolate_fov(azim, zen, radiance)

        data_plot = select(azim_, zen_, radiance_, config)

        num_points = len(data_plot[0])

        # interpolate data
        rad_interp = scipy.interpolate.interp1d(data_plot[0], data_plot[1], kind='cubic')
        axis_new = sorted(np.linspace(data_plot[0][0], data_plot[0][-1], num=num_points))

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
    """Plot the FOV saved in the data file """

    waves = 400 + (500 / 1024) * dt[:, 0]

    plt.plot(waves, dt[:, 1], '.b', markersize=3)
    plt.title("Spectral FOV")

    windows_len = int(len(dt[:, 1]) / 5)

    if windows_len % 2 == 0:
        windows_len = windows_len - 1
    else:
        windows_len = windows_len


    # smooth the FOV data
    sm = scipy.signal.savgol_filter(dt[:, 1], window_length=windows_len, polyorder=6, mode='nearest')
    sm_inter = scipy.interpolate.interp1d(waves, sm, kind='cubic')

    waves_new = np.arange(400, 900, 1000)
    plt.plot(waves, sm_inter(waves), '-r')
    plt.xlabel('wavelength[nm]')
    plt.ylabel('FOV[deg]')
    plt.show()


def save_fov(data, config):
    """Save the spectral FOV calculated with spectral_fov()"""

    with nc.Dataset('results/{}.nc'.format(config['name']), 'w', format="NETCDF4") as fov_nc:
        fov_nc.createDimension('wavelength_pixel', None)
        fov_nc.createDimension('info', 4)
        fov_nc.createVariable('data', 'f4', ('wavelength_pixel', 'info'), fill_value=np.nan)

        fov_nc['data'][:] = data

        fov_nc.Columns = ('wavelength_pixel', 'fov_az', 'position_az', 'position_zn')
        fov_nc.PositionFibre = config['meas_azim'], config['meas_zen']

        print(fov_nc['data'])

