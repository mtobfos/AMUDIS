
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal


# %% Define functions

def all_FOV(data, config, correc=False):
    """Plot all data together into a figure"""

    # Correct the data

    azim = config['positions'][:, 0]
    zen = config['positions'][:, 1]

    # angular correction
    if correc is True:
        azim = angle_correction(azim, d=90, r=12.5, centre=config['peaks'][0])
        zen = angle_correction(zen, d=90, r=12.5, centre=config['peaks'][1])

    radiance = data[:, config['pixel_index'], config['wave']] / data[:, config['pixel_index'], config['wave']].max()
    config['meas_azim'] = 180

    azim = ((azim - config['meas_azim']) * np.sin(np.radians(zen))) + config['meas_azim']

    azim, zen, radiance = select_range(azim, zen, radiance, config)

    # Define figure
    plt.figure(figsize=(12, 9))
    grid = plt.GridSpec(3, 4, wspace=0.6, hspace=0.5)

    # Plot FOV shape
    plt.subplot(grid[0:2, 0:2])
    FOV_plot(azim, zen, radiance, config, add_points=True)

    #os.makedirs(os.path.dirname(cwd + '/results/'), exist_ok=True)

    # FOV azimuth ::::::::::::::::::::::::::::::::::::::::::::::::::::
    #try:
    config['var'] = 'azimuth'
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
    plt.subplot(grid[2, 0:2])

    plt.plot(data_plot[0], data_plot[1], '.-r', label=config['var'])
    plt.plot(axis_new, sm_inter(axis_new), label="smoothing")
    plt.ylabel('Radiance[1]')
    plt.xlabel('Angle[deg]')
    plt.xlim(config['meas_azim'] - config['delta'], config['meas_azim'] + config['delta'])
    plt.title("FOV along Azimuth Axis")
    plt.plot([ind_f[0], ind_f[1]], [0.5, 0.5], '-g')
    # except:
    #      print("Could not calculate azimuth FOV")
    #      FOV_az = np.nan

    # FOV zenith :::::::::::::::::::::::::::::::::::::::::::::::::::::::
    try:
        config['var'] = 'zenith'
        config['ang_step'] = 0.2
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
        FOV_zn = fov_val

        # Figure zenith
        plt.subplot(grid[2, 2:4])

        plt.plot(data_plot[0], data_plot[1], '.-r', label=config['var'])
        plt.plot(axis_new, sm_inter(axis_new), label="smoothing")
        plt.ylabel('Radiance[1]')
        plt.xlabel('Angle[deg]')
        plt.xlim(config['meas_zen'] - config['delta'], config['meas_zen'] + config['delta'])
        plt.title("FOV along Zenith Axis")
        plt.plot([ind_f[0], ind_f[1]], [0.5, 0.5], '-g')

    except:
        print("Could not calculate zenith FOV")
        FOV_zn = np.nan

    # :::::::::::::: Spectral FOV::::::::::::::::::::::

    config['var'] = 'azimuth'
    config['wave'] = 500
    #radiance = data[:, config['pixel_index'], config['wave']] / data[:, config['pixel_index'], config['wave']].max()

    #azim, zen, radiance = select_range(azim, zen, radiance, config)
    #config['peaks'] = find_centre_fov(azim, zen, radiance)

    dt = spectral_FOV(data, config, correc=correc)

    # delete nan values in array
    dt = dt[~np.isnan(dt[:, 1])]

    np.savetxt('{name}spectral_fov_{index:d}.txt'.format(name=config['name'],
                                                          index=config['pixel_index']),
               X=dt, delimiter=',', header="Wavelength Index, FOV")

    waves = 400 + (500 / 1024) * dt[:, 0]

    config['wave'] = 500
    #radiance = data[:, config['pixel_index'], config['wave']] / data[:, config['pixel_index'], config['wave']].max()
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
                                   FOV_az=FOV_az, FOV_zn=FOV_zn)
    plt.text(0.1, 0.80, text, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='left')

    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())

    plt.savefig('fig_{name}_new.png'.format(name=config['name']), dpi=300)


def FOV_plot(azim, zen, radiance, config, add_points=False):

    """Plot the FOV measured in a contour plot """

    delta = config['delta']
    name = 'corrected_FOV'

    ylim_min = config['meas_zen'] - delta
    ylim_max = config['meas_zen'] + delta
    xlim_min = config['meas_azim'] - delta
    xlim_max = config['meas_azim'] + delta

    # parameters
    levels = 30
    cmap = 'rainbow' #'gray'#'nipy_spectral'

    plt.tricontourf(azim, zen, radiance, levels, cmap=cmap)
    plt.ylim(ylim_min, ylim_max)
    plt.title('FOV AZ{} ZN{}'.format(config['meas_azim'], config['meas_zen']))
    plt.xlabel('azimuth angle[deg]', fontsize=12)
    plt.ylabel('zenith angle[deg]', fontsize=12)
    plt.gca().invert_yaxis() #invert y axis

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


def interpolate_fov(azim, zen, radiance):
    """ Interpolate the measured data into a regular grid using scipy.interpol-
    ate.griddata()"""
    num_points = 100
    # Interpolation of data  #
    x = np.linspace(azim.min(), azim.max(), num=num_points)
    y = np.linspace(zen.min(), zen.max(), num=num_points)
    xx, yy = np.meshgrid(x, y)
    plane = scipy.interpolate.griddata((azim, zen), radiance, (xx, yy),
                                       method='cubic')

    return xx, yy, plane


def angle_correction(ang, d, r, centre):
    """ Correct the angles viewed by the optical fibre when the movement is
    centred on the input optic. If the movement is centred on the optical fibre
    it is not needed """

    return ang + np.degrees(np.arctan((r * np.sin(np.radians(ang - centre)))
                                      / (d + r * (1 - np.cos(np.radians(ang - centre))))))


def find_centre_fov(azim, zen, radiance):
    """ Find the FOV using the center of mass, 2d array"""
    peak_azim = np.nansum(azim * radiance) / np.nansum(radiance)
    peak_zen = np.nansum(zen * radiance) / np.nansum(radiance)

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


def select(azim, zen, radiance, config):
    """ Select values for FOV curves"""

    config['peaks'] = find_centre_fov(azim, zen, radiance)
    # peak_azim, peak_zen

    # find the values
    tol = config['ang_step'] / 2

    # azimuth FOV
    if config['var'] == 'azimuth':
        axis = zen
        axis_t = config['peaks'][1]

        index = np.where((axis >= axis_t - tol) &
                         (axis <= axis_t + tol))
        # eliminate nan values
        x = azim[index]
        y = radiance[index]
        par = np.column_stack((x, y))
        par = par[~np.isnan(y)]
        data_peaks = np.asfarray((sorted(par[:, 0]), par[:, 1]))

    # zenith FOV
    if config['var'] == 'zenith':
        axis = azim
        axis_t = config['peaks'][0]

        index = np.where((axis > axis_t - tol) &
                         (axis <= axis_t + tol))

        # eliminate nan values
        x = zen[index]
        y = radiance[index]
        par = np.column_stack((x, y))
        par = par[~np.isnan(y)]
        data_peaks = np.asfarray((par[:, 0], par[:, 1]))

    return data_peaks


def spectral_FOV(data, config, correc=False):
    """Spectral FOV"""
    w_i = 100
    w_f = 1000
    st = config['step_waves']

    positions = config['positions']

    azim = positions[:, 0]
    zen = positions[:, 1]

    azim = ((azim - config['meas_azim']) * np.sin(np.radians(zen))) + config[
        'meas_azim']

    # angular correction
    if correc is True:
        azim = angle_correction(azim, d=90, r=12.5, centre=config['peaks'][0])
        zen = angle_correction(zen, d=90, r=12.5, centre=config['peaks'][1])
        # config['ang_step'] = 0.2

    config['var'] = 'azimuth'
    fovs = np.zeros((int((w_f - w_i) / st), 2))
    cnt = 0
    azim_0 = positions[:, 0]
    zen_0 = positions[:, 1]

    for wave in np.arange(w_i, w_f, st):
        radiance_0 = data[:, config['pixel_index'], wave] / data[:, config['pixel_index'], wave].max()
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
        fovs[cnt] = wave, fov_val
        cnt += 1
    return fovs
