""" Analyses white noise stimulus.
"""


import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve
from scipy.optimize import curve_fit
from scipy.linalg import norm
import pathlib


###############################################################################
# Global variables
###############################################################################
# Length of autocorrelation in ms.
AUTOCORR_LENGTH_MS = 50
# Duration of STA in ms. STA ends at frame of spike.
STA_DURATION_MS = 500
# Setup and stimulus info
STIXELSIZE = 2          # Size of the stimulus squares in pixels (edge length)
RUNNINGIMAGES = 3825    # Does not consider blinks
FROZENIMAGES = 652      # Does not consider blinks
NBLINKS = 4             # Number of frames for which each noise image was shown
SAMPLING_RATE = 25000   # Sampling rate of the recording
FRAME_RATE = 85         # Frame rate of the screen
# Converting units etc
AUTOCORR_LENGTH = int(AUTOCORR_LENGTH_MS / 1000 * SAMPLING_RATE)    # In samples
STA_DURATION = int(STA_DURATION_MS / 1000 * FRAME_RATE)             # In frames
X_RES = int(800/STIXELSIZE)     # Number of stimulus squares in x-direction
Y_RES = int(600/STIXELSIZE)     # Number of stimulus squares in y-direction
PIXELSIZE = 7.5                         # In micrometers
STIXELSIZE_UM = STIXELSIZE * PIXELSIZE  # In micrometers
CYCLEFRAMES = (RUNNINGIMAGES + FROZENIMAGES) * NBLINKS  # Number of frames per stimulus cycle (running + frozen noise)


###############################################################################
# Functions
###############################################################################
def Twod_gaussian(data_tuple, x0, y0, sigma_x, sigma_y, theta, amplitude):
    """ Calculate the value of a 2D Gaussian at the given position.

    Parameters
    ----------
    data_tuple : ndarray
        Location at which the Gaussian is evaluated of the form (x, y).
    x0 : float
        X-position of the center of the Gaussian.
    y0 : float
        Y-position of the center of the Gaussian.
    sigma_x : float
        Standard deviation in x-direction.
    sigma_y : float
        Standard deviation in y-direction.
    theta : float
        Rotation angle in radians clockwise.
    amplitude : float
        Amplitude/height of the Gaussian.

    Returns
    -------
    ndarray
        1D array of the values of the gaussian at the specified positions.
    """

    (x, y) = data_tuple
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))

    return g.ravel()


def Ellipse_xy_size(h_rad, v_rad, angle):
    """ Calculate the size in x- and y-direction of a rotated ellipse.

    Parameters
    ----------
    h_width : float
        Horizontal radius of the ellipse before rotation.
    v_width : float
        Vertical radius of the ellipse before rotation.
    angle : float
        Rotational angle of the ellipse in radians from a horizontal ellipse
        in clockwise or anti-clockwise direction.

    Returns
    -------
    float
        Width of the rotated ellipse in x-direction in pixels.
    float
        Height of the rotated ellipse in y-direction in pixels.
    """

    if h_rad >= v_rad:
        semimajor = h_rad
        semiminor = v_rad
    else:
        semimajor = v_rad
        semiminor = h_rad
        angle += np.pi/2

    t = np.arctan(-semiminor/semimajor * np.tan(angle))
    x = (semimajor*np.cos(t)*np.cos(angle)
         - semiminor*np.sin(t)*np.sin(angle))
    x_size = np.abs(2*x)

    t = np.arctan(-semiminor/semimajor * np.tan(angle + np.pi/2))
    y = (semimajor*np.cos(t)*np.sin(angle)
         + semiminor*np.sin(t)*np.cos(angle))
    y_size = np.abs(2*y)

    return x_size, y_size


###############################################################################
# Main program
###############################################################################
if __name__ == '__main__':

    pathlib.Path("../results").mkdir(parents=True, exist_ok=True)

    ###########################################################################
    # Loading and preprocessing data
    ###########################################################################
    print("Loading and preprocessing data...")

    # Loading and preprocessing the stimulus data
    frame_times = np.load("../data/05 - FrozenNoise/stimchanges.npy")
    num_full_cycles = int(frame_times.size/(CYCLEFRAMES))
    frame_lengths = np.diff(frame_times[:-1])
    avg_frame_length = np.mean(frame_lengths)
    stimulus_npz = np.load("../data/05 - FrozenNoise/stimulus.npz")
    stimulus_full = stimulus_npz['running_noise']

    # Loading spikes
    cluster_ids = np.loadtxt("../data/cell_classification.txt",
                             usecols=0, dtype=int)
    spikes_all = np.load("../data/05 - FrozenNoise/spikes.npy",
                         allow_pickle=True)
    num_units = cluster_ids.size

    # Find out the frame each spike belongs to
    spike_frames_all = [np.digitize(spikes_all[i], frame_times) - 1
                        for i in range(num_units)]
    spike_frames_all = np.array(spike_frames_all, dtype='object')

    ###########################################################################
    # Spatio-temporal STA calculation
    ###########################################################################
    print("Calculating spatiotemporal STA...")

    # Loop running over the different cycles
    stas = np.zeros((num_units, STA_DURATION, X_RES, Y_RES))
    num_spikes = np.zeros(num_units)
    for cycle in range(num_full_cycles):
        print(f'{cycle/num_full_cycles*100:.1f}%')
        first_frame = cycle * CYCLEFRAMES
        # Extract the next batch of frames
        stimulus = stimulus_full[cycle].astype(int) * 2 - 1
        stimulus = np.repeat(stimulus, NBLINKS, axis=0)
        # Loop over all units
        for i in range(num_units):
            # Extract the frame of all spikes in the relevant window
            spike_frames = spike_frames_all[i]
            spike_frames = spike_frames[(spike_frames >= first_frame + STA_DURATION - 1)
                                        & (spike_frames < first_frame + RUNNINGIMAGES * NBLINKS)]
            spike_frames -= first_frame
            # Add all pre-spike stimuli to the STA
            for spike_frame in spike_frames:
                stas[i] += stimulus[spike_frame - STA_DURATION + 1:
                                    spike_frame + 1, :, :]
            num_spikes[i] += spike_frames.size
    # Normalisation
    stas[num_spikes > 0] /= np.transpose([[[num_spikes[num_spikes > 0]]]])

    print("100.0%")

    ###########################################################################
    # STA separation
    ###########################################################################
    print("Performing STA separation")

    # Loop over all units
    stas_xy = np.zeros((num_units, X_RES, Y_RES))
    stas_t = np.zeros((num_units, STA_DURATION))
    for i in range(num_units):
        print(f'{i/num_units*100:.1f}%')
        sta = stas[i]
        sta_smooth = gaussian_filter(sta, sigma=(0, 60/STIXELSIZE_UM,
                                                 60/STIXELSIZE_UM),
                                     mode='constant')
        max_pix = np.unravel_index(np.argmax(np.abs(sta_smooth), axis=None),
                                   sta_smooth.shape)
        sta_xy = np.copy(sta[max_pix[0], :, :])
        sta_t = np.copy(sta[:, max_pix[1], max_pix[2]])
        # Normalize temporal STA
        if np.sum(np.square(sta_t)) != 0:
            sta_t = sta_t / np.sqrt(np.sum(np.square(sta_t)))
        # Correct the sign so the spatial STA is always positive
        sign = np.sign(sta_smooth[max_pix])
        sta_xy *= sign

        stas_xy[i] = sta_xy
        stas_t[i] = sta_t

    print("100.0%")

    ###########################################################################
    # Fitting Gaussians to the spatial STAs
    ###########################################################################
    print("Fitting Gaussians to spatial STA...")

    # Loop over all units
    xx, yy = np.mgrid[0:X_RES, 0:Y_RES]
    gauss_params = np.zeros((num_units, 6))
    bounds = ([0, 0, 0, 0, -np.pi, 0],
              [X_RES, Y_RES, np.inf, np.inf, np.pi, np.inf])
    for i in range(num_units):
        print(f'{i/num_units*100:.1f}%')
        if np.all(stas_xy[i] == 0):
            gauss_params[i] = np.full(6, np.NaN)
            print(f"Optimal Gaussian parameters for unit {cluster_ids[i]}"
                  + " not found! Unit has no STA.")
        else:
            smooth_sta_xy = gaussian_filter(stas_xy[i], sigma=(60/STIXELSIZE_UM,
                                                               60/STIXELSIZE_UM),
                                            mode='constant')
            peak_loc = np.unravel_index(np.argmax(smooth_sta_xy),
                                        smooth_sta_xy.shape)
            initial_guess = (peak_loc[0], peak_loc[1], 1, 1, 0,
                             np.max(stas_xy[i]))
            try:
                gauss_params[i], _ = curve_fit(Twod_gaussian, (xx, yy),
                                               stas_xy[i].ravel(),
                                               p0=initial_guess, bounds=bounds)
            except RuntimeError:
                gauss_params[i] = np.full(6, np.NaN)
                print(f"Optimal Gaussian parameters for unit {cluster_ids[i]}"
                      + " not found!")
    print("100.0%")

    ###########################################################################
    # Estimating the nonlinearity
    ###########################################################################
    print("Estimating the nonlinearity...")

    # First calculate the crop window for each spatial STA
    rf_windows_x = np.empty((num_units, 2), dtype=int)
    rf_windows_y = np.empty((num_units, 2), dtype=int)
    for i in range(num_units):
        if np.any(np.isnan(gauss_params[i])):
            rf_windows_x[i] = [0, X_RES]
            rf_windows_y[i] = [0, Y_RES]
        else:
            x_size, y_size = Ellipse_xy_size(gauss_params[i, 2]*3,
                                             gauss_params[i, 3]*3,
                                             gauss_params[i, 4])
            rf_windows_x[i] = [np.floor(gauss_params[i, 0] - x_size/2),
                               np.ceil(gauss_params[i, 0] + x_size/2)]
            rf_windows_y[i] = [np.floor(gauss_params[i, 1] - y_size/2),
                               np.ceil(gauss_params[i, 1] + y_size/2)]
    rf_windows_x = np.clip(rf_windows_x, 0, X_RES)
    rf_windows_y = np.clip(rf_windows_y, 0, Y_RES)

    # Calculate generator signal and spike count
    stas_t_half = np.copy(stas_t[:, int(STA_DURATION/2):])
    normalize = norm(stas_t_half, axis=1)
    normalize[normalize==0] = 1
    stas_t_half /= np.transpose([normalize])
    stas_xy_window = [stas_xy[i, rf_windows_x[i, 0]:rf_windows_x[i, 1],
                              rf_windows_y[i, 0]:rf_windows_y[i, 1]]
                      for i in range(num_units)]
    stas_xy_window = np.array(stas_xy_window, dtype=object)
    stas_xy_window /= [norm(stas_xy_window[i]) if not np.all(stas_xy_window[i] == 0) else 1
                       for i in range(num_units)]
    generators = np.empty((num_units, num_full_cycles,
                           RUNNINGIMAGES * NBLINKS - stas_t_half.shape[1] + 1))
    spike_count = np.empty((num_units, num_full_cycles,
                            RUNNINGIMAGES * NBLINKS - stas_t_half.shape[1] + 1),
                           dtype=int)
    for cycle in range(num_full_cycles):
        print(f'{cycle/num_full_cycles*100:.1f}%')
        first_frame = cycle * CYCLEFRAMES
        # Extract the next batch of frames
        stimulus = stimulus_full[cycle].astype(int) * 2 - 1
        stimulus = np.repeat(stimulus, NBLINKS, axis=0)
        for i in range(num_units):
            stimulus_window = stimulus[:,
                                       rf_windows_x[i, 0]:rf_windows_x[i, 1],
                                       rf_windows_y[i, 0]:rf_windows_y[i, 1]]
            temp_signal = convolve(np.transpose(stimulus_window),
                                   np.transpose(np.flip([stas_xy_window[i]])),
                                   mode='valid')[0, 0]
            generators[i, cycle] = convolve(temp_signal,
                                            np.flip(stas_t_half[i]),
                                            mode='valid')
            spike_count[i, cycle], _ = np.histogram(spike_frames_all[i],
                                                    bins=np.arange(first_frame + stas_t_half.shape[1] - 1,
                                                                   first_frame + RUNNINGIMAGES * NBLINKS + 1))
    generators = np.reshape(generators, (num_units, -1))
    spike_count = np.reshape(spike_count, (num_units, -1))

    # Calculate firing rate for similar generator signals
    num_bins = 10
    nl_firing_rates = np.empty((num_units, num_bins))
    nl_bin_centers = np.empty((num_units, num_bins))
    for i in range(num_units):
        generator_sort = np.sort(generators[i])
        bin_edges = generator_sort[::round(generators[i].size/num_bins)]
        # Correct the last bin edge
        if bin_edges.size == num_bins + 1:
            bin_edges[-1] = generator_sort[-1]
        else:
            bin_edges = np.append(bin_edges, generator_sort[-1])
        # Bin centers are the weighted centers
        for j in range(num_bins):
            nl_bin_centers[i, j] = np.mean(generator_sort[(generator_sort >= bin_edges[j])
                                                          & (generator_sort < bin_edges[j+1])])
        # Count the number of spikes in each generator bin
        summed_spike_count, _ = np.histogram(generators[i],
                                             bins=bin_edges,
                                             weights=spike_count[i])
        # Divide by the number of occurences of each generator bin
        # and the duration of a frame
        nl_firing_rates[i] = (summed_spike_count
                              / np.histogram(generators[i], bins=bin_edges)[0]
                              / avg_frame_length)

    print("100.0%")

    ###########################################################################
    # Autocorrelation calculation
    ###########################################################################
    print("Calculating autocorrelation...")

    # Loop running over all units and cycles
    auto_corrs = np.zeros((num_units, AUTOCORR_LENGTH + 1))
    for i in range(num_units):
        print(f'{i/num_units*100:.1f}%')
        spikes = np.rint(spikes_all[i] * SAMPLING_RATE).astype(int)
        for cycle in range(num_full_cycles):
            # Calculations are done in samples to avoid numerical errors
            start_time = int(round(frame_times[cycle * CYCLEFRAMES]
                                   * SAMPLING_RATE))
            end_time = int(round(frame_times[(cycle * CYCLEFRAMES
                                              + RUNNINGIMAGES * NBLINKS)]
                                 * SAMPLING_RATE))
            spike_trace = np.zeros(end_time - start_time, dtype=bool)
            spike_times = spikes[(spikes >= start_time) & (spikes < end_time)]
            spike_trace[spike_times - start_time] = 1
            auto_corrs[i] += np.correlate(spike_trace[:-AUTOCORR_LENGTH],
                                          spike_trace)
    auto_corrs = auto_corrs[:, :-1]

    # Smoothing and normalizing
    auto_corrs = gaussian_filter(auto_corrs, (0, 10))
    auto_corrs /= np.transpose([np.sum(auto_corrs, axis=1)])

    print("100.0%")

    ###########################################################################
    # Finishing
    ###########################################################################
    # Transform parameters of Gaussian fit from stixels to pixels
    gauss_params_px = np.copy(gauss_params)
    gauss_params_px[:, :4] *= STIXELSIZE
    # Upscaling leads to an effective shift of the positions of about half a stixel
    gauss_params_px[:, :2] += STIXELSIZE/2 - 0.5

    # Calculate the receptive field width and height
    ellipse_sizes = np.zeros((num_units, 2))
    for i in range(num_units):
        ellipse_sizes[i] = Ellipse_xy_size(gauss_params_px[i, 2]*1.5,
                                           gauss_params_px[i, 3]*1.5,
                                           gauss_params_px[i, 4])

    # Calculate the RF ellipse areas in mm²
    ellipse_areas = (np.pi * (gauss_params_px[:, 2] * 1.5)
                     * (gauss_params_px[:, 3] * 1.5))
    ellipse_areas = ellipse_areas * (PIXELSIZE/1000)**2
    # Calculate the diameter of a circle with the same area in micrometers
    eff_diameters = np.sqrt(4*ellipse_areas/np.pi) * 1000

    # Save results
    np.savez("../results/Whitenoise.npz",
             stixelsize=STIXELSIZE,           # Size of a stixel in pixels
             cluster_ids=cluster_ids,         # Indices of the clusters included in this analysis
             frame_length=avg_frame_length,   # Average length of a frame in seconds
             auto_corrs_smoothed=auto_corrs,  # Smooth autocorrelation per sample normalized to sum
             stas_xy=stas_xy,                 # Spatial component of STA in stixels
             stas_t=stas_t,                   # Temporal component of STA in frames
             gauss_params=gauss_params_px,    # Gaussian fit parameters in pixels/radians clockwise
             eff_diameters=eff_diameters,     # Diameter of a circle with same area as RF ellipse in micrometers
             nl_bin_centers=nl_bin_centers,   # Generator values of the datapoints of the nonlinearity
             nl_firing_rates=nl_firing_rates) # Firing rates correpsonding to the generator values
