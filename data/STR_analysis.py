""" Analyzes the Ricker stripes stimulus.
"""


import numpy as np
from skimage.transform import iradon
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
import pathlib


###############################################################################
# Parameters
###############################################################################
# Bin size of any PSTHs calculated in seconds.
PSTH_BIN_SIZE = 0.01
# Gaussian sigma for smoothing the sinograms. First value is in position-
# second in angle-direction. In units of sinogram-pixels.
SMOOTHING = (1.5, 1.0)
# Determines how long after a bar flash/onset the counting of spikes starts to
# calculate the response (in seconds).
FLASH_INTEGRATION_WINDOW_START = 0.0
# Determines the end of counting spikes after bar onset as a factor of the
# average bar presentation duration.
FLASH_INTEGRATION_WINDOW_END = 1
# Setup and stimulus info
X_RES = 800             # Resolution of the screen in x-direction
Y_RES = 600             # Resolution of the screen in y-direction
PIXELSIZE = 7.5         # In micrometers
NUM_SHIFTS = 75         # Number of stripe positions
NUM_ANGLES = 36         # Number of stripe angles
BAR_DISTANCE = 50       # Distance between stripes in pixels


###############################################################################
# Functions
###############################################################################
def Psth(spike_train, stimulus_times, bin_length, num_bins):
    """Calculate the PSTH after the given stimuli.

    Parameters
    ----------
    spike_train : ndarray
        1D array containing all spike times of the unit.
    stimulus_times: ndarray
        1D array containing all stimulus times for which the PSTH is
        calculated.
    bin_length : float
        Length of a bin of the PSTH in seconds.
    num_bins : int
        Length of the PSTH in bins.

    Returns
    -------
    ndarray
        1D array containing the PSTH for the given stimuli.
    ndarray
        1D array containing the bin centers of the histogram.
    """

    # Find the indices of the stimuli preceding the spikes
    idxe_spike_stim = np.digitize(spike_train, stimulus_times) - 1
    # Calculate the distance between the spikes and the preceding stimulus
    spike_stim_dist = spike_train - stimulus_times[idxe_spike_stim]
    # Exclude all spikes that are not within the PSTH period
    spike_stim_dist = spike_stim_dist[spike_stim_dist < bin_length * num_bins]
    # Calculate the PSTH length that can be covered by this number of bins
    psth_length_trunc = num_bins * bin_length
    # Calculate the PSTH
    psth, bin_edges = np.histogram(spike_stim_dist, bins=num_bins,
                                   range=(0, psth_length_trunc))
    # Normalize the PSTH to Hz.
    psth = psth / (stimulus_times.size * bin_length)
    # Calculate the PSTH bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return psth, bin_centers


def Flash_sinogram(spike_train, bar_times, bar_params, integration_window):
    """ Calculate the sinogram from the spikes for a flashed stimulus.

    Parameters
    ----------
    spike_train : ndarray
        1D array containing all spike times of the unit.
    bar_times : ndarray
        1D array containing the times of bar presentations.
    bar_params : ndarray
        2D array containing the position and angle of each presented bar. First
        dimension is bar presentation, second position and angle.
    integration_window : array-like
        Array with two entries containing the start time and end time for
        summing the spikes belonging to a bar presentation after the beginning
        of its presentation.

    Returns
    -------
    ndarray
        2D array containing the sinogram of the unit. First dimension is bar
        position, second bar angle.
    """

    # First calculate the number of spikes in response to each bar presentation
    idxe_spike_bar = np.digitize(spike_train, bar_times) - 1
    spike_bar_dist = spike_train - bar_times[idxe_spike_bar]
    idxe_spike_bar = idxe_spike_bar[(spike_bar_dist >= integration_window[0])
                                    & (spike_bar_dist < integration_window[1])]
    unique, unique_spikes = np.unique(idxe_spike_bar, return_counts=True)
    num_spikes = np.zeros_like(bar_times)
    num_spikes[unique] = unique_spikes

    # For each parameter combination, calculate the mean number of spikes
    sinogram = np.empty((np.unique(bar_params[:, 0]).size,
                         np.unique(bar_params[:, 1]).size))
    for counter_pos, pos in enumerate(np.unique(bar_params[:, 0])):
        for counter_ang, ang in enumerate(np.unique(bar_params[:, 1])):
            mean_spikes = np.mean(num_spikes[np.all(bar_params == [pos, ang],
                                                    axis=1)])
            sinogram[counter_pos, counter_ang] = mean_spikes

    # Convert number of spikes into firing rate
    sinogram /= integration_window[1] - integration_window[0]

    return sinogram


def Correct_sinogram(sinogram, rf_center):
    """ Correct the sinogram such that the RF center is always in the middle.

    Parameters
    ----------
    sinogram : ndarray
        2D array containing the uncorrected sinogram. First dimension is bar
        position, second is bar angle.
    rf_center : ndarray
        1D array containing the receptive field center in x and y.

    Returns
    -------
    ndarray
        Corrected sinogram. Center of receptive field is at ceiled middle in
        position-dimension.
    """

    sino_corr = np.copy(sinogram)
    # Recover positions and angles used
    positions = np.linspace(-0.5, 0.5, num=NUM_SHIFTS, endpoint=False)
    angles = np.linspace(0, np.pi, num=NUM_ANGLES, endpoint=False)
    for counter_ang, angle in enumerate(angles):
        # Find distance of RF to center line (projection according to current angle)
        # Shift the screen center to the origin
        rf_pos = rf_center - [(X_RES-1)/2, (Y_RES-1)/2]
        # Rotate the RF position against the rotation of the bar
        rf_pos = [np.cos(angle)*rf_pos[0] - np.sin(angle)*rf_pos[1],
                  np.sin(angle)*rf_pos[0] + np.cos(angle)*rf_pos[1]]
        # Distance to the center line is then the new x coordinate
        dist_to_center = rf_pos[0]
        # Calculate the theoretical bar position the RF is at
        bar_position = 0.5 - ((dist_to_center/BAR_DISTANCE) % 1)
        # Find the nearest presented position
        position_idxe = np.argmin(np.abs(np.append(positions, 0.5)
                                         - bar_position))
        if position_idxe == positions.size:
            position_idxe = 0
        # Roll that position to where you want it
        sino_corr[:, counter_ang] = np.roll(sino_corr[:, counter_ang],
                                            int(NUM_SHIFTS/2) - position_idxe,
                                            axis=0)

    return sino_corr


def Find_hotspots(fbp):
    """ Localizes hotspots in the FBP by finding local maxima higher than 30%
    of the global maximum in the inner 90% circle.

    Parameters
    ----------
    fbp : ndarray
        2D array containing the FBP.

    Returns
    -------
    ndarray
        2D array containing the x- and y-coordinates of all identified
        hotspots.
    """

    global_max = np.max(fbp)
    coordinates = peak_local_max(fbp, min_distance=1)
    coordinates = [coord for coord in coordinates
                   if fbp[coord[0], coord[1]] >= 0.3*global_max]
    coordinates = np.array(coordinates)
    if coordinates.size > 0:
        distances = np.sqrt(np.sum(np.square(coordinates - (fbp.shape[0]-1)/2),
                                   axis=1))
        coordinates = coordinates[distances <= 0.9 * (fbp.shape[0]-1)/2]

    return coordinates


def Sinogram_analysis(spikes, bar_times, rf_center, gaussian_sigma, bar_params,
                      integration_window, analyse_hotspots=False):
    """ Performs the entire tomographic analysis for one cell.

    Parameters
    ----------
    spikes : ndarray
        1D array containing all spike times of the cell.
    bar_times : ndarray
        1D array containing the times of bar presentation.
    rf_center : array_like
        X- and Y-position of the receptive field center.
    gaussian_sigma : array_like
        Sigma of the gaussian smoothing in position- and angle-direction
        applied to the sinogram in pixel.
    bar_params : ndarray
        2D array containing the parameters of the presented bars in order of
        presentation. First dimension is presented bars (same size as
        *bar_times*), second dimension is bar position and bar angle.
    integration_window : array_like
        Contains two values that specify the start and end of the integration
        window used to calculate the number of spikes in response to a flashed
        bar. In seconds.
    analyse_hotspots : bool, optional
        If True, identifies hotspots in the reconstruction and returns their
        locations and average nearest neighbour distance.

    Returns
    -------
    fbp : ndarray
        2D array containing the filtered back-projection of the calculated
        sinogram.
    sinogram : ndarray
        2D array containing the sinogram of the cell. First index is bar
        position, second is bar angle. Sinogram is corrected to have the bar
        that hits the center of the RF in the middle.
    sinogram_uncorrected : ndarray, optional
        2D array containing a sinogram of the cell. Like *sinogram*, but not
        corrected for the RF position and not smoothed.
    hotspots : ndarray, optional
        Locations of the identified hotspots in the reconstruction. Only
        provided if *analyse_hotspots* is True.
    avg_nn_distance : float, optional
        Average nearest neighbour distance of the identified hotspots in
        micrometers. Only provided if *analyse_hotspots* is True.
    """

    # Calculate the sinogram
    sinogram_uncorrected = Flash_sinogram(spikes, bar_times, bar_params,
                                          integration_window)
    # Correct the sinogram such that the RF center is always at position 0,
    # i.e. in the middle
    sinogram = Correct_sinogram(sinogram_uncorrected, rf_center)
    # Smoothing sinogram to make structure more visible
    sinogram = gaussian_filter(sinogram, sigma=gaussian_sigma, mode='wrap')
    # Calculate the filtered back-projection
    fbp = np.transpose(iradon(sinogram, circle=True))
    # Correcting for the 180° rotation that the FBP contains
    fbp = np.flip(fbp)
    # Analysing hotspots in the FBP
    if analyse_hotspots:
        hotspots = Find_hotspots(fbp)
        num_hotspots = hotspots.shape[0]
        if num_hotspots > 1:
            # Calculate the average nearest neighbour distance
            distances = (np.reshape(np.repeat(hotspots, num_hotspots, axis=0),
                                    (num_hotspots, num_hotspots, 2))
                         - hotspots)
            distances = np.linalg.norm(distances, axis=2)
            distances = distances[np.logical_not(np.identity(num_hotspots,
                                                             dtype=bool))]
            distances = np.reshape(distances, (num_hotspots, num_hotspots-1))
            nearest_neighbours = np.min(distances, axis=1)
            avg_nn_distance = np.mean(nearest_neighbours)
            avg_nn_distance *= BAR_DISTANCE / NUM_SHIFTS * PIXELSIZE
        else:
            avg_nn_distance = np.NaN

    # Return results
    if analyse_hotspots:
        return fbp, sinogram, sinogram_uncorrected, hotspots, avg_nn_distance
    else:
        return fbp, sinogram, sinogram_uncorrected


###############################################################################
# Main program
###############################################################################
pathlib.Path("../results").mkdir(parents=True, exist_ok=True)

# Loading and preprocessing the stimulus data
pulse_times = np.load("data/04 - RickerStripes/stimchanges.npy")

# Count only completed cycles
num_bars = int((pulse_times.size - 1)/2)
black_bar_pulses = pulse_times[1::2][:num_bars]
grey_pulses = pulse_times[::2][:num_bars]
avg_grey_duration = np.mean(np.diff(pulse_times)[::2])
black_bar_offset_pulses = pulse_times[2::2][:num_bars]
avg_bar_duration = np.mean(np.diff(pulse_times)[1::2])
avg_cycle_length = avg_grey_duration + avg_bar_duration
first_half_idxb = np.ones(num_bars, dtype=bool)
first_half_idxb[int(num_bars/2):] = False
second_half_idxb = np.logical_not(first_half_idxb)
flash_integration_window = [FLASH_INTEGRATION_WINDOW_START,
                            FLASH_INTEGRATION_WINDOW_END*avg_bar_duration]

# Reconstructing the stimulus
stimulus_npz = np.load("data/04 - RickerStripes/stimulus.npz")
positions = stimulus_npz['positions']
angles = stimulus_npz['angles']
black_bar_params = np.transpose(np.stack((positions, angles)))

# Loading RF data
#whitenoise_npz = np.load("../results/Whitenoise.npz")
#rf_params = whitenoise_npz['gauss_params']

# Loading spike data
spikes_all = np.load("data/04 - RickerStripes/spikes.npy",
                     allow_pickle=True)
num_units = len(spikes_all)

# Setup variables that store the results to be saved later
psths = np.empty((num_units, int(avg_cycle_length / PSTH_BIN_SIZE)))
bin_centers = None
black_sinograms = np.empty((num_units, NUM_SHIFTS, NUM_ANGLES))
black_fbps = np.empty(num_units, dtype=object)
black_sinograms_uncorrected = np.empty((num_units, NUM_SHIFTS, NUM_ANGLES))
black_fbps_1st = np.empty(num_units, dtype=object)
black_fbps_2nd = np.empty(num_units, dtype=object)
black_hotspots = np.empty(num_units, dtype=object)
black_avg_nn_distances = np.empty(num_units)
black_sinograms_offset = np.empty((num_units, NUM_SHIFTS, NUM_ANGLES))
black_fbps_offset = np.empty(num_units, dtype=object)
black_fbps_1st_offset = np.empty(num_units, dtype=object)
black_fbps_2nd_offset = np.empty(num_units, dtype=object)
black_hotspots_offset = np.empty(num_units, dtype=object)
black_avg_nn_distances_offset = np.empty(num_units)

# Loop that analyses each unit individually
for counter in range(num_units):
    print(f'{counter/num_units*100:.1f}%')
    spikes = spikes_all[counter]

    # Calculate a PSTH for the presentation of the stimulus
    psths[counter], bin_centers = Psth(spikes, grey_pulses,
                                       PSTH_BIN_SIZE,
                                       int(avg_cycle_length / PSTH_BIN_SIZE))

    # Calculate the sinogram and FBP for black bars
    temp = Sinogram_analysis(spikes,
                             black_bar_pulses,
                             rf_params[counter, :2],
                             SMOOTHING,
                             black_bar_params,
                             flash_integration_window,
                             analyse_hotspots=True)
    black_fbps[counter], black_sinograms[counter], black_sinograms_uncorrected[counter], black_hotspots[counter], black_avg_nn_distances[counter] = temp

    # Analyze the two halfs of the recording
    temp = Sinogram_analysis(spikes,
                             black_bar_pulses[first_half_idxb],
                             rf_params[counter, :2],
                             SMOOTHING,
                             black_bar_params[first_half_idxb],
                             flash_integration_window)
    black_fbps_1st[counter] = temp[0]
    temp = Sinogram_analysis(spikes,
                             black_bar_pulses[second_half_idxb],
                             rf_params[counter, :2],
                             SMOOTHING,
                             black_bar_params[second_half_idxb],
                             flash_integration_window)
    black_fbps_2nd[counter] = temp[0]

    # Calculate the sinogram and FBP for the offset of black bars
    temp = Sinogram_analysis(spikes,
                             black_bar_offset_pulses,
                             rf_params[counter, :2],
                             SMOOTHING,
                             black_bar_params,
                             flash_integration_window,
                             analyse_hotspots=True)
    black_fbps_offset[counter], black_sinograms_offset[counter], _, black_hotspots_offset[counter], black_avg_nn_distances_offset[counter] = temp
    # Analyze the two halfs of the recording
    temp = Sinogram_analysis(spikes,
                             black_bar_offset_pulses[first_half_idxb],
                             rf_params[counter, :2],
                             SMOOTHING,
                             black_bar_params[first_half_idxb],
                             flash_integration_window)
    black_fbps_1st_offset[counter] = temp[0]
    temp = Sinogram_analysis(spikes,
                             black_bar_offset_pulses[second_half_idxb],
                             rf_params[counter, :2],
                             SMOOTHING,
                             black_bar_params[second_half_idxb],
                             flash_integration_window)
    black_fbps_2nd_offset[counter] = temp[0]

print("100%")

# Save results
np.savez("../results/STR.npz",
         num_shifts=NUM_SHIFTS,                         # Number of bar positions
         bar_distance=BAR_DISTANCE,                     # Distance of bars in pixels
         avg_grey_duration=avg_grey_duration,           # Average duration of grey presentations
         avg_bar_duration=avg_bar_duration,             # Average duration of bar presentations
         psths=psths,                                   # PSTHs of all units
         bin_centers=bin_centers,                       # Common bin centers of the PSTHs
         black_sinograms=black_sinograms,               # Sinograms for black bars
         black_sinograms_uncorrected=black_sinograms_uncorrected,       # Uncorrected sinograms for black bars
         black_fbps=black_fbps,                         # FBPs of black sinograms
         black_fbps_1st=black_fbps_1st,                 # FBPs of black sinograms of 1st half of recording
         black_fbps_2nd=black_fbps_2nd,                 # FBPs of black sinograms of 2nd half of recording
         black_hotspots=black_hotspots,                 # Locations of identified hotspots in FBPs of black sinograms
         black_avg_nn_distances=black_avg_nn_distances, # Average nearest neighbour distances of identified hotspots in FBPs of black sinograms
         black_fbps_offset=black_fbps_offset,           # FBPs of black offset sinograms
         black_fbps_1st_offset=black_fbps_1st_offset,   # FBPs of black offset sinograms of 1st half of recording
         black_fbps_2nd_offset=black_fbps_2nd_offset,   # FBPs of black offset sinograms of 2nd half of recording
         black_hotspots_offset=black_hotspots_offset,   # Locations of identified hotspots in FBPs of white offset sinograms
         black_avg_nn_distances_offset=black_avg_nn_distances_offset)   # Average nearest neighbour distances of identified hotspots in FBPs of black offset sinograms
