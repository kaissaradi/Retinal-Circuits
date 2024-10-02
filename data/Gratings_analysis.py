""" Analyzes the reversing grating with varying spatial period stimulus.
"""


import numpy as np
import pathlib


###############################################################################
# Global variables
###############################################################################
# Length of PSTH bins in seconds.
PSTH_BIN_LENGTH = 0.01
# Stimulus info
NUM_REVERSALS = 25                      # Number of reversals of each grating
NUM_REPEATS = 2                         # Number of repeats of the entire stimulus
STRIPE_WIDTHS = [1, 2, 4, 8, 16, 32, 64, 800]   # Barwidths of the gratings in pixels
NUM_PHASES = [1, 1, 2, 2, 4, 4, 8, 1]   # Number of spatial phases per grating size


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


def Psth_list(spike_train, pulse_times, grating_length,
              num_repeats=NUM_REPEATS):
    """ Calculate the PSTH for each stripe width and phase.

    Binsize of PSTH is given by global parameter *PSTH_BIN_LENGTH*.

    Parameters
    ----------
    spike_train : ndarray
        1D array containing all spike times of the unit.
    pulse_times : ndarray
        1D array containing the times of all stimulus changes.
    grating_length : float
        Average length of grating presentation.
    num_repeats : int, optional
        Number of trials/repeats of the stimulus scheme. Can be used to only
        analysis parts of the recording.

    Returns
    -------
    ndarray
        2D array containing all PSTHs. First index runs over stripe width and
        phase in the order of stimulus presentation (i.e. phase first then
        width), second index is time after stimulus.
    ndarray
        1D array containing the common bin centers of all the PSTHs.
    """

    num_bins = int(round(2*grating_length / PSTH_BIN_LENGTH))
    psth_list = np.empty((np.sum(NUM_PHASES), num_bins))
    for stim_counter in range(np.sum(NUM_PHASES)):
        # Find the indices of the relevant pulses
        stim_idxe = np.empty(0, dtype=int)
        for repeat in range(num_repeats):
            grey_idxe = (repeat*np.sum(NUM_PHASES)*(NUM_REVERSALS+1)
                         + stim_counter*(NUM_REVERSALS+1))
            # Disregard the onset of the grating and don't calculate PSTHs
            # further than where the reversals are happening
            new_stim_idxe = [grey_idxe + 2 + 2*i
                             for i in range(int((NUM_REVERSALS-1)/2))]
            stim_idxe = np.append(stim_idxe, new_stim_idxe)
        # Check that the stimulus hasn't been stopped early
        stim_idxe = stim_idxe[stim_idxe < pulse_times.size]
        psth_list[stim_counter], bin_centers = Psth(spike_train,
                                                    pulse_times[stim_idxe],
                                                    PSTH_BIN_LENGTH,
                                                    num_bins)

    return psth_list, bin_centers


###############################################################################
# Main program
###############################################################################
pathlib.Path("../results").mkdir(parents=True, exist_ok=True)

# Loading the stimulus data
pulse_times = np.load("../data/02 - reversinggratingswithvaryingspatialperiod/"
                      + "stimchanges.npy")

# Reconstructing the stimulus
grey_pulses_idxb = np.zeros(pulse_times.size, dtype=bool)
grey_pulses_idxb[::NUM_REVERSALS + 1] = True
stripe_width_seq = np.tile(np.repeat(STRIPE_WIDTHS, NUM_PHASES), NUM_REPEATS)
phase_seq = np.concatenate([np.arange(n) for n in NUM_PHASES])

# Calculate average stimulus times
grating_lengths = np.diff(pulse_times)[np.logical_not(grey_pulses_idxb[:-1])]
avg_grating_length = np.mean(grating_lengths)

# Loading spike data
spikes_all = np.load("../data/02 - reversinggratingswithvaryingspatialperiod/"
                     + "spikes.npy",
                     allow_pickle=True)
num_units = len(spikes_all)

# Arrays for information about all units
psth_lists = np.empty((num_units, np.sum(NUM_PHASES),
                       int(round(2*avg_grating_length / PSTH_BIN_LENGTH))))
bin_centers = np.empty((num_units,
                       int(round(2*avg_grating_length / PSTH_BIN_LENGTH))))


###############################################################################
# Main analysis loop going over all units.
###############################################################################
for counter in range(num_units):
    print(f'{counter/num_units*100:.1f}%')
    spikes = spikes_all[counter]

    # Analyze responses
    psth_lists[counter], bin_centers[counter] = Psth_list(spikes, pulse_times,
                                                          avg_grating_length)

# Storing all information
np.savez("../results/Gratings.npz",
         stripe_widths=STRIPE_WIDTHS,
         num_phases=NUM_PHASES,
         grating_length=avg_grating_length,
         psth_lists=psth_lists,
         bin_centers=bin_centers)
