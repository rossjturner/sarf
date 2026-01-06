# sarf module
# Ross Turner, 29 October 2025

# import packages
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.colors as mc
#import matplotlib.dates as mdates
#import colorsys, datetime, os, obspy, pytz
import os, warnings
#import seaborn as sns
#from functools import partial
#from math import factorial
from obspy import read, read_inventory, Trace, Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.mass_downloader import GlobalDomain, Restrictions, RectangularDomain, MassDownloader
#from obspy.geodetics import locations2degrees, degrees2kilometers
#from obspy.signal.trigger import trigger_onset
#from obspy.signal.cross_correlation import correlate
#from obspy.signal.polarization import flinn
#from obspy.signal.filter import envelope
#from matplotlib import cm, rc
#from matplotlib.backends.backend_pdf import PdfPages
#from matplotlib.colors import ListedColormap,LinearSegmentedColormap
#from multiprocessing import cpu_count, Pool
#from numpy.fft import rfft, rfftfreq

from numba import njit

#from seismic_attributes import get_waveforms, group_components, __group_seismometers

# define constants
__chunklength_in_sec = 86400

## Define Seismic Array Response Function functions
def sarf(network, station, location, channel, starttime, endtime, coords=None, station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None, download=True, frequency=None, depression=None):
    """
    Parameters
    ----------
    network : str or list
        Network code(s) of the seismic station (e.g., 'IU'). Can set as null value if coords is provided.
    station : str or list
        Station code(s) (e.g., 'CASY'). Can set as null value if coords is provided.
    location : str or list
        Location code(s) of the station (may be empty string '').
    channel : str or list
        Channel code(s) (e.g., 'BHZ', 'HHZ'). Can set as null value if coords is provided.
    starttime : UTCDateTime or str
        Start time of the requested waveform. Can be ObsPy UTCDateTime or ISO-formatted string.
    endtime : UTCDateTime or str
        End time of the requested waveform. Can be ObsPy UTCDateTime or ISO-formatted string.
    coords : list of dict[str, float], optional
        List of station coordinates dictionaries with keys: 'latitude', 'longitude', 'elevation', 'local_depth'. 
    station_name : str, optional
        Name of the output station metadata file or directory. Default is 'stations'.
    providers : list of str, optional
        List of FDSN data providers to try in order (default is ['IRIS', 'LMU', 'GFZ']).
    user : str, optional
        Username for FDSN providers requiring authentication. Default is None.
    password : str, optional
        Password for FDSN providers requiring authentication. Default is None.
    download : bool, optional
        If True, downloads data from providers; if False, only prepares request objects. Default is True.
    frequency : float, optional
        Monochromatic frequency in Hz. Default is None.
    depression : float, optional
        Depression angle of incoming wave. Default is None.
    """

    # Request user input (if variables not defined)
    if frequency == None or depression == None:
        frequency = float(input('Enter monochromatic frequency in Hz: '))
        depression = float(input('Enter depression angle in degrees: '))

    # Manage possible input types for seismic stations
    network, station, location = __test_inputs(network, station, location)
    
    # Determine relative coordinates of each seismic station
    if not isinstance(coords, (list, np.ndarray, dict)):
        coords = get_stations(network, station, location, channel, starttime, endtime, station_name=station_name, providers=providers, user=user, password=password, download=download)
    elif isinstance(coords, dict):
        coords = [coords]
        
    # Calculate position vector for seismic array
    r_E, r_N, r_Z = __array_config(coords)
    
    # Define slowness and horizontal azimuth vectors for the SARF
    s_range = np.arange(0, 0.5 + 1e-9, 0.002) # s/km
    theta_range = np.arange(0, 360 + 1e-9, 1)
    
    # Calculate slowness vector for incoming wave (with assumed depression angle)
    s_E = s_range[:, None] * np.cos(np.deg2rad(depression)) * np.cos(np.deg2rad(90 - theta_range[None, :]))
    s_N = s_range[:, None] * np.cos(np.deg2rad(depression)) * np.sin(np.deg2rad(90 - theta_range[None, :]))
    s_Z = -s_range[:, None] * np.sin(np.deg2rad(depression)) * (1 + 0 * theta_range[None, :])

    # Calculate sigma for each slowness and angle of azimuth
    sigma = 1/len(coords) * np.abs(np.sum(np.exp(-1j * 2 * np.pi * frequency * (r_E[:, None, None] * s_E[None, :, :] + r_N[:, None, None] * s_N[None, :, :] + r_Z[:, None, None] * s_Z[None, :, :])), axis=0))**2

    # Calculate sigma in dB relative to maximum value
    sigma_dB = 10 * np.log10(sigma / np.max(sigma))
        
    return s_range, theta_range, sigma_dB


def plot_sarf(s_range, theta_range, sigma_dB, cmap='viridis'):

    # Set of plot layout
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # Convert degrees to radians for the polar axis
    theta_rad = np.deg2rad(theta_range)
    r = s_range

    # Create 2D grids for plotting
    Theta, R = np.meshgrid(theta_rad, r)

    # Plot using pcolormesh for polar coordinates
    pcm = ax.pcolor(Theta, R, sigma_dB, cmap=cmap, shading='auto', vmin=-10, vmax=0)

    # Add colorbar
    cbar = plt.colorbar(pcm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Relative Power (dB)')

    # Customize plot
    #ax.set_title('Seismic Array Response Function', va='bottom')
    ax.set_theta_zero_location('N')     # 0° at the top (North)
    ax.set_theta_direction(-1)          # Clockwise azimuth
    ax.set_xlabel('')                   # Not used in polar
    ax.set_ylabel('')                   # Not used in polar

    # Set radius limits and labels
    ax.set_rlabel_position(225)
    ax.set_ylim(r.min(), r.max())

    plt.tight_layout()
    plt.show()


def get_stations(network, station, location, channel, starttime, endtime, station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None, download=True):
        
    # Suppress warnings due to (e.g.) multiple location codes
    warnings.filterwarnings('ignore', message='Found more than one matching channel metadata')
    
    # Manage possible input types for seismic stations
    network, station, location = __test_inputs(network, station, location)
    if not isinstance(channel, (list, np.ndarray)):
        channel = [channel]
    
    # Define client for accessing client server
    if isinstance(providers, (list, np.ndarray)):
        client = []
        for provider in providers:
            client.append(Client(provider, user=user, password=password))
    else:
        client = [Client(provider, user=user, password=password)]
    
    # Read-in or download station data
    coords = []
    for i in range(0, len(station)):
        for j in range(0, len(channel)):
            # Read-in data from local file if it exists
            try:
                inv = read_inventory(station_name+'/'+network[i]+'.'+station[i]+'.'+location[i]+'.'+channel[j]+'.xml')
                
            # Download data from client server
            except Exception:
                print('Accessing data from client server.')
                
                for k in range(0, len(client)):
                    try:
                        inv = client[k].get_stations(
                            network=network[i],
                            station=station[i],
                            location=location[i],
                            channel=channel[j],
                            level='response')
                        
                        break
                    except Exception:
                        inv = None
                
                # Write station data to file
                if download == True:
                    if not inv == None:
                        inv.write(station_name+'/'+network[i]+'.'+station[i]+'.'+location[i]+'.'+channel[j]+'.xml', format="STATIONXML")
                    else:
                        print('Cannot access station data for '+network[i]+'.'+station[i]+'.'+location[i]+'.'+channel[j]+'.')
        
        # Append station coordinates
        if not inv == None:
            coords.append(inv.get_coordinates(network[i]+'.'+station[i]+'.'+location[i]+'.'+channel[j]))
            
    return coords
    

def __test_inputs(network, station, location):
    
    # Convert strings to lists
    if not isinstance(network, (list, np.ndarray)):
        network = [network]
    if not isinstance(station, (list, np.ndarray)):
        station = [station]
    if not isinstance(location, (list, np.ndarray)):
        location = [location]
        
    # Determine target length
    max_len = np.max([len(network), len(station), len(location)])
    
    # correct length of each vector
    if len(network) == 1:
        network = [network[0]] * max_len
    if len(station) == 1:
        station = [station[0]] * max_len
    if len(location) == 1:
        location = [location[0]] * max_len
        
    # Check all lists have same length
    if not (len(network) == len(station) == len(location)):
        raise ValueError("The seismometer network, station and location inputs must be strings or lists of the same length.")

    return network, station, location
    
    
def __array_config(coords):
    
    # Define parameters for WGS84 Earth
    major_axis = 6378.137 # km
    eccentricity = 0.0818191908426
    
    # Extract station locations
    lats, lons = np.array([c['latitude'] for c in coords]), np.array([c['longitude'] for c in coords])
    if 'elevation' in coords[0]:
        elevs = np.array([c['elevation'] for c in coords]) / 1000.0  # m → km
    else:
        elevs = np.zeros_like(lats)
    
    # Convert to Earth-centred coordinates
    local_radius = major_axis / np.sqrt(1 - eccentricity**2 * np.sin(np.deg2rad(lats))**2)
    xs = (local_radius + elevs) * np.cos(np.deg2rad(lats)) * np.cos(np.deg2rad(lons))
    ys = (local_radius + elevs) * np.cos(np.deg2rad(lats)) * np.sin(np.deg2rad(lons))
    zs = (local_radius * (1 - eccentricity**2) + elevs) * np.sin(np.deg2rad(lats))

    # Set reference station location
    ref_lat, ref_lon, ref_elev = lats[0], lons[0], elevs[0]
    ref_x, ref_y, ref_z = xs[0], ys[0], zs[0]
    
    # Calculate relative station locations
    rel_x, rel_y, rel_z = xs - ref_x, ys - ref_y, zs - ref_z
    
    # Convert to east, north, vertical coordinates (in frame of reference station)
    rel_E = -np.sin(np.deg2rad(ref_lon)) * rel_x + np.cos(np.deg2rad(ref_lon)) * rel_y
    rel_N = -np.sin(np.deg2rad(ref_lat)) * np.cos(np.deg2rad(ref_lon)) * rel_x - np.sin(np.deg2rad(ref_lat)) * np.sin(np.deg2rad(ref_lon)) * rel_y + np.cos(np.deg2rad(ref_lat)) * rel_z
    rel_Z = np.cos(np.deg2rad(ref_lat)) * np.cos(np.deg2rad(ref_lon)) * rel_x + np.cos(np.deg2rad(ref_lat)) * np.sin(np.deg2rad(ref_lon)) * rel_y + np.sin(np.deg2rad(ref_lat)) * rel_z
    
    return rel_E, rel_N, rel_Z


## Define IAS Capon functions
def IAS_Capon(network, station, location, channel, starttime, endtime, waveform_name='waveforms', station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None, download=True, frequency=None, dfrequency=None, slowness=60., dslowness=0.5, window_length=None, overlap=True, tapering=True, diagonal_loading=1.):
    """
    Parameters
    ----------
    network : str or list
        Network code(s) of the seismic station (e.g., 'IU'). Can set as null value if coords is provided.
    station : str or list
        Station code(s) (e.g., 'CASY'). Can set as null value if coords is provided.
    location : str or list
        Location code(s) of the station (may be empty string '').
    channel : str or list
        Channel code(s) (e.g., 'BHZ', 'HHZ'). Can set as null value if coords is provided.
    starttime : UTCDateTime or str
        Start time of the requested waveform. Can be ObsPy UTCDateTime or ISO-formatted string.
    endtime : UTCDateTime or str
        End time of the requested waveform. Can be ObsPy UTCDateTime or ISO-formatted string.
    waveform_name : str, optional
        Name of the output waveform timeseries file or directory. Default is 'waveforms'.
    station_name : str, optional
        Name of the output station metadata file or directory. Default is 'stations'.
    providers : list of str, optional
        List of FDSN data providers to try in order (default is ['IRIS', 'LMU', 'GFZ']).
    user : str, optional
        Username for FDSN providers requiring authentication. Default is None.
    password : str, optional
        Password for FDSN providers requiring authentication. Default is None.
    download : bool, optional
        If True, downloads data from providers; if False, only prepares request objects. Default is True.
    frequency : float, optional
        Monochromatic frequency in Hz. Default is None.
    dfrequency : float, optional
        Half-bandwidth in Hz. Default is None.
    slowness : float, optional
        Maximum slowness in seconds/degree. Default is 60 seconds.
    dslowness : float, optional
        Step-size in slowness in seconds/degree. Default is 0.5 seconds.
    window_length : float, optional
        Duration of each time window in seconds. Default is None.
    overlap : boolean, optional
        Option to include/exclude overlap between samples. Default is True.
    tapering : boolean, optional
        Option to apply a Hanning tapering/windowing. Default is True.
    diagonal_loading : float, optional
        Energy added to the cross-spectral density matrix to stabilise the inversion. Default is 1.0.
    """
    
    # Request user input (if variables not defined)
    if frequency == None or dfrequency == None or window_length == None:
        frequency = float(input('Enter monochromatic frequency in Hz: '))
        dfrequency = float(input('Enter half-bandwidth in Hz: '))
        window_length = float(input('Enter length of time window in seconds: '))

    # Determine relative coordinates of each seismic station
    coords = get_stations(network, station, location, channel, starttime, endtime, station_name=station_name, providers=providers, user=user, password=password, download=download)
    
    # Read-in or download waveform data as a stream
    if isinstance(channel, (list, np.ndarray)):
        channel = [channel[0]] # only take first channel
        print('IAS-Capon can only be performed for a single channel: using '+channel[0]+'.')
    stream = get_waveforms(network, station, location, channel, starttime, endtime, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password, download=download)
    
    # Remove coordinates of stations with no data in requested time window
    __missing_data(network, station, location, coords, stream)
    
    # Calculate position vector for seismic array
    r_E, r_N, r_Z = __array_config(coords)
    
    # Slice waveform data into time windows separated for each station
    traces, nstations, nwindows, nsamples = __time_windows(stream, window_length, overlap=overlap, tapering=tapering)
    
    # Calculate Fast Fourier Transform of trace in each time window
    rft, ift = __fast_fourier_transform(traces, nstations, nwindows, nsamples)
    
    # Calculate cross-spectral density between stations, and accumulate over time windows
    rsm, ism, fw, fe = __spectral_matrices_IAS(traces, nstations, nwindows, frequency, dfrequency, rft, ift, window_length)
    
    if diagonal_loading > 0: ### This weights based on the average across frequencies
        print('Diagonal Loading On!')
        rsm += np.identity(nstations) * (rsm.trace() / nsamples) * diagonal_loading
            
    # Perform FK analysis on the inverse of cross power spectral density / spectral covariance matrix
    fk, max_val, s_opt, vel, baz = __FK_analysis_IAS(rsm, ism, nstations, r_E, r_N, r_Z, nwindows, nsamples, frequency, dfrequency, slowness, dslowness, window_length)
    
    return 10 * np.log10(fk.real / max_val), slowness, s_opt, vel, baz, np.log10(fw), np.log10(fe)
    
    

# Define function to remove coordinates of stations with no data in requested time window.
def __missing_data(network, station, location, coords, stream):
    
    # Manage possible input types for seismic stations
    network, station, location = __test_inputs(network, station, location)
    
    # Remove coordinates that do not appear in the data stream
    for i in reversed(range(0, len(station))): # reverse to safely remove
        if (network[i], station[i]) not in {(tr.stats.network, tr.stats.station) for tr in stream}:
            coords.pop(i)
            print('Station '+network[i]+'.'+station[i]+'.'+location[i]+' removed as no waveform data is available.')


# Define function to slice the waveform data into time windows for later covariance estimation.
def __time_windows(stream, window_length, overlap=True, tapering=True):
    
    # Calculate number of arrays and and sample per time window
    nstations = stream.count()
    delta = 0
    for i in range(0, nstations):
        if i == 0:
            delta = stream[0].stats.delta
        else:
            if not stream[i].stats.delta == delta:
                raise RuntimeError('Traces for do not have the same sample rate for all stations.')
    nsamples = int(window_length/delta + 0.5)

    # Slice traces into time windows, de-mean and apply Hanning filter (as required)
    if overlap == True:
        nwindows = int(np.array(stream[0].stats.npts / nsamples)) * 2 - 1 # 50% overlap
        traces = np.zeros((nstations, nwindows, nsamples))
        for i in range(0, nstations):
            for j in range(0, nwindows):
                trace = stream[i][int(j * nsamples / 2) : int((j + 2) * nsamples / 2)]
                traces[i, j, 0:len(trace)] = trace[:]
                traces[i, j, 0:len(trace)] -= np.mean(traces[i, j, 0:len(trace)])
                if tapering == True:
                    traces[i, j, :] *= np.hanning(nsamples)
    else:
        nwindows = int(np.array(stream[0].stats.npts / nsamples))
        traces = np.zeros((nstations, nwindows, nsamples))
        for i in range(0, nstations):
            for j in range(0, nwindows):
                traces[i, j, :] = stream[i, j * nsamp : (j + 1) * nsamples]
                traces[i, j, :] -= np.mean(traces[i, j, :])
                if tapering == True:
                    traces[i, j, :] *= np.hanning(nsamples)

    return traces, nstations, nwindows, nsamples
    

# Define function to calculate the Fast Fourier Transform of trace in each time window
def __fast_fourier_transform(traces, nstations, nwindows, nsamples):
    
    # Calculate Fast Fourier Transform of trace in each time window
    rft = np.zeros((nwindows, nstations, int(nsamples / 2 + 1)))
    ift = np.zeros((nwindows, nstations, int(nsamples / 2 + 1)))
    for i in range(0, nstations):
        for j in range(0, nwindows):
            tp = np.fft.rfft(traces[i, j, :], nsamples)
            rft[j, i, :] = tp.real
            ift[j, i, :] = tp.imag
        
    return rft, ift
    

# Define function to calculate cross-spectral density between stations, and accumulate over time windows.
@njit(fastmath=True)
def __spectral_matrices_IAS(traces, nstations, nwindows, frequency, dfrequency, rft, ift, window_length):
    
    # Calculate frequency and dfrequency in Fourier indices
    freq_idx = int(frequency * window_length + 0.5)
    dfreq_idx = int(dfrequency * window_length + 0.5)
    
    # Calculate the cross-spectral density matrix between stations, and accumulate over time windows
    rsm = np.zeros((2 * dfreq_idx + 1, nstations, nstations))
    ism = np.zeros((2 * dfreq_idx + 1, nstations, nstations))
    for n in range(0, nwindows):
        for i in range(0, nstations):
            for j in range(0, nstations):
                for l in range(freq_idx - dfreq_idx, freq_idx + dfreq_idx + 1):
                    idx = l - freq_idx + dfreq_idx
                    rsm[idx, i, j] += rft[n, i, l] * rft[n, j, l] + ift[n, i, l] * ift[n, j, l]
                    ism[idx, i, j] += rft[n, j, l] * ift[n, i, l] - rft[n, i, l] * ift[n, j, l]
    
    # Calculate and apply per-frequency diagonal pre-whitening
    rsm, ism, fw, fe = __diagonal_whitening(rsm/nwindows, ism/nwindows, nstations, dfreq_idx)
    
    return rsm, ism, fw, fe
    

# Define function to calculate and apply diagonal pre-whitening to the covariance matrix
@njit(fastmath=True)
### FIX - read paper to check for correct whitening
def __diagonal_whitening(rsm, ism, nstations, dfreq_idx):

    # Calculate per-frequency diagonal pre-whitening
    fw, fe = 0., 0.
    for m in range(0, 2 * dfreq_idx + 1):
    
        # Calculate diagonal weights for each station
        weight = np.zeros(nstations)
        for i in range(0, nstations):
            weight[i] = (rsm[m, i, i] ** 2 + ism[m, i, i] ** 2) ** (-0.25) # weight is 1/sqrt(C)
            fw += 1. / (weight[i] ** 2)
            fe += rsm[m, i, i] ** 2 + ism[m, i, i] ** 2
        
        # Apply diagonal pre-whitening to convariance matrix; i.e., downweight noisy stations
        for i in range(0, nstations):
            for j in range(0, nstations):
                rsm[m, i, j] *= weight[i] * weight[j]
                ism[m, i, j] *= weight[i] * weight[j]
    
    fw = fw / (nstations * (2 * dfreq_idx + 1))
    fe = fe / nstations
    
    return rsm, ism, fw, fe


def __FK_analysis_IAS(rsm, ism, nstations, r_E, r_N, r_Z, nwindows, nsamples, frequency, dfrequency, slowness, dslowness, window_length):
    
    # Define parameters for WGS84 Earth
    deg2km = 111.133
    
    # Calculate frequency and dfrequency in Fourier indices
    freq_idx = int(frequency * window_length + 0.5)
    dfreq_idx = int(dfrequency * window_length + 0.5)
    
    # Calculate number of wavenumber bins
    k_max = 2 * np.pi * slowness * frequency / deg2km
    k_inc = 2 * np.pi * dslowness * frequency / deg2km
    nk = int(2*k_max / k_inc + 0.5) + 1 # fence post problem
    
    # Calculate the inverse of the cross-spectral density matrix
    rism, iism = __inverse_matrix(rsm, ism, nstations, dfreq_idx)
    
    # Perform frequency-wavenumber analysis
    fk = np.zeros((nk, nk))
    for l in range(freq_idx - dfreq_idx, freq_idx + dfreq_idx + 1):
        idx = l - freq_idx + dfreq_idx

        # Calculate local frequency and wavenumber
        freq_local = l / float(window_length)
        k_max_local = 2 * np.pi * slowness * freq_local / deg2km
        k_inc_local = 2 * np.pi * dslowness * freq_local / deg2km
        
        # Perform frequency-wavenumber analysis
        __fk = __FK_analysis(rism, iism, nk, idx, k_max_local, k_inc_local, nstations, r_E, r_N, r_Z)
        fk[:, :] += __fk[:, :]
    
    # Calculate the velocity and back azimuth of the maximum
    max_val = 0.
    for i in range(0, nk):
        for j in range(0, nk):
            if fk[i, j].real > max_val:
                max_val = fk[i, j].real
                sx_opt = -slowness + i * dslowness
                sy_opt = -slowness + j * dslowness
            if fk[i, j].real < 0:
                fk[i, j] = 0
    
    s_opt = np.sqrt(sx_opt ** 2 + sy_opt ** 2)
    vel = deg2km / s_opt # in km/s
    baz = np.degrees(np.arctan2(sx_opt, sy_opt)) % 360
        
    return fk, max_val, s_opt, vel, baz


# Define function to calculate the inverse of the cross-spectral density matrix.
@njit(fastmath=True)
def __inverse_matrix(rsm, ism, nstations, dfreq_idx):
    
    # Calculate Fast Fourier Transform of trace in each time window
    rism = np.zeros((2 * dfreq_idx + 1, nstations, nstations))
    iism = np.zeros((2 * dfreq_idx + 1, nstations, nstations))
    for m in range(2 * dfreq_idx + 1):
        imtx = np.linalg.inv(rsm[m] + 1j * ism[m])
        rism[m, :, :] = imtx.real
        iism[m, :, :] = imtx.imag
        
    return rism, iism
    

# Define function to perform frequency-wavenumber analysis.
@njit(fastmath=True)
def __FK_analysis(rism, iism, nk, idx, k_max, k_inc, nstations, r_E, r_N, r_Z):
    
    # Perform frequency-wavenumber analysis
    fk = np.zeros((nk, nk))
    for i in range(0, nk):
        k_E = -(-k_max + i * k_inc) ### check sign of wavenumber
        for j in range(0, nk):
            k_N = -(-k_max + j * k_inc)
            
            for m in range(0, nstations):
                # Add diagonal terms (real as matrix is Hermitian)
                fk[i, j] += rism[idx, m, m]

                for n in range(m + 1, nstations):
                    # Add off-diagonal terms (factor of 2 for symmetric terms; mn and nm)
                    phase = (k_E * (r_E[m] - r_E[n]) + k_N * (r_N[m] - r_N[n])) # can add a velocity model for k_Z
                    fk[i, j] += 2. * (rism[idx, m, n] * np.cos(phase) - iism[idx, m, n] * np.sin(phase))
                    #print((r_E[m] - r_E[n]), (r_N[m] - r_N[n]), rism[idx, m, n], iism[idx, m, n], 2. * (rism[idx, m, n] * np.cos(phase) - iism[idx, m, n] * np.sin(phase)))
            
            fk[i, j] = 1. / fk[i, j]
        
    return fk
    

# Define function to read-in or download waveform data as a stream
def get_waveforms(network, station, location, channel, starttime, endtime, waveform_name='waveforms', station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None, download=True):
        
    # Manage possible input types for seismic stations
    network, station, location = __test_inputs(network, station, location)
    if not isinstance(channel, (list, np.ndarray)):
        channel = [channel]
    
    # Read-in or download waveform data
    for i in range(0, len(station)):
        for j in range(0, len(channel)):
            if i == 0 and j == 0:
                stream = __get_waveforms(network[i], station[i], location[i], channel[j], starttime, endtime, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password, download=download)
            else:
                stream += __get_waveforms(network[i], station[i], location[i], channel[j], starttime, endtime, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password, download=download)
            
    return stream.slice(starttime, endtime)


# Define function to read-in or download waveform data; taken from seismic_attributes.py
def __get_waveforms(network, station, location, channel, t1, t2, waveform_name='waveforms', station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None, download=True):

    # Create empty stream to store waveform
    stream = Stream()

    # Set start and end time of each file; these start and end on calendar dates
    start_time = UTCDateTime(t1.year, t1.month, t1.day)
    end_time = start_time + __chunklength_in_sec

    # Read-in waveform data from downloaded files
    while (start_time < t2):
        filename = waveform_name+'/'+network+'.'+station+'.'+location+'.'+channel+'__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.mseed'

        # If file exists add to stream
        if os.path.isfile(filename):
            stream += read(filename)
        # otherwise attempt to download file then read-in if data exists
        else:
            if download == True:
                __download_waveforms(network, station, location, channel, start_time, end_time, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password)
                if os.path.isfile(filename):
                    stream += read(filename)
            else:
                # Issue warning that file is not available in local directory
                warnings.filterwarnings('always', category=UserWarning)
                warnings.warn(filename+' not available in local directory.', category=UserWarning)
                warnings.filterwarnings('ignore', category=Warning)

        # Update start and end time of each file
        start_time += __chunklength_in_sec
        end_time += __chunklength_in_sec
            
    return stream
    

# Define function to download waveforms using the mass downloader; taken from seismic_attributes.py
def __download_waveforms(network, station, location, channel, t1, t2, waveform_name='waveforms', station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None):

    # Specify rectangular domain containing any location in the world.
    domain = RectangularDomain(minlatitude=-90, maxlatitude=90, minlongitude=-180, maxlongitude=180)

    # Apply restrictions on start/end times, chunk length, station name, and minimum station separation
    restrictions = Restrictions(
        starttime=t1,
        endtime=t2,
        chunklength_in_sec=__chunklength_in_sec,
        network=network, station=station, location=location, channel=channel,
        reject_channels_with_gaps=False,
        minimum_length=0.0,
        minimum_interstation_distance_in_m=100.0)

    # Download requested waveform and station data to specified locations
    if isinstance(providers, list):
        if (not user == None) and (not password == None):
            client = []
            for provider in providers:
                client.append(Client(provider, user=user, password=password))
            mdl = MassDownloader(providers=client)
        else:
            mdl = MassDownloader(providers=providers)
    else:
        if (not user == None) and (not password == None):
            mdl = MassDownloader(providers=[Client(providers, user=user, password=password)])
        else:
            mdl = MassDownloader(providers=[providers])
    mdl.download(domain, restrictions, mseed_storage=waveform_name, stationxml_storage=station_name+'/{network}.{station}.'+location+'.'+channel+'.xml')
