"""
Display utilities
"""

def spectroviz(data, sr=None, hop_length=None, 
            x_axis='time', y_axis='hz', 
            ax=None, filepath=None, 
            cmap='viridis', **kwargs
):
    """
    Display a spectrogram
    
    Parameters
    ----------
    data : np.ndarray
        Spectrogram data (freq x time)
    sr : int
        Sample rate
    hop_length : int
        Hop length in samples
    x_axis : str
        X-axis type ('time' or 'frames')
    y_axis : str
        Y-axis type ('hz' or 'linear')
    cmap : str
        Colormap name
    ax : matplotlib axis or None
        Axis to plot on
    **kwargs : additional arguments
        Passed to imshow
        
    Returns
    -------
    im : matplotlib image
        Image object
    """


    # resolve sr: explicit -> filepath probe -> module native -> raise
    if sr is None:
        if filepath is not None:
            from .io import _sample_rate
            sr = _sample_rate(filepath)
        else:
            from .fourier import get_native_sr
            sr = get_native_sr()
            if sr is None:
                raise ValueError("sr unresolvable: provide sr, filepath, or call set_native_sr()")


    import matplotlib.pyplot as plt
    
    if ax is None:
        ax = plt.gca()
    
    n_freq, n_time = data.shape
    
    # Calculate extents
    if x_axis == 'time':
        time_extent = n_time * hop_length / sr
        x_coords = [0, time_extent]
    else:
        x_coords = [0, n_time]
    
    if y_axis == 'hz':
        freq_extent = sr / 2
        y_coords = [0, freq_extent]
    else:
        y_coords = [0, n_freq]
    
    extent = [x_coords[0], x_coords[1], y_coords[0], y_coords[1]]
    
    # Display image
    im = ax.imshow(data, 
                    aspect='auto', 
                    origin='lower',
                    extent=extent,
                    cmap=cmap,
                    **kwargs)
    
    # Set labels
    if x_axis == 'time':
        ax.set_xlabel('Time (s)')
    if y_axis == 'hz':
        ax.set_ylabel('Frequency (Hz)')
    
    return im