"""pysoniq - Lightweight cross-platform audio library"""

from .play import play
from .stop import stop
from .pause import pause, resume, is_paused
from .io import load, save
from .loop import set_loop, is_looping

from .gain import (
    set_gain, get_gain, 
    set_volume_db, get_volume_db,
    adjust_gain_level, normalize, compress, limiter,
    db_to_linear, linear_to_db
)
from .fourier_stuff import stft, fft_frequencies, frames_to_time, amplitude_to_db

__version__ = '0.1.0'
__all__ = [
    'play', 'stop', 
    'pause', 'resume', 'is_paused',
    'load', 'save', 
    'set_loop', 'is_looping',
    
    'set_gain', 'get_gain', 'set_volume_db', 'get_volume_db',
    'adjust_gain_level', 'normalize', 'compress', 'limiter',
    'db_to_linear', 'linear_to_db', 

    'stft', 'fft_frequencies', 'frames_to_time', 'amplitude_to_db'
]

