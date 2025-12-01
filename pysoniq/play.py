"""Audio playback functionality"""

import numpy as np
import sys
from pathlib import Path
from .io import load
from . import loop as loop_module
from .utils import to_pcm, int32_to_24bit_bytes
from . import pause as pause_module

# Track current playback process for stopping
_current_process = None

def play(data, samplerate=None, blocking=False):
    """
    Play audio from file or array
    
    Args:
        data: str/Path (audio file) or numpy array (audio data)
        samplerate: int, required if data is array
        blocking: bool, if True wait for playback to finish
    """
    # Reset stop flag when starting new playback
    loop_module.reset_stop()
    
    # If string/Path, load file first
    if isinstance(data, (str, Path)):
        audio_array, sr = load(data)

        # Store state for pause/resume
        pause_module.set_playback_state(audio_array, sr)
        
        # If looping, start loop thread
        if loop_module.is_looping():
            loop_module.start_loop(audio_array, sr, _play_array)
            return
        
        return _play_array(audio_array, sr, blocking)
    
    # Otherwise assume numpy array
    if samplerate is None:
        raise ValueError("samplerate required when playing numpy array")
    
    # Store state for pause/resume
    pause_module.set_playback_state(data, samplerate)

    # If looping, start loop thread
    if loop_module.is_looping():
        loop_module.start_loop(data, samplerate, _play_array)
        return
    
    return _play_array(data, samplerate, blocking)


def _play_array(data, samplerate, blocking):
    """Internal: play numpy array"""
    # Check if stopped
    if loop_module.is_stopped():
        return
    # Apply main gain from gain module
    from . import gain as gain_module
    data = gain_module.adjust_gain_level(data)

    # Determine channels
    n_channels = 1 if data.ndim == 1 else data.shape[1]
    
    # Convert to mono if needed
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    # Convert to PCM
    sampwidth, data_int = to_pcm(data)
    
    # Platform-specific playback
    if sys.platform == 'win32':
        _play_windows(data_int, samplerate, sampwidth, n_channels, blocking)
    elif sys.platform == 'darwin':
        _play_macos(data_int, samplerate, sampwidth, n_channels, blocking)
    else:  # Linux
        _play_linux(data_int, samplerate, sampwidth, n_channels, blocking)


def _play_windows(data, samplerate, sampwidth, n_channels, blocking):
    """Windows playback using winsound"""
    import winsound
    import wave
    import tempfile
    import os
    
    if loop_module.is_stopped():
        return
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
        
        with wave.open(temp_path, 'wb') as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(sampwidth)
            wav.setframerate(samplerate)
            
            if sampwidth == 3:
                wav.writeframes(int32_to_24bit_bytes(data))
            else:
                wav.writeframes(data.tobytes())
    
    try:
        flags = winsound.SND_FILENAME
        if not blocking:
            flags |= winsound.SND_ASYNC
        
        winsound.PlaySound(temp_path, flags)
    finally:
        if not blocking:
            import threading
            def cleanup():
                import time
                time.sleep(len(data) / samplerate + 0.5)
                try:
                    os.unlink(temp_path)
                except:
                    pass
            threading.Thread(target=cleanup, daemon=True).start()
        else:
            try:
                os.unlink(temp_path)
            except:
                pass


def _play_macos(data, samplerate, sampwidth, n_channels, blocking):
    """macOS playback using afplay"""
    import subprocess
    import wave
    import tempfile
    import os
    
    global _current_process
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
        
        with wave.open(temp_path, 'wb') as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(sampwidth)
            wav.setframerate(samplerate)
            
            if sampwidth == 3:
                wav.writeframes(int32_to_24bit_bytes(data))
            else:
                wav.writeframes(data.tobytes())
    
    try:
        if blocking:
            _current_process = subprocess.Popen(['afplay', temp_path])
            _current_process.wait()
            _current_process = None
            os.unlink(temp_path)
        else:
            _current_process = subprocess.Popen(['afplay', temp_path])
            import threading
            def cleanup():
                import time
                time.sleep(len(data) / samplerate + 0.5)
                try:
                    os.unlink(temp_path)
                except:
                    pass
            threading.Thread(target=cleanup, daemon=True).start()
    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e


def _play_linux(data, samplerate, sampwidth, n_channels, blocking):
    """Linux playback using aplay"""
    import subprocess
    import wave
    import tempfile
    import os
    
    global _current_process
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name
        
        with wave.open(temp_path, 'wb') as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(sampwidth)
            wav.setframerate(samplerate)
            
            if sampwidth == 3:
                wav.writeframes(int32_to_24bit_bytes(data))
            else:
                wav.writeframes(data.tobytes())
    
    try:
        if blocking:
            _current_process = subprocess.Popen(['aplay', temp_path])
            _current_process.wait()
            _current_process = None
            os.unlink(temp_path)
        else:
            _current_process = subprocess.Popen(['aplay', temp_path])
            import threading
            def cleanup():
                import time
                time.sleep(len(data) / samplerate + 0.5)
                try:
                    os.unlink(temp_path)
                except:
                    pass
            threading.Thread(target=cleanup, daemon=True).start()
    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e