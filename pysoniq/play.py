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

def play(audio_data, samplerate=None, blocking=False):
    """
    Play audio from file or array
    
    Args:
        audio_data: str/Path (audio file) or numpy array (audio audio_data)
        samplerate: int, required if audio_data is array
        blocking: bool, if True wait for playback to finish
    """
    # Reset stop flag when starting new playback
    loop_module.reset_stop()
    
    # If string/Path, load file first
    if isinstance(audio_data, (str, Path)):
        audio_array, sr = load_audio(audio_data)

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
    pause_module.set_playback_state(audio_data, samplerate)

    # If looping, start loop thread
    if loop_module.is_looping():
        loop_module.start_loop(audio_data, samplerate, _play_array)
        return
    
    return _play_array(audio_data, samplerate, blocking)


def _play_array(audio_data, samplerate, blocking):
    """Internal: play numpy array"""
    # Check if stopped
    if loop_module.is_stopped():
        return
    # Apply main gain from gain module
    from . import gain as gain_module
    audio_data = gain_module.adjust_gain_level(audio_data)

    # Determine channels
    n_channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
    
    # Convert to mono if needed
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Convert to PCM
    sampwidth, audio_data_int = to_pcm(audio_data)
    
    # Platform-specific playback
    if sys.platform == 'win32':
        _play_windows(audio_data_int, samplerate, sampwidth, n_channels, blocking)
    elif sys.platform == 'darwin':
        _play_macos(audio_data_int, samplerate, sampwidth, n_channels, blocking)
    else:  # Linux
        _play_linux(audio_data_int, samplerate, sampwidth, n_channels, blocking)


def _play_windows(audio_data, samplerate, sampwidth, n_channels, blocking):
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
                wav.writeframes(int32_to_24bit_bytes(audio_data))
            else:
                wav.writeframes(audio_data.tobytes())
    
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
                time.sleep(len(audio_data) / samplerate + 0.5)
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


def _play_macos(audio_data, samplerate, sampwidth, n_channels, blocking):
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
                wav.writeframes(int32_to_24bit_bytes(audio_data))
            else:
                wav.writeframes(audio_data.tobytes())
    
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
                time.sleep(len(audio_data) / samplerate + 0.5)
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


def _play_linux(audio_data, samplerate, sampwidth, n_channels, blocking):
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
                wav.writeframes(int32_to_24bit_bytes(audio_data))
            else:
                wav.writeframes(audio_data.tobytes())
    
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
                time.sleep(len(audio_data) / samplerate + 0.5)
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