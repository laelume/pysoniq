"""Stop playback functionality"""

import sys
from . import loop as loop_module


def stop():
    """Stop audio playback (preserves loop state)"""
    # Stop playback but preserve loop setting
    loop_module.stop()
    
    # Platform-specific immediate stop
    if sys.platform == 'win32':
        _stop_windows()
    else:
        _stop_unix()


def _stop_windows():
    """Stop Windows playback"""
    try:
        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)
    except:
        pass


def _stop_unix():
    """Stop macOS/Linux playback"""
    from . import play
    if hasattr(play, '_current_process') and play._current_process:
        try:
            play._current_process.terminate()
            play._current_process = None
        except:
            pass