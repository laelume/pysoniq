"""Loop control for audio playback"""

import threading
import time

# Module-level state
_loop_enabled = False
_stop_requested = False
_playback_thread = None
_current_audio = None
_current_samplerate = None
_play_function = None


def set_loop(enabled):
    """Enable or disable looping"""
    global _loop_enabled
    _loop_enabled = enabled


def is_looping():
    """Check if looping is enabled"""
    return _loop_enabled


def stop():
    """Stop playback but preserve loop state"""
    global _stop_requested
    # Note: we do NOT touch _loop_enabled here
    _stop_requested = True
    
    print(f"DEBUG: loop.stop() called, loop_enabled={_loop_enabled} (preserved)")

# def stop():
#     """Stop playback and looping"""
#     global _stop_requested, _loop_enabled
    
#     print(f"DEBUG: loop.stop() called from loop.py, was loop_enabled={_loop_enabled}")
    
#     _stop_requested = True
#     _loop_enabled = False
    
#     # Platform-specific stop
#     import sys
#     if sys.platform == 'win32':
#         print("DEBUG: Calling winsound.PlaySound(None, SND_PURGE)")
#         try:
#             import winsound
#             winsound.PlaySound(None, winsound.SND_PURGE)
#             print("DEBUG: winsound stop completed")
#         except Exception as e:
#             print(f"DEBUG: winsound stop failed: {e}")
#     else:
#         # Kill subprocess for macOS/Linux
#         from . import playback
#         if hasattr(playback, '_current_process') and playback._current_process:
#             try:
#                 print(f"DEBUG: Terminating process {playback._current_process.pid}")
#                 playback._current_process.terminate()
#                 playback._current_process = None
#                 print("DEBUG: Process terminated")
#             except Exception as e:
#                 print(f"DEBUG: Process terminate failed: {e}")
    
#     print(f"DEBUG: loop.stop() completed, stop_requested={_stop_requested}, loop_enabled={_loop_enabled}")


def is_stopped():
    """Check if stop was requested"""
    return _stop_requested


def reset_stop():
    """Reset stop flag"""
    global _stop_requested
    _stop_requested = False


def start_loop(audio_data, samplerate, play_func):
    """Start looping playback in background thread"""
    global _playback_thread, _current_audio, _current_samplerate, _play_function
    
    print(f"DEBUG: start_loop called, loop_enabled={_loop_enabled}")
    
    _current_audio = audio_data
    _current_samplerate = samplerate
    _play_function = play_func
    
    # Only stop existing playback thread if one is running
    if _playback_thread is not None and _playback_thread.is_alive():
        global _stop_requested
        _stop_requested = True
        time.sleep(0.1)
        # Don't call stop() - just set the flag
    
    # Reset stop flag for new playback
    _stop_requested = False
    
    print(f"DEBUG: Starting thread, loop_enabled={_loop_enabled}")
    
    # Start loop thread
    _playback_thread = threading.Thread(target=_loop_worker, daemon=True)
    _playback_thread.start()


def _loop_worker():
    """Worker thread for looping playback"""
    global _loop_enabled, _stop_requested
    
    print(f"DEBUG: Loop worker started, loop_enabled={_loop_enabled}")  # Add debug
    
    while _loop_enabled and not _stop_requested:
        try:
            # Play audio (blocking) - call the function directly, don't go through play()
            _play_function(_current_audio, _current_samplerate, blocking=True)
            
            print(f"DEBUG: Playback finished, loop_enabled={_loop_enabled}, stop={_stop_requested}")  # Add debug
            
            # Check if we should continue looping
            if not _loop_enabled or _stop_requested:
                break
                
        except Exception as e:
            print(f"Loop playback error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("DEBUG: Loop worker exiting")  # Add debug
    # Cleanup
    #_loop_enabled = False