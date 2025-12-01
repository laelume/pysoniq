"""Pause/resume playback"""

import numpy as np
import time

# Module state
_paused = False
_pause_position = 0.0
_pause_audio = None
_pause_samplerate = None
_pause_start_time = None
_loop_was_enabled = False  # Track if loop was on when paused


def pause():
    """Pause current playback"""
    global _paused, _pause_start_time, _pause_position, _loop_was_enabled
    
    if not _paused:
        _paused = True
        
        # Save loop state before stopping
        from . import loop as loop_module
        _loop_was_enabled = loop_module.is_looping()
        
        # Estimate current position
        if _pause_start_time is not None:
            elapsed = time.time() - _pause_start_time
            _pause_position += elapsed
        
        # Stop playback (this will disable loop)
        from .stop import stop as stop_func
        stop_func()
        
        print(f"Paused at {_pause_position:.2f}s (loop was {_loop_was_enabled})")


def resume():
    """Resume playback from pause point"""
    global _paused, _pause_position, _pause_start_time, _loop_was_enabled
    
    if _paused and _pause_audio is not None:
        _paused = False
        
        # Restore loop state
        from . import loop as loop_module
        if _loop_was_enabled:
            loop_module.set_loop(True)
        
        # Calculate where to resume from
        sample_position = int(_pause_position * _pause_samplerate)
        
        # Trim audio to resume point
        if sample_position < len(_pause_audio):
            remaining_audio = _pause_audio[sample_position:]
            
            # Play remaining audio
            from . import play as play_module
            _pause_start_time = time.time()
            play_module.play(remaining_audio, _pause_samplerate)
            
            print(f"Resumed from {_pause_position:.2f}s (loop={_loop_was_enabled})")
        else:
            # Already at end
            reset()


def is_paused():
    """Check if paused"""
    return _paused


def was_looping():
    """Check if loop was enabled when paused"""
    return _loop_was_enabled


def set_playback_state(audio, samplerate):
    """Store current playback state for pause/resume"""
    global _pause_audio, _pause_samplerate, _pause_position, _pause_start_time
    _pause_audio = audio.copy() if audio is not None else None
    _pause_samplerate = samplerate
    _pause_position = 0.0
    _pause_start_time = time.time()


def reset():
    """Reset pause state"""
    global _paused, _pause_position, _pause_audio, _pause_samplerate, _pause_start_time, _loop_was_enabled
    _paused = False
    _pause_position = 0.0
    _pause_audio = None
    _pause_samplerate = None
    _pause_start_time = None
    _loop_was_enabled = False