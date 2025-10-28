"""Video frame utilities."""

import av


def ensure_even_dimensions(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Ensure frame has even dimensions for H.264 yuv420p encoding.
    
    Crops by 1 pixel if width or height is odd.
    """
    needs_width_adjust = frame.width % 2 != 0
    needs_height_adjust = frame.height % 2 != 0
    
    if not needs_width_adjust and not needs_height_adjust:
        return frame
    
    new_width = frame.width - (1 if needs_width_adjust else 0)
    new_height = frame.height - (1 if needs_height_adjust else 0)
    
    cropped = frame.reformat(width=new_width, height=new_height)
    cropped.pts = frame.pts
    if frame.time_base is not None:
        cropped.time_base = frame.time_base
    
    return cropped

