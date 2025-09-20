"""
Timing utilities for performance monitoring and latency tracking.
"""

import time
import logging
from functools import wraps
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class TimingData:
    """Data structure for timing measurements."""
    start_time: float
    end_time: float
    duration: float
    method_name: str
    frame_id: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None

class TimingMetrics:
    """Collects and analyzes timing metrics."""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.timings: deque = deque(maxlen=max_samples)
        self.frame_timings: Dict[str, TimingData] = {}
        self.total_calls = 0
        self.total_duration = 0.0
    
    def add_timing(self, timing_data: TimingData):
        """Add a timing measurement."""
        self.timings.append(timing_data)
        self.total_calls += 1
        self.total_duration += timing_data.duration
        
        if timing_data.frame_id:
            self.frame_timings[timing_data.frame_id] = timing_data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get timing statistics."""
        if not self.timings:
            return {"error": "No timing data available"}
        
        durations = [t.duration for t in self.timings]
        durations.sort()
        
        return {
            "total_calls": self.total_calls,
            "total_duration": self.total_duration,
            "average_duration": self.total_duration / self.total_calls,
            "min_duration": min(durations),
            "max_duration": max(durations),
            "median_duration": durations[len(durations) // 2],
            "p95_duration": durations[int(len(durations) * 0.95)],
            "p99_duration": durations[int(len(durations) * 0.99)],
            "recent_samples": len(self.timings)
        }
    
    def log_performance_warning(self, method_name: str, duration: float, threshold: float = 0.1):
        """Log performance warnings for slow operations."""
        if duration > threshold:
            logger.warning(f"🐌 SLOW OPERATION: {method_name} took {duration:.3f}s (threshold: {threshold:.3f}s)")

def timing_decorator(method_name: Optional[str] = None, threshold: float = 0.1, log_level: str = "debug"):
    """Decorator to add timing to methods."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.monotonic()
            name = method_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = await func(*args, **kwargs)
                end_time = time.monotonic()
                duration = end_time - start_time
                
                # Create timing data
                timing_data = TimingData(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    method_name=name
                )
                
                # Log timing based on level
                if log_level == "info":
                    logger.info(f"⏱️ TIMING: {name} took {duration:.3f}s")
                elif log_level == "warning" and duration > threshold:
                    logger.warning(f"⏱️ SLOW: {name} took {duration:.3f}s (threshold: {threshold:.3f}s)")
                else:
                    logger.debug(f"⏱️ TIMING: {name} took {duration:.3f}s")
                
                return result
                
            except Exception as e:
                end_time = time.monotonic()
                duration = end_time - start_time
                logger.error(f"⏱️ ERROR: {name} failed after {duration:.3f}s: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.monotonic()
            name = method_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                end_time = time.monotonic()
                duration = end_time - start_time
                
                # Log timing based on level
                if log_level == "info":
                    logger.info(f"⏱️ TIMING: {name} took {duration:.3f}s")
                elif log_level == "warning" and duration > threshold:
                    logger.warning(f"⏱️ SLOW: {name} took {duration:.3f}s (threshold: {threshold:.3f}s)")
                else:
                    logger.debug(f"⏱️ TIMING: {name} took {duration:.3f}s")
                
                return result
                
            except Exception as e:
                end_time = time.monotonic()
                duration = end_time - start_time
                logger.error(f"⏱️ ERROR: {name} failed after {duration:.3f}s: {e}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def frame_timing_decorator(frame_id_key: str = "frame_id", threshold: float = 0.05):
    """Specialized decorator for frame processing timing."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.monotonic()
            name = f"{func.__module__}.{func.__name__}"
            
            # Extract frame ID if available
            frame_id = None
            if args and hasattr(args[0], frame_id_key):
                frame_id = getattr(args[0], frame_id_key)
            
            try:
                result = await func(*args, **kwargs)
                end_time = time.monotonic()
                duration = end_time - start_time
                
                # Log frame timing with more detail
                if duration > threshold:
                    logger.warning(f"🎥 SLOW FRAME: {name} frame_id={frame_id} took {duration:.3f}s (threshold: {threshold:.3f}s)")
                else:
                    logger.debug(f"🎥 FRAME TIMING: {name} frame_id={frame_id} took {duration:.3f}s")
                
                return result
                
            except Exception as e:
                end_time = time.monotonic()
                duration = end_time - start_time
                logger.error(f"🎥 FRAME ERROR: {name} frame_id={frame_id} failed after {duration:.3f}s: {e}")
                raise
        
        return async_wrapper
    return decorator

# Global timing metrics instance
global_timing_metrics = TimingMetrics()

def get_global_timing_stats() -> Dict[str, Any]:
    """Get global timing statistics."""
    return global_timing_metrics.get_stats()

def log_performance_summary():
    """Log a performance summary."""
    stats = get_global_timing_stats()
    if "error" not in stats:
        logger.info(f"📊 PERFORMANCE SUMMARY: {stats['total_calls']} calls, "
                   f"avg: {stats['average_duration']:.3f}s, "
                   f"p95: {stats['p95_duration']:.3f}s, "
                   f"max: {stats['max_duration']:.3f}s")