"""Tests for the Timer class in observability metrics."""

import asyncio
import pytest
from unittest.mock import MagicMock
from vision_agents.core.observability.metrics import Timer


@pytest.fixture
def mock_histogram():
    """Create a mock histogram for testing."""
    return MagicMock()


class TestTimerContextManager:
    """Tests for Timer used as a context manager."""

    def test_context_manager_records_timing(self, mock_histogram):
        """Test that Timer records elapsed time when used as context manager."""
        with Timer(mock_histogram) as timer:
            pass  # Do nothing, just measure overhead

        # Verify record was called
        mock_histogram.record.assert_called_once()
        call_args = mock_histogram.record.call_args

        # First argument should be elapsed time in ms
        elapsed_ms = call_args[0][0]
        assert isinstance(elapsed_ms, float)
        assert elapsed_ms >= 0

        # Should have recorded the elapsed time
        assert timer.last_elapsed_ms is not None
        assert timer.last_elapsed_ms >= 0

    def test_context_manager_with_base_attributes(self, mock_histogram):
        """Test that base attributes are included in recording."""
        base_attrs = {"provider": "test", "version": "1.0"}

        with Timer(mock_histogram, base_attrs):
            pass

        # Verify attributes were passed
        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]

        assert "provider" in recorded_attrs
        assert recorded_attrs["provider"] == "test"
        assert "version" in recorded_attrs
        assert recorded_attrs["version"] == "1.0"

    def test_context_manager_with_dynamic_attributes(self, mock_histogram):
        """Test that dynamic attributes can be added during execution."""
        with Timer(mock_histogram, {"base": "value"}) as timer:
            timer.attributes["dynamic"] = "added"
            timer.attributes["count"] = 42

        # Verify both base and dynamic attributes were recorded
        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]

        assert recorded_attrs["base"] == "value"
        assert recorded_attrs["dynamic"] == "added"
        assert recorded_attrs["count"] == 42

    def test_context_manager_exception_tracking(self, mock_histogram):
        """Test that exceptions are tracked in attributes."""
        try:
            with Timer(mock_histogram, record_exceptions=True):
                raise ValueError("test error")
        except ValueError:
            pass

        # Verify exception was recorded
        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]

        assert recorded_attrs["exception"] == "true"
        assert recorded_attrs["exception_type"] == "ValueError"

    def test_context_manager_no_exception(self, mock_histogram):
        """Test that no exception is recorded when code succeeds."""
        with Timer(mock_histogram, record_exceptions=True):
            pass

        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]

        assert recorded_attrs["exception"] == "false"
        assert "exception_type" not in recorded_attrs

    def test_direct_call_pattern(self, mock_histogram):
        """Test Timer used with direct call pattern."""
        timer = Timer(mock_histogram, {"base": "attr"})

        # Simulate some work
        import time

        time.sleep(0.01)

        # Call with extra attributes
        elapsed = timer({"phase": "init"})

        # Verify recording
        assert elapsed > 0
        mock_histogram.record.assert_called_once()

        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]

        assert recorded_attrs["base"] == "attr"
        assert recorded_attrs["phase"] == "init"

    def test_stop_is_idempotent(self, mock_histogram):
        """Test that calling stop multiple times only records once."""
        timer = Timer(mock_histogram)

        timer.stop()
        timer.stop()
        timer.stop()

        # Should only be called once
        assert mock_histogram.record.call_count == 1


class TestTimerDecorator:
    """Tests for Timer used as a decorator."""

    def test_sync_function_decorator(self, mock_histogram):
        """Test decorating a synchronous function."""

        @Timer(mock_histogram, {"func": "test"})
        def my_function(x, y):
            return x + y

        result = my_function(2, 3)

        assert result == 5
        mock_histogram.record.assert_called_once()

        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]
        assert recorded_attrs["func"] == "test"

    async def test_async_function_decorator(self, mock_histogram):
        """Test decorating an async function."""

        @Timer(mock_histogram, {"func": "async_test"})
        async def my_async_function(x):
            await asyncio.sleep(0.01)
            return x * 2

        result = await my_async_function(5)

        assert result == 10
        mock_histogram.record.assert_called_once()

        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]
        assert recorded_attrs["func"] == "async_test"

    def test_method_decorator_adds_class_name(self, mock_histogram):
        """Test that decorating a method automatically adds class name."""

        class MyClass:
            @Timer(mock_histogram, {"method": "process"})
            def process(self):
                return "processed"

        instance = MyClass()
        result = instance.process()

        assert result == "processed"
        mock_histogram.record.assert_called_once()

        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]

        # Should automatically add fully qualified class path
        assert "class" in recorded_attrs
        # Check it ends with the class name (module path will vary)
        assert recorded_attrs["class"].endswith(".MyClass")
        assert recorded_attrs["method"] == "process"

    async def test_async_method_decorator_adds_class_name(self, mock_histogram):
        """Test that decorating an async method adds class name."""

        class MyAsyncClass:
            @Timer(mock_histogram)
            async def async_process(self):
                await asyncio.sleep(0.01)
                return "async_processed"

        instance = MyAsyncClass()
        result = await instance.async_process()

        assert result == "async_processed"
        mock_histogram.record.assert_called_once()

        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]

        assert "class" in recorded_attrs
        assert recorded_attrs["class"].endswith(".MyAsyncClass")


class TestTimerInheritedMethods:
    """Tests for Timer with inherited methods - the bug fix."""

    def test_inherited_method_reports_subclass_name(self, mock_histogram):
        """Test that inherited methods report the actual subclass name."""

        class BaseClass:
            @Timer(mock_histogram)
            def process(self):
                return "processed"

        class SubClassA(BaseClass):
            pass

        class SubClassB(BaseClass):
            pass

        # Test SubClassA
        instance_a = SubClassA()
        instance_a.process()

        # Test SubClassB
        instance_b = SubClassB()
        instance_b.process()

        # Should have been called twice
        assert mock_histogram.record.call_count == 2

        # Check first call (SubClassA)
        first_call = mock_histogram.record.call_args_list[0]
        first_attrs = first_call[1]["attributes"]
        assert first_attrs["class"].endswith(".SubClassA")

        # Check second call (SubClassB)
        second_call = mock_histogram.record.call_args_list[1]
        second_attrs = second_call[1]["attributes"]
        assert second_attrs["class"].endswith(".SubClassB")

    async def test_inherited_async_method_reports_subclass_name(self, mock_histogram):
        """Test that inherited async methods report the actual subclass name."""

        class AsyncBaseClass:
            @Timer(mock_histogram)
            async def process(self):
                await asyncio.sleep(0.01)
                return "processed"

        class AsyncSubClass(AsyncBaseClass):
            pass

        instance = AsyncSubClass()
        await instance.process()

        mock_histogram.record.assert_called_once()

        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]

        # Should report the subclass path, not the base class
        assert recorded_attrs["class"].endswith(".AsyncSubClass")

    def test_deeply_nested_inheritance(self, mock_histogram):
        """Test that deep inheritance chains still report the correct class."""

        class GrandParent:
            @Timer(mock_histogram)
            def process(self):
                return "processed"

        class Parent(GrandParent):
            pass

        class Child(Parent):
            pass

        instance = Child()
        instance.process()

        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]

        # Should report the most specific class path
        assert recorded_attrs["class"].endswith(".Child")


class TestTimerUnits:
    """Tests for Timer unit conversions."""

    def test_millisecond_unit_default(self, mock_histogram):
        """Test that default unit is milliseconds."""
        with Timer(mock_histogram):
            pass

        call_args = mock_histogram.record.call_args
        elapsed = call_args[0][0]

        # Value should be in milliseconds (small positive number)
        assert elapsed >= 0
        assert elapsed < 1000  # Should be less than 1 second for this test

    def test_second_unit_conversion(self, mock_histogram):
        """Test that seconds unit converts correctly."""
        with Timer(mock_histogram, unit="s"):
            import time

            time.sleep(0.01)  # Sleep 10ms

        call_args = mock_histogram.record.call_args
        elapsed_seconds = call_args[0][0]

        # Should be approximately 0.01 seconds
        assert 0.005 < elapsed_seconds < 0.05


class TestTimerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_timer_without_stop_in_context_manager(self, mock_histogram):
        """Test that __exit__ always calls stop."""
        with Timer(mock_histogram) as timer:
            # Don't call stop manually
            pass

        # Should have been called by __exit__
        mock_histogram.record.assert_called_once()
        assert timer.last_elapsed_ms is not None

    def test_restart_clears_attributes(self, mock_histogram):
        """Test that restart clears dynamic attributes."""
        timer = Timer(mock_histogram)

        # First use
        timer.attributes["first"] = "value1"
        timer.stop()

        # Restart and use again
        timer._restart()
        timer.attributes["second"] = "value2"
        timer.stop({"extra": "attr"})

        # Second call should only have "second" and "extra", not "first"
        second_call = mock_histogram.record.call_args_list[1]
        second_attrs = second_call[1]["attributes"]

        assert "second" in second_attrs
        assert "extra" in second_attrs
        assert "first" not in second_attrs

    def test_elapsed_ms_while_running(self, mock_histogram):
        """Test that elapsed_ms can be called while timer is running."""
        with Timer(mock_histogram) as timer:
            import time

            time.sleep(0.01)
            elapsed = timer.elapsed_ms()
            assert elapsed > 0

        # Final elapsed should be >= interim elapsed
        assert timer.last_elapsed_ms >= elapsed

    def test_callable_check_in_call(self, mock_histogram):
        """Test that __call__ with callable argument triggers decoration."""

        def my_func():
            return 42

        timer = Timer(mock_histogram)
        decorated = timer(my_func)

        # Should return a wrapped function
        assert callable(decorated)
        assert decorated() == 42
        mock_histogram.record.assert_called_once()


class TestTimerRealWorldScenarios:
    """Tests simulating real-world usage patterns."""

    async def test_stt_pattern(self, mock_histogram):
        """Test the pattern used in STT base class."""

        class STT:
            async def process_audio(self, audio_data):
                with Timer(mock_histogram) as timer:
                    timer.attributes["provider"] = self.__class__.__name__
                    timer.attributes["samples"] = len(audio_data)

                    # Simulate processing
                    await asyncio.sleep(0.01)

        class DeepgramSTT(STT):
            pass

        stt = DeepgramSTT()
        await stt.process_audio([1, 2, 3, 4, 5])

        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]

        assert recorded_attrs["provider"] == "DeepgramSTT"
        assert recorded_attrs["samples"] == 5

    def test_turn_detection_pattern(self, mock_histogram):
        """Test the pattern used in TurnDetector base class."""

        class TurnDetector:
            @Timer(mock_histogram)
            async def process_audio(self, audio_data):
                await asyncio.sleep(0.01)
                return "turn_detected"

        class SmartTurnDetection(TurnDetector):
            pass

        detector = SmartTurnDetection()
        result = asyncio.run(detector.process_audio([1, 2, 3]))

        assert result == "turn_detected"

        call_args = mock_histogram.record.call_args
        recorded_attrs = call_args[1]["attributes"]

        # Should report the actual implementation class path
        assert recorded_attrs["class"].endswith(".SmartTurnDetection")

    def test_multiple_nested_timers(self, mock_histogram):
        """Test that nested timers work independently."""
        with Timer(mock_histogram, {"outer": "timer"}):
            with Timer(mock_histogram, {"inner": "timer"}):
                pass

        # Both should have recorded
        assert mock_histogram.record.call_count == 2

        # Check both calls had different attributes
        first_call_attrs = mock_histogram.record.call_args_list[0][1]["attributes"]
        second_call_attrs = mock_histogram.record.call_args_list[1][1]["attributes"]

        assert first_call_attrs["inner"] == "timer"
        assert second_call_attrs["outer"] == "timer"
