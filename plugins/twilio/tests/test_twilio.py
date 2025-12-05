import pytest
from datetime import datetime
from dotenv import load_dotenv

from vision_agents.plugins import twilio


load_dotenv()


class TestTwilioPlugin:
    def test_import(self):
        """Test that the plugin can be imported."""
        assert twilio.TwilioCall is not None
        assert twilio.TwilioCallRegistry is not None
        assert twilio.TwilioMediaStream is not None


class TestTwilioCall:
    def test_create_call(self):
        """Test creating a TwilioCall."""
        call = twilio.TwilioCall(
            call_sid="CA123456",
            form_data={
                "From": "+1234567890",
                "To": "+0987654321",
                "CallStatus": "ringing",
                "CallerCity": "New York",
            }
        )
        
        assert call.call_sid == "CA123456"
        assert call.from_number == "+1234567890"
        assert call.to_number == "+0987654321"
        assert call.call_status == "ringing"
        assert call.ended_at is None
    
    def test_end_call(self):
        """Test ending a call."""
        call = twilio.TwilioCall(call_sid="CA123456")
        assert call.ended_at is None
        
        call.end()
        assert call.ended_at is not None
        assert isinstance(call.ended_at, datetime)


class TestTwilioCallRegistry:
    def test_create_and_get(self):
        """Test creating and retrieving calls."""
        registry = twilio.TwilioCallRegistry()
        
        call = registry.create("CA123", {"From": "+123"})
        assert call.call_sid == "CA123"
        
        retrieved = registry.get("CA123")
        assert retrieved is call
    
    def test_get_unknown(self):
        """Test getting unknown call returns None."""
        registry = twilio.TwilioCallRegistry()
        assert registry.get("unknown") is None
    
    def test_remove(self):
        """Test removing a call."""
        registry = twilio.TwilioCallRegistry()
        registry.create("CA123", {})
        
        removed = registry.remove("CA123")
        assert removed is not None
        assert removed.ended_at is not None
        assert registry.get("CA123") is None
    
    def test_list_active(self):
        """Test listing active calls."""
        registry = twilio.TwilioCallRegistry()
        registry.create("CA1", {})
        registry.create("CA2", {})
        registry.create("CA3", {})
        
        # End one call
        registry.remove("CA2")
        
        active = registry.list_active()
        assert len(active) == 2
        assert all(c.ended_at is None for c in active)


class TestAudioConversion:
    def test_mulaw_to_pcm(self):
        """Test mulaw to PCM conversion."""
        # Create some test mulaw bytes
        import numpy as np
        mulaw_bytes = bytes([0xFF, 0x7F, 0x00, 0x80])  # Various mulaw values
        
        pcm = twilio.mulaw_to_pcm(mulaw_bytes)
        
        assert pcm.sample_rate == 8000
        assert pcm.channels == 1
        assert len(pcm.samples) == 4
    
    def test_pcm_to_mulaw(self):
        """Test PCM to mulaw conversion."""
        from getstream.video.rtc.track_util import PcmData, AudioFormat
        import numpy as np
        
        # Create test PCM data
        samples = np.array([0, 1000, -1000, 16000], dtype=np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=8000,
            channels=1,
            format=AudioFormat.S16,
        )
        
        mulaw_bytes = twilio.pcm_to_mulaw(pcm)
        
        assert isinstance(mulaw_bytes, bytes)
        assert len(mulaw_bytes) == 4


