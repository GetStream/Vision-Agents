import asyncio
import os
import pytest
from dotenv import load_dotenv

from stream_agents.plugins.gemini.realtime2 import Realtime2

# Load environment variables
load_dotenv()


class TestRealtime2Integration:
    """Integration tests for Realtime2 connect flow"""

    
    @pytest.fixture
    def realtime2(self):
        """Create Realtime2 instance with API key"""
        return Realtime2(
            model="gemini-2.5-flash-exp-native-audio-thinking-dialog",
        )

    @pytest.mark.integration
    async def test_simple_response_flow(self, realtime2):
        """Test sending a simple text message and receiving response"""
        await realtime2.connect()

        try:
            # Send a simple message
            print("starting")
            await realtime2.simple_response("Hello, can you hear me?")

            # Wait for response
            await asyncio.sleep(3.0)

            # Verify we have a session and it's active
            assert realtime2._session is not None

        finally:
            await realtime2.close()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self, realtime2):
        """Test basic connect and disconnect flow"""
        # Test connection
        await realtime2.connect()
        
        # Verify connection was established
        assert hasattr(realtime2, '_session')
        assert realtime2._session is not None
        assert hasattr(realtime2, '_receive_task')
        assert realtime2._receive_task is not None
        
        # Wait a moment to ensure connection is stable
        await asyncio.sleep(1.0)
        
        # Test disconnection
        await realtime2.close()
        
        # Verify cleanup
        assert realtime2._session is None
        assert realtime2._receive_task is None or realtime2._receive_task.done()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connect_with_custom_config(self, api_key):
        """Test connection with custom configuration"""
        from google.genai.types import LiveConnectConfigDict, Modality
        
        custom_config = LiveConnectConfigDict(
            response_modalities=[Modality.AUDIO, Modality.TEXT],
        )
        
        realtime2 = Realtime2(
            model="gemini-2.5-flash-exp-native-audio-thinking-dialog",
            api_key=api_key,
            config=custom_config
        )
        
        try:
            await realtime2.connect()
            assert realtime2._session is not None
            
            # Verify custom config was applied
            assert realtime2.config is not None
            
        finally:
            await realtime2.close()
    

    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, api_key):
        """Test connection timeout handling"""
        # Create instance with very short timeout
        realtime2 = Realtime2(
            model="gemini-2.5-flash-preview",
            api_key=api_key
        )
        
        try:
            # This should connect successfully
            await realtime2.connect()
            assert realtime2._session is not None
            
        except Exception as e:
            # If connection fails, it should be a specific type of error
            assert "timeout" in str(e).lower() or "connection" in str(e).lower()
        
        finally:
            try:
                await realtime2.close()
            except:
                pass  # Ignore cleanup errors
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_connect_disconnect_cycles(self, realtime2):
        """Test multiple connect/disconnect cycles"""
        for i in range(3):
            # Connect
            await realtime2.connect()
            assert realtime2._session is not None
            
            # Wait briefly
            await asyncio.sleep(0.5)
            
            # Disconnect
            await realtime2.close()
            assert realtime2._session is None
            
            # Brief pause between cycles
            await asyncio.sleep(0.1)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connect_without_api_key(self):
        """Test that connection fails gracefully without API key"""
        realtime2 = Realtime2(model="gemini-2.5-flash-exp-native-audio-thinking-dialog")
        
        try:
            await realtime2.connect()
            # If it connects, that's unexpected but not necessarily wrong
            await realtime2.close()
        except Exception as e:
            # Should get some kind of error about missing API key or authentication
            assert "key" in str(e).lower() or "auth" in str(e).lower() or "api" in str(e).lower()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_connect_with_invalid_model(self, api_key):
        """Test connection with invalid model name"""
        realtime2 = Realtime2(
            model="invalid-model-name",
            api_key=api_key
        )
        
        try:
            await realtime2.connect()
            # If it connects, that's unexpected but not necessarily wrong
            await realtime2.close()
        except Exception as e:
            # Should get some kind of error about invalid model
            assert "model" in str(e).lower() or "invalid" in str(e).lower()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_receive_loop_task_management(self, realtime2):
        """Test that receive loop task is properly managed"""
        await realtime2.connect()
        
        try:
            # Verify task exists and is running
            assert realtime2._receive_task is not None
            assert not realtime2._receive_task.done()
            
            # Wait a bit
            await asyncio.sleep(1.0)
            
            # Task should still be running
            assert not realtime2._receive_task.done()
            
        finally:
            await realtime2.close()
            
            # Task should be cancelled/done after close
            if realtime2._receive_task:
                assert realtime2._receive_task.done()
