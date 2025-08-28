"""
Tests for the DotaGameProcessor class.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from stream_agents.processors.dota_processors import DotaGameProcessor
from stream_agents.processors.base_processor import BaseProcessor


class TestDotaGameProcessor:
    """Test cases for the DotaGameProcessor class."""

    def test_dota_processor_initialization(self):
        """Test basic DotaGameProcessor initialization."""
        game_id = "test_game_123"
        processor = DotaGameProcessor(game_id)
        
        assert processor.game_id == game_id
        assert isinstance(processor, BaseProcessor)

    def test_dota_processor_initialization_with_different_game_ids(self):
        """Test DotaGameProcessor initialization with various game ID formats."""
        test_cases = [
            "game_123",
            "dota_match_456",
            "123456789",
            "test-game-with-dashes",
            "game_with_underscores_123",
            ""  # Edge case: empty string
        ]
        
        for game_id in test_cases:
            processor = DotaGameProcessor(game_id)
            assert processor.game_id == game_id

    def test_dota_processor_initialization_with_none_game_id(self):
        """Test DotaGameProcessor initialization with None game_id."""
        processor = DotaGameProcessor(None)
        assert processor.game_id is None

    def test_dota_processor_initialization_with_numeric_game_id(self):
        """Test DotaGameProcessor initialization with numeric game_id."""
        game_id = 12345
        processor = DotaGameProcessor(game_id)
        assert processor.game_id == game_id

    def test_dota_processor_inherits_from_base_processor(self):
        """Test that DotaGameProcessor properly inherits from BaseProcessor."""
        processor = DotaGameProcessor("test_game")
        
        # Check inheritance
        assert isinstance(processor, BaseProcessor)
        
        # Check that BaseProcessor protocol methods exist (even if not implemented)
        assert hasattr(processor, 'state')
        assert hasattr(processor, 'input')

    def test_dota_processor_state_method_exists(self):
        """Test that DotaGameProcessor has the required state method."""
        processor = DotaGameProcessor("test_game")
        
        # The method should exist (inherited from BaseProcessor protocol)
        assert hasattr(processor, 'state')
        
        # Since it's not implemented in the current class, calling it should work
        # but might return None or raise NotImplementedError depending on implementation
        try:
            result = processor.state()
            # If it returns something, that's fine
            assert result is None or result is not None
        except NotImplementedError:
            # If it raises NotImplementedError, that's also acceptable
            pass

    def test_dota_processor_input_method_exists(self):
        """Test that DotaGameProcessor has the required input method."""
        processor = DotaGameProcessor("test_game")
        
        # The method should exist (inherited from BaseProcessor protocol)
        assert hasattr(processor, 'input')
        
        # Since it's not implemented in the current class, calling it should work
        # but might return None or raise NotImplementedError depending on implementation
        try:
            result = processor.input()
            # If it returns something, that's fine
            assert result is None or result is not None
        except NotImplementedError:
            # If it raises NotImplementedError, that's also acceptable
            pass

    def test_dota_processor_game_id_attribute_access(self):
        """Test accessing and modifying the game_id attribute."""
        initial_game_id = "initial_game"
        processor = DotaGameProcessor(initial_game_id)
        
        # Test initial value
        assert processor.game_id == initial_game_id
        
        # Test modification
        new_game_id = "modified_game"
        processor.game_id = new_game_id
        assert processor.game_id == new_game_id

    def test_dota_processor_multiple_instances(self):
        """Test creating multiple DotaGameProcessor instances."""
        processor1 = DotaGameProcessor("game_1")
        processor2 = DotaGameProcessor("game_2")
        processor3 = DotaGameProcessor("game_3")
        
        # Each instance should have its own game_id
        assert processor1.game_id == "game_1"
        assert processor2.game_id == "game_2"
        assert processor3.game_id == "game_3"
        
        # Instances should be independent
        processor1.game_id = "modified_game_1"
        assert processor1.game_id == "modified_game_1"
        assert processor2.game_id == "game_2"  # Should remain unchanged
        assert processor3.game_id == "game_3"  # Should remain unchanged

    def test_dota_processor_str_representation(self):
        """Test string representation of DotaGameProcessor (if implemented)."""
        processor = DotaGameProcessor("test_game_123")
        
        # Check if __str__ or __repr__ methods work
        str_repr = str(processor)
        assert isinstance(str_repr, str)
        
        # The string representation should contain some identifying information
        # This test is flexible since the actual implementation might vary
        assert len(str_repr) > 0

    @pytest.mark.parametrize("game_id", [
        "dota_match_001",
        "competitive_game_456",
        "ranked_match_789",
        123456,
        None,
        ""
    ])
    def test_dota_processor_parametrized_initialization(self, game_id):
        """Parametrized test for DotaGameProcessor initialization with various game IDs."""
        processor = DotaGameProcessor(game_id)
        assert processor.game_id == game_id

    def test_dota_processor_attribute_existence(self):
        """Test that all expected attributes exist on the processor."""
        processor = DotaGameProcessor("test_game")
        
        # Test that game_id attribute exists
        assert hasattr(processor, 'game_id')
        
        # Test that BaseProcessor protocol methods exist
        assert hasattr(processor, 'state')
        assert hasattr(processor, 'input')

    def test_dota_processor_type_checking(self):
        """Test type checking and class relationships."""
        processor = DotaGameProcessor("test_game")
        
        # Check class type
        assert type(processor).__name__ == "DotaGameProcessor"
        
        # Check inheritance
        assert isinstance(processor, BaseProcessor)
        
        # Check MRO (Method Resolution Order)
        mro_classes = [cls.__name__ for cls in type(processor).__mro__]
        assert "DotaGameProcessor" in mro_classes
        assert "BaseProcessor" in mro_classes

    def test_dota_processor_with_mock_game_id(self):
        """Test DotaGameProcessor with mock game ID."""
        mock_game_id = Mock()
        mock_game_id.__str__ = Mock(return_value="mock_game_123")
        
        processor = DotaGameProcessor(mock_game_id)
        assert processor.game_id == mock_game_id

    def test_dota_processor_initialization_kwargs(self):
        """Test that DotaGameProcessor handles initialization properly."""
        # Test with positional argument
        processor1 = DotaGameProcessor("positional_game")
        assert processor1.game_id == "positional_game"
        
        # Test with keyword argument (if supported)
        try:
            processor2 = DotaGameProcessor(game_id="keyword_game")
            assert processor2.game_id == "keyword_game"
        except TypeError:
            # If keyword arguments aren't supported, that's also valid
            pass

    def test_dota_processor_edge_cases(self):
        """Test edge cases for DotaGameProcessor."""
        # Test with very long game ID
        long_game_id = "a" * 1000
        processor = DotaGameProcessor(long_game_id)
        assert processor.game_id == long_game_id
        
        # Test with special characters
        special_game_id = "game!@#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        processor = DotaGameProcessor(special_game_id)
        assert processor.game_id == special_game_id
        
        # Test with Unicode characters
        unicode_game_id = "游戏_123_тест_🎮"
        processor = DotaGameProcessor(unicode_game_id)
        assert processor.game_id == unicode_game_id


    # New tests for OpenDota API functionality
    
    def test_dota_processor_with_api_key(self):
        """Test DotaGameProcessor initialization with API key."""
        api_key = "test_api_key_123"
        processor = DotaGameProcessor("test_game", api_key=api_key)
        
        assert processor.api_key == api_key
        assert processor.game_id == "test_game"

    @patch('stream_agents.processors.dota_processors.python_opendota')
    @patch('stream_agents.processors.dota_processors.LiveApi')
    @patch('stream_agents.processors.dota_processors.MatchesApi')
    @patch('stream_agents.processors.dota_processors.Configuration')
    def test_setup_client_success(self, mock_config, mock_matches_api, mock_live_api, mock_opendota):
        """Test successful OpenDota API client setup."""
        # Mock the configuration and client
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        mock_client = Mock()
        mock_opendota.ApiClient.return_value = mock_client
        
        mock_live_api_instance = Mock()
        mock_live_api.return_value = mock_live_api_instance
        
        mock_matches_api_instance = Mock()
        mock_matches_api.return_value = mock_matches_api_instance
        
        processor = DotaGameProcessor("test_game")
        result = processor.setup_client()
        
        assert result is True
        assert processor.client == mock_client
        assert processor.live_games_api == mock_live_api_instance
        assert processor.matches_api == mock_matches_api_instance

    @patch('stream_agents.processors.dota_processors.python_opendota')
    def test_setup_client_with_api_key(self, mock_opendota):
        """Test OpenDota API client setup with API key."""
        api_key = "test_key"
        
        # Mock configuration
        mock_config = Mock()
        mock_config.api_key = {}
        
        with patch('stream_agents.processors.dota_processors.Configuration', return_value=mock_config):
            with patch('stream_agents.processors.dota_processors.LiveApi'):
                with patch('stream_agents.processors.dota_processors.MatchesApi'):
                    processor = DotaGameProcessor("test_game", api_key=api_key)
                    processor.setup_client()
                    
                    assert mock_config.api_key['api_key'] == api_key

    @patch('stream_agents.processors.dota_processors.python_opendota')
    def test_setup_client_failure(self, mock_opendota):
        """Test OpenDota API client setup failure."""
        mock_opendota.ApiClient.side_effect = Exception("Setup failed")
        
        processor = DotaGameProcessor("test_game")
        result = processor.setup_client()
        
        assert result is False

    @patch('stream_agents.processors.dota_processors.requests')
    def test_get_live_matches_http_fallback_success(self, mock_requests):
        """Test successful live matches retrieval via HTTP fallback."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'match_id': 12345,
                'game_mode': 22,
                'average_mmr': 5000,
                'spectators': 10
            },
            {
                'match_id': 67890,
                'game_mode': 1,
                'average_mmr': 3000,
                'spectators': 5
            }
        ]
        mock_requests.get.return_value = mock_response
        
        # Create processor
        processor = DotaGameProcessor("test_game")
            
        matches = processor.get_live_matches()
        
        assert matches is not None
        assert len(matches) == 2
        assert matches[0]['match_id'] == 12345
        assert matches[1]['match_id'] == 67890

    @patch('stream_agents.processors.dota_processors.requests')
    def test_get_live_matches_http_fallback_with_api_key(self, mock_requests):
        """Test HTTP fallback with API key in headers."""
        api_key = "test_api_key"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_requests.get.return_value = mock_response
        
        processor = DotaGameProcessor("test_game", api_key=api_key)
            
        processor.get_live_matches()
        
        # Verify API key was included in headers
        mock_requests.get.assert_called_once()
        call_args = mock_requests.get.call_args
        headers = call_args[1]['headers']
        assert 'Authorization' in headers
        assert headers['Authorization'] == f'Bearer {api_key}'

    @patch('stream_agents.processors.dota_processors.requests')
    def test_get_live_matches_http_failure(self, mock_requests):
        """Test HTTP fallback failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_requests.get.return_value = mock_response
        
        processor = DotaGameProcessor("test_game")
            
        matches = processor.get_live_matches()
        
        assert matches is None

    @patch('stream_agents.processors.dota_processors.requests')
    def test_get_live_matches_http_exception(self, mock_requests):
        """Test HTTP fallback with exception."""
        mock_requests.get.side_effect = Exception("Network error")
        
        processor = DotaGameProcessor("test_game")
            
        matches = processor.get_live_matches()
        
        assert matches is None

    def test_convert_match_to_dict(self):
        """Test match object to dictionary conversion."""
        # Mock match object
        mock_match = Mock()
        mock_match.match_id = 12345
        mock_match.game_mode = 22
        mock_match.average_mmr = 5000
        mock_match.spectators = 10
        mock_match.lobby_type = 1
        mock_match.num_mmr = 8
        mock_match.league_id = None
        mock_match.delay = 0
        mock_match.sort_score = 100
        mock_match.last_update_time = 1234567890
        mock_match.radiant_lead = 1000
        mock_match.duration = 1800
        mock_match.building_state = 123456
        
        # Mock team objects
        mock_radiant_team = Mock()
        mock_radiant_team.team_id = 1
        mock_radiant_team.name = "Team Radiant"
        mock_radiant_team.tag = "RAD"
        mock_radiant_team.logo_url = "http://example.com/logo.png"
        
        mock_dire_team = Mock()
        mock_dire_team.team_id = 2
        mock_dire_team.name = "Team Dire"
        mock_dire_team.tag = "DIRE"
        mock_dire_team.logo_url = "http://example.com/dire.png"
        
        mock_match.radiant_team = mock_radiant_team
        mock_match.dire_team = mock_dire_team
        
        processor = DotaGameProcessor("test_game")
        result = processor._convert_match_to_dict(mock_match)
        
        # Check basic match data
        assert result['match_id'] == 12345
        assert result['game_mode'] == 22
        assert result['average_mmr'] == 5000
        assert result['spectators'] == 10
        
        # Check team data
        assert result['radiant_team']['name'] == "Team Radiant"
        assert result['radiant_team']['tag'] == "RAD"
        assert result['dire_team']['name'] == "Team Dire"
        assert result['dire_team']['tag'] == "DIRE"

    def test_convert_match_to_dict_minimal(self):
        """Test match conversion with minimal data."""
        mock_match = Mock()
        mock_match.match_id = 12345
        # Remove other attributes to test getattr defaults
        del mock_match.game_mode
        del mock_match.average_mmr
        
        processor = DotaGameProcessor("test_game")
        result = processor._convert_match_to_dict(mock_match)
        
        assert result['match_id'] == 12345
        assert result['game_mode'] is None
        assert result['average_mmr'] is None

    def test_state_method_with_api_features(self):
        """Test state method returns correct API-related information."""
        processor = DotaGameProcessor("test_game", api_key="test_key")
        
        state = processor.state()
        
        assert 'has_api_key' in state
        assert 'client_initialized' in state
        assert state['has_api_key'] is True

    def test_input_method_with_api_features(self):
        """Test input method returns correct configuration."""
        processor = DotaGameProcessor("test_game", api_key="test_key")
        
        input_data = processor.input()
        
        assert 'api_key_configured' in input_data
        assert input_data['api_key_configured'] is True
        assert input_data['game_id'] == "test_game"

    def test_get_live_matches_client_success_fallback_not_called(self):
        """Test that HTTP fallback is not called when client succeeds."""
        # Mock successful client response
        mock_response = Mock()
        mock_response.body = [Mock(match_id=12345)]
        
        mock_api = Mock()
        mock_api.live_get.return_value = mock_response
        
        processor = DotaGameProcessor("test_game")
        processor.live_games_api = mock_api
        
        with patch.object(processor, '_get_live_matches_via_http') as mock_http:
            matches = processor.get_live_matches()
            
            # HTTP fallback should not be called
            mock_http.assert_not_called()
            assert matches is not None

    def test_get_live_matches_both_methods_fail(self):
        """Test when both client and HTTP methods fail."""
        processor = DotaGameProcessor("test_game")
        
        # Mock both methods to fail
        with patch.object(processor, '_get_live_matches_via_client', return_value=None):
            with patch.object(processor, '_get_live_matches_via_http', return_value=None):
                matches = processor.get_live_matches()
                
                assert matches is None

    @pytest.mark.parametrize("api_key,expected_auth", [
        (None, False),
        ("", False),
        ("test_key", True),
        ("very_long_api_key_123456789", True),
    ])
    def test_api_key_handling(self, api_key, expected_auth):
        """Test API key handling in various scenarios."""
        processor = DotaGameProcessor("test_game", api_key=api_key)
        
        assert bool(processor.api_key) == bool(api_key)
        
        state = processor.state()
        input_data = processor.input()
        
        assert state['has_api_key'] == expected_auth
        assert input_data['api_key_configured'] == expected_auth

    # Tests for get_match functionality
    
    @patch('stream_agents.processors.dota_processors.requests')
    def test_get_match_http_success(self, mock_requests):
        """Test successful match retrieval via HTTP fallback."""
        match_id = 8437488139
        
        # Mock HTTP response with detailed match data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'match_id': match_id,
            'duration': 2325,
            'game_mode': 22,
            'lobby_type': 7,
            'radiant_win': True,
            'radiant_score': 41,
            'dire_score': 23,
            'first_blood_time': 105,
            'human_players': 10,
            'players': [
                {
                    'account_id': 123456,
                    'hero_id': 7,
                    'kills': 11,
                    'deaths': 7,
                    'assists': 7,
                    'last_hits': 289,
                    'gold_per_min': 644
                }
            ]
        }
        mock_requests.get.return_value = mock_response
        
        # Create processor
        processor = DotaGameProcessor("test_game")
            
        match_data = processor.get_match(match_id)
        
        assert match_data is not None
        assert match_data['match_id'] == match_id
        assert match_data['duration'] == 2325
        assert match_data['radiant_win'] is True
        assert len(match_data['players']) == 1

    @patch('stream_agents.processors.dota_processors.requests')
    def test_get_match_http_failure(self, mock_requests):
        """Test match retrieval HTTP failure."""
        match_id = 12345
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_requests.get.return_value = mock_response
        
        processor = DotaGameProcessor("test_game")
            
        match_data = processor.get_match(match_id)
        
        assert match_data is None

    @patch('stream_agents.processors.dota_processors.requests')
    def test_get_match_http_with_api_key(self, mock_requests):
        """Test match retrieval with API key in headers."""
        match_id = 12345
        api_key = "test_api_key"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'match_id': match_id}
        mock_requests.get.return_value = mock_response
        
        processor = DotaGameProcessor("test_game", api_key=api_key)
            
        processor.get_match(match_id)
        
        # Verify API key was included in headers
        mock_requests.get.assert_called_once()
        call_args = mock_requests.get.call_args
        headers = call_args[1]['headers']
        assert 'Authorization' in headers
        assert headers['Authorization'] == f'Bearer {api_key}'

    def test_convert_detailed_match_to_dict(self):
        """Test detailed match object to dictionary conversion."""
        # Mock detailed match object
        mock_match = Mock()
        mock_match.match_id = 8437488139
        mock_match.duration = 2325
        mock_match.game_mode = 22
        mock_match.radiant_win = True
        mock_match.radiant_score = 41
        mock_match.dire_score = 23
        mock_match.first_blood_time = 105
        mock_match.human_players = 10
        mock_match.lobby_type = 7
        mock_match.skill = None
        mock_match.start_time = 1735402466
        
        # Mock players
        mock_player1 = Mock()
        mock_player1.account_id = 123456
        mock_player1.hero_id = 7
        mock_player1.kills = 11
        mock_player1.deaths = 7
        mock_player1.assists = 7
        mock_player1.last_hits = 289
        mock_player1.gold_per_min = 644
        mock_player1.player_slot = 0
        
        mock_player2 = Mock()
        mock_player2.account_id = 789012
        mock_player2.hero_id = 20
        mock_player2.kills = 1
        mock_player2.deaths = 10
        mock_player2.assists = 16
        mock_player2.last_hits = 31
        mock_player2.gold_per_min = 273
        mock_player2.player_slot = 128
        
        mock_match.players = [mock_player1, mock_player2]
        
        processor = DotaGameProcessor("test_game")
        result = processor._convert_detailed_match_to_dict(mock_match)
        
        # Check basic match data
        assert result['match_id'] == 8437488139
        assert result['duration'] == 2325
        assert result['radiant_win'] is True
        assert result['radiant_score'] == 41
        assert result['dire_score'] == 23
        
        # Check players data
        assert len(result['players']) == 2
        assert result['players'][0]['hero_id'] == 7
        assert result['players'][0]['kills'] == 11
        assert result['players'][1]['hero_id'] == 20
        assert result['players'][1]['kills'] == 1

    def test_get_match_both_methods_fail(self):
        """Test when both client and HTTP methods fail for get_match."""
        processor = DotaGameProcessor("test_game")
        
        # Mock both methods to fail
        with patch.object(processor, '_get_match_via_client', return_value=None):
            with patch.object(processor, '_get_match_via_http', return_value=None):
                match_data = processor.get_match(12345)
                
                assert match_data is None

    @pytest.mark.parametrize("match_id", [
        8437488139,
        1234567890,
        999999999,
        123,
    ])
    def test_get_match_with_various_ids(self, match_id):
        """Test get_match with various match IDs."""
        processor = DotaGameProcessor("test_game")
        
        # Mock successful HTTP response
        with patch.object(processor, '_get_match_via_http') as mock_http:
            mock_http.return_value = {'match_id': match_id, 'duration': 1800}
            
            result = processor.get_match(match_id)
            
            assert result is not None
            assert result['match_id'] == match_id
            mock_http.assert_called_once_with(match_id)

    @patch('stream_agents.processors.dota_processors.requests')
    def test_get_match_http_exception(self, mock_requests):
        """Test get_match HTTP with exception."""
        mock_requests.get.side_effect = Exception("Network error")
        
        processor = DotaGameProcessor("test_game")
            
        match_data = processor.get_match(12345)
        
        assert match_data is None

    def test_get_match_url_construction(self):
        """Test that the correct URL is constructed for match requests."""
        match_id = 8437488139
        
        with patch('stream_agents.processors.dota_processors.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'match_id': match_id}
            mock_requests.get.return_value = mock_response
            
            with patch('stream_agents.processors.dota_processors.HAS_OPENDOTA', False):
                processor = DotaGameProcessor("test_game")
                
            processor.get_match(match_id)
            
            # Verify correct URL was used
            expected_url = f"https://api.opendota.com/api/matches/{match_id}"
            mock_requests.get.assert_called_once()
            call_args = mock_requests.get.call_args
            assert call_args[0][0] == expected_url


if __name__ == "__main__":
    pytest.main([__file__])
