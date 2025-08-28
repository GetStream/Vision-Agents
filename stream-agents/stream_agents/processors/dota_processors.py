import logging
import requests
from typing import Optional, List, Dict, Any
from stream_agents.processors import BaseProcessor

try:
    import stratz
    from stratz.lang import code_English
    HAS_STRATZ = True
except ImportError:
    HAS_STRATZ = False
    logging.warning("stratz package not available. Install with: pip install stratz")


class DotaGameProcessor(BaseProcessor):
    """
    Dota 2 game processor that integrates with Stratz API.
    
    Provides functionality to:
    - Set up Stratz API client
    - Retrieve live matches with lower latency
    - Process game data for match analysis
    """
    
    def __init__(self, game_id: Optional[str] = None, api_token: Optional[str] = None):
        """
        Initialize the Dota game processor.
        
        Args:
            game_id: Optional game ID to track
            api_token: Optional Stratz API token for authenticated requests
        """
        self.game_id = game_id
        self.api_token = api_token
        self.stratz_api = None
        self.logger = logging.getLogger(__name__)
        
        if HAS_STRATZ:
            self.setup_client()
        else:
            self.logger.error("Stratz API not available. Install stratz package.")

    def setup_client(self) -> bool:
        """
        Set up the Stratz API client.
        
        Returns:
            bool: True if client setup was successful, False otherwise
        """
        if not HAS_STRATZ:
            self.logger.error("Stratz package not available")
            return False
            
        try:
            # Create Stratz API client
            if self.api_token:
                self.stratz_api = stratz.Api(lang=code_English, api_token=self.api_token)
                self.logger.info("Stratz API client configured with API token")
            else:
                self.stratz_api = stratz.Api(lang=code_English)
                self.logger.info("Stratz API client configured without API token (public access)")
            
            self.logger.info("Stratz API client setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup Stratz API client: {e}")
            return False

    def get_live_matches(self) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve current live matches from Stratz API.
        
        Returns:
            Optional[List[Dict[str, Any]]]: List of live match data, or None if failed
        """
        self.logger.info("Fetching live matches from Stratz API...")
        
        # Try Stratz client first, then fallback to raw GraphQL HTTP
        matches_data = self._get_live_matches_via_stratz()
        
        if matches_data is None:
            self.logger.info("Stratz client approach failed, trying GraphQL HTTP fallback...")
            matches_data = self._get_live_matches_via_graphql()
        
        if matches_data:
            self.logger.info(f"Successfully retrieved {len(matches_data)} live matches")
        else:
            self.logger.error("Failed to retrieve live matches via all methods")
        
        return matches_data

    def _get_live_matches_via_stratz(self) -> Optional[List[Dict[str, Any]]]:
        """
        Try to get live matches using the Stratz client.
        
        Returns:
            Optional[List[Dict[str, Any]]]: List of live match data, or None if failed
        """
        if not self.stratz_api:
            self.logger.debug("Stratz API client not initialized")
            return None
            
        try:
            # Call the Stratz live games API
            live_matches = self.stratz_api.get_live_matches()
            
            if not live_matches:
                return []
            
            # Convert to standardized format
            matches_data = []
            for match in live_matches:
                match_dict = self._convert_stratz_match_to_dict(match)
                matches_data.append(match_dict)
            
            return matches_data
            
        except Exception as e:
            self.logger.debug(f"Stratz client approach failed: {e}")
            return None

    def _get_live_matches_via_graphql(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get live matches using raw GraphQL HTTP requests as fallback.
        
        Returns:
            Optional[List[Dict[str, Any]]]: List of live match data, or None if failed
        """
        try:
            url = "https://api.stratz.com/graphql"
            
            # GraphQL query for live matches
            query = """
            query GetLiveMatches {
                live {
                    matches {
                        matchId
                        gameMode
                        lobbyType
                        spectators
                        gameTime
                        radiantScore
                        direScore
                        radiantTeam {
                            id
                            name
                            tag
                        }
                        direTeam {
                            id
                            name
                            tag
                        }
                        players {
                            steamAccountId
                            heroId
                            kills
                            deaths
                            assists
                            netWorth
                            level
                            isRadiant
                        }
                    }
                }
            }
            """
            
            headers = {
                'Content-Type': 'application/json',
            }
            
            # Add API token if available
            if self.api_token:
                headers['Authorization'] = f'Bearer {self.api_token}'
            
            payload = {'query': query}
            
            # Make the request
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and 'live' in data['data'] and 'matches' in data['data']['live']:
                    live_matches = data['data']['live']['matches']
                    return live_matches
                else:
                    self.logger.error("Unexpected GraphQL response structure")
                    return None
            else:
                self.logger.error(f"GraphQL request failed with status: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"GraphQL fallback failed: {e}")
            return None

    def _convert_stratz_match_to_dict(self, match) -> Dict[str, Any]:
        """
        Convert a Stratz match object to standardized dictionary format.
        
        Args:
            match: Match object from Stratz API
            
        Returns:
            Dict[str, Any]: Match data as dictionary
        """
        # Handle both object attributes and dictionary keys
        def safe_get(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key, default)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default
        
        match_dict = {
            'match_id': safe_get(match, 'matchId') or safe_get(match, 'match_id'),
            'lobby_type': safe_get(match, 'lobbyType') or safe_get(match, 'lobby_type'),
            'game_mode': safe_get(match, 'gameMode') or safe_get(match, 'game_mode'),
            'spectators': safe_get(match, 'spectators'),
            'game_time': safe_get(match, 'gameTime') or safe_get(match, 'duration'),
            'radiant_score': safe_get(match, 'radiantScore') or safe_get(match, 'radiant_score'),
            'dire_score': safe_get(match, 'direScore') or safe_get(match, 'dire_score'),
        }
        
        # Add team information if available
        radiant_team = safe_get(match, 'radiantTeam') or safe_get(match, 'radiant_team')
        if radiant_team:
            match_dict['radiant_team'] = {
                'team_id': safe_get(radiant_team, 'id') or safe_get(radiant_team, 'team_id'),
                'name': safe_get(radiant_team, 'name'),
                'tag': safe_get(radiant_team, 'tag'),
            }
        
        dire_team = safe_get(match, 'direTeam') or safe_get(match, 'dire_team')
        if dire_team:
            match_dict['dire_team'] = {
                'team_id': safe_get(dire_team, 'id') or safe_get(dire_team, 'team_id'),
                'name': safe_get(dire_team, 'name'),
                'tag': safe_get(dire_team, 'tag'),
            }
        
        # Add players if available
        players = safe_get(match, 'players')
        if players:
            players_data = []
            for player in players:
                player_dict = {
                    'account_id': safe_get(player, 'steamAccountId') or safe_get(player, 'account_id'),
                    'hero_id': safe_get(player, 'heroId') or safe_get(player, 'hero_id'),
                    'kills': safe_get(player, 'kills'),
                    'deaths': safe_get(player, 'deaths'),
                    'assists': safe_get(player, 'assists'),
                    'net_worth': safe_get(player, 'netWorth') or safe_get(player, 'net_worth'),
                    'level': safe_get(player, 'level'),
                    'is_radiant': safe_get(player, 'isRadiant') or safe_get(player, 'is_radiant'),
                }
                players_data.append(player_dict)
            match_dict['players'] = players_data
        
        return match_dict

    def get_match(self, match_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed information for a specific match by ID.
        
        Args:
            match_id: The match ID to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Match data dictionary, or None if failed
        """
        self.logger.info(f"Fetching match {match_id} from Stratz API...")
        
        # Try Stratz client first, then fallback to GraphQL HTTP
        match_data = self._get_match_via_stratz(match_id)
        
        if match_data is None:
            self.logger.info("Stratz client approach failed, trying GraphQL HTTP fallback...")
            match_data = self._get_match_via_graphql_http(match_id)
        
        if match_data:
            self.logger.info(f"Successfully retrieved match {match_id}")
        else:
            self.logger.error(f"Failed to retrieve match {match_id} via all methods")
        
        return match_data

    def _get_match_via_stratz(self, match_id: int) -> Optional[Dict[str, Any]]:
        """
        Try to get match data using the Stratz client.
        
        Args:
            match_id: The match ID to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Match data, or None if failed
        """
        if not self.stratz_api:
            self.logger.debug("Stratz API client not initialized")
            return None
            
        try:
            # Call the Stratz match API
            match_data = self.stratz_api.get_match(match_id)
            
            if not match_data:
                return None
            
            # Convert to standardized dictionary format
            result = self._convert_stratz_detailed_match_to_dict(match_data)
            
            return result
            
        except Exception as e:
            self.logger.debug(f"Stratz client approach failed for match {match_id}: {e}")
            return None

    def _get_match_via_graphql_http(self, match_id: int) -> Optional[Dict[str, Any]]:
        """
        Get match data using raw GraphQL HTTP requests as fallback.
        
        Args:
            match_id: The match ID to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Match data, or None if failed
        """
        try:
            url = "https://api.stratz.com/graphql"
            
            # GraphQL query for specific match
            query = """
            query GetMatch($matchId: Long!) {
                match(id: $matchId) {
                    id
                    gameMode
                    lobbyType
                    durationSeconds
                    startDateTime
                    radiantKills
                    direKills
                    winRateByHeroPair {
                        heroId1
                        heroId2
                        winRate
                    }
                    players {
                        steamAccountId
                        heroId
                        kills
                        deaths
                        assists
                        lastHits
                        denies
                        goldPerMinute
                        experiencePerMinute
                        level
                        netWorth
                        isRadiant
                        item0Id
                        item1Id
                        item2Id
                        item3Id
                        item4Id
                        item5Id
                    }
                    radiantTeam {
                        id
                        name
                        tag
                    }
                    direTeam {
                        id
                        name
                        tag
                    }
                }
            }
            """
            
            headers = {
                'Content-Type': 'application/json',
            }
            
            # Add API token if available
            if self.api_token:
                headers['Authorization'] = f'Bearer {self.api_token}'
            
            payload = {
                'query': query,
                'variables': {'matchId': match_id}
            }
            
            # Make the request
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and 'match' in data['data'] and data['data']['match']:
                    match_data = data['data']['match']
                    return match_data
                else:
                    self.logger.error(f"Match {match_id} not found or unexpected GraphQL response structure")
                    return None
            else:
                self.logger.error(f"GraphQL request failed for match {match_id} with status: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"GraphQL HTTP fallback failed for match {match_id}: {e}")
            return None

    def _convert_stratz_detailed_match_to_dict(self, match) -> Dict[str, Any]:
        """
        Convert a detailed Stratz match object to dictionary format.
        
        Args:
            match: Match object from Stratz API
            
        Returns:
            Dict[str, Any]: Match data as dictionary
        """
        # Handle both object attributes and dictionary keys
        def safe_get(obj, key, default=None):
            if hasattr(obj, key):
                return getattr(obj, key, default)
            elif isinstance(obj, dict):
                return obj.get(key, default)
            return default
        
        match_dict = {
            'match_id': safe_get(match, 'id') or safe_get(match, 'match_id'),
            'game_mode': safe_get(match, 'gameMode') or safe_get(match, 'game_mode'),
            'lobby_type': safe_get(match, 'lobbyType') or safe_get(match, 'lobby_type'),
            'duration': safe_get(match, 'durationSeconds') or safe_get(match, 'duration'),
            'start_time': safe_get(match, 'startDateTime') or safe_get(match, 'start_time'),
            'radiant_score': safe_get(match, 'radiantKills') or safe_get(match, 'radiant_score'),
            'dire_score': safe_get(match, 'direKills') or safe_get(match, 'dire_score'),
        }
        
        # Add team information if available
        radiant_team = safe_get(match, 'radiantTeam') or safe_get(match, 'radiant_team')
        if radiant_team:
            match_dict['radiant_team'] = {
                'team_id': safe_get(radiant_team, 'id') or safe_get(radiant_team, 'team_id'),
                'name': safe_get(radiant_team, 'name'),
                'tag': safe_get(radiant_team, 'tag'),
            }
        
        dire_team = safe_get(match, 'direTeam') or safe_get(match, 'dire_team')
        if dire_team:
            match_dict['dire_team'] = {
                'team_id': safe_get(dire_team, 'id') or safe_get(dire_team, 'team_id'),
                'name': safe_get(dire_team, 'name'),
                'tag': safe_get(dire_team, 'tag'),
            }
        
        # Convert players list if available
        players = safe_get(match, 'players')
        if players:
            players_data = []
            for player in players:
                player_dict = {
                    'account_id': safe_get(player, 'steamAccountId') or safe_get(player, 'account_id'),
                    'hero_id': safe_get(player, 'heroId') or safe_get(player, 'hero_id'),
                    'kills': safe_get(player, 'kills'),
                    'deaths': safe_get(player, 'deaths'),
                    'assists': safe_get(player, 'assists'),
                    'last_hits': safe_get(player, 'lastHits') or safe_get(player, 'last_hits'),
                    'denies': safe_get(player, 'denies'),
                    'gold_per_min': safe_get(player, 'goldPerMinute') or safe_get(player, 'gold_per_min'),
                    'xp_per_min': safe_get(player, 'experiencePerMinute') or safe_get(player, 'xp_per_min'),
                    'level': safe_get(player, 'level'),
                    'net_worth': safe_get(player, 'netWorth') or safe_get(player, 'net_worth'),
                    'is_radiant': safe_get(player, 'isRadiant') or safe_get(player, 'is_radiant'),
                    'item_0': safe_get(player, 'item0Id') or safe_get(player, 'item_0'),
                    'item_1': safe_get(player, 'item1Id') or safe_get(player, 'item_1'),
                    'item_2': safe_get(player, 'item2Id') or safe_get(player, 'item_2'),
                    'item_3': safe_get(player, 'item3Id') or safe_get(player, 'item_3'),
                    'item_4': safe_get(player, 'item4Id') or safe_get(player, 'item_4'),
                    'item_5': safe_get(player, 'item5Id') or safe_get(player, 'item_5'),
                }
                players_data.append(player_dict)
            match_dict['players'] = players_data
        
        return match_dict

    def state(self) -> Dict[str, Any]:
        """
        Get current state of the processor.
        
        Returns:
            Dict[str, Any]: Current state information
        """
        return {
            'game_id': self.game_id,
            'has_api_token': bool(self.api_token),
            'stratz_client_initialized': bool(self.stratz_api),
            'stratz_available': HAS_STRATZ,
        }

    def input(self) -> Dict[str, Any]:
        """
        Get input configuration for the processor.
        
        Returns:
            Dict[str, Any]: Input configuration
        """
        return {
            'game_id': self.game_id,
            'api_token_configured': bool(self.api_token),
        }