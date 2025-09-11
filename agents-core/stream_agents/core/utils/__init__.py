"""
Stream Agents Utilities Package

This package provides utility functions and scripts for Stream Agents.
"""

import os
import webbrowser
from urllib.parse import urlencode
from uuid import uuid4

from getstream.models import UserRequest
from getstream.video.call import Call

import logging

logger = logging.getLogger(__name__)


