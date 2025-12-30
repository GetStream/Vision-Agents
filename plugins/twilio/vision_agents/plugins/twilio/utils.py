def normalize_websocket_url(url: str) -> str:
    """Normalize a URL to use the wss:// scheme for Twilio media streams."""
    url = url.strip()
    if url.startswith("wss://https://"):
        url = "wss://" + url[14:]
    elif url.startswith("wss://http://"):
        url = "wss://" + url[13:]
    elif url.startswith("https://"):
        url = "wss://" + url[8:]
    elif url.startswith("http://"):
        url = "wss://" + url[8:]
    return url
