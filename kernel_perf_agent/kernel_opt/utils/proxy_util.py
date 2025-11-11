"""
Meta Proxy Configuration

Authors: Laura Wang, Jie Liu
"""

import os
import subprocess
from typing import Dict, Optional


def get_meta_proxy_config() -> Optional[Dict[str, str]]:
    """
    Get Meta's proxy configuration if available.

    Returns:
        Dictionary with proxy settings or None if not available
    """
    try:
        # Check if with-proxy command exists (Meta environment)
        result = subprocess.run(
            ["which", "with-proxy"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        # Get proxy environment variables from with-proxy
        result = subprocess.run(
            ["with-proxy", "env"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        # Parse proxy settings
        proxy_config = {}
        for line in result.stdout.split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                if key.lower() in ["http_proxy", "https_proxy"]:
                    proxy_config[key.lower()] = value

        return proxy_config if proxy_config else None

    except Exception:
        return None


def devgpu_proxy_setup() -> Dict[str, str]:
    original_proxy_env = {}
    proxy_config = get_meta_proxy_config()
    if proxy_config:
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
            original_proxy_env[key] = os.environ.get(key)
            proxy_url = proxy_config.get(key)
            if proxy_url:
                os.environ[key] = proxy_url
    return original_proxy_env
