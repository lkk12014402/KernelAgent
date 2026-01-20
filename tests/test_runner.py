# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pytest
from unittest.mock import patch


def test_allowlist_env_preserves_ld_library_path():
    """Test that LD_LIBRARY_PATH is preserved in the allowlist."""
    from Fuser.runner import _allowlist_env
    
    # Set a test LD_LIBRARY_PATH
    test_path = "/usr/local/lib:/opt/intel/lib"
    
    with patch.dict(os.environ, {"LD_LIBRARY_PATH": test_path}):
        # Get the allowlisted environment
        allowed_env = _allowlist_env()
        
        # Verify LD_LIBRARY_PATH is preserved
        assert "LD_LIBRARY_PATH" in allowed_env
        assert allowed_env["LD_LIBRARY_PATH"] == test_path


def test_allowlist_env_preserves_path():
    """Test that PATH is preserved in the allowlist."""
    from Fuser.runner import _allowlist_env
    
    # Get the allowlisted environment
    allowed_env = _allowlist_env()
    
    # Verify PATH is preserved
    assert "PATH" in allowed_env
    assert allowed_env["PATH"] == os.environ.get("PATH")


def test_allowlist_env_without_ld_library_path():
    """Test that _allowlist_env works when LD_LIBRARY_PATH is not set."""
    from Fuser.runner import _allowlist_env
    
    # Create a copy of the environment without LD_LIBRARY_PATH
    env_without_ld = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    
    with patch.dict(os.environ, env_without_ld, clear=True):
        # Get the allowlisted environment
        allowed_env = _allowlist_env()
        
        # Verify LD_LIBRARY_PATH is not in the result when not set
        assert "LD_LIBRARY_PATH" not in allowed_env
