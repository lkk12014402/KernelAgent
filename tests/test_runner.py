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


def test_allowlist_env_preserves_ld_library_path():
    """Test that LD_LIBRARY_PATH is preserved in the allowlist."""
    from Fuser.runner import _allowlist_env
    
    # Save original environment
    original_env = os.environ.copy()
    
    try:
        # Set a test LD_LIBRARY_PATH
        test_path = "/usr/local/lib:/opt/intel/lib"
        os.environ["LD_LIBRARY_PATH"] = test_path
        
        # Get the allowlisted environment
        allowed_env = _allowlist_env()
        
        # Verify LD_LIBRARY_PATH is preserved
        assert "LD_LIBRARY_PATH" in allowed_env
        assert allowed_env["LD_LIBRARY_PATH"] == test_path
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


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
    
    # Save original environment
    original_env = os.environ.copy()
    
    try:
        # Remove LD_LIBRARY_PATH if it exists
        os.environ.pop("LD_LIBRARY_PATH", None)
        
        # Get the allowlisted environment
        allowed_env = _allowlist_env()
        
        # Verify LD_LIBRARY_PATH is not in the result when not set
        assert "LD_LIBRARY_PATH" not in allowed_env
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)
