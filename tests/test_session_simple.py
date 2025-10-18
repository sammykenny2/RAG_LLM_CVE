"""
Simple test for SessionManager (no Unicode output).
"""

import sys
import os
import uuid

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test imports
print("Test 1: Import SessionManager...")
try:
    from core.session_manager import SessionManager
    print("[PASS] SessionManager imported")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    exit(1)

print("\nTest 2: Import configuration...")
try:
    from config import (
        SESSION_MAX_FILES,
        SESSION_MAX_FILE_SIZE_MB,
        SESSION_TIMEOUT_HOURS
    )
    print("[PASS] Configuration imported")
    print(f"  SESSION_MAX_FILES: {SESSION_MAX_FILES}")
    print(f"  SESSION_MAX_FILE_SIZE_MB: {SESSION_MAX_FILE_SIZE_MB}")
    print(f"  SESSION_TIMEOUT_HOURS: {SESSION_TIMEOUT_HOURS}")
except ImportError as e:
    print(f"[FAIL] Config import error: {e}")
    exit(1)

print("\nTest 3: Initialize SessionManager...")
try:
    session_id = str(uuid.uuid4())
    session = SessionManager(session_id=session_id)
    print(f"[PASS] Session initialized: {session_id}")
    print(f"  Collection: {session.collection_name}")
    print(f"  Files: {len(session.files)}")

    # Cleanup
    session.cleanup()
    print("[PASS] Session cleanup successful")

except Exception as e:
    print(f"[FAIL] Initialization error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
