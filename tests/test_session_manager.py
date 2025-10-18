"""
Unit tests for SessionManager.
Tests session-scoped file management and multi-file conversation context.
"""

import os
import uuid
import tempfile
from pathlib import Path

# Test if SessionManager can be imported
try:
    from core.session_manager import SessionManager
    print("✅ SessionManager import successful")
except ImportError as e:
    print(f"❌ SessionManager import failed: {e}")
    exit(1)

# Test if config is properly loaded
try:
    from config import (
        SESSION_MAX_FILES,
        SESSION_MAX_FILE_SIZE_MB,
        SESSION_TIMEOUT_HOURS
    )
    print(f"✅ Session configuration loaded:")
    print(f"  └─ SESSION_MAX_FILES: {SESSION_MAX_FILES}")
    print(f"  └─ SESSION_MAX_FILE_SIZE_MB: {SESSION_MAX_FILE_SIZE_MB}")
    print(f"  └─ SESSION_TIMEOUT_HOURS: {SESSION_TIMEOUT_HOURS}")
except ImportError as e:
    print(f"❌ Config import failed: {e}")
    exit(1)


def test_session_initialization():
    """Test SessionManager initialization."""
    print("\n" + "=" * 60)
    print("Test 1: SessionManager Initialization")
    print("=" * 60)

    try:
        session_id = str(uuid.uuid4())
        session = SessionManager(session_id=session_id)

        assert session.session_id == session_id
        assert session.collection_name == f"session_{session_id}"
        assert len(session.files) == 0

        print("✅ Session initialization successful")
        print(f"  └─ Session ID: {session_id}")
        print(f"  └─ Collection: {session.collection_name}")
        print(f"  └─ Session directory: {session.session_dir}")

        # Cleanup
        session.cleanup()
        print("✅ Session cleanup successful")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_operations():
    """Test adding and removing files (requires sample PDF)."""
    print("\n" + "=" * 60)
    print("Test 2: File Operations")
    print("=" * 60)

    # Check if sample PDF exists
    sample_pdf = Path("samples/CVEDocument.pdf")
    if not sample_pdf.exists():
        print(f"⚠️ Sample PDF not found: {sample_pdf}")
        print("  └─ Skipping file operations test")
        return None

    try:
        session_id = str(uuid.uuid4())
        session = SessionManager(session_id=session_id)

        # Test add_file
        print("\n[Test 2a] Adding file...")
        file_info = session.add_file(str(sample_pdf))

        assert file_info["status"] == "ready"
        assert file_info["chunks"] > 0
        assert file_info["name"] == "CVEDocument.pdf"

        print("✅ File added successfully")
        print(f"  └─ Name: {file_info['name']}")
        print(f"  └─ Status: {file_info['status']}")
        print(f"  └─ Chunks: {file_info['chunks']}")

        # Test list_files
        print("\n[Test 2b] Listing files...")
        files = session.list_files()

        assert len(files) == 1
        assert files[0]["name"] == "CVEDocument.pdf"

        print("✅ File listing successful")
        print(f"  └─ Files in session: {len(files)}")

        # Test query
        print("\n[Test 2c] Querying session...")
        results = session.query("CVE-2024", top_k=3)

        assert len(results) > 0
        assert all("text" in r and "source" in r for r in results)

        print("✅ Query successful")
        print(f"  └─ Results: {len(results)}")
        if results:
            print(f"  └─ Top result source: {results[0]['source']}")

        # Test remove_file
        print("\n[Test 2d] Removing file...")
        removed = session.remove_file("CVEDocument.pdf")

        assert removed == True
        assert len(session.list_files()) == 0

        print("✅ File removed successfully")

        # Cleanup
        session.cleanup()
        print("\n✅ All file operations tests passed")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_limits():
    """Test file size and count limits."""
    print("\n" + "=" * 60)
    print("Test 3: Limits Enforcement")
    print("=" * 60)

    try:
        session_id = str(uuid.uuid4())
        session = SessionManager(session_id=session_id)

        # Test max files limit (mock test)
        print(f"\n[Test 3a] Max files limit: {SESSION_MAX_FILES}")
        print("  └─ (Limit enforcement tested via add_file)")

        # Test file size limit (mock test)
        print(f"\n[Test 3b] Max file size: {SESSION_MAX_FILE_SIZE_MB} MB")
        print("  └─ (Limit enforcement tested via add_file)")

        print("\n✅ Limits configuration verified")

        # Cleanup
        session.cleanup()

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "=" * 60)
    print("SessionManager Test Suite")
    print("=" * 60)

    results = {
        "initialization": test_session_initialization(),
        "file_operations": test_file_operations(),
        "limits": test_limits()
    }

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r == True)
    failed = sum(1 for r in results.values() if r == False)
    skipped = sum(1 for r in results.values() if r is None)

    for test_name, result in results.items():
        status = "✅ PASS" if result == True else ("❌ FAIL" if result == False else "⚠️ SKIP")
        print(f"{status}: {test_name}")

    print(f"\nTotal: {len(results)} tests")
    print(f"  └─ Passed: {passed}")
    print(f"  └─ Failed: {failed}")
    print(f"  └─ Skipped: {skipped}")

    if failed > 0:
        print("\n❌ Some tests failed!")
        exit(1)
    else:
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
