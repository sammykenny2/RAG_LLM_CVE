"""
Test dual-source retrieval (permanent KB + session files) for RAG implementations.

Tests both PureRAG and LangChainRAG with SessionManager integration.
"""

import sys
import os
from pathlib import Path
import uuid

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from core.session_manager import SessionManager
from rag.pure_python import PureRAG
from rag.langchain_impl import LangChainRAG


def test_pure_rag_merge_results():
    """Test PureRAG._merge_results() method."""
    print("\n[TEST] PureRAG._merge_results()")

    # Create RAG instance
    rag = PureRAG()

    # Mock KB results (list of strings)
    kb_results = [
        "CVE-2024-1234 is a critical vulnerability in Apache.",
        "CVE-2024-5678 affects multiple versions of OpenSSL."
    ]

    # Mock session results (list of dicts)
    session_results = [
        {
            'text': "CVE-2024-9999 is a recently discovered vulnerability.",
            'source': 'report1.pdf',
            'source_type': 'session',
            'score': 0.95
        },
        {
            'text': "Security analysis of CVE-2024-1111.",
            'source': 'report2.pdf',
            'source_type': 'session',
            'score': 0.85
        }
    ]

    # Test merge
    merged = rag._merge_results(kb_results, session_results)

    # Verify results
    assert len(merged) == 4, f"Expected 4 results, got {len(merged)}"

    # Check format (all should be dicts with required keys)
    for item in merged:
        assert 'text' in item, "Missing 'text' key"
        assert 'source' in item, "Missing 'source' key"
        assert 'source_type' in item, "Missing 'source_type' key"
        assert 'score' in item, "Missing 'score' key"

    # Check sorting (highest score first)
    # KB results have score 1.0, which is higher than session results' 0.95
    assert merged[0]['score'] == 1.0, "Results not sorted by score (KB should be first)"
    assert merged[1]['score'] == 1.0, "Second KB result should have score 1.0"
    assert merged[2]['score'] == 0.95, "First session result should have score 0.95"

    print("  [PASS] Merge results works correctly")
    print(f"  [INFO] Merged {len(merged)} results from KB and session files")


def test_pure_rag_build_prompt_with_sources():
    """Test PureRAG._build_prompt_with_sources() method."""
    print("\n[TEST] PureRAG._build_prompt_with_sources()")

    # Create RAG instance
    rag = PureRAG()

    # Mock context items
    context_items = [
        {
            'text': "CVE-2024-1234 is critical.",
            'source': 'Knowledge Base',
            'source_type': 'permanent',
            'score': 1.0
        },
        {
            'text': "Exploit details for CVE-2024-1234.",
            'source': 'report1.pdf',
            'source_type': 'session',
            'score': 0.92
        }
    ]

    # Build prompt
    prompt = rag._build_prompt_with_sources(context_items)

    # Verify format
    assert "From Knowledge Base:" in prompt, "Missing KB source attribution"
    assert "From report1.pdf:" in prompt, "Missing session file source attribution"
    assert "CVE-2024-1234 is critical" in prompt, "Missing KB content"
    assert "Exploit details" in prompt, "Missing session content"

    print("  [PASS] Build prompt with sources works correctly")
    print(f"  [INFO] Generated prompt with {len(context_items)} sources")


def test_langchain_rag_merge_results():
    """Test LangChainRAG._merge_results() method."""
    print("\n[TEST] LangChainRAG._merge_results()")

    # Create RAG instance (no initialization needed for this test)
    rag = LangChainRAG()

    # Mock KB results
    kb_results = [
        "CVE-2024-1234 vulnerability analysis.",
        "CVE-2024-5678 mitigation strategies."
    ]

    # Mock session results
    session_results = [
        {
            'text': "Recent CVE-2024-9999 exploit.",
            'source': 'threat_report.pdf',
            'source_type': 'session',
            'score': 0.88
        }
    ]

    # Test merge
    merged = rag._merge_results(kb_results, session_results)

    # Verify results
    assert len(merged) == 3, f"Expected 3 results, got {len(merged)}"

    # Check format
    for item in merged:
        assert 'text' in item
        assert 'source' in item
        assert 'source_type' in item
        assert 'score' in item

    # Check KB results have correct source
    kb_items = [item for item in merged if item['source_type'] == 'permanent']
    assert len(kb_items) == 2, f"Expected 2 KB items, got {len(kb_items)}"

    # Check session results have correct source
    session_items = [item for item in merged if item['source_type'] == 'session']
    assert len(session_items) == 1, f"Expected 1 session item, got {len(session_items)}"

    print("  [PASS] Merge results works correctly")
    print(f"  [INFO] Merged {len(merged)} results (KB: {len(kb_items)}, Session: {len(session_items)})")


def test_langchain_rag_build_prompt_with_sources():
    """Test LangChainRAG._build_prompt_with_sources() method."""
    print("\n[TEST] LangChainRAG._build_prompt_with_sources()")

    # Create RAG instance
    rag = LangChainRAG()

    # Mock context items
    context_items = [
        {
            'text': "CVE-2024-1111 affects Windows.",
            'source': 'Knowledge Base',
            'source_type': 'permanent',
            'score': 1.0
        },
        {
            'text': "Additional details on CVE-2024-1111.",
            'source': 'analysis.pdf',
            'source_type': 'session',
            'score': 0.91
        },
        {
            'text': "CVE-2024-2222 related vulnerabilities.",
            'source': 'Knowledge Base',
            'source_type': 'permanent',
            'score': 1.0
        }
    ]

    # Build prompt
    prompt = rag._build_prompt_with_sources(context_items)

    # Verify format
    assert "From Knowledge Base:" in prompt
    assert "From analysis.pdf:" in prompt
    assert prompt.count("From Knowledge Base:") == 2, "Should have 2 KB sources"
    assert prompt.count("From analysis.pdf:") == 1, "Should have 1 session source"

    print("  [PASS] Build prompt with sources works correctly")
    print(f"  [INFO] Generated prompt with {len(context_items)} sources")


def test_session_manager_integration():
    """Test SessionManager integration (basic connectivity test)."""
    print("\n[TEST] SessionManager integration")

    # Create session manager
    session_id = str(uuid.uuid4())
    session = SessionManager(session_id=session_id)

    print(f"  [INFO] Created session: {session_id}")
    print(f"  [INFO] Collection name: {session.collection_name}")
    print(f"  [INFO] Session directory: {session.session_dir}")

    # Verify collection exists
    assert session.client is not None, "Chroma client not initialized"
    assert session.collection is not None, "Chroma collection not initialized"
    assert session.embedder is not None, "EmbeddingModel not initialized"

    print("  [PASS] SessionManager created successfully")

    # Cleanup
    session.cleanup()
    print("  [INFO] Session cleaned up")


def test_pure_rag_with_session_manager():
    """Test PureRAG initialization with SessionManager."""
    print("\n[TEST] PureRAG with SessionManager")

    # Create session manager
    session_id = str(uuid.uuid4())
    session = SessionManager(session_id=session_id)

    # Create PureRAG with session manager
    rag = PureRAG(session_manager=session)

    # Verify integration
    assert rag.session_manager is not None, "SessionManager not attached"
    assert rag.session_manager.session_id == session_id, "Session ID mismatch"

    print("  [PASS] PureRAG successfully integrated with SessionManager")
    print(f"  [INFO] Session ID: {session_id}")

    # Cleanup
    session.cleanup()


def test_langchain_rag_with_session_manager():
    """Test LangChainRAG initialization with SessionManager."""
    print("\n[TEST] LangChainRAG with SessionManager")

    # Create session manager
    session_id = str(uuid.uuid4())
    session = SessionManager(session_id=session_id)

    # Create LangChainRAG with session manager
    rag = LangChainRAG(session_manager=session)

    # Verify integration
    assert rag.session_manager is not None, "SessionManager not attached"
    assert rag.session_manager.session_id == session_id, "Session ID mismatch"

    print("  [PASS] LangChainRAG successfully integrated with SessionManager")
    print(f"  [INFO] Session ID: {session_id}")

    # Cleanup
    session.cleanup()


def run_all_tests():
    """Run all dual-source retrieval tests."""
    print("=" * 70)
    print("Dual-Source Retrieval Tests (PR 2)")
    print("=" * 70)

    tests = [
        ("PureRAG merge results", test_pure_rag_merge_results),
        ("PureRAG build prompt", test_pure_rag_build_prompt_with_sources),
        ("LangChainRAG merge results", test_langchain_rag_merge_results),
        ("LangChainRAG build prompt", test_langchain_rag_build_prompt_with_sources),
        ("SessionManager integration", test_session_manager_integration),
        ("PureRAG + SessionManager", test_pure_rag_with_session_manager),
        ("LangChainRAG + SessionManager", test_langchain_rag_with_session_manager),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} total")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
