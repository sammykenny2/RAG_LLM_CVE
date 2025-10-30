"""
Remove documents from knowledge base by source name.

Usage:
    python cli/remove_from_embeddings.py
    # Interactive: lists all sources and lets you choose

    python cli/remove_from_embeddings.py --source="report.pdf"
    # Direct: removes specific source
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from core.chroma_manager import ChromaManager
from config import VERBOSE_LOGGING

def list_sources(manager: ChromaManager):
    """List all sources in the knowledge base."""
    stats = manager.get_stats()
    sources = stats['sources']

    if not sources:
        print("[WARNING] Knowledge base is empty")
        return []

    print(f"\n{'='*60}")
    print(f"Knowledge Base Sources")
    print(f"{'='*60}\n")
    print(f"Total documents: {stats['total_docs']}")
    print(f"\nBy type:")
    for src_type, count in stats['by_source_type'].items():
        print(f"  - {src_type}: {count}")

    print(f"\n{'='*60}")
    print(f"Individual Sources:")
    print(f"{'='*60}\n")

    source_list = []
    for i, (name, info) in enumerate(sorted(sources.items()), 1):
        print(f"{i}. {name}")
        print(f"   Type: {info['type']}, Chunks: {info['count']}, Added: {info['added_date'][:10]}")
        source_list.append(name)

    return source_list

def remove_source(manager: ChromaManager, source_name: str) -> bool:
    """Remove a source from the knowledge base."""
    print(f"\n{'='*60}")
    print(f"Removing: {source_name}")
    print(f"{'='*60}\n")

    # Confirm deletion
    confirm = input(f"[WARNING] This will permanently delete all chunks from '{source_name}'. Continue? (yes/no): ").strip().lower()

    if confirm not in ['yes', 'y']:
        print("[ERROR] Cancelled")
        return False

    # Delete
    count = manager.delete_by_source(source_name)

    if count > 0:
        print(f"[OK] Removed {count} chunks from '{source_name}'")
        return True
    else:
        print(f"[WARNING] Source '{source_name}' not found")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Remove documents from knowledge base',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli/remove_from_embeddings.py                  # Interactive mode
  python cli/remove_from_embeddings.py --source="report.pdf"  # Direct removal
  python cli/remove_from_embeddings.py --list           # List sources only
        """
    )
    parser.add_argument(
        '--source',
        type=str,
        help='Source name to remove (e.g., "report.pdf" or "CVE_2024_v5")'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all sources and exit'
    )
    args = parser.parse_args()

    # Initialize Chroma manager
    print("Connecting to knowledge base...")
    manager = ChromaManager()
    manager.initialize(create_if_not_exists=False)

    # List sources
    source_list = list_sources(manager)

    if args.list:
        # List only mode
        return

    if not source_list:
        return

    # Determine which source to remove
    if args.source:
        # Direct removal
        source_name = args.source
    else:
        # Interactive selection
        print(f"\n{'='*60}")
        choice = input("Enter number to remove (or 'q' to quit): ").strip()

        if choice.lower() == 'q':
            print("Cancelled")
            return

        try:
            index = int(choice) - 1
            if 0 <= index < len(source_list):
                source_name = source_list[index]
            else:
                print(f"[ERROR] Invalid number: {choice}")
                return
        except ValueError:
            print(f"[ERROR] Invalid input: {choice}")
            return

    # Remove source
    if remove_source(manager, source_name):
        # Show updated stats
        print(f"\n{'='*60}")
        print("Updated Knowledge Base:")
        print(f"{'='*60}")
        new_stats = manager.get_stats()
        print(f"Total documents: {new_stats['total_docs']}")
        print(f"Total sources: {len(new_stats['sources'])}")

if __name__ == "__main__":
    main()
