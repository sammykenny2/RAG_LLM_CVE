import sys
from pathlib import Path
# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm

# Import configuration
from config import (
    CVE_V5_PATH,
    CVE_V4_PATH,
    CVE_DESCRIPTION_PATH,
    DEFAULT_SCHEMA
)

# Import CVE lookup utilities
from core.cve_lookup import extract_cve_fields



def get_available_years(base_path: str) -> list[int]:
    """Scan directory for available year folders."""
    if not os.path.exists(base_path):
        return []

    years = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 4:
            years.append(int(item))
    return sorted(years)


def process_cve_directory(folder_path: str, source_label: str, output_file: str, processed_cves: set, extension: str, verbose: bool = False):
    """Process CVE files from a directory and append to output file.

    Args:
        folder_path: Path to CVE directory
        source_label: Label for logging (e.g., "V5 CVE Schema (2024)")
        output_file: Output file path
        processed_cves: Set of already processed CVE IDs (for deduplication)
        extension: Output format ('txt' or 'jsonl')
        verbose: Enable detailed logging (default: False)

    Returns:
        tuple: (processed_count, skipped_count)
    """
    if not os.path.exists(folder_path):
        print(f"The directory {folder_path} does not exist. Skipping {source_label}.")
        return 0, 0

    processed_count = 0
    skipped_count = 0
    print(f"\nProcessing {source_label} from: {folder_path}")

    # Collect all files first
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            all_files.append((root, file_name))

    # Process files with progress bar or verbose output
    file_iterator = all_files if verbose else tqdm(all_files, desc=f"  {source_label}", unit="file")

    for root, file_name in file_iterator:
        file_path = os.path.join(root, file_name)

        if verbose:
            print(f"Currently reading file: {file_name}")
            print(f"Full path: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Use shared CVE extraction function from core module
            cve_number, vendor_name, product_name, description = extract_cve_fields(data, file_name)

            # Check if already processed (V5 takes priority)
            if cve_number in processed_cves:
                if verbose:
                    print(f"Skipped (already processed): {cve_number}")
                skipped_count += 1
                continue

            # Format output based on extension
            if extension == 'txt':
                # Text format (lossy, human-readable)
                text_content = (
                    f"- CVE Number: {cve_number}, Vendor: {vendor_name}, "
                    f"Product: {product_name}, Description: {description}\n\n"
                )
            else:  # jsonl
                # JSONL format (lossless, machine-readable)
                # Determine schema type from data
                schema_type = 'v5' if 'cveMetadata' in data else 'v4'

                # Extract year from CVE ID
                year = cve_number.split('-')[1] if '-' in cve_number else 'unknown'

                json_record = {
                    'cve_id': cve_number,
                    'vendor': vendor_name,
                    'product': product_name,
                    'description': description,
                    'schema': schema_type,
                    'year': year
                }
                text_content = json.dumps(json_record, ensure_ascii=False) + '\n'

            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(text_content)
                if verbose:
                    print("placed in")

            # Track processed CVE
            processed_cves.add(cve_number)
            processed_count += 1

        except (KeyError, IndexError, TypeError) as e:
            if verbose:
                print(f"Could not parse CVE data in {file_path}: {e}")
        except Exception as e:
            if verbose:
                print(f"Could not read file {file_path}: {e}")

    return processed_count, skipped_count


def main():
    """Main function with command-line argument support."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Extract CVE descriptions from v5 and v4 schema JSON feeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_cve.py                       # Extract current year from V5 (fastest)
  python extract_cve.py --year=2024           # Extract 2024 from V5
  python extract_cve.py --schema=all          # Extract from both V5 and V4 (with dedup)
  python extract_cve.py --schema=v4           # Extract from V4 only
  python extract_cve.py --year=all --schema=all  # Extract all years from both schemas
        """
    )
    parser.add_argument(
        '--year',
        type=str,
        default=str(datetime.now().year),
        help='Year(s) to process: single year (2024), comma-separated (2023,2024), or "all" (default: current year)'
    )
    parser.add_argument(
        '--schema',
        type=str,
        choices=['v5', 'v4', 'all'],
        default='v5',
        help='Schema to process: v5 (fastest), v4, or all (both with deduplication) (default: v5)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable detailed logging (default: show progress only)'
    )
    parser.add_argument(
        '--extension',
        type=str,
        choices=['txt', 'jsonl'],
        default='jsonl',
        help='Output format: txt (human-readable, lossy) or jsonl (machine-readable, lossless) (default: jsonl)'
    )
    args = parser.parse_args()

    # Determine which years to process
    if args.year.lower() == 'all':
        # Scan directories based on schema selection
        v5_years = get_available_years(str(CVE_V5_PATH)) if args.schema in ['v5', 'all'] else []
        v4_years = get_available_years(str(CVE_V4_PATH)) if args.schema in ['v4', 'all'] else []
        years = sorted(set(v5_years + v4_years))
        if not years:
            print(f"Error: No year directories found in CVE feeds for schema '{args.schema}'.")
            return
        print(f"Processing all available years: {years}")
    else:
        # Parse comma-separated years
        try:
            years = [int(y.strip()) for y in args.year.split(',')]
        except ValueError:
            print(f"Error: Invalid year format '{args.year}'. Use single year (2024), comma-separated (2023,2024), or 'all'")
            return

    # Determine output file name using config path
    output_file = f"{CVE_DESCRIPTION_PATH}.{args.extension}"

    # Ensure output directory exists
    output_dir = CVE_DESCRIPTION_PATH.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Clear output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Track processed CVEs to avoid duplicates (only when schema=all)
    processed_cves = set() if args.schema == 'all' else None

    total_v5_count = 0
    total_v4_count = 0
    total_v4_skipped = 0

    # Print header
    print(f"\n{'='*60}")
    print(f"Extract CVE Descriptions - Export to File")
    print(f"{'='*60}\n")
    print(f"Years:       {years}")
    print(f"Schema:      {args.schema}")
    print(f"Format:      {args.extension} ({'human-readable' if args.extension == 'txt' else 'machine-readable'})")
    print(f"Output:      {output_file}")
    print(f"{'='*60}")

    # Process each year
    for year in years:
        print(f"\n{'='*60}")
        print(f"Processing Year: {year}")
        print(f"{'='*60}")

        v5_folder = str(CVE_V5_PATH / str(year))
        v4_folder = str(CVE_V4_PATH / str(year))

        # Process based on schema selection
        if args.schema in ['v5', 'all']:
            # Process V5 schema
            if processed_cves is not None:
                v5_count, _ = process_cve_directory(v5_folder, f"V5 CVE Schema ({year})", output_file, processed_cves, args.extension, args.verbose)
            else:
                # No dedup needed for v5-only
                v5_count, _ = process_cve_directory(v5_folder, f"V5 CVE Schema ({year})", output_file, set(), args.extension, args.verbose)
            total_v5_count += v5_count

        if args.schema in ['v4', 'all']:
            # Process V4 schema
            if processed_cves is not None:
                v4_count, v4_skipped = process_cve_directory(v4_folder, f"V4 CVE Schema ({year})", output_file, processed_cves, args.extension, args.verbose)
                total_v4_skipped += v4_skipped
            else:
                # No dedup needed for v4-only
                v4_count, _ = process_cve_directory(v4_folder, f"V4 CVE Schema ({year})", output_file, set(), args.extension, args.verbose)
            total_v4_count += v4_count

        # Print summary based on schema
        if args.schema == 'v5':
            print(f"\nYear {year} Summary: V5={v5_count}")
        elif args.schema == 'v4':
            print(f"\nYear {year} Summary: V4={v4_count}")
        else:  # all
            print(f"\nYear {year} Summary: V5={v5_count}, V4={v4_count} (skipped {v4_skipped} duplicates), Total unique={v5_count + v4_count}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    print(f"Years:       {years}")
    print(f"Schema:      {args.schema}")

    if args.schema == 'v5':
        print(f"Total CVEs:  {total_v5_count}")
    elif args.schema == 'v4':
        print(f"Total CVEs:  {total_v4_count}")
    else:  # all
        print(f"V5 CVEs:     {total_v5_count}")
        print(f"V4 CVEs:     {total_v4_count} (skipped {total_v4_skipped} duplicates)")
        print(f"Unique:      {total_v5_count + total_v4_count}")

    print(f"Output:      {output_file}")
    print(f"{'='*60}")

    print(f"\n✅ Extraction completed successfully!")
    print(f"\n💡 File created at: {output_file}")
    if args.extension == 'jsonl':
        print(f"\n💡 JSONL format preserves all metadata (schema, year, etc.)")
        print(f"   Use with build_embeddings.py option 3 for lossless processing")


if __name__ == "__main__":
    main()
