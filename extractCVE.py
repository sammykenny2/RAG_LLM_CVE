import os
import json
import argparse
from datetime import datetime

def extract_cve_info(data: dict) -> tuple[str, str, str, str] | None:
    """Extract CVE information from v5 or v4 schema.

    Returns: (cve_number, vendor_name, product_name, description) or None if failed
    """
    try:
        # Try v5 schema first
        if 'cveMetadata' in data:
            cve_number = data['cveMetadata']['cveId']
            affected_info = data['containers']['cna']['affected'][0]
            vendor_name = affected_info.get('vendor') or affected_info.get('vendor_name', 'Unknown')
            product_name = affected_info.get('product') or affected_info.get('product_name', 'Unknown')
            description = data['containers']['cna']['descriptions'][0]['value']
            return cve_number, vendor_name, product_name, description

        # Try v4 schema
        elif 'CVE_data_meta' in data:
            cve_number = data['CVE_data_meta']['ID']
            vendor_data = data['affects']['vendor']['vendor_data'][0]
            vendor_name = vendor_data['vendor_name']
            product_name = vendor_data['product']['product_data'][0]['product_name']

            description_data = data['description']['description_data']
            description = next(
                (item['value'] for item in description_data if item.get('lang') == 'eng'),
                description_data[0]['value'] if description_data else 'No description available.'
            )
            return cve_number, vendor_name, product_name, description

        else:
            return None

    except (KeyError, IndexError, TypeError) as e:
        return None


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


def process_cve_directory(folder_path: str, source_label: str, output_file: str, processed_cves: set):
    """Process CVE files from a directory and append to output file.

    Args:
        folder_path: Path to CVE directory
        source_label: Label for logging (e.g., "V5 CVE Schema (2024)")
        output_file: Output file path
        processed_cves: Set of already processed CVE IDs (for deduplication)

    Returns:
        tuple: (processed_count, skipped_count)
    """
    if not os.path.exists(folder_path):
        print(f"The directory {folder_path} does not exist. Skipping {source_label}.")
        return 0, 0

    processed_count = 0
    skipped_count = 0
    print(f"\nProcessing {source_label} from: {folder_path}")

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            print(f"Currently reading file: {file_name}")
            print(f"Full path: {file_path}")

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                result = extract_cve_info(data)

                if result:
                    cve_number, vendor_name, product_name, description = result

                    # Check if already processed (V5 takes priority)
                    if cve_number in processed_cves:
                        print(f"Skipped (already processed): {cve_number}")
                        skipped_count += 1
                        continue

                    text_content = (
                        f"- CVE Number: {cve_number}, Vendor: {vendor_name}, "
                        f"Product: {product_name}, Description: {description}\n\n"
                    )

                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(text_content)
                        print("placed in")

                    # Track processed CVE
                    processed_cves.add(cve_number)
                    processed_count += 1
                else:
                    print(f"Unknown schema format in {file_path}")

            except Exception as e:
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
  python extractCVE.py                       # Extract current year from V5 (fastest)
  python extractCVE.py --year=2024           # Extract 2024 from V5
  python extractCVE.py --schema=all          # Extract from both V5 and V4 (with dedup)
  python extractCVE.py --schema=v4           # Extract from V4 only
  python extractCVE.py --year=all --schema=all  # Extract all years from both schemas
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
        help='Schema to process: v5 (default, fastest), v4, or all (both with deduplication)'
    )
    args = parser.parse_args()

    # Determine which years to process
    if args.year.lower() == 'all':
        # Scan directories based on schema selection
        v5_years = get_available_years("../cvelistV5/cves") if args.schema in ['v5', 'all'] else []
        v4_years = get_available_years("../cvelist") if args.schema in ['v4', 'all'] else []
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

    # Determine output file name
    if len(years) == 1:
        output_file = f"CVEDescription{years[0]}.txt"
    else:
        year_range = f"{min(years)}-{max(years)}"
        output_file = f"CVEDescription{year_range}.txt"

    # Clear output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Cleared existing output file: {output_file}")

    # Track processed CVEs to avoid duplicates (only when schema=all)
    processed_cves = set() if args.schema == 'all' else None

    total_v5_count = 0
    total_v4_count = 0
    total_v4_skipped = 0

    # Process each year
    for year in years:
        print(f"\n{'='*60}")
        print(f"Processing Year: {year}")
        print(f"{'='*60}")

        v5_folder = f"../cvelistV5/cves/{year}"
        v4_folder = f"../cvelist/{year}"

        # Process based on schema selection
        if args.schema in ['v5', 'all']:
            # Process V5 schema
            if processed_cves is not None:
                v5_count, _ = process_cve_directory(v5_folder, f"V5 CVE Schema ({year})", output_file, processed_cves)
            else:
                # No dedup needed for v5-only
                v5_count, _ = process_cve_directory(v5_folder, f"V5 CVE Schema ({year})", output_file, set())
            total_v5_count += v5_count

        if args.schema in ['v4', 'all']:
            # Process V4 schema
            if processed_cves is not None:
                v4_count, v4_skipped = process_cve_directory(v4_folder, f"V4 CVE Schema ({year})", output_file, processed_cves)
                total_v4_skipped += v4_skipped
            else:
                # No dedup needed for v4-only
                v4_count, _ = process_cve_directory(v4_folder, f"V4 CVE Schema ({year})", output_file, set())
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
    print(f"=== Extraction Complete ===")
    print(f"{'='*60}")
    print(f"Years processed: {years}")
    print(f"Schema: {args.schema}")

    if args.schema == 'v5':
        print(f"Total CVEs extracted: {total_v5_count}")
    elif args.schema == 'v4':
        print(f"Total CVEs extracted: {total_v4_count}")
    else:  # all
        print(f"Total V5 CVEs: {total_v5_count}")
        print(f"Total V4 CVEs: {total_v4_count} (skipped {total_v4_skipped} duplicates)")
        print(f"Unique CVEs extracted: {total_v5_count + total_v4_count}")
        if processed_cves:
            print(f"Memory usage: Tracked {len(processed_cves)} CVE IDs (~{len(processed_cves) * 20 // 1024} KB)")

    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    main()
