"""
CVE lookup and parsing module.
Handles extraction and parsing of CVE records from JSON feeds (V4 and V5 schemas).
"""

import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict

# Import configuration
from config import (
    CVE_V5_PATH,
    CVE_V4_PATH,
    DEFAULT_SCHEMA,
    VERBOSE_LOGGING
)


def extract_cve_numbers(cve_string: str) -> Tuple[int, str]:
    """
    Extract year and ID from CVE string.

    Args:
        cve_string: CVE identifier (e.g., "CVE-2024-12345")

    Returns:
        tuple: (year, id) where year is int and id is str

    Raises:
        ValueError: If string doesn't match CVE pattern
    """
    pattern = r'CVE-(\d{4})-(\d{4,7})'
    match = re.match(pattern, cve_string)
    if match:
        first_set = int(match.group(1))  # year
        second_set = match.group(2)  # id
        return first_set, second_set
    else:
        raise ValueError(f"String does not match the CVE pattern: {cve_string}")


def format_second_set(second_set: str) -> str:
    """
    Format CVE ID for directory structure.

    Examples:
        "0001" -> "0xxx"
        "12345" -> "12xxx"

    Args:
        second_set: CVE ID portion

    Returns:
        str: Formatted directory name
    """
    if len(second_set) == 4:
        return second_set[0] + 'xxx'
    else:
        return second_set[:2] + 'xxx'


def load_cve_record(
    cve: str,
    first_set: int,
    second_set: str,
    schema: str = None
) -> Optional[Dict]:
    """
    Load CVE JSON record based on schema configuration.

    Args:
        cve: CVE identifier (e.g., "CVE-2024-12345")
        first_set: Year (e.g., 2024)
        second_set: ID (e.g., "12345")
        schema: 'v5', 'v4', or 'all' (default from config)

    Returns:
        dict: CVE JSON data, or None if not found
    """
    schema = schema if schema is not None else DEFAULT_SCHEMA
    formatted_x = format_second_set(second_set)

    if schema == 'v5':
        # V5 only
        v5_path = CVE_V5_PATH / str(first_set) / formatted_x / f"CVE-{first_set}-{second_set}.json"
        if v5_path.exists():
            with open(v5_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    elif schema == 'v4':
        # V4 only
        v4_path = CVE_V4_PATH / str(first_set) / formatted_x / f"CVE-{first_set}-{second_set}.json"
        if v4_path.exists():
            with open(v4_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    else:  # schema == 'all'
        # Try V5 first, fallback to V4
        v5_path = CVE_V5_PATH / str(first_set) / formatted_x / f"CVE-{first_set}-{second_set}.json"
        if v5_path.exists():
            with open(v5_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Fallback to V4
        v4_path = CVE_V4_PATH / str(first_set) / formatted_x / f"CVE-{first_set}-{second_set}.json"
        if v4_path.exists():
            with open(v4_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        return None


def extract_cve_fields(data: Dict, fallback_id: str) -> Tuple[str, str, str, str]:
    """
    Extract CVE fields from V5 or V4 schema (auto-detects schema).

    Args:
        data: CVE JSON data
        fallback_id: CVE ID to use if not found in JSON

    Returns:
        tuple: (cve_number, vendor, product, description)
    """
    # Try V5 schema first
    if "cveMetadata" in data:
        cve_number = data["cveMetadata"].get("cveId", fallback_id)
        cna = data.get("containers", {}).get("cna", {})
        descriptions = cna.get("descriptions", [])
        description = descriptions[0].get("value") if descriptions else "No description available."
        affected = cna.get("affected", [])
        entry = next((item for item in affected if isinstance(item, dict)), {})
        vendor = entry.get("vendor") or entry.get("vendor_name", "Unknown")
        product = entry.get("product") or entry.get("product_name", "Unknown")
        return cve_number, vendor, product, description

    # V4 schema
    legacy_meta = data.get("CVE_data_meta", {})
    cve_number = legacy_meta.get("ID", fallback_id)
    description_data = data.get("description", {}).get("description_data", [])
    description = description_data[0].get("value") if description_data else "No description available."
    vendor_data = data.get("affects", {}).get("vendor", {}).get("vendor_data", [])
    vendor_entry = vendor_data[0] if vendor_data else {}
    vendor = vendor_entry.get("vendor_name", "Unknown")
    product_data = vendor_entry.get("product", {}).get("product_data", [])
    product_entry = product_data[0] if product_data else {}
    product = product_entry.get("product_name", "Unknown")
    return cve_number, vendor, product, description


def lookup_cve(cve: str, schema: str = None) -> Optional[Dict[str, str]]:
    """
    Lookup CVE and return structured information.

    Args:
        cve: CVE identifier (e.g., "CVE-2024-12345")
        schema: 'v5', 'v4', or 'all' (default from config)

    Returns:
        dict: CVE information with keys:
            - cve_id: CVE identifier
            - vendor: Vendor name
            - product: Product name
            - description: Vulnerability description
        Returns None if CVE not found
    """
    try:
        first_set, second_set = extract_cve_numbers(cve)
    except ValueError as e:
        if VERBOSE_LOGGING:
            print(f"[WARNING] Invalid CVE format: {cve} - {e}")
        return None

    data = load_cve_record(cve, first_set, second_set, schema)

    if data is None:
        return None

    cve_number, vendor, product, description = extract_cve_fields(data, cve)

    return {
        'cve_id': cve_number,
        'vendor': vendor,
        'product': product,
        'description': description
    }


def extract_cves_regex(text: str) -> list[str]:
    """
    Extract all CVE identifiers from text using regex.

    Args:
        text: Text to search for CVEs

    Returns:
        list: Unique CVE identifiers found
    """
    pattern = r'CVE-\d{4}-\d{4,7}'
    cve_matches = re.findall(pattern, text)
    return list(set(cve_matches))


def format_cve_description(cve_info: Dict[str, str]) -> str:
    """
    Format CVE information as a structured string.

    Args:
        cve_info: CVE information dict from lookup_cve()

    Returns:
        str: Formatted description
    """
    return (
        f"-CVE Number: {cve_info['cve_id']}, "
        f"Vendor: {cve_info['vendor']}, "
        f"Product: {cve_info['product']}, "
        f"Description: {cve_info['description']}"
    )


def batch_lookup_cves(cves: list[str], schema: str = None) -> Dict[str, Optional[Dict[str, str]]]:
    """
    Lookup multiple CVEs at once.

    Args:
        cves: List of CVE identifiers
        schema: 'v5', 'v4', or 'all' (default from config)

    Returns:
        dict: Mapping of CVE ID to CVE info dict (None if not found)
    """
    results = {}
    for cve in cves:
        results[cve] = lookup_cve(cve, schema)

    return results
