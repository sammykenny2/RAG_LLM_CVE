import os
import json

MAIN_FOLDER_PATH = "../cvelist/2024"
OUTPUT_FILE = "CVEDescription2024.txt"

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


if not os.path.exists(MAIN_FOLDER_PATH):
    print(f"The directory {MAIN_FOLDER_PATH} does not exist.")
else:
    for root, dirs, files in os.walk(MAIN_FOLDER_PATH):
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
                    text_content = (
                        f"- CVE Number: {cve_number}, Vendor: {vendor_name}, "
                        f"Product: {product_name}, Description: {description}\n\n"
                    )

                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                        f.write(text_content)
                        print("placed in")
                else:
                    print(f"Unknown schema format in {file_path}")

            except Exception as e:
                print(f"Could not read file {file_path}: {e}")
