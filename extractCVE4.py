import os
import json

# Specify the path to the main folder
main_folder_path = "../cvelist/2024"

if not os.path.exists(main_folder_path):
    print(f"The directory {main_folder_path} does not exist.")
else:
    for root, dirs, files in os.walk(main_folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            print(f"Currently reading file: {file_name}")
            print(f"Full path: {file_path}")

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                    cve_number = data['CVE_data_meta']['ID']

                    vendor_data = data['affects']['vendor']['vendor_data'][0]
                    vendor_name = vendor_data['vendor_name']
                    product_name = vendor_data['product']['product_data'][0]['product_name']

                    description_data = data['description']['description_data']
                    description = next(
                        (item['value'] for item in description_data if item.get('lang') == 'eng'),
                        description_data[0]['value'] if description_data else 'No description available.'
                    )

                    text_content = (
                        f"- CVE Number: {cve_number}, Vendor: {vendor_name}, "
                        f"Product: {product_name}, Description: {description}\n\n"
                    )

                    with open('CVEDescription2024.txt', 'a', encoding='utf-8') as f:
                        f.write(text_content)
                        print("placed in")

            except Exception as e:
                print(f"Could not read file {file_path}: {e}")
