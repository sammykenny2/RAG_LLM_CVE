import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Specify the path to the main folder
main_folder_path = "../cvelistV5/cves/2024"

if not os.path.exists(main_folder_path):
    print(f"The directory {main_folder_path} does not exist.")
else:
    # Initialize an empty string to store all CVE descriptions
    cve_descriptions = ""

    # Iterate through the directory tree
    for root, dirs, files in os.walk(main_folder_path):
        for file_name in files:
            # Construct the full file path
            file_path = os.path.join(root, file_name)
            
            # Print the current file being read, including its name and full path
            print(f"Currently reading file: {file_name}")
            print(f"Full path: {file_path}")
            
            # Open and read the file
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    # Load JSON data
                    data = json.load(file)

                    # Extracting the CVE number
                    cve_number = data['cveMetadata']['cveId']

                    # Extracting vendor and product information
                    affected_info = data['containers']['cna']['affected'][0]
                
                    # Handling multiple affected entries
                    try:
                        vendor_name = affected_info['vendor']
                        product_name = affected_info['product']
                    except KeyError:
                        affected_info = data['containers']['cna']['affected'][1]
                        vendor_name = affected_info['vendor']
                        product_name = affected_info['product']

                    description = data['containers']['cna']['descriptions'][0]['value']

                    # Format the extracted information into a string
                    cve_description = f"- CVE Number: {cve_number}, Vendor: {vendor_name}, Product: {product_name}, Description: {description}\n\n"

                    text_content = f"- CVE Number: {cve_number}, Vendor: {vendor_name}, Product: {product_name}, Description: {description}\n\n"
                    output_file = "CVEDescription2024.txt"

                    with open(output_file, 'a') as f:
                        f.write(text_content)
                        print("placed in")
                    
                    # Append the description to the cumulative string
                    cve_descriptions += cve_description

            except Exception as e:
                print(f"Could not read file {file_path}: {e}")

