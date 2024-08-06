# This Menu Option Code is set to up to summarize and validate the CVE Mentioned in the Report
# I am going to allow the user an option to ask a general question about the report next; done
# I am making LLama take out the CVE's but the issue i am running into is that LLama is giving out random output too
    #  i might run the CVE extractor on it lol; i am in the process of doing it


#################### Fake TI Report with Fake CVE ##################################
fakeTIReport= """
Threat Details
Campaign Name: Operation Nightfall

Target Sector: Critical Infrastructure (Energy, Telecommunications)

Geographic Scope: Global

Tactics, Techniques, and Procedures (TTPs):

Initial Access: Spear-phishing emails with malicious attachments exploiting CVE-2024-0007.
Execution: Exploitation of zero-day vulnerabilities, specifically CVE-2024-0008, in SCADA systems used in the energy sector.
Persistence: Installation of backdoor malware (e.g., NightStalker RAT) allowing continuous access.
Privilege Escalation: Exploitation of CVE-2024-0009 to escalate privileges on compromised systems.
Defense Evasion: Use of code obfuscation and polymorphic malware to avoid detection.
Credential Access: Deployment of keyloggers and credential dumping tools exploiting CVE-2024-0010.
Discovery: Network reconnaissance using tools like Nmap and custom scripts.
Lateral Movement: Use of stolen credentials to move laterally within the network, exploiting CVE-2024-0012.
Exfiltration: Data exfiltration through encrypted channels (HTTPS and DNS tunneling).
Impact: Data theft and potential disruption of critical services.
Indicators of Compromise (IOCs)
IP Addresses:

203.0.113.15
198.51.100.45
192.0.2.75
Domains:

nightfall-apt.com
secure-infrastructure.net
dataleakage.info
File Hashes:

MD5: d41d8cd98f00b204e9800998ecf8427e
SHA-1: da39a3ee5e6b4b0d3255bfef95601890afd80709
SHA-256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
Malware Samples:

NightStalker RAT: Remote Access Trojan used for persistent access.
PhishLoader: Malware used to deliver malicious payloads via spear-phishing emails.
Mitigation Recommendations
Update and Patch Systems: Ensure all systems, especially those involving SCADA and critical infrastructure, are patched against CVE-2024-0007, CVE-2024-0008, CVE-2024-0009, and CVE-2024-0010.
Enhance Email Security: Implement advanced email filtering and phishing detection mechanisms.
Monitor Network Traffic: Use Intrusion Detection Systems (IDS) and Intrusion Prevention Systems (IPS) to monitor and block suspicious network activity.
Implement Multi-Factor Authentication (MFA): Enforce MFA to reduce the risk of credential theft and unauthorized access.
Conduct Security Training: Regularly train employees on recognizing and responding to phishing attempts and other social engineering tactics.
Backup Critical Data: Ensure regular backups of critical data and verify the integrity of backup procedures.
Conclusion
Operation Nightfall represents a significant threat to critical infrastructure sectors worldwide. By following the recommended mitigation steps and staying vigilant, organizations can reduce the risk of compromise and maintain the integrity of their systems.

Prepared by:
Cyber Science Lab
Cyber Threat Intelligence Team

"""



fakeTIReport2 = """
Certainly! Here's a detailed fake Threat Intelligence Report containing fictional CVEs for illustrative purposes.

---

## Threat Intelligence Report

### Executive Summary

This report highlights recent vulnerabilities identified in widely used software systems, emphasizing their potential impact, exploitation methods, and mitigation strategies. The vulnerabilities, assigned fictional CVE identifiers, are critical and require immediate attention to ensure system security and integrity.

### Vulnerability Details

#### 1. **CVE-2024-0007: Buffer Overflow in XYZ Web Server**

- **Description**: A buffer overflow vulnerability has been identified in XYZ Web Server version 3.4.1, which allows remote attackers to execute arbitrary code on the affected server.
- **Impact**: Successful exploitation could lead to a complete system compromise, allowing attackers to gain full control over the server and access sensitive data.
- **Exploitation Method**: An attacker can exploit this vulnerability by sending a specially crafted HTTP request to the server, causing a buffer overflow and enabling the execution of malicious code.
- **Mitigation**: Upgrade to XYZ Web Server version 3.4.2, which includes a patch for this vulnerability. Additionally, implement network segmentation and apply input validation to mitigate the risk.

#### 2. **CVE-2024-0008: SQL Injection in ABC Database Management System**

- **Description**: An SQL injection vulnerability exists in ABC Database Management System version 2.8.0, which allows attackers to execute arbitrary SQL commands.
- **Impact**: Exploiting this vulnerability can lead to unauthorized access to the database, data manipulation, and potential data breaches.
- **Exploitation Method**: Attackers can exploit this vulnerability by injecting malicious SQL code through input fields in the application, bypassing authentication mechanisms and executing arbitrary commands.
- **Mitigation**: Apply the security patch provided by ABC Inc. in version 2.8.1. Additionally, implement parameterized queries and input sanitization to prevent SQL injection attacks.

#### 3. **CVE-2024-0009: Remote Code Execution in GHI Software**

- **Description**: A remote code execution vulnerability exists in GHI Software version 4.2.1, allowing attackers to execute arbitrary commands on the affected system.
- **Impact**: This vulnerability can lead to full system compromise, data exfiltration, and lateral movement within the network.
- **Exploitation Method**: Attackers can exploit this vulnerability by sending specially crafted payloads to the application, which are then executed with elevated privileges.
- **Mitigation**: Upgrade to GHI Software version 4.2.2. Employ network security measures such as firewalls and intrusion detection systems to detect and prevent exploitation attempts.

#### 4. **CVE-2024-0010: Denial of Service in JKL Network Appliance**

- **Description**: A denial of service vulnerability has been identified in JKL Network Appliance version 5.6.3, which can be exploited to crash the system or degrade its performance.
- **Impact**: Successful exploitation can result in the unavailability of network services, impacting business operations and causing significant downtime.
- **Exploitation Method**: Attackers can exploit this vulnerability by sending a high volume of malformed packets to the network appliance, overwhelming its processing capabilities.
- **Mitigation**: Apply the latest firmware update provided by JKL Inc. Additionally, implement rate limiting and traffic filtering to protect against denial of service attacks.

### Conclusion

The vulnerabilities outlined in this report pose significant threats to the security and integrity of affected systems. Immediate action is required to apply the recommended patches and implement mitigation strategies to prevent exploitation. Continuous monitoring and security best practices are essential to safeguard against emerging threats.

### References

- XYZ Web Server: [www.xyzwebserver.com](http://www.xyzwebserver.com)
- ABC Database Management System: [www.abcdatabase.com](http://www.abcdatabase.com)
- DEF Web Application: [www.defwebapp.com](http://www.defwebapp.com)
- GHI Software: [www.ghisoftware.com](http://www.ghisoftware.com)
- JKL Network Appliance: [www.jklnetworks.com](http://www.jklnetworks.com)

---

This report is entirely fictional and intended for illustrative purposes only. The CVEs mentioned do not correspond to real vulnerabilities.


"""


fakeTIReport3 = """
Threat Intelligence Report

Report ID: TI-2024-0723

Date: July 23, 2024

Subject: Emerging Cyber Threats and Vulnerabilities

Summary:
This report outlines recent developments in cyber threats and vulnerabilities affecting global IT infrastructure. Key findings include the discovery of a critical vulnerability in widely used Apache Log4j library and emerging trends in ransomware-as-a-service (RaaS) targeting healthcare providers.

Details:

Apache Log4j Vulnerability (CVE-2024-0008):

Description: A critical flaw, designated CVE-2024-0008, has been identified in Apache Log4j versions 2.10.0 to 2.12.1, allowing remote code execution.
Impact: Attackers can exploit this vulnerability to execute arbitrary code on affected systems, potentially compromising sensitive data and disrupting operations.
Recommendation: Organizations are advised to immediately update to patched versions (2.12.2 or later) of Apache Log4j and implement network segmentation to mitigate potential exploits.
Ransomware-as-a-Service (RaaS) Targeting Healthcare:

Description: An increase in RaaS attacks targeting healthcare institutions has been observed, leveraging social engineering tactics and vulnerabilities in legacy systems.
Impact: These attacks not only encrypt critical patient data but also pose significant operational and financial risks to healthcare providers.
Recommendation: Healthcare organizations should prioritize cybersecurity awareness training, enhance endpoint protection measures, and ensure robust data backup and recovery strategies.
Phishing Campaigns Exploiting COVID-19 Themes:

Description: Cybercriminals continue to exploit global health crises, such as COVID-19, to launch phishing campaigns delivering malware payloads.
Impact: End users are deceived into downloading malicious attachments or clicking on links that harvest credentials or deploy ransomware.
Recommendation: Heightened vigilance, email filtering, and regular phishing simulations are essential to mitigate the risks associated with these campaigns.
Recommendations:

Patch Management: Implement a robust patch management strategy to promptly apply security updates for critical vulnerabilities like CVE-2024-0008.

Threat Intelligence Sharing: Foster collaboration among industry peers and threat intelligence providers to stay ahead of emerging threats and vulnerabilities.

Incident Response Preparation: Develop and test incident response plans tailored to address specific cyber threats, ensuring readiness to contain and mitigate potential breaches.

Conclusion:
The evolving cybersecurity landscape demands proactive measures to safeguard against sophisticated cyber threats and vulnerabilities. By prioritizing cybersecurity hygiene, adopting a proactive defense posture, and fostering collaboration across sectors, organizations can effectively mitigate risks and protect their digital assets.

"""







########################################## Begining of Menu Code #####################################################
from testingRagPart2 import askingLlamaForAdvice


#*******************Extract all text from the pdf


userPDFName = input("Please Enter the name of the pdf that you would like to analyze (please include the .pdf at the end as well).")

import fitz  # PyMuPDF
pdf_document = fitz.open(userPDFName)

# Initialize an empty string to store the text
all_text = ""

# Iterate through all the pages
for page_num in range(len(pdf_document)):
    page = pdf_document.load_page(page_num)
    text = page.get_text("text")
    all_text += text



#******************** Extract all CVE's mentioned in the report using only Llama3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import re


#Clear the cache
torch.cuda.empty_cache()


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    #torch_dtype=torch.float32,
    device_map="auto",
    #device_map = "auto",
    )


messages = [
    {
        "role": "system", "content": f"""You are a chatbot"""

    },
    {
    "role": "user", 
    "content": f"""

    Please extract all the CVEs mentioned in the text provided, seperated by commas. Do not output anything else except for the CVE numbers

    Here is a an example of an output to follow, onyl give the CVE numbers as output, dont say anyting else: 
    CVE-2021-44228,CVE-2023-23852,CVE-2022-22965,CVE-2022-41893

    Provided text: {all_text}

        
        
        
        
    """
    },
    ]
    
input_ids = tokenizer.apply_chat_template(
messages,
add_generation_prompt=True,
return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

outputs = model.generate(
input_ids,
max_new_tokens=700,
eos_token_id=terminators,
do_sample=True,
temperature=0.3,
top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
llmOutput = tokenizer.decode(response, skip_special_tokens=True)




#******************** Extract all CVE's mentioned in the report
# Might have change this to a simple way of llama3 taking the text and taking the CVE's out in a comma format and just putting that into it

import re

def extract_cves(text):
    # Regular expression pattern for CVE identifiers
    pattern = r'CVE-\d{4}-\d{4,7}'
    
    # Find all matches using the regex pattern
    cve_matches = re.findall(pattern, text)
    
    # Return unique CVE identifiers
    return list(set(cve_matches))


#cves = extract_cves(all_text)
cves = extract_cves(llmOutput)
#print("Extracted CVEs:", cves)

for cve in cves:
    print(cve)



#******************* Format the CVE's in a way to put them into varaibles that will allow us to open all the correct CVEs in the pdf

def extract_cve_numbers(cve_string):
    # Define the regex pattern
    pattern = r'CVE-(\d{4})-(\d{4,7})'
    
    # Search for the pattern in the string
    match = re.match(pattern, cve_string)
    
    if match:
        # Extract the numbers and convert them to integers
        first_set = int(match.group(1))
        second_set = match.group(2)  # Keep this as a string to preserve leading zeros
        return first_set, second_set
    else:
        raise ValueError("String does not match the CVE pattern")


def format_second_set(second_set):
    # Check the length of the second set string
    if len(second_set) == 4:
        # Take the first character and append 'xxx'
        formatted_str = second_set[0] + 'xxx'
    else:
        # Take the first two characters and append 'xxx'
        formatted_str = second_set[:2] + 'xxx'
    
    return formatted_str


#************************************ Knowing what CVE's we have, open our database and gather the relevant information about them
# some new stuff at the gym
import re



# Define the CVE pattern
cve_pattern = re.compile(r'CVE-\d{4}-\d{4,7}')

# Initialize a set to store unique CVEs
unique_cves = set()

# Find all CVEs in the input string
trueCves = cve_pattern.findall(all_text)
unique_cves.update(trueCves)

# Convert the set to a sorted list (optional)
unique_cves_list = sorted(unique_cves)

# Print or process the unique CVEs
for cve in unique_cves_list:
    print(cve)


# Check if the two lists have the same contents (ignoring order and duplicates)
def have_same_contents(list1, list2):
    return set(list1) == set(list2)

# Find strings in list1 that are not in list2
def find_difference(list1, list2):
    return list(set(list1) - set(list2))



#Llama, could you describe the cve not mentined in the list in 2 sentences. Do not give any other ouput
#the list would be the llama list compring the actual text


#********************** Menu Time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

menuOptionNumber = input(""" Welcome our CVE Rag. Please Select how you would like to analyze the Report. Options: 1. Give a Summary of the Report. Option 2. Try to Validate the Use of the CVE's being mentioned. Option 3. Ask a general question about the Report. Option 4. Exit Type 1, 2, 3 or 4""")

userContinue = "1"

while userContinue == "1":

    if menuOptionNumber == "1":
    
        #Clear the cache
        torch.cuda.empty_cache()
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            #torch_dtype=torch.float32,
            #device_map="cpu",
            device_map = "auto",
        )

        messages = [
        {
            "role": "system", 
            "content": f"""You are a ChatBot that summarizes threat Intellegence Reports. Your task is to summarize the report given you by the user. 
        
            """
        },

        # I am thinking of telling it to tell what the CVE's do as well
        {
            "role": "user", 
            "content": f""" Please Summarize the following Threat Intellegence Report {all_text}
        
            """
        },
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=700,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        print(tokenizer.decode(response, skip_special_tokens=True))


        userContinue = input("Would you like to continue? Please Enter 1 for yes and 2 for now")
    


    if menuOptionNumber == "2":


        ########################********************
        cveDescription = ""

        for cve in cves:

            try:

        ############################
                # Example usage
                #cve_string = "CVE-2021-34567"
                first_set, second_set = extract_cve_numbers(cve)
                #print(f"First set: {first_set}, Second set: {second_set}")

                formatedX = format_second_set(second_set)


                print(first_set)
                print(second_set)
                print(formatedX)

                ############################

                #with open('aFolder/CVE-2024-0007.json', 'r') as f:
                print("trying to open folder %s" %(cve))
                #print("%s.json" % (cve))
                #with open('afolder/%s.json' % (cve), 'r') as f:
        
                pizza = "cvelistV5/cves/%d/%s/CVE-%d-%s.json" % (first_set,formatedX,first_set,second_set)
                print("the pizza is " +  pizza)


                with open('cvelistV5/cves/%d/%s/CVE-%d-%s.json' % (first_set,formatedX,first_set,second_set), 'r') as f:


                    # Load JSON data
                    data = json.load(f)

                    # Extracting the CVE number
                    cve_number = data['cveMetadata']['cveId']

                    # Extracting vendor and product information
                    affected_info = data['containers']['cna']['affected'][0]
            
                    #### new stuff for cve-2024-3094, update the cve rag
                    try:
                        vendor_name = affected_info['vendor']
                        product_name = affected_info['product']

                    except KeyError as e:
                        affected_info = data['containers']['cna']['affected'][1]
                        vendor_name = affected_info['vendor']
                        product_name = affected_info['product']

            

                    description = data['containers']['cna']['descriptions'][0]['value']

                    # Format the extracted information into a string
                    #formatted_string = ""
                    cveDescription = cveDescription+ f"-CVE Number: {cve_number}, Vendor: {vendor_name}, Product: {product_name}, Description: {description}\n\n\n"

                    # Print the formatted string
                    output_file = "cvePDF.txt"
                    with open(output_file, 'a') as f:
                        f.write(text_content)
                        print("placed in")
                    print(cveDescription)


                    #print(cve)

            except FileNotFoundError:
                print("Could Not Find %s " %(cve))
                torch.cuda.empty_cache()

                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                )

                messages = [
                {"role": "system", "content": "You are a chat bot"},
                {"role": "user", "content": f"""These are some CVEs {cves}. I am about to give you a report, if there is a CVE that is not mentioned in the cves that I had provided to you before, could you describe in 2 sentences what that CVE is about. Provided Text: {all_text}"""},
                ]

                input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
                ).to(model.device)

                terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                outputs = model.generate(
                input_ids,
                max_new_tokens=500,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                )
                response = outputs[0][input_ids.shape[-1]:]
                print(tokenizer.decode(response, skip_special_tokens=True))
                llamaDesc = tokenizer.decode(response, skip_special_tokens=True)

                torch.cuda.empty_cache()



                askingLlamaForAdvice(llamaDesc)







        ########################**********************

        #Clear the cache
        torch.cuda.empty_cache()


        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            #torch_dtype=torch.float32,
            #device_map="cpu",
            device_map = "auto",
        )

        messages = [
            {
                "role": "system", 
                "content": f"""You are a chatbot that verifies the use of CVEs mentioned in a Threat Intelligence Report. In a Threat Intelligence Report, a CVE and its vulnerability is used correctly when it closely matches the Correct CVE description that is provided to you before the user prompt. Incorrect usage includes mentioning non-existent CVEs, misrepresenting the Correct CVE Description, or applying the CVE inaccurately in relation to the Correct CVE Description. A CVE in a report is also incorrect even if it describes a different vulnerability. If the two descriptions are note similar, it is incorrect even if the report correctly describes the vulnerability and its impact, and provides recommendations for mitigation

                Correct CVE Descriptions: {cveDescription}. 

                Verify each CVE mentioned in the report given by the user and indicate whether they are used correctly or not. If any CVE is used correctly or incorrectly, provide a detailed explanation with direct quotes from both the report and the Correct CVE Description.
        
                Example of correct usage:
                - **CVE-2023-1234: Buffer overflow vulnerability in software y. 
                Correct usage: This CVE is related to a buffer overflow vulnerability in software X. The correct CVE description provided earlier is a buffer overflow vulnerability in software y.
                * Explanation: The Threat Intelligence Report states "software y is facing buffer overlfow" and in the correct CVE states "software x is facing buffer overflow vulnerabilities". Therefore the two descriptions are very similar and hence the CVE is being correctly used in the report. 


                Example of Incorrect Usage:
                **CVE-2024-2342: Buffer Overflow in XYZ Web Server**
                * Incorrect usage: This CVE is not related to a buffer overflow vulnerability in XYZ Web Server. The correct CVE description provided earlier is a cross-site scripting (XSS) vulnerability in Palo Alto Networks PAN-OS software.
                * Explanation: The report incorrectly assigns a fictional CVE to a non-existent vulnerability in XYZ Web Server. The Threat Intelligence Report states "Buffer Overflow in XYZ Web Server" but the correct CVE description states "cross-site scripting (XSS) vulnerability". Therefore  the two descriptions are not similar and hence the CVE is being used incorrectly in the report. 



                Example of Incorrect Ussage:
                Based on the Correct CVE Description provided earlier, I will verify the usage of CVE-2024-0123 in the Threat Intelligence Report.

                **CVE-2024-0123: Server management interface in Microsoft do not connect with some users, making it prone to outages.**

                The report mentions:

                **TensorFlow Vulnerability (CVE-2024-0008):**

                Description: A critical flaw, designated CVE-2024-0008, has been identified in TensorFlow versions 0.1 - 0.2, allowing phishing scams.

                Impact: Attackers can exploit this vulnerability to scam users into giving personal information.

                **Incorrect usage:** The report incorrectly assigns CVE-2024-0008 to a vulnerability in TensorFlow, which is a different vulnerability from the Correct CVE Description. The Correct CVE Description describes a vulnerability in servers, whereas the report describes a vulnerability in phishing scams and versions.

                **Explanation:** The two descriptions are not similar, and the report incorrectly assigns the CVE to a different vulnerability. Therefore, the usage of CVE-2024-0008 in the report is incorrect.
        
        
                """
        },
        
        {
            "role": "user", 
            "content": f"""

            Please verify if the following CVEs have been used correctly in the following Threat Intelligence Report: {all_text}.
        
            """
        },
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=700,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

        response = outputs[0][input_ids.shape[-1]:]
        print(tokenizer.decode(response, skip_special_tokens=True))

        userContinue = input("Would you like to continue? Please Enter 1 for yes and 2 for now")



    if menuOptionNumber == "3":

        userQuestion = input("Please Enter the question you have about the report. ")

        #Clear the cache
        torch.cuda.empty_cache()


        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            #torch_dtype=torch.float32,
            #device_map="cpu",
            device_map = "auto",
        )

        messages = [
            {
                "role": "system", 
                "content": f"""You are a chatbot that asnwers question based off the text provided to you. 
                """
        },
        
        {
            "role": "user", 
            "content": f"""

            Provided Text: {all_text}. 

            Question by user: {userQuestion}
        
            """
        },
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=700,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

        response = outputs[0][input_ids.shape[-1]:]
        print(tokenizer.decode(response, skip_special_tokens=True))

        userContinue = input("Would you like to continue? Please Enter 1 for yes and 2 for now")


    if menuOptionNumber == "4":
        print("you have succsefully exited the application")
        userContinue = "4"


########################################## End of Menu Code ###########################################################

