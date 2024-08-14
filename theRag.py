# trying to input a suggestion
#Converge-Threat-Intel-Report-2024-MAY.pdf
#Converge-Threat-Intel-Report-2024-APR.pdf"
#fakeReport2.pdf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers.utils import logging
logging.set_verbosity_error()

import re
import fitz 
import json 
import gc
import numpy as np
import pandas as pd
from sentence_transformers import util, SentenceTransformer


torch.cuda.empty_cache()

llamaSug = ""

# Initialize the model once
def initialize_model():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return tokenizer, model

def cleanup_model(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()

tokenizer, model = initialize_model()

# Extract text from the PDF
def extract_text_from_pdf(pdf_name):
    pdf_document = fitz.open(pdf_name)
    all_text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        all_text += text
    return all_text

userPDFName = input("Please Enter the name of the pdf that you would like to analyze (please include the .pdf at the end as well).")
all_text = extract_text_from_pdf(userPDFName)

# Extract CVEs using LLama
def extract_cves_llama(text, tokenizer, model):
    messages = [
        {"role": "system", "content": "You are a chatbot"},
        {"role": "user", "content": f"Please extract all the CVEs mentioned in the text provided, separated by commas. Do not output anything else except for the CVE numbers.\n\nProvided text: {text}"}
    ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=700, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.3, top_p=0.9)
    response = outputs[0][input_ids.shape[-1]:]
    llm_output = tokenizer.decode(response, skip_special_tokens=True)
    
    return llm_output

llm_output = extract_cves_llama(all_text, tokenizer, model)

# Extract CVEs using regex
def extract_cves_regex(text):
    pattern = r'CVE-\d{4}-\d{4,7}'
    cve_matches = re.findall(pattern, text)
    return list(set(cve_matches))

cves = extract_cves_regex(llm_output)

# Format CVE numbers for database lookup
def extract_cve_numbers(cve_string):
    pattern = r'CVE-(\d{4})-(\d{4,7})'
    match = re.match(pattern, cve_string)
    if match:
        first_set = int(match.group(1))
        second_set = match.group(2)
        return first_set, second_set
    else:
        raise ValueError("String does not match the CVE pattern")

def format_second_set(second_set):
    if len(second_set) == 4:
        return second_set[0] + 'xxx'
    else:
        return second_set[:2] + 'xxx'
    

def asking_llama_for_advice(cveDesp: str) -> str:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        csv_path = 'test6.csv'

        chunk_embeddings_df = pd.read_csv(csv_path)

        chunk_embeddings_df["embedding"] = chunk_embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

        embeddings = torch.tensor(np.stack(chunk_embeddings_df["embedding"].tolist(), axis=0), dtype=torch.float32).to(device)

        text_chunks = chunk_embeddings_df.to_dict(orient="records")

        embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=device)

        indices = retrieve_context(query=cveDesp, embeddings=embeddings, model=embedding_model)

        context_items = [text_chunks[i] for i in indices]

        context_str = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])


        messages = [
            {"role": "system", "content": f""""You are a Q&A Assistant. You will be provided with relevant information about various CVEs. Based on this information, your task is to recommend the CVE number that most closely matches the description of the vulnerability.
            Provided Relevant Information: {context_str} 
            """},
            {"role": "user", "content": f""" Hello, could you recomend me a CVE that most closly resembles this chunk of text based of the relevant information that you have. Chunk of text: {cveDesp}"""},
            ]   

        ############
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        outputs = model.generate(input_ids, max_new_tokens=700, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.3, top_p=0.9)
        response = outputs[0][input_ids.shape[-1]:]
        llamaSug = tokenizer.decode(response, skip_special_tokens=True)
    
        return llamaSug


def retrieve_context(query: str,
                embeddings: torch.tensor,
                model: SentenceTransformer,
                n_resources_to_return: int=5):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

 
    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    _, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

    return indices

    




        ##########


# Load CVE descriptions from database
cve_description = ""

for cve in cves:
    try:
        first_set, second_set = extract_cve_numbers(cve)
        formatted_x = format_second_set(second_set)
        file_path = f"../cvelistV5/cves/{first_set}/{formatted_x}/CVE-{first_set}-{second_set}.json"

        with open(file_path, 'r') as f:
            data = json.load(f)
            cve_number = data['cveMetadata']['cveId']
            affected_info = data['containers']['cna']['affected'][0]

            try:
                vendor_name = affected_info['vendor']
                product_name = affected_info['product']
            except KeyError:
                affected_info = data['containers']['cna']['affected'][1]
                vendor_name = affected_info['vendor']
                product_name = affected_info['product']

            description = data['containers']['cna']['descriptions'][0]['value']
            cve_description += f"-CVE Number: {cve_number}, Vendor: {vendor_name}, Product: {product_name}, Description: {description}\n\n\n"

    except FileNotFoundError:
        print(f"Could Not Find {cve}, I will report it you later")
        
        torch.cuda.empty_cache()

        messages = [
            {"role": "system", "content": "You are a chat bot."},
            {"role": "user", "content": f"""In 2 sentences, could you describe the use of {cve} in the provided text. Provided text: {all_text}.
"""},
            ]
        

        

        ############
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        outputs = model.generate(input_ids, max_new_tokens=700, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.3, top_p=0.9)
        response = outputs[0][input_ids.shape[-1]:]
        llamaDesc = tokenizer.decode(response, skip_special_tokens=True)
        




        ##########
        llamaSug = asking_llama_for_advice(llamaDesc)


        

# Menu options
def menu_option_1(all_text, tokenizer, model):
    messages = [
        {"role": "system", "content": "You are a ChatBot that summarizes threat intelligence reports. Your task is to summarize the report given to you by the user."},
        {"role": "user", "content": f"Please summarize the following Threat Intelligence Report: {all_text}"}
    ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=700, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.3, top_p=0.9)
    response = outputs[0][input_ids.shape[-1]:]
    summary = tokenizer.decode(response, skip_special_tokens=True)
    print(summary)

def menu_option_2(all_text, cve_description, tokenizer, model):
    
    messages = [
            {
                "role": "system", 
                "content": f"""You are a chatbot that verifies the correct use of CVEs (Common Vulnerabilities and Exposures) mentioned in a Threat Intelligence Report. A CVE is used correctly when it closely matches the provided Correct CVE Description. Incorrect usage includes citing non-existent CVEs, misrepresenting the Correct CVE Description, or inaccurately applying the CVE.
                            Correct CVE Descriptions:
                            {cve_description}
                            Instructions:
                            1. Verify each CVE mentioned in the user-provided report.
                            2. Indicate whether each CVE is used correctly or not.
                            3. You must provide a detailed explanation with direct quotes from both the report and the Correct CVE Description.
                            A CVE in the report is incorrect if it describes a different vulnerability, even if the report accurately describes the vulnerability and its impact, and provides mitigation recommendations.

                            
                            Example Output:
                            CVE-2023-1234: Buffer Overflow in Software X

                            Correct CVE Description:

                            CVE-2023-1234: Buffer overflow vulnerability in software Y.
                            Report Excerpt:

                            "Software X is facing a buffer overflow vulnerability."
                            Verification:

                            Correct Usage
                            Explanation: The report mentions a buffer overflow in Software X, which closely matches the correct CVE description of a buffer overflow in software Y. Although the software names differ, the nature of the vulnerability is accurately represented.
                            CVE-2024-2342: Cross-Site Scripting (XSS) in XYZ Web Server

                            Correct CVE Description:

                            CVE-2024-2342: Cross-Site Scripting (XSS) vulnerability in Palo Alto Networks PAN-OS software.
                            Report Excerpt:

                            "XYZ Web Server is vulnerable to a buffer overflow attack."
                            Verification:

                            Incorrect Usage
                            Explanation: The report incorrectly describes a buffer overflow vulnerability in XYZ Web Server, while the correct CVE description refers to a Cross-Site Scripting (XSS) vulnerability in Palo Alto Networks PAN-OS software. The descriptions do not match, indicating incorrect usage of the CVE.

        
                """
        },
        
        {
            "role": "user", 
            "content": f"""

            Please verify if the following CVEs have been used correctly in the following Threat Intelligence Report: {all_text}.
        
            """
        },
        ]


    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=700, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.3, top_p=0.9)
    response = outputs[0][input_ids.shape[-1]:]
    verification = tokenizer.decode(response, skip_special_tokens=True)
    print(verification)

    print(llamaSug)

def menu_option_3(all_text, user_question, tokenizer, model):
    messages = [
        {"role": "system", "content": "You are a chatbot that answers questions based on the text provided to you."},
        {"role": "user", "content": f"{user_question}\n\nReport: {all_text}"}
    ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=700, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.3, top_p=0.9)
    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)
    print(answer)

# Main menu
user_continue = "1"

while user_continue == "1":
    torch.cuda.empty_cache()
    menu_option_number = input("Welcome to our CVE Rag. Please select how you would like to analyze the report. Options: 1. Give a Summary of the Report. 2. Validate the Use of the CVEs mentioned. 3. Ask a general question about the Report. 4. Exit. Type 1, 2, 3, or 4: ")
    if menu_option_number == "1":
        menu_option_1(all_text, tokenizer, model)
    elif menu_option_number == "2":
        menu_option_2(all_text, cve_description, tokenizer, model)
    elif menu_option_number == "3":
        user_question = input("Please enter the question you have about the report: ")
        menu_option_3(all_text, user_question, tokenizer, model)
    else:
        print("Exiting the program.")
        break

    user_continue = input("Would you like to continue? Please enter 1 for yes and 2 for no: ")

# Cleanup
cleanup_model(model)
