# trying to input a suggestion
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers.utils import logging
logging.set_verbosity_error()

import re
import fitz  # PyMuPDF
import json 
import os
import gc
import numpy as np
import pandas as pd


#from testingRagPart2 import askingLlamaForAdvice

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




##################################################

#print("/n/n The length is /n/n" + len(all_text))

#def split_with_overlap(s, chunk_size, overlap_size):
    # Ensure the chunk size is greater than the overlap size
#    if chunk_size <= overlap_size:
#        raise ValueError("Chunk size must be greater than overlap size")

#    chunks = []
#    start = 0

 #   while start < len(s):
#        end = start + chunk_size
#        chunks.append(s[start:end])
#        start += chunk_size - overlap_size

 #   return chunks

# Example usage
#string = "This is a long string that we want to split into overlapping chunks."
#chunk_size = 10 000
#overlap_size = 1000

#result = split_with_overlap(string, chunk_size, overlap_size)
#print(result)





##################################################








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

#for cve in cves:
#    print(cve)

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
    

def askingLlamaForAdvice(cveDesp):
    ##print(cveDesp)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    user_input_csv = "test2.csv"

    text_chunks_and_embedding_df = pd.read_csv(user_input_csv)


    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    embeddings = torch.tensor(np.stack(text_chunks_and_embedding_df["embedding"].tolist(), axis=0), dtype=torch.float32).to(device)

    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

    # Create model
    from sentence_transformers import util, SentenceTransformer

    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device=device)

   

    device = "cuda"
    

    def retrieve(query: str,
                embeddings: torch.tensor,
                model: SentenceTransformer=embedding_model,
                n_resources_to_return: int=5):
        """
        Embeds a query with model and returns top k scores and indices from embeddings.
        """

        # Embed the query
        query_embedding = model.encode(query, convert_to_tensor=True)

 
        dot_scores = util.dot_score(query_embedding, embeddings)[0]

        scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)

        return indices


    #from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    query = cveDesp

    # Get just the scores and indices of top related results
    indices = retrieve(query=query,
                                              embeddings=embeddings)

    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]


    #print(prompt)
 
    def prompt_Context(query: str,
                         context_items: list[dict]) -> str:
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
        return context


    theContext= prompt_Context(query=query,
                          context_items=context_items)


    #print(theContext)

    ##print(cveDesp)



    messages = [
        {"role": "system", "content": f""""You are a Q&A Assistant. You will be provided with relevant information about various CVEs. Based on this information, your task is to recommend the CVE number that most closely matches the description of the vulnerability.
        Provided Relevant Information: {theContext} 
    """},
        {"role": "user", "content": f""" Hello, could you recomend me a CVE that most closly resembles this chunk of text based of the relevant information that you have. Chunk of text: {cveDesp}"""},
        #{"role": "user", "content": f""" Can you give me a Indicators of Compromise (IoC) Report of Palo Alto Networks given the context; you must provide relevant CVE numbers associated with each threat. Please ignore the CVE's that are note related to the company. The context is the following: {theContext}"""},
    ]   

    ############
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=700, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.3, top_p=0.9)
    response = outputs[0][input_ids.shape[-1]:]
    llamaSug = tokenizer.decode(response, skip_special_tokens=True)
    
    return llamaSug




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
            {"role": "user", "content": f"""Here is a list of CVEs: {cves}. I will provide you with a report shortly. If you encounter any CVEs not included in the list I previously provided, please describe each of them in two sentences. The provided text is as follows: {all_text}.
"""},
            ]

        #input_ids = tokenizer.apply_chat_template(
        #messages,
        #add_generation_prompt=True,
        #return_tensors="pt"
        #).to(model.device)

       # terminators = [
        #    tokenizer.eos_token_id,
        #    tokenizer.convert_tokens_to_ids("<|eot_id|>")
        #    ]

       # outputs = model.generate(
         #   input_ids,
          #  max_new_tokens=500,
           # eos_token_id=terminators,
           # do_sample=True,
          #  temperature=0.3,
         #   top_p=0.9,
        #    )
        #response = outputs[0][input_ids.shape[-1]:]
        #print(tokenizer.decode(response, skip_special_tokens=True))
        #llamaDesc = tokenizer.decode(response, skip_special_tokens=True)

        ############
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        outputs = model.generate(input_ids, max_new_tokens=700, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.3, top_p=0.9)
        response = outputs[0][input_ids.shape[-1]:]
        llamaDesc = tokenizer.decode(response, skip_special_tokens=True)
        




        ##########

        ##print(f"Hello ladies and gentleman, the llama has spoken, the output is {llamaDesc}")

        

        llamaSug = askingLlamaForAdvice(llamaDesc)

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
