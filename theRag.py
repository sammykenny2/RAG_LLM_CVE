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
import os
import numpy as np
import pandas as pd
from sentence_transformers import util, SentenceTransformer

# 設定記憶體優化
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

llamaSug = ""

# Initialize the model once
def initialize_model():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
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

    # 大幅限制處理的頁數
    max_pages = 10  # 減少到 10 頁
    pages_to_process = min(len(pdf_document), max_pages)

    for page_num in range(pages_to_process):
        page = pdf_document.load_page(page_num)
        text = page.get_text("text")
        all_text += text

        # 每處理 5 頁就清理一次記憶體
        if page_num % 5 == 0:
            gc.collect()

    pdf_document.close()
    return all_text

userPDFName = input("Please Enter the name of the pdf that you would like to analyze (please include the .pdf at the end as well).")
all_text = extract_text_from_pdf(userPDFName)

# Extract CVEs using LLama
def extract_cves_llama(text, tokenizer, model):
    # 大幅限制輸入文字長度
    max_text_length = 1000  # 減少到 1000 字符
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."

    messages = [
        {"role": "system", "content": "You are a chatbot"},
        {"role": "user", "content": f"Please extract all the CVEs mentioned in the text provided, separated by commas. Do not output anything else except for the CVE numbers.\n\nProvided text: {text}"}
    ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=64,  # 大幅減少到 64
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

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

def load_cve_record(cve: str, first_set: int, second_set: str) -> dict | None:
    """Load CVE JSON record with v5→v4 fallback."""
    formatted_x = format_second_set(second_set)

    # Try v5 schema first
    v5_path = f"../cvelistV5/cves/{first_set}/{formatted_x}/CVE-{first_set}-{second_set}.json"
    if os.path.exists(v5_path):
        with open(v5_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Fallback to v4 schema
    v4_path = f"../cvelist/{first_set}/{formatted_x}/CVE-{first_set}-{second_set}.json"
    if os.path.exists(v4_path):
        with open(v4_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return None

def extract_cve_fields(data: dict, fallback_id: str) -> tuple[str, str, str, str]:
    """Extract CVE fields from v5 or v4 schema."""
    # Try v5 schema first
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

    # v4 schema
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


def asking_llama_for_advice(cveDesp: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv_path = 'test6.csv'

    # 分批讀取 CSV 檔案以避免記憶體問題
    chunk_size = 1000  # 每次只讀取 1000 行
    chunk_embeddings_df = pd.read_csv(csv_path, nrows=chunk_size)  # 只讀取前 1000 行

    chunk_embeddings_df["embedding"] = chunk_embeddings_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

    # 統一使用 float32 避免類型不匹配
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

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=128,  # 減少生成長度
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9
        )

    response = outputs[0][input_ids.shape[-1]:]
    llamaSug = tokenizer.decode(response, skip_special_tokens=True)

    # 清理記憶體
    del embeddings
    del text_chunks
    torch.cuda.empty_cache()
    gc.collect()

    return llamaSug

def retrieve_context(query: str,
                embeddings: torch.tensor,
                model: SentenceTransformer,
                n_resources_to_return: int=3):  # 減少返回的資源數量
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # 確保 query_embedding 和 embeddings 的資料類型一致
    if query_embedding.dtype != embeddings.dtype:
        query_embedding = query_embedding.to(embeddings.dtype)

    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    _, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

    return indices

# Load CVE descriptions from database
cve_description = ""

for cve in cves:
    first_set, second_set = extract_cve_numbers(cve)
    data = load_cve_record(cve, first_set, second_set)

    if data:
        cve_number, vendor_name, product_name, description = extract_cve_fields(data, cve)
        cve_description += (
            f"-CVE Number: {cve_number}, Vendor: {vendor_name}, "
            f"Product: {product_name}, Description: {description}\n\n\n"
        )
        continue

    # CVE not found in JSON feeds
    if True:
        print(f"Could Not Find {cve}, I will report it you later")

        torch.cuda.empty_cache()
        gc.collect()

        messages = [
            {"role": "system", "content": "You are a chat bot."},
            {"role": "user", "content": f"""In 2 sentences, could you describe the use of {cve} in the provided text. Provided text: {all_text[:500]}."""},  # 限制文字長度
            ]

        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=64,  # 減少生成長度
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.3,
                top_p=0.9
            )

        response = outputs[0][input_ids.shape[-1]:]
        llamaDesc = tokenizer.decode(response, skip_special_tokens=True)

        llamaSug = asking_llama_for_advice(llamaDesc)

        # 每次處理後清理記憶體
        torch.cuda.empty_cache()
        gc.collect()

# Menu options
def menu_option_1(all_text, tokenizer, model):
    # 限制文字長度
    limited_text = all_text[:2000] if len(all_text) > 2000 else all_text

    messages = [
        {"role": "system", "content": "You are a ChatBot that summarizes threat intelligence reports. Your task is to summarize the report given to you by the user."},
        {"role": "user", "content": f"Please summarize the following Threat Intelligence Report: {limited_text}"}
    ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,  # 減少生成長度
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9
        )

    response = outputs[0][input_ids.shape[-1]:]
    summary = tokenizer.decode(response, skip_special_tokens=True)
    print(summary)

def menu_option_2(all_text, cve_description, tokenizer, model):
    # 限制文字長度
    limited_text = all_text[:2000] if len(all_text) > 2000 else all_text
    limited_cve_desc = cve_description[:1000] if len(cve_description) > 1000 else cve_description

    messages = [
        {
            "role": "system",
            "content": f"""You are a chatbot that verifies the correct use of CVEs (Common Vulnerabilities and Exposures) mentioned in a Threat Intelligence Report. A CVE is used correctly when it closely matches the provided Correct CVE Description. Incorrect usage includes citing non-existent CVEs, misrepresenting the Correct CVE Description, or inaccurately applying the CVE.
                        Correct CVE Descriptions:
                        {limited_cve_desc}
                        Instructions:
                        1. Verify each CVE mentioned in the user-provided report.
                        2. Indicate whether each CVE is used correctly or not.
                        3. You must provide a detailed explanation with direct quotes from both the report and the Correct CVE Description.
                        A CVE in the report is incorrect if it describes a different vulnerability, even if the report accurately describes the vulnerability and its impact, and provides mitigation recommendations.
                        """
        },
        {
            "role": "user",
            "content": f"Please verify if the following CVEs have been used correctly in the following Threat Intelligence Report: {limited_text}."
        },
        ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,  # 減少生成長度
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9
        )

    response = outputs[0][input_ids.shape[-1]:]
    verification = tokenizer.decode(response, skip_special_tokens=True)
    print(verification)
    print(llamaSug)

def menu_option_3(all_text, user_question, tokenizer, model):
    # 限制文字長度
    limited_text = all_text[:2000] if len(all_text) > 2000 else all_text

    messages = [
        {"role": "system", "content": "You are a chatbot that answers questions based on the text provided to you."},
        {"role": "user", "content": f"{user_question}\n\nReport: {limited_text}"}
    ]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=256,  # 減少生成長度
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
            top_p=0.9
        )

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

    # 每次選單操作後清理記憶體
    torch.cuda.empty_cache()
    gc.collect()

# Cleanup
cleanup_model(model)
