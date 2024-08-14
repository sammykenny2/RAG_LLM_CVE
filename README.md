# RAG_LLM_CVE
 Developing RAG to address LLM hallucinations in SOCs
![image](https://github.com/user-attachments/assets/4227c17b-a781-48ee-a7f2-b7bd84d9487d)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) model designed to validate the use of Common Vulnerabilities and Exposures (CVEs) in security reports. The system leverages advanced natural language processing techniques to ensure that CVEs are accurately referenced and used in context within threat intelligence reports.

## Features

- **CVE Retrieval:** Retrieves relevant CVE information from a database or API.
- **Contextual Validation:** Validates that CVEs are used correctly in the context of security reports.
- **Report Integration:** Integrates with security reports to provide feedback on CVE usage.

## Requirements

- **Python 3.8+**
- **PyTorch**: For model training and inference.
- **Transformers**: For accessing pre-trained models and tokenizers.
- **spaCy**: For advanced text processing.
- **pandas**: For handling data and managing reports.
- **requests**: For interacting with APIs.
