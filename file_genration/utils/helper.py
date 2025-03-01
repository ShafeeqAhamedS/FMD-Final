import json
import pandas as pd
import logging
import os

logger = logging.getLogger()

def extract_code_cells(ipynb_file):
    logger.debug(f"Extracting code cells from {ipynb_file}")
    try:
        with open(ipynb_file, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
        logger.info("IPYNB file loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading IPYNB file: {e}")
        return []

    extracted_data = []
    
    try:
        for cell in notebook_data.get("cells", []):
            if cell.get("cell_type") == "code":
                cell_data = {"source": cell.get("source", []), "outputs": []}
                
                for output in cell.get("outputs", []):
                    if "text/plain" in output.get("data", {}):
                        cell_data["outputs"].append(output["data"]["text/plain"])
                    elif "text/html" in output.get("data", {}):
                        cell_data["outputs"].append(output["data"]["text/html"])
                
                extracted_data.append(cell_data)
        logger.info("Code cells extracted successfully.")
    except Exception as e:
        logger.error(f"Error extracting code cells: {e}")
        return []

    return json.dumps(extracted_data, indent=4)

def read_dataset_sample(file_path):
    """
    Load the dataset from a CSV file.
    Return column names and the first 5 rows of the dataset as string
    """
    logger.debug(f"Reading dataset sample from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully.")
        return df.to_string(index=False, header=True, max_rows=5)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return ""

def read_file(file_path):
    """Read the file which is a markdown file."""
    logger.debug(f"Reading file from {file_path}")
    try:
        with open(file_path, 'r') as f:
            prompt_data = f.read()
        logger.info("Prompt file loaded successfully.")
        return prompt_data
    except Exception as e:
        logger.error(f"Error loading prompt file: {e}")
        return ""

def write_file(file_path, data):
    """Write the data to a file."""
    logger.debug(f"Writing data to {file_path}")
    try:
        with open(file_path, 'w') as f:
            f.write(data)
        logger.info("File written successfully.")
    except Exception as e:
        logger.error(f"Error writing file: {e}")

def prepare_environment(user_input_path, backend_path):
    logger.debug("Preparing environment")
    try:
        logger.info("Copying files to respective directories")
        os.system(f"cp {user_input_path} ./")
        os.system(f"cp {user_input_path} {backend_path}")
        logger.info("Files copied to respective directories")
    except Exception as e:
        logger.error(f"Error preparing environment: {e}")

def execute_python_file(file_path, output_file):
    logger.debug(f"Executing python file {file_path}")
    try:
        os.system(f"python {file_path} > {output_file} 2>&1")
        logger.info(f"Output written to {output_file}")
    except Exception as e:
        logger.error(f"Error executing python file: {e}")

def get_prompt_execute_store(chat_session, base_prompt_file, template_code_file, user_code, data_preview, full_prompt_file, output_file, eval_output=None, replace_code_path=None):
    try:
        logger.debug(f"Reading base prompt from {base_prompt_file}")
        base_prompt = read_file(base_prompt_file)
        logger.debug(f"Reading template code from {template_code_file}")
        template_code = read_file(template_code_file)
        data = f"### Prompt\n\n{base_prompt}\n\n#### 1. Dataset Preview\n\n{data_preview}\n\n#### 2. Code Blocks JSON\n\n{user_code} \n\n {eval_output}#### 3. Template Code\n\n{template_code}"
        logger.debug(f"Writing full prompt to {full_prompt_file}")
        write_file(full_prompt_file, data)

        logger.debug("Sending message to chat session")
        response = chat_session.send_message(data)
        logger.debug(f"Writing response to {output_file}")
        with open(output_file, 'w') as f:
            code_block = None
            
            for lang in ['```python', '```js', '```py', '```javascript']:
                if (response.text).startswith(lang):
                    try:
                        code_block = response.text.split(lang)[1].split('```')[0]
                        break  # Exit loop once found
                    except IndexError:
                        continue  # Move to the next language

            if code_block:
                f.write(code_block)
            else:
                f.write(response.text)  # Fallback: write the entire response
        logger.info(f"Response written to {output_file}")

        if replace_code_path:
            logger.debug(f"Replacing code in {replace_code_path}")
            os.system(f"cat {output_file} > {replace_code_path}")
            logger.info(f"Code replaced in {replace_code_path}")
    except Exception as e:
        logger.error(f"Error in get_prompt_execute_store: {e}")
