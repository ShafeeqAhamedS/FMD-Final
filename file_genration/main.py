import os
import glob
import google.generativeai as genai # type: ignore
import logging
from dotenv import load_dotenv # type: ignore
from utils.helper import extract_code_cells, read_dataset_sample, read_file, prepare_environment, execute_python_file, get_prompt_execute_store
from utils.config import logger, generation_config

load_dotenv()

def main():
    try:
        work_dir = os.getcwd()+"/workspace/file_genration"
        genai.configure(api_key=os.getenv("API_KEY"))

        model = genai.GenerativeModel(
            model_name=os.getenv("MODEL_NAME"),
            generation_config=generation_config,
        )

        user_input_path = work_dir + "/generate_code/user_input/*"

        backend_path = work_dir+"/backend"

        prepare_environment(user_input_path, backend_path)

        eval_prompt_file = work_dir+"/generate_code/eval/prompt.md"
        eval_template_code = work_dir+"/generate_code/eval/template_code.md"
        eval_output_file = work_dir+"/generate_code/eval/output.py"
        eval_full_prompt_file = work_dir+"/generate_code/eval/full_prompt.md"
        eval_output_log = work_dir+"/generate_code/eval/output.log"
        replace_new_code_path = work_dir+"/generate_code/eval/new_code.py"

        backend_prompt_file = work_dir+"/generate_code/backend/prompt.md"
        backend_template_code = work_dir+"/generate_code/backend/template_code.md"
        backend_output_file = work_dir+"/generate_code/backend/output.py"
        backend_full_prompt_file = work_dir+"/generate_code/backend/full_prompt.md"  
        replace_backend_code_path = work_dir+"/backend/main.py"

        frontend_prompt_file = work_dir+"/generate_code/frontend/prompt.md"
        frontend_template_code = work_dir+"/generate_code/frontend/template_code.md"
        frontend_output_file = work_dir+"/generate_code/frontend/output.jsx"
        frontend_full_prompt_file = work_dir+"/generate_code/frontend/full_prompt.md" 
        replace_frontend_code_path = work_dir+"/frontend/src/App.jsx"
        
        logger.debug("Starting main execution")

        user_input_path = work_dir + '/generate_code/user_input/'

        # Find the .ipynb file
        ipynb_files = glob.glob(os.path.join(user_input_path, "*.ipynb"))
        if ipynb_files:
            ipynb_file = ipynb_files[0]  # Assuming the first .ipynb file is chosen
            print("Found .ipynb file:", ipynb_file)
        else:
            print("No .ipynb file found.")

        # Find the dataset files (.csv, .json, .xl)
        dataset_file = None
        csv_files = glob.glob(os.path.join(user_input_path, "*.csv"))
        json_files = glob.glob(os.path.join(user_input_path, "*.json"))
        xl_files = glob.glob(os.path.join(user_input_path, "*.xl*"))

        if csv_files:
            dataset_file = csv_files[0]
        elif json_files:
            dataset_file = json_files[0]
        elif xl_files:
            dataset_file = xl_files[0]

        if dataset_file:
            print("Found dataset file:", dataset_file)
        else:
            print("No dataset file found.")

        extracted_data = extract_code_cells(ipynb_file)
        df = read_dataset_sample(dataset_file)

        logger.info("Starting chat session")

        # Start a chat session 
        chat_session = model.start_chat() 

        get_prompt_execute_store(chat_session, eval_prompt_file, eval_template_code, extracted_data, df, eval_full_prompt_file, eval_output_file, replace_code_path=replace_new_code_path)

        # Run eval.py and get the output
        execute_python_file(eval_output_file, eval_output_log)

        eval_output = read_file(eval_output_log)

        get_prompt_execute_store(chat_session, backend_prompt_file, backend_template_code, extracted_data, df, backend_full_prompt_file, backend_output_file, eval_output, replace_backend_code_path)
        
        get_prompt_execute_store(chat_session, frontend_prompt_file, frontend_template_code, extracted_data, df, frontend_full_prompt_file, frontend_output_file, eval_output, replace_frontend_code_path)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()