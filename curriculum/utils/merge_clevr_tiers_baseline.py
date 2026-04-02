import json
import random
import os

def merge_clevr_tiers(input_folder=".", output_file="CLEVR_val_baseline_questions.json"):
    """
    input_folder: The directory where your L1-L5 files are stored.
    output_file: The name of the resulting merged file.
    """
    all_questions = []
    
    input_folder = os.path.abspath(input_folder)
    
    # Iterate through your 5 tier files
    for i in range(1, 6):
        filename = f"CLEVR_val_questions_L{i}.json"
        # Join the folder path with the filename
        file_path = os.path.join(input_folder, filename)
        
        if not os.path.exists(file_path):
            print(f"Error: Could not find {file_path}")
            continue
            
        print(f"Loading {file_path}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            questions = data.get('questions', [])
            all_questions.extend(questions)
    
    if not all_questions:
        print("No questions found! Check your folder path.")
        return

    print(f"Total questions collected: {len(all_questions)}")
    
    # Shuffle for the random baseline
    random.seed(42) 
    random.shuffle(all_questions)
    
    output_data = {
        "info": "Merged baseline for T1-T5 pooled dataset",
        "questions": all_questions
    }
    
    # Save the output file
    with open(output_file, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Success! Baseline saved as {output_file}")

if __name__ == "__main__":
    PATH_TO_FILES = r"C:\Users\Senal\Downloads\CLEVRD\CLEVR_v1.0\downsampled"
    
    merge_clevr_tiers(input_folder=PATH_TO_FILES)