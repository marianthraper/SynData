import os
import numpy as np
import pandas as pd
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from starlette.middleware.cors import CORSMiddleware
from ctgan import CTGAN
import chromadb
import google.generativeai as genai
from pydantic import BaseModel
from scipy.stats import kurtosis, skew
from scipy.stats import ks_2samp


# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
DATA_DIR = "generated_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize ChromaDB (Persistent Storage)
chroma_client = chromadb.PersistentClient(path=".cache/chroma")
dataset_collection = chroma_client.get_or_create_collection(name="datasets")

# Initialize Gemini AI
GENAI_API_KEY = "*"  # Replace with your API key
genai.configure(api_key=GENAI_API_KEY)

from scipy.stats import ks_2samp
# -------------------- KS TEST AGENT -------------------- #
def perform_ks_test(original_column, synthetic_column):
    ks_statistic, p_value = ks_2samp(original_column.dropna(), synthetic_column.dropna())
    return {
        "ks_statistic": ks_statistic,
            }

# -------------------- DISTRIBUTION METRICS AGENT -------------------- #
def compute_distribution_metrics(df: pd.DataFrame):
    metrics = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        metrics[column] = {
            "mean": df[column].mean(),
            "variance": df[column].var()
        }
    return metrics

# -------------------- DISTRIBUTION COMPARISON AGENT -------------------- #
def compare_distributions(original_metrics, synthetic_metrics, original_df, synthetic_df, threshold=0.1):
    comparison_results = {}
    for column in original_metrics:
        if column in synthetic_metrics:
            comparison_results[column] = {}

            # Compare mean and variance
            for metric in original_metrics[column]:
                orig_value = original_metrics[column][metric]
                synth_value = synthetic_metrics[column][metric]

                # Compute relative difference
                if orig_value != 0:
                    relative_diff = abs(orig_value - synth_value) / abs(orig_value)
                else:
                    relative_diff = abs(orig_value - synth_value)

                comparison_results[column][metric] = {
                    "original": orig_value,
                    "synthetic": synth_value
                }

            # Add KS test result
            ks_result = perform_ks_test(original_df[column], synthetic_df[column])
            comparison_results[column]["ks_test"] = {
                "ks_statistic": ks_result["ks_statistic"],
                            }

    return comparison_results
# -------------------- AGENTIC PREPROCESSING FUNCTION -------------------- #
def agentic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    try:
        print("Agentic preprocessing applied.")
        
        # Check for non-numeric columns and ask Gemini to remove them
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        if non_numeric_columns.any():
            print(f"Found non-numeric columns: {', '.join(non_numeric_columns)}")
            # Ask Gemini if non-numeric columns should be removed
            response_non_numeric = ask_gemini_to_remove_non_numeric(non_numeric_columns)
            
            if response_non_numeric.lower() == "yes":
                df = df.drop(columns=non_numeric_columns)
                print(f"Non-numeric columns removed: {', '.join(non_numeric_columns)}")
            else:
                print("Non-numeric columns retained.")

        # Handle NaN values for numeric columns based on Gemini's response
        response_nan = ask_gemini_to_handle_nan()
        if response_nan.lower() == "yes":
            # Handle NaN values by filling with median values for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                df[column] = df[column].fillna(df[column].median())
            print(f"NaN values handled and filled with median values for columns: {', '.join(numeric_columns)}")
        else:
            print("Skipping NaN value preprocessing.")

        return df

    except Exception as e:
        print(f"Error during agentic preprocessing: {str(e)}")
        raise ValueError(f"Agentic preprocessing failed: {str(e)}")

def ask_gemini_to_remove_non_numeric(non_numeric_columns):
    """ Ask Gemini if it should remove non-numeric columns """
    # Prepare the query to Gemini
    query = (
        f"Do you recommend removing the following non-numeric columns from the dataset?"
        f" Columns: {', '.join(non_numeric_columns)}. Please respond with 'yes' or 'no'."
    )
    
    # Call Gemini to get a response
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(query)
    
    return response.text.strip()

def ask_gemini_to_handle_nan():
    """ Ask Gemini if it should handle NaN values in the dataset """
    query = "Do you recommend handling NaN values by filling them with median values? Please respond with 'yes' or 'no'."
    
    # Call Gemini to get a response
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(query)
    
    return response.text.strip()

#----------------------Column Type Agent------------------------------#
def ask_gemini_column_type_decision(column_names):
    """ Ask Gemini which columns should be treated as integers """
    query = (
        "Based on these column names, which ones should logically be treated as integers?\n"
        f"Columns: {', '.join(column_names)}.\n"
        "Return only a comma-separated list of column names to convert to integers. If none, return 'none'."
    )
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(query)
    return response.text.strip()

def enforce_agentic_column_types(df: pd.DataFrame) -> pd.DataFrame:
    try:
        print("Applying agentic type enforcement...")
        column_names = df.columns.tolist()
        response = ask_gemini_column_type_decision(column_names)

        if response.lower() != "none":
            columns_to_convert = [col.strip() for col in response.split(",") if col.strip() in df.columns]
            for col in columns_to_convert:
                df[col] = df[col].round().astype(int)
                print(f"Column '{col}' converted to integer.")
        else:
            print("Gemini suggests no columns require integer conversion.")
        return df

    except Exception as e:
        print(f"Error during agentic type enforcement: {str(e)}")
        return df

# -------------------- NATURAL LANGUAGE CONSTRAINTS AGENT -------------------- #
def ask_gemini_for_constraints_json(natural_language_constraints, columns):
    """
    Convert natural language constraints to JSON format using Gemini
    """
    if not natural_language_constraints or natural_language_constraints.strip().lower() == "none":
        return {}
    
    prompt = (
        f"Convert the following natural language constraints into a JSON object for data filtering:\n"
        f"Columns available: {', '.join(columns)}\n"
        f"Natural language constraints: '{natural_language_constraints}'\n\n"
        f"For example, if the user says 'age between 20 and 60, salary above 30000', "
        f"you would respond with: {{ \"age\": {{ \"min\": 20, \"max\": 60 }}, \"salary\": {{ \"min\": 30000 }} }}\n\n"
        f"Return ONLY the JSON object without any explanations or additional text. "
        f"If no valid constraints, return an empty JSON object {{}}."
    )
    
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    response_text = response.text.strip()
    
    # Extract JSON if wrapped in code blocks
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].strip()
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Error parsing JSON from Gemini response: {response_text}")
        return {}

def filter_by_constraints(df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """
    Filter a dataframe based on constraints dictionary.
    
    Args:
        df: The dataframe to filter
        constraints: Dictionary of column constraints (e.g. {"age": {"min": 40}})
        
    Returns:
        Filtered dataframe meeting all constraints
    """
    filtered = df.copy()
    
    # Apply each constraint
    for col, rules in constraints.items():
        if col in filtered.columns:
            # Apply minimum constraint if present
            if "min" in rules:
                min_value = rules["min"]
                print(f"Applying min constraint on {col}: >= {min_value}")
                try:
                    # Convert column to numeric if possible
                    filtered[col] = pd.to_numeric(filtered[col], errors='coerce')
                    filtered = filtered[filtered[col] >= min_value]
                    print(f"After min filter: {len(filtered)} rows")
                except Exception as e:
                    print(f"Error applying min constraint on {col}: {str(e)}")
            
            # Apply maximum constraint if present
            if "max" in rules:
                max_value = rules["max"]
                print(f"Applying max constraint on {col}: <= {max_value}")
                try:
                    # Convert column to numeric if possible
                    filtered[col] = pd.to_numeric(filtered[col], errors='coerce')
                    filtered = filtered[filtered[col] <= max_value]
                    print(f"After max filter: {len(filtered)} rows")
                except Exception as e:
                    print(f"Error applying max constraint on {col}: {str(e)}")
    
    return filtered


def enforce_constraints_2(df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """
    Strictly enforces the constraint rules in the constraints dictionary
    by filtering the dataframe to only include rows that satisfy all constraints.
    
    Args:
        df: DataFrame to filter
        constraints: Dictionary of column constraints (e.g. {"customer_age": {"min": 40}})
        
    Returns:
        DataFrame with only rows that satisfy all constraints
    """
    try:
        print(f"Enforcing constraints: {constraints}")
        result_df = df.copy()
        
        for column, rules in constraints.items():
            if column not in result_df.columns:
                print(f"Warning: Column '{column}' not found in dataframe")
                continue
                
            # Ensure column is numeric for comparison
            result_df[column] = pd.to_numeric(result_df[column], errors='coerce')
            
            # Apply minimum constraint
            if "min" in rules:
                min_value = rules["min"]
                print(f"Enforcing minimum value of {min_value} for '{column}'")
                result_df = result_df[result_df[column] >= min_value]
                print(f"  Rows after min filter: {len(result_df)}")
                
            # Apply maximum constraint
            if "max" in rules:
                max_value = rules["max"]
                print(f"Enforcing maximum value of {max_value} for '{column}'")
                result_df = result_df[result_df[column] <= max_value]
                print(f"  Rows after max filter: {len(result_df)}")
                
        print(f"Constraint enforcement complete. {len(result_df)} rows remain.")
        return result_df
        
    except Exception as e:
        print(f"Error during constraint enforcement: {str(e)}")
        # Return original dataframe if there's an error
        return df

# -------------------- TOOL 1: Data Preprocessing -------------------- #
def preprocess_data(file: UploadFile):
    try:
        # Load CSV file into DataFrame
        df = pd.read_csv(file.file, encoding="latin1", on_bad_lines="skip")
        if df.empty:
            raise ValueError("Uploaded dataset is empty!")

        # Apply agentic preprocessing here before any other operations
        print("Applying agentic preprocessing...")
        df = agentic_preprocessing(df)
        print(f"Agentic preprocessing complete. DataFrame shape: {df.shape}")

        # Select numeric columns and handle missing values
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty:
            raise ValueError("No numeric columns found in the dataset!")
        df_numeric.fillna(df_numeric.median(numeric_only=True), inplace=True)

        filename = os.path.join(DATA_DIR, f"cleaned_{file.filename}")
        df_numeric.to_csv(filename, index=False)
        return filename, df_numeric

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during preprocessing: {str(e)}")

# Store dataset metadata in ChromaDB
def store_dataset_in_rag(filename, column_names):
    column_string = ", ".join(column_names)  # Store as a string
    dataset_collection.add(
        documents=[column_string],  
        metadatas=[{"columns": column_string, "filename": filename}],  
        ids=[filename]
    )

# -------------------- TOOL 2: Modified Synthetic Data Generation -------------------- #
def generate_synthetic_data(cleaned_file: str, num_samples: int, selected_columns=None):
    df = pd.read_csv(cleaned_file)

    if df.empty:
        raise ValueError("Preprocessed dataset is empty, cannot generate synthetic data.")

    # Filter columns if specific ones are requested
    if selected_columns:
        selected_columns = [col.strip() for col in selected_columns.split(",")]
        missing_columns = [col for col in selected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Requested columns {missing_columns} not found in dataset.")
        df = df[selected_columns]

    # Apply log transformation for strictly positive columns
    log_transformed_cols = []
    for col in df.columns:
        if (df[col] > 0).all():  
            df[col] = np.log1p(df[col])  
            log_transformed_cols.append(col)

    model = CTGAN()
    model.fit(df, epochs=100)
    
    # Generate synthetic data with 30% more samples than requested to account for filtering
    synthetic_data = model.sample(int(num_samples))
    
    decimal_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns 
        if not (df[col].dropna() % 1 == 0).all()  # Ignore NaNs and check for decimal values
    ]
    
    # Reverse log transformation & ensure correct data types
    for col in df.columns:
        if col in log_transformed_cols:
            synthetic_data[col] = np.expm1(synthetic_data[col])  
        if df[col].nunique() == 2 and set(df[col].unique()) == {0, 1}:
            synthetic_data[col] = (synthetic_data[col] > 0.5).astype(int)
        elif col not in decimal_cols:
            synthetic_data[col] = synthetic_data[col].round().astype(int)

        if (df[col] >= 0).all():
            synthetic_data[col] = synthetic_data[col].clip(lower=0)

    # Apply agentic type enforcement after generation
    synthetic_data = enforce_agentic_column_types(synthetic_data)

    # Save final synthetic data
    synthetic_filename = os.path.join(DATA_DIR, f"synthetic_{os.path.basename(cleaned_file)}")
    synthetic_data.to_csv(synthetic_filename, index=False)

    return synthetic_filename

# -------------------- New function to generate data with constraints -------------------- #
def generate_data_with_constraints(model, df, num_samples, constraints, log_transformed_cols, decimal_cols):
    """
    Generates synthetic data that meets the constraints, ensuring we have the requested number of rows
    """
    # Start with a batch larger than what we need - scale based on constraints
    initial_multiplier = 10.0  # Generate 10x the requested samples initially
    current_multiplier = initial_multiplier
    max_attempts = 5
    attempts = 0
    
    collected_rows = pd.DataFrame(columns=df.columns)
    
    while attempts < max_attempts and len(collected_rows) < num_samples:
        # Generate a batch of synthetic data
        batch_size = int((num_samples - len(collected_rows)) * current_multiplier)
        print(f"Generating {batch_size} samples (attempt {attempts+1}/{max_attempts})")
        
        synthetic_batch = model.sample(batch_size)
        
        # Post-processing
        for col in df.columns:
            if col in log_transformed_cols:
                synthetic_batch[col] = np.expm1(synthetic_batch[col])
            if df[col].nunique() == 2 and set(df[col].unique()) == {0, 1}:
                synthetic_batch[col] = (synthetic_batch[col] > 0.5).astype(int)
            elif col not in decimal_cols:
                synthetic_batch[col] = synthetic_batch[col].round().astype(int)
            if (df[col] >= 0).all():
                synthetic_batch[col] = synthetic_batch[col].clip(lower=0)
        
        # Apply type enforcement
        synthetic_batch = enforce_agentic_column_types(synthetic_batch)
        
        # Apply constraints - using enforce_constraints_2 instead of filter_by_constraints
        # for more strict enforcement
        filtered_batch = enforce_constraints_2(synthetic_batch, constraints)
        
        # Add to our collection
        if not filtered_batch.empty:
            collected_rows = pd.concat([collected_rows, filtered_batch], ignore_index=True)
            print(f"Collected {len(collected_rows)}/{num_samples} rows so far")
        
        # If we don't have enough samples, increase the multiplier and try again
        attempts += 1
        if len(collected_rows) < num_samples:
            current_multiplier *= 2
            print(f"Increasing multiplier to {current_multiplier}")
    
    # Ensure we don't return more rows than requested
    if len(collected_rows) >= num_samples:
        print(f"Successfully generated {num_samples} constrained samples")
        return collected_rows.head(num_samples)
    
    # If we exhausted attempts and still don't have enough data
    print(f"Warning: Could only generate {len(collected_rows)}/{num_samples} samples after constraints")
    return collected_rows

# Retrieve dataset based on column similarity
def retrieve_similar_data(column_names):
    input_columns = [col.strip() for col in column_names.split(",")]

    results = dataset_collection.query(query_texts=[", ".join(input_columns)], n_results=1)

    if not results["ids"]:
        return None

    best_match = results["metadatas"][0][0]["filename"]
    return os.path.join(DATA_DIR, best_match)


# -------------------- FASTAPI ENDPOINTS -------------------- #
@app.post("/generate")
async def generate(file: UploadFile = File(...), num_samples: int = Form(...)):
    try:
        cleaned_file, df = preprocess_data(file)
        store_dataset_in_rag(os.path.basename(cleaned_file), df.columns.tolist())  
        
        # Generate synthetic data first
        synthetic_filename = generate_synthetic_data(cleaned_file, num_samples)
        
        return JSONResponse(content={
            "temp_file": os.path.basename(synthetic_filename),
            "columns": df.columns.tolist()
        })
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"Internal Server Error: {str(e)}"}, status_code=500)

@app.post("/generate-with-columns")
async def generate_with_columns(column_names: str = Form(...), num_samples: int = Form(...)):
    try:
        column_names = column_names.strip()
        columns_list = [col.strip() for col in column_names.split(",")]
        
        matched_file = retrieve_similar_data(column_names)
        if not matched_file:
            return JSONResponse(content={"error": "No relevant dataset found."}, status_code=404)
        
        synthetic_filename = generate_synthetic_data(matched_file, num_samples, selected_columns=column_names)
        
        return JSONResponse(content={
            "temp_file": os.path.basename(synthetic_filename),
            "columns": columns_list
        })
    except Exception as e:
        return JSONResponse(content={"error": f"Internal Server Error: {str(e)}"}, status_code=500)

@app.post("/process-natural-language-constraints")
async def process_natural_language_constraints(
    natural_constraints: str = Form(...), 
    columns: str = Form(...)):
    """
    Convert natural language constraints to JSON constraints
    """
    try:
        columns_list = json.loads(columns)
        constraints_json = ask_gemini_for_constraints_json(natural_constraints, columns_list)
        return JSONResponse(content={"constraints_json": constraints_json})
    except Exception as e:
        return JSONResponse(content={"error": f"Error processing constraints: {str(e)}"}, status_code=500)

@app.post("/submit-constraints")
async def submit_constraints(constraints: str = Form(...), filename: str = Form(...), num_samples: int = Form(...)):
    try:
        # Convert num_samples to int (as a safety measure)
        try:
            num_samples = int(num_samples)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid num_samples value: {num_samples}")
        
        print(f"Processing constraints with num_samples: {num_samples}")
        
        # Parse the constraints JSON
        constraints_dict = json.loads(constraints)
        print(f"Processing constraints: {constraints_dict}")
        
        # Load the synthetic data
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Synthetic data file not found")
        
        # Load existing synthetic data
        synthetic_df = pd.read_csv(file_path)
        
        # Get the original file path (before synthetic generation)
        if filename.startswith("synthetic_"):
            # Extract the base name, removing "synthetic_" prefix
            base_name = filename[len("synthetic_"):]
            
            # Try different possible paths for the cleaned file
            possible_paths = [
                os.path.join(DATA_DIR, base_name),                  # Direct base name
                os.path.join(DATA_DIR, f"cleaned_{base_name}"),     # With cleaned_ prefix
                os.path.join(DATA_DIR, base_name.replace("cleaned_", "")) # Remove duplicate cleaned_
            ]
            
            cleaned_path = None
            for path in possible_paths:
                print(f"Checking path: {path}")
                if os.path.exists(path):
                    cleaned_path = path
                    print(f"Found original cleaned file: {cleaned_path}")
                    break
            
            if cleaned_path:
                df = pd.read_csv(cleaned_path)
                        
                # Identify columns that need log transformation
                log_transformed_cols = []
                for col in df.columns:
                    if (df[col] > 0).all():  
                        log_transformed_cols.append(col)
                        df[col] = np.log1p(df[col])
                
                # Identify decimal columns
                decimal_cols = [
                    col for col in df.select_dtypes(include=[np.number]).columns 
                    if not (df[col].dropna() % 1 == 0).all()
                ]
                
                # Train CTGAN model
                print("Training CTGAN model...")
                model = CTGAN()
                model.fit(df, epochs=100)
                print("Model training complete")
                
                # Generate data with constraints ensuring we have exactly num_samples rows
                print(f"Generating constrained synthetic data for {num_samples} samples...")
                synthetic_data = generate_data_with_constraints(
                    model, df, num_samples, constraints_dict, log_transformed_cols, decimal_cols
                )
                print(f"Generated {len(synthetic_data)} samples after constraints")
            else:
                print(f"Original cleaned file not found: {cleaned_path}")
                # If we can't find the original file, apply constraints to existing synthetic data
                synthetic_data = filter_by_constraints(synthetic_df, constraints_dict)
                
                # If filtering resulted in fewer rows than needed, we'll use what we have
                if len(synthetic_data) < num_samples:
                    print(f"Warning: Could only generate {len(synthetic_data)}/{num_samples} samples after constraints")
                else:
                    # Otherwise, take exactly the number we need
                    synthetic_data = synthetic_data.head(num_samples)
        else:
            # If not a synthetic file, just apply constraints
            synthetic_data = filter_by_constraints(synthetic_df, constraints_dict)
            
            # Take the first num_samples rows (or all if we have fewer)
            if len(synthetic_data) > num_samples:
                synthetic_data = synthetic_data.head(num_samples)
        
        # Crucial step: ensure all constraints are enforced before returning
        synthetic_data = enforce_constraints_2(synthetic_data, constraints_dict)
        
        # Make sure we don't return more rows than requested
        if len(synthetic_data) > num_samples:
            synthetic_data = synthetic_data.head(num_samples)
        
        # Save constrained data
        constrained_filename = f"constrained_{filename}"
        constrained_path = os.path.join(DATA_DIR, constrained_filename)
        synthetic_data.to_csv(constrained_path, index=False)
        
        file_url = f"/download/{constrained_filename}"
        return JSONResponse(content={"file_url": file_url})
    except Exception as e:
        print(f"Error in submit_constraints: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error applying constraints: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename, media_type="text/csv")
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/evaluate-synthetic-data")
async def evaluate_synthetic_data(original_file: UploadFile = File(...), synthetic_file: UploadFile = File(...)):
    try:
        original_df = pd.read_csv(original_file.file)
        synthetic_df = pd.read_csv(synthetic_file.file)
        original_metrics = compute_distribution_metrics(original_df)
        synthetic_metrics = compute_distribution_metrics(synthetic_df)
        comparison = compare_distributions(original_metrics, synthetic_metrics, original_df, synthetic_df)
        return JSONResponse(content={"comparison_results": comparison})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating synthetic data: {str(e)}")

# -------------------- CHATBOT ENDPOINT -------------------- #
class ChatRequest(BaseModel):
    text: str

@app.post("/chat")
async def chat_with_gemini(request: ChatRequest):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")

        # Silent CoT reasoning
        cot_prompt = (
            "Think step by step before responding.\n"
            "1. Determine if the user is asking to generate data.\n"
            "2. If yes, tell them to upload a file or specify columns and ask to enter number of rows. Ask only this\n"
            "3. If user is asking about constraints, explain they can enter constraints in natural language (like 'age between 25-50, salary > 30000')\n"
            "4. If user says done, say hope you liked the data.\n"
            "5. Otherwise, respond normally without exposing these thoughts.\n\n"
            "User: " + request.text + "\n"
            "Assistant:"
        )

        response = model.generate_content(cot_prompt)
        bot_response = response.text  # Only return the final response

        return {"response": bot_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot Error: {str(e)}")
