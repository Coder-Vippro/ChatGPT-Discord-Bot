import re
import os
import logging
import tempfile
import uuid
from typing import List, Tuple, Optional, Dict, Any
import time
from datetime import datetime

# Directory to store temporary user data files for code execution and analysis
DATA_FILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src', 'temp_data_files')

# Create the directory if it doesn't exist
os.makedirs(DATA_FILES_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='logs/code_execution.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Regular expressions for safety checks
PYTHON_UNSAFE_IMPORTS = [
    r'import\s+os', 
    r'from\s+os\s+import', 
    r'import\s+subprocess', 
    r'from\s+subprocess\s+import',
    r'import\s+shutil',
    r'from\s+shutil\s+import',
    r'__import__\([\'"]os[\'"]\)',
    r'__import__\([\'"]subprocess[\'"]\)',
    r'__import__\([\'"]shutil[\'"]\)',
    r'import\s+sys',
    r'from\s+sys\s+import'
]

PYTHON_UNSAFE_FUNCTIONS = [
    r'os\.', 
    r'subprocess\.', 
    r'shutil\.rmtree', 
    r'shutil\.move',
    r'eval\(',
    r'exec\(',
    r'sys\.'
]

CPP_UNSAFE_FUNCTIONS = [
    r'system\(', 
    r'popen\(', 
    r'execl\(', 
    r'execlp\(', 
    r'execle\(', 
    r'execv\(',
    r'execvp\(',
    r'execvpe\(',
    r'fork\(',
    r'unlink\('
]

CPP_UNSAFE_INCLUDES = [
    r'#include\s+<unistd\.h>',
    r'#include\s+<stdlib\.h>'
]

def sanitize_code(code: str, language: str) -> Tuple[bool, str]:
    """
    Check code for potentially unsafe operations.
    
    Args:
        code: The code to check
        language: Programming language of the code
        
    Returns:
        Tuple of (is_safe, sanitized_code_or_error_message)
    """
    if language.lower() in ['python', 'py']:
        # Check for unsafe imports
        for pattern in PYTHON_UNSAFE_IMPORTS:
            if re.search(pattern, code):
                return False, f"Forbidden import or system access detected: {pattern}"
                
        # Check for unsafe function calls
        for pattern in PYTHON_UNSAFE_FUNCTIONS:
            if re.search(pattern, code):
                return False, f"Forbidden function call detected: {pattern}"
                
        # Add safety imports
        safe_imports = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"""
        return True, safe_imports + "\n" + code
        
    elif language.lower() in ['cpp', 'c++']:
        # Check for unsafe includes
        for pattern in CPP_UNSAFE_INCLUDES:
            if re.search(pattern, code):
                return False, f"Forbidden include detected: {pattern}"
                
        # Check for unsafe function calls
        for pattern in CPP_UNSAFE_FUNCTIONS:
            if re.search(pattern, code):
                return False, f"Forbidden function call detected: {pattern}"
                
        return True, code
        
    return False, "Unsupported language"

def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Extract code blocks from a markdown-formatted string.
    
    Args:
        text: The text containing code blocks
        
    Returns:
        List of tuples (language, code)
    """
    # Pattern to match code blocks with optional language
    pattern = r'```(\w*)\n(.*?)```'
    blocks = []
    
    # Find all code blocks
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        language = match.group(1) or 'text'  # Default to 'text' if no language specified
        code = match.group(2).strip()
        blocks.append((language.lower(), code))
        
    return blocks

def get_temporary_file_path(file_extension: str = '.py', user_id: Optional[int] = None) -> str:
    """
    Generate a temporary file path.
    
    Args:
        file_extension: The file extension to use
        user_id: Optional user ID to include in filename
        
    Returns:
        str: Path to temporary file
    """
    filename = f"temp_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    if user_id:
        filename = f"{user_id}_{filename}"
    return os.path.join(DATA_FILES_DIR, filename + file_extension)

def clean_old_files(max_age_hours: int = 23) -> None:
    """
    Remove old temporary files.
    
    Args:
        max_age_hours: Maximum age in hours before file deletion (default: 23)
    """
    if not os.path.exists(DATA_FILES_DIR):
        return
        
    current_time = time.time()
    for filename in os.listdir(DATA_FILES_DIR):
        file_path = os.path.join(DATA_FILES_DIR, filename)
        try:
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > (max_age_hours * 3600):  # Convert hours to seconds
                os.remove(file_path)
                logging.info(f"Removed old file: {file_path} (age: {file_age/3600:.1f} hours)")
        except Exception as e:
            logging.error(f"Error removing file {file_path}: {str(e)}")

def init_data_directory() -> None:
    """Initialize the data directory and set up logging"""
    # Ensure data directory exists
    os.makedirs(DATA_FILES_DIR, exist_ok=True)
    
    # Set up logging specifically for data operations
    data_log_file = 'logs/code_execution.log'
    os.makedirs(os.path.dirname(data_log_file), exist_ok=True)
    
    file_handler = logging.FileHandler(data_log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    logger = logging.getLogger('code_utils')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Log directory initialization
    logger.info(f"Initialized data directory at {DATA_FILES_DIR}")

# Initialize on module import
init_data_directory()

def generate_analysis_code(file_path: str, analysis_request: str) -> str:
    """
    Generate Python code for data analysis based on user request.
    
    Args:
        file_path: Path to the data file
        analysis_request: Natural language description of desired analysis
        
    Returns:
        str: Generated Python code
    """
    # Get file extension to determine data format
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    # Basic template for data analysis
    template = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('default')  # Use default matplotlib style
sns.set_theme()  # Apply seaborn theme on top

# Read the data file
print(f"Reading data from {file_path}...")
"""
    
    # Add file reading code based on file type
    if file_extension == '.csv':
        template += f"df = pd.read_csv('{file_path}')\n"
    elif file_extension in ['.xlsx', '.xls']:
        template += f"df = pd.read_excel('{file_path}')\n"
    else:
        # Default to CSV
        template += f"df = pd.read_csv('{file_path}')\n"
    
    # Add basic data exploration
    template += """
# Display basic information
print("\\nDataset Info:")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\\nColumns:", df.columns.tolist())

# Display data types
print("\\nData Types:")
print(df.dtypes)

# Check for missing values
print("\\nMissing Values:")
print(df.isnull().sum())

# Basic statistics for numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    print("\\nSummary Statistics:")
    print(df[numeric_cols].describe())
"""
    
    # Add visualization code based on the analysis request
    viz_code = generate_visualization_code(analysis_request.lower())
    template += "\n" + viz_code
    
    return template

def generate_visualization_code(analysis_request: str) -> str:
    """
    Generate visualization code based on analysis request.
    
    Args:
        analysis_request: The analysis request string
        
    Returns:
        str: Generated visualization code
    """
    viz_code = """
# Create visualizations based on the data types
plt.figure(figsize=(12, 6))
"""
    
    # Add specific visualizations based on keywords in the request
    if any(word in analysis_request for word in ['distribution', 'histogram', 'spread']):
        viz_code += """
# Create histograms for numeric columns
for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'histogram_{col}.png')
    plt.close()
"""
    
    if any(word in analysis_request for word in ['correlation', 'relationship', 'compare']):
        viz_code += """
# Create correlation heatmap
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 10))
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
"""
    
    if any(word in analysis_request for word in ['time series', 'trend', 'over time']):
        viz_code += """
# Check for datetime columns
date_cols = df.select_dtypes(include=['datetime64']).columns
if len(date_cols) > 0:
    date_col = date_cols[0]
    for col in numeric_cols[:2]:
        plt.figure(figsize=(12, 6))
        plt.plot(df[date_col], df[col])
        plt.title(f'{col} Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'timeseries_{col}.png')
        plt.close()
"""
    
    if any(word in analysis_request for word in ['bar', 'count', 'frequency']):
        viz_code += """
# Create bar plots for categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
    plt.figure(figsize=(12, 6))
    value_counts = df[col].value_counts().head(10)  # Top 10 categories
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f'Top 10 Categories in {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'barplot_{col}.png')
    plt.close()
"""
    
    if any(word in analysis_request for word in ['scatter', 'relationship']):
        viz_code += """
# Create scatter plots if multiple numeric columns exist
if len(numeric_cols) >= 2:
    for i in range(min(2, len(numeric_cols))):
        for j in range(i+1, min(3, len(numeric_cols))):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=numeric_cols[i], y=numeric_cols[j])
            plt.title(f'Scatter Plot: {numeric_cols[i]} vs {numeric_cols[j]}')
            plt.tight_layout()
            plt.savefig(f'scatter_{numeric_cols[i]}_{numeric_cols[j]}.png')
            plt.close()
"""
    
    # Add catch-all visualization if no specific type was requested
    if 'box' in analysis_request or not any(word in analysis_request for word in ['distribution', 'correlation', 'time series', 'bar', 'scatter']):
        viz_code += """
# Create box plots for numeric columns
if len(numeric_cols) > 0:
    plt.figure(figsize=(12, 6))
    df[numeric_cols].boxplot()
    plt.title('Box Plots of Numeric Variables')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('boxplots.png')
    plt.close()
"""
    
    return viz_code

def analyze_data(file_path: str, user_id: Optional[int] = None, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """
    Analyze a data file and generate visualizations.
    
    Args:
        file_path: Path to the data file
        user_id: Optional user ID for file management
        analysis_type: Type of analysis to perform (e.g., 'summary', 'correlation', 'distribution')
        
    Returns:
        Dict containing analysis results and visualization paths
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Read the data
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            return {"error": f"Unsupported file type: {file_extension}"}
        
        # Basic statistics
        summary = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        # Create visualizations based on requested type
        plots = []
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Generate ONLY the requested chart type (unless comprehensive mode)
        if analysis_type == "comprehensive":
            # For comprehensive, limit to just 1-2 of each type to avoid too many charts
            # Distribution plot (just one)
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x=col, kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                plot_path = os.path.join(DATA_FILES_DIR, f'dist_{user_id}_{col}_{int(time.time())}.png')
                plt.savefig(plot_path)
                plt.close()
                plots.append(plot_path)
            
            # Correlation matrix (just one)
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                ax.set_title('Correlation Matrix')
                plot_path = os.path.join(DATA_FILES_DIR, f'corr_{user_id}_{int(time.time())}.png')
                plt.savefig(plot_path)
                plt.close()
                plots.append(plot_path)
                
            # For comprehensive mode, add just one chart of other types as well
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                col = categorical_cols[0]
                plt.figure(figsize=(12, 6))
                value_counts = df[col].value_counts().head(10)
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f'Top 10 Categories in {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_path = os.path.join(DATA_FILES_DIR, f'bar_{user_id}_{col}_{int(time.time())}.png')
                plt.savefig(plot_path)
                plt.close()
                plots.append(plot_path)
                
            # Box plot as well
            if len(numeric_cols) > 0:
                plt.figure(figsize=(12, 6))
                df[numeric_cols[:5]].boxplot()  # Limit to 5 columns
                plt.title('Box Plots of Numeric Variables')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_path = os.path.join(DATA_FILES_DIR, f'boxplot_{user_id}_{int(time.time())}.png')
                plt.savefig(plot_path)
                plt.close()
                plots.append(plot_path)
        
        # Handle specific chart types (when not in comprehensive mode)
        elif analysis_type == "distribution":
            # Distribution plots for numeric columns
            for col in numeric_cols[:3]:  # Up to 3 distribution charts
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x=col, kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                plot_path = os.path.join(DATA_FILES_DIR, f'dist_{user_id}_{col}_{int(time.time())}.png')
                plt.savefig(plot_path)
                plt.close()
                plots.append(plot_path)
        
        elif analysis_type == "correlation":
            # Correlation matrix
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                ax.set_title('Correlation Matrix')
                plot_path = os.path.join(DATA_FILES_DIR, f'corr_{user_id}_{int(time.time())}.png')
                plt.savefig(plot_path)
                plt.close()
                plots.append(plot_path)
        
        elif analysis_type == "bar":
            # Bar charts for categorical data
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols[:3]:  # Up to 3 bar charts
                plt.figure(figsize=(12, 6))
                value_counts = df[col].value_counts().head(10)  # Top 10 categories
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f'Top 10 Categories in {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_path = os.path.join(DATA_FILES_DIR, f'bar_{user_id}_{col}_{int(time.time())}.png')
                plt.savefig(plot_path)
                plt.close()
                plots.append(plot_path)
                
        elif analysis_type == "scatter":
            # Scatter plots if multiple numeric columns
            if len(numeric_cols) >= 2:
                for i in range(min(2, len(numeric_cols))):
                    for j in range(i+1, min(i+3, len(numeric_cols))):
                        plt.figure(figsize=(10, 6))
                        sns.scatterplot(data=df, x=numeric_cols[i], y=numeric_cols[j])
                        plt.title(f'Scatter Plot: {numeric_cols[i]} vs {numeric_cols[j]}')
                        plt.tight_layout()
                        plot_path = os.path.join(DATA_FILES_DIR, f'scatter_{user_id}_{numeric_cols[i]}_{numeric_cols[j]}_{int(time.time())}.png')
                        plt.savefig(plot_path)
                        plt.close()
                        plots.append(plot_path)
        
        elif analysis_type == "box":
            # Box plots for numeric columns
            if len(numeric_cols) > 0:
                plt.figure(figsize=(12, 6))
                df[numeric_cols].boxplot()
                plt.title('Box Plots of Numeric Variables')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_path = os.path.join(DATA_FILES_DIR, f'boxplot_{user_id}_{int(time.time())}.png')
                plt.savefig(plot_path)
                plt.close()
                plots.append(plot_path)
        
        return {
            "success": True,
            "summary": summary,
            "plots": plots
        }
        
    except Exception as e:
        logging.error(f"Error analyzing data: {str(e)}")
        return {"error": str(e)}

def clean_old_files(max_age_hours=23):
    """Clean up old data files and visualizations"""
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        # Clean up files in DATA_FILES_DIR
        if os.path.exists(DATA_FILES_DIR):
            for filename in os.listdir(DATA_FILES_DIR):
                file_path = os.path.join(DATA_FILES_DIR, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        try:
                            os.remove(file_path)
                            logging.info(f"Removed old file: {file_path}")
                        except Exception as e:
                            logging.error(f"Error removing file {file_path}: {str(e)}")

    except Exception as e:
        logging.error(f"Error in clean_old_files: {str(e)}")

def format_output_path(output_path: str) -> str:
    """Format file paths in output to remove sandbox references"""
    if not output_path:
        return output_path
        
    # Remove sandbox path references
    output_path = re.sub(r'\(sandbox:.*?/temp_data_files/', '(', output_path)
    
    # Keep only the filename
    output_path = os.path.basename(output_path)
    
    return output_path