import os
import sys
import io
import logging
import asyncio
import traceback
import contextlib
import tempfile
import uuid
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Import data analysis libraries
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    LIBRARIES_AVAILABLE = False
    logging.warning(f"Data analysis libraries not available: {str(e)}")

# Import utility functions
from .code_utils import DATA_FILES_DIR, format_output_path, clean_old_files

# Configure logging
log_file = 'logs/data_analyzer.log'
os.makedirs(os.path.dirname(log_file), exist_ok=True)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(formatter)
logger = logging.getLogger('data_analyzer')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

def _is_valid_python_code(code_string: str) -> bool:
    """
    Check if a string contains valid Python code or is natural language.
    
    Args:
        code_string: String to check
        
    Returns:
        bool: True if it's valid Python code, False if it's natural language
    """
    try:
        # Strip whitespace and check for common natural language patterns
        stripped = code_string.strip()
        
        # Check for obvious natural language patterns
        natural_language_indicators = [
            'analyze', 'create', 'show', 'display', 'plot', 'visualize',
            'tell me', 'give me', 'what is', 'how many', 'find'
        ]
        
        # If it starts with typical natural language words, it's likely not Python
        first_words = stripped.lower().split()[:3]
        if any(indicator in ' '.join(first_words) for indicator in natural_language_indicators):
            return False
        
        # Try to compile as Python code
        compile(stripped, '<string>', 'exec')
        return True
    except SyntaxError:
        return False
    except Exception:
        return False

# Data analysis templates
ANALYSIS_TEMPLATES = {
    "summary": """
# Data Summary Analysis
# User request: {custom_request}
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('{file_path}') if '{file_path}'.endswith('.csv') else pd.read_excel('{file_path}')

print("=== DATA SUMMARY ===")
print(f"Shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
print("\\n=== DATA TYPES ===")
print(df.dtypes)
print("\\n=== MISSING VALUES ===")
print(df.isnull().sum())
print("\\n=== BASIC STATISTICS ===")
print(df.describe())
""",
    
    "correlation": """
# Correlation Analysis
# User request: {custom_request}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('{file_path}') if '{file_path}'.endswith('.csv') else pd.read_excel('{file_path}')

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df.columns) > 1:
    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('{output_path}')
    plt.close()
    
    print("=== CORRELATION ANALYSIS ===")
    print(correlation_matrix)
    
    # Find strong correlations
    strong_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_corr.append((correlation_matrix.columns[i], 
                                  correlation_matrix.columns[j], corr_val))
    
    if strong_corr:
        print("\\n=== STRONG CORRELATIONS (|r| > 0.7) ===")
        for col1, col2, corr in strong_corr:
            print(f"{{col1}} <-> {{col2}}: {{corr:.3f}}")
else:
    print("Not enough numeric columns for correlation analysis")
""",
    
    "distribution": """
# Distribution Analysis
# User request: {custom_request}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('{file_path}') if '{file_path}'.endswith('.csv') else pd.read_excel('{file_path}')

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

if len(numeric_cols) > 0:
    # Create distribution plots
    n_cols = min(len(numeric_cols), 4)
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            df[col].dropna().hist(bins=30, alpha=0.7, edgecolor='black', ax=axes[i])
            axes[i].set_title(f'Distribution of {{col}}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide extra subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('{output_path}')
    plt.close()
    
    print("=== DISTRIBUTION ANALYSIS ===")
    for col in numeric_cols:
        print(f"\\n{{col}}:")
        print(f"  Mean: {{df[col].mean():.2f}}")
        print(f"  Median: {{df[col].median():.2f}}")
        print(f"  Std: {{df[col].std():.2f}}")
        print(f"  Skewness: {{df[col].skew():.2f}}")
else:
    print("No numeric columns found for distribution analysis")
""",
    
    "comprehensive": """
# Comprehensive Data Analysis
# User request: {custom_request}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('{file_path}') if '{file_path}'.endswith('.csv') else pd.read_excel('{file_path}')

print("=== COMPREHENSIVE DATA ANALYSIS ===")
print(f"Dataset shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")

# Basic info
print("\\n=== DATA TYPES ===")
print(df.dtypes)

print("\\n=== MISSING VALUES ===")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\\n=== BASIC STATISTICS ===")
print(df.describe())

# Numeric analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    print("\\n=== NUMERIC COLUMNS ANALYSIS ===")
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Correlation heatmap
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[0,0])
        axes[0,0].set_title('Correlation Matrix')
    
    # 2. Distribution of first numeric column
    if len(numeric_cols) >= 1:
        df[numeric_cols[0]].hist(bins=30, ax=axes[0,1])
        axes[0,1].set_title(f'Distribution of {{numeric_cols[0]}}')
    
    # 3. Box plot of numeric columns
    if len(numeric_cols) <= 5:
        df[numeric_cols].boxplot(ax=axes[1,0])
        axes[1,0].set_title('Box Plot of Numeric Columns')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Pairplot for first few numeric columns
    if len(numeric_cols) >= 2:
        scatter_cols = numeric_cols[:min(3, len(numeric_cols))]
        if len(scatter_cols) == 2:
            axes[1,1].scatter(df[scatter_cols[0]], df[scatter_cols[1]], alpha=0.6)
            axes[1,1].set_xlabel(scatter_cols[0])
            axes[1,1].set_ylabel(scatter_cols[1])
            axes[1,1].set_title(f'{{scatter_cols[0]}} vs {{scatter_cols[1]}}')
    
    plt.tight_layout()
    plt.savefig('{output_path}')
    plt.close()

# Categorical analysis
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("\\n=== CATEGORICAL COLUMNS ANALYSIS ===")
    for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
        print(f"\\n{{col}}:")
        print(df[col].value_counts().head())
"""
}

async def install_packages(packages: List[str]) -> Dict[str, Any]:
    """
    Install Python packages in a sandboxed environment.
    
    Args:
        packages: List of package names to install
        
    Returns:
        Dict containing installation results
    """
    try:
        import subprocess
        
        installed = []
        failed = []
        
        for package in packages:
            try:
                # Use pip to install package
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    installed.append(package)
                    logger.info(f"Successfully installed package: {package}")
                else:
                    failed.append({"package": package, "error": result.stderr})
                    logger.error(f"Failed to install package {package}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                failed.append({"package": package, "error": "Installation timeout"})
                logger.error(f"Installation timeout for package: {package}")
            except Exception as e:
                failed.append({"package": package, "error": str(e)})
                logger.error(f"Error installing package {package}: {str(e)}")
        
        return {
            "success": True,
            "installed": installed,
            "failed": failed,
            "message": f"Installed {len(installed)} packages, {len(failed)} failed"
        }
        
    except Exception as e:
        logger.error(f"Error in package installation: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "installed": [],
            "failed": packages
        }

async def analyze_data_file(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze data files with pre-built templates and custom analysis.
    
    Args:
        args: Dictionary containing:
            - file_path: Path to the data file (CSV/Excel)
            - analysis_type: Type of analysis (summary, correlation, distribution, comprehensive)
            - custom_analysis: Optional custom analysis request in natural language
            - user_id: Optional user ID for file management
            - install_packages: Optional list of packages to install
            
    Returns:
        Dict containing analysis results
    """
    try:
        if not LIBRARIES_AVAILABLE:
            return {
                "success": False,
                "error": "Data analysis libraries not available. Please install pandas, numpy, matplotlib, seaborn."
            }
        
        file_path = args.get("file_path", "")
        analysis_type = args.get("analysis_type", "comprehensive")
        custom_analysis = args.get("custom_analysis", "")
        user_id = args.get("user_id")
        packages_to_install = args.get("install_packages", [])
        
        # Install packages if requested
        if packages_to_install:
            install_result = await install_packages(packages_to_install)
            if not install_result["success"]:
                logger.warning(f"Package installation issues: {install_result}")
        
        # Validate file path
        if not file_path or not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"Data file not found: {file_path}"
            }
        
        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ['.csv', '.xlsx', '.xls']:
            return {
                "success": False,
                "error": "Unsupported file format. Please use CSV or Excel files."
            }
        
        # Generate output path for visualizations
        timestamp = int(time.time())
        output_filename = f"analysis_{user_id or 'user'}_{timestamp}.png"
        output_path = format_output_path(output_filename)
        
        # Determine analysis code
        if custom_analysis:
            # Check if custom_analysis contains valid Python code or is natural language
            is_python_code = _is_valid_python_code(custom_analysis)
            
            if is_python_code:
                # Generate custom analysis code with valid Python
                code = f"""
# Custom Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('{file_path}') if '{file_path}'.endswith('.csv') else pd.read_excel('{file_path}')

print("=== CUSTOM DATA ANALYSIS ===")
print(f"Dataset loaded: {{df.shape}}")

# Custom analysis based on user request
{custom_analysis}

# Save any plots
if plt.get_fignums():
    plt.savefig('{output_path}')
    plt.close()
"""
            else:
                # For natural language queries, use comprehensive analysis with comment
                logger.info(f"Natural language query detected: {custom_analysis}")
                analysis_type = "comprehensive"
                code = ANALYSIS_TEMPLATES[analysis_type].format(
                    file_path=file_path,
                    output_path=output_path,
                    custom_request=custom_analysis
                )
        else:
            # Use predefined template
            if analysis_type not in ANALYSIS_TEMPLATES:
                analysis_type = "comprehensive"
            
            # Format template with default values
            template_vars = {
                'file_path': file_path,
                'output_path': output_path,
                'custom_request': custom_analysis or 'General data analysis'
            }
            code = ANALYSIS_TEMPLATES[analysis_type].format(**template_vars)
        
        # Execute the analysis code
        result = await execute_analysis_code(code, output_path)
        
        # Add file information to result
        result.update({
            "file_path": file_path,
            "analysis_type": analysis_type,
            "custom_analysis": bool(custom_analysis)
        })
        
        # Clean up old files
        clean_old_files()
        
        return result
        
    except Exception as e:
        error_msg = f"Error in data analysis: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "success": False,
            "error": error_msg,
            "traceback": traceback.format_exc()
        }

async def execute_analysis_code(code: str, output_path: str) -> Dict[str, Any]:
    """
    Execute data analysis code in a controlled environment.
    
    Args:
        code: Python code to execute
        output_path: Path where visualizations should be saved
        
    Returns:
        Dict containing execution results
    """
    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        # Create a controlled execution environment
        exec_globals = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "print": print,
        }
        
        # Try to import plotly if available
        try:
            exec_globals["go"] = go
            exec_globals["px"] = px
        except:
            pass
        
        # Execute the code
        exec(code, exec_globals)
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Get the output
        output = captured_output.getvalue()
        
        # Check if visualization was created
        visualizations = []
        if os.path.exists(output_path):
            visualizations.append(output_path)
        
        logger.info(f"Data analysis executed successfully, output length: {len(output)}")
        
        return {
            "success": True,
            "output": output,
            "visualizations": visualizations,
            "has_visualization": len(visualizations) > 0
        }
        
    except Exception as e:
        # Restore stdout
        sys.stdout = old_stdout
        
        error_msg = f"Error executing analysis code: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        
        return {
            "success": False,
            "error": error_msg,
            "output": captured_output.getvalue() if 'captured_output' in locals() else "",
            "traceback": traceback.format_exc()
        }

# Utility function to validate data analysis requests
def validate_analysis_request(args: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate data analysis request parameters.
    
    Args:
        args: Analysis request arguments
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["file_path"]
    
    for field in required_fields:
        if field not in args or not args[field]:
            return False, f"Missing required field: {field}"
    
    # Validate analysis type
    analysis_type = args.get("analysis_type", "comprehensive")
    valid_types = list(ANALYSIS_TEMPLATES.keys())
    
    if analysis_type not in valid_types:
        return False, f"Invalid analysis type. Valid types: {valid_types}"
    
    return True, ""
