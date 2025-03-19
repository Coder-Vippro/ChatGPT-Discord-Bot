import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import io
import logging
import asyncio
import functools
import os
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List

# Ensure matplotlib doesn't require a GUI backend
matplotlib.use('Agg')

async def process_data_file(file_bytes: bytes, filename: str, query: str) -> Tuple[str, Optional[bytes], Optional[Dict[str, Any]]]:
    """
    Analyze and visualize data from CSV/Excel files.
    
    Args:
        file_bytes: File content as bytes
        filename: File name
        query: User command/query
        
    Returns:
        Tuple containing text summary, image bytes (if any) and metadata
    """
    try:
        # Use thread pool to avoid blocking event loop with CPU-bound tasks
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(_process_data_file_sync, file_bytes, filename, query)
        )
    except Exception as e:
        logging.error(f"Error processing data file: {str(e)}")
        return f"Error processing file {filename}: {str(e)}", None, None

def _process_data_file_sync(file_bytes: bytes, filename: str, query: str) -> Tuple[str, Optional[bytes], Optional[Dict[str, Any]]]:
    """Synchronous version of process_data_file to run in thread pool"""
    file_obj = io.BytesIO(file_bytes)
    
    try:
        # Read file based on format with improved error handling
        if filename.lower().endswith('.csv'):
            try:
                # Try multiple encodings and separator detection
                df = pd.read_csv(file_obj, encoding='utf-8')
            except UnicodeDecodeError:
                # Reset file pointer and try different encoding
                file_obj.seek(0)
                df = pd.read_csv(file_obj, encoding='latin1')
        elif filename.lower().endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(file_obj)
            except Exception as excel_err:
                logging.error(f"Excel read error: {excel_err}")
                # Try with engine specification
                file_obj.seek(0)
                df = pd.read_excel(file_obj, engine='openpyxl')
        else:
            return "Unsupported file format. Please use CSV or Excel.", None, None
        
        if df.empty:
            return "The file does not contain any data.", None, None
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
            
        # Create metadata for dataframe
        rows = len(df)
        columns = len(df.columns)
        column_names = list(df.columns)
        
        # Data preprocessing for better analysis
        # Convert potential date columns
        for col in df.columns:
            # Try to convert columns that might be dates but are stored as strings
            if df[col].dtype == 'object':
                try:
                    # Check if the column might contain dates
                    sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample and isinstance(sample, str) and ('/' in sample or '-' in sample):
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                except Exception:
                    pass  # Skip if conversion fails
        
        # Format data for analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Create basic information about the data
        summary = f"Data analysis for {filename}:\n"
        summary += f"- Rows: {rows}\n"
        summary += f"- Columns: {columns}\n"
        summary += f"- Column names: {', '.join(column_names)}\n\n"
        
        # Basic descriptive statistics
        if len(numeric_cols) > 0:
            summary += "Statistics for numeric data:\n"
            desc_stats = df[numeric_cols].describe().round(2)
            summary += desc_stats.to_string() + "\n\n"
            
        # Statistics for categorical data
        if len(categorical_cols) > 0:
            summary += "Value distribution for categorical columns:\n"
            for col in categorical_cols[:3]:  # Limit displayed columns
                value_counts = df[col].value_counts().head(5)
                summary += f"{col}: {dict(value_counts)}\n"
            if len(categorical_cols) > 3:
                summary += f"...and {len(categorical_cols) - 3} other categorical columns.\n"
            summary += "\n"
            
        # Determine if a chart should be created
        chart_keywords = ["chart", "graph", "plot", "visualization", "visualize", 
                         "histogram", "bar chart", "line chart", "pie chart", "scatter"]
        create_chart = any(keyword in query.lower() for keyword in chart_keywords)
        
        # Metadata to return
        metadata = {
            "filename": filename,
            "rows": rows,
            "columns": columns,
            "column_names": column_names,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "date_columns": date_cols
        }
        
        # Create chart if requested or by default
        chart_image = None
        if (create_chart or len(query) < 10) and len(numeric_cols) > 0:  # Default to chart for short queries
            plt.figure(figsize=(10, 6))
            
            # Better chart styling
            plt.style.use('seaborn-v0_8')
            
            # Determine chart type based on keywords in the query
            if any(keyword in query.lower() for keyword in ["pie", "circle"]):
                # Pie chart - works best with categorical data
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    cat_col = categorical_cols[0]
                    num_col = numeric_cols[0]
                    # Better handling of pie chart data
                    top_categories = df.groupby(cat_col)[num_col].sum().nlargest(5)
                    # Add "Other" category if there are more than 5 categories
                    if len(df[cat_col].unique()) > 5:
                        other_sum = df.groupby(cat_col)[num_col].sum().sum() - top_categories.sum()
                        if other_sum > 0:
                            top_categories["Other"] = other_sum
                    
                    plt.figure(figsize=(10, 7))
                    plt.pie(top_categories, labels=top_categories.index, autopct='%1.1f%%',
                           shadow=True, startangle=90)
                    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                    plt.title(f"Pie Chart: {num_col} by {cat_col}")
            elif any(keyword in query.lower() for keyword in ["bar", "column"]):
                # Bar chart
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    cat_col = categorical_cols[0]
                    num_col = numeric_cols[0]
                    # Sort for better visualization
                    top_values = df.groupby(cat_col)[num_col].sum().nlargest(10)
                    top_values.plot.bar(color='skyblue', edgecolor='black')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.title(f"Bar Chart: {num_col} by {cat_col} (Top 10)")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                else:
                    df[numeric_cols[0]].nlargest(10).plot.bar(color='skyblue', edgecolor='black')
                    plt.title(f"Top 10 highest values of {numeric_cols[0]}")
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
            elif any(keyword in query.lower() for keyword in ["scatter", "dispersion"]):
                # Scatter plot
                if len(numeric_cols) >= 2:
                    plt.figure(figsize=(9, 6))
                    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6, 
                               edgecolor='w', s=50)
                    plt.xlabel(numeric_cols[0])
                    plt.ylabel(numeric_cols[1])
                    plt.title(f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}")
                    # Add trend line if there seems to be a correlation
                    if abs(df[numeric_cols[0]].corr(df[numeric_cols[1]])) > 0.3:
                        z = np.polyfit(df[numeric_cols[0]].dropna(), df[numeric_cols[1]].dropna(), 1)
                        p = np.poly1d(z)
                        plt.plot(df[numeric_cols[0]].sort_values(), 
                                p(df[numeric_cols[0]].sort_values()),
                                "r--", linewidth=1)
                    plt.grid(True, alpha=0.3)
            elif any(keyword in query.lower() for keyword in ["histogram", "hist", "distribution"]):
                # Histogram with better binning
                plt.figure(figsize=(10, 6))
                # Calculate optimal number of bins using Sturges' rule
                data = df[numeric_cols[0]].dropna()
                bins = int(np.ceil(np.log2(len(data))) + 1) if len(data) > 0 else 10
                
                plt.hist(data, bins=min(bins, 30), color='skyblue', edgecolor='black')
                plt.title(f"Distribution of {numeric_cols[0]}")
                plt.xlabel(numeric_cols[0])
                plt.ylabel("Frequency")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
            elif len(date_cols) > 0 and len(numeric_cols) > 0:
                # Time series chart if we have dates and numeric data
                plt.figure(figsize=(12, 6))
                date_col = date_cols[0]
                num_col = numeric_cols[0]
                
                # Sort by date and plot
                temp_df = df[[date_col, num_col]].dropna().sort_values(date_col)
                plt.plot(temp_df[date_col], temp_df[num_col], marker='o', markersize=3, 
                        linestyle='-', linewidth=1)
                plt.title(f"{num_col} over time")
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
            else:
                # Default: line chart if we have multiple numeric values
                if len(numeric_cols) > 0:
                    plt.figure(figsize=(10, 6))
                    df[numeric_cols[0]].plot(color='#1f77b4', alpha=0.8)
                    plt.title(f"Line chart for {numeric_cols[0]}")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
            
            # Save chart to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            chart_image = buf.read()
            plt.close()
            
            # Create a timestamp for the chart file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = f"chart_{timestamp}.png"
            
            # Save chart to temporary file
            chart_dir = os.path.join(os.getcwd(), "temp_charts")
            if not os.path.exists(chart_dir):
                os.makedirs(chart_dir)
                
            chart_path = os.path.join(chart_dir, chart_filename)
            with open(chart_path, "wb") as f:
                f.write(chart_image)
            
            summary += f"Chart created based on the data with improved visualization."
            
            # Add chart filename to metadata
            metadata["chart_filename"] = chart_filename
            metadata["chart_path"] = chart_path
            metadata["chart_created_at"] = datetime.now().timestamp()
            
        return summary, chart_image, metadata
        
    except Exception as e:
        logging.error(f"Error in _process_data_file_sync: {str(e)}")
        return f"Could not analyze file {filename}. Error: {str(e)}", None, None

async def cleanup_old_charts(max_age_hours=1):
    """
    Clean up chart images older than the specified time
    
    Args:
        max_age_hours: Maximum age in hours before deleting charts
    """
    try:
        chart_dir = os.path.join(os.getcwd(), "temp_charts")
        if not os.path.exists(chart_dir):
            return
            
        now = datetime.now().timestamp()
        deleted_count = 0
        
        for filename in os.listdir(chart_dir):
            if filename.startswith("chart_") and filename.endswith(".png"):
                file_path = os.path.join(chart_dir, filename)
                file_modified_time = os.path.getmtime(file_path)
                
                # If file is older than max_age_hours
                if now - file_modified_time > (max_age_hours * 3600):
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        logging.error(f"Error deleting chart file {filename}: {str(e)}")
        
        if deleted_count > 0:
            logging.info(f"Cleaned up {deleted_count} chart files older than {max_age_hours} hours")
    except Exception as e:
        logging.error(f"Error in cleanup_old_charts: {str(e)}")