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

# Set global matplotlib parameters for better readability
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

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
            # Use a better visual style
            plt.style.use('ggplot')
            
            # Determine chart type based on keywords in the query
            if any(keyword in query.lower() for keyword in ["pie", "circle"]):
                # Pie chart - works best with categorical data
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    cat_col = categorical_cols[0]
                    num_col = numeric_cols[0]
                    
                    # Better handling of pie chart data - limit to top 6 categories max
                    top_categories = df.groupby(cat_col)[num_col].sum().nlargest(6)
                    
                    # Add "Other" category if there are more than 6 categories
                    if len(df[cat_col].unique()) > 6:
                        other_sum = df.groupby(cat_col)[num_col].sum().sum() - top_categories.sum()
                        if other_sum > 0:
                            top_categories["Other"] = other_sum
                    
                    # Larger figure size for better readability
                    plt.figure(figsize=(12, 9))
                    
                    # Enhanced pie chart
                    wedges, texts, autotexts = plt.pie(
                        top_categories, 
                        labels=None,  # We'll add a legend instead of cluttering the pie
                        autopct='%1.1f%%',
                        shadow=False, 
                        startangle=90,
                        explode=[0.05] * len(top_categories),  # Slight separation for visibility
                        textprops={'color': 'white', 'weight': 'bold', 'fontsize': 14},
                        wedgeprops={'width': 0.6, 'edgecolor': 'white', 'linewidth': 2}
                    )
                    
                    # Make the percentage labels more visible
                    for autotext in autotexts:
                        autotext.set_fontsize(12)
                        autotext.set_weight('bold')
                    
                    # Add a legend outside the pie for better readability
                    plt.legend(
                        wedges, 
                        top_categories.index, 
                        title=cat_col,
                        loc="center left",
                        bbox_to_anchor=(1, 0, 0.5, 1)
                    )
                    
                    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                    plt.title(f"Distribution of {num_col} by {cat_col}", pad=20)
                    plt.tight_layout()
                    
            elif any(keyword in query.lower() for keyword in ["bar", "column"]):
                # Bar chart
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    cat_col = categorical_cols[0]
                    num_col = numeric_cols[0]
                    
                    # Determine optimal figure size based on number of categories
                    category_count = min(10, len(df[cat_col].unique()))
                    fig_height = max(6, category_count * 0.4 + 4)  # Dynamic height based on categories
                    plt.figure(figsize=(12, fig_height))
                    
                    # Sort for better visualization
                    top_values = df.groupby(cat_col)[num_col].sum().nlargest(10)
                    
                    # Use a horizontal bar chart for better label readability with many categories
                    if len(top_values) > 5:
                        ax = top_values.plot.barh(
                            color='#5975a4', 
                            edgecolor='#344e7a',
                            linewidth=1.5
                        )
                        # Add data labels at the end of each bar
                        for i, v in enumerate(top_values):
                            ax.text(v * 1.01, i, f'{v:,.1f}', va='center', fontweight='bold')
                        plt.xlabel(num_col)
                        plt.ylabel(cat_col)
                    else:
                        # For fewer categories, use vertical bars
                        ax = top_values.plot.bar(
                            color='#5975a4', 
                            edgecolor='#344e7a',
                            linewidth=1.5
                        )
                        # Add data labels on top of each bar
                        for i, v in enumerate(top_values):
                            ax.text(i, v * 1.01, f'{v:,.1f}', ha='center', fontweight='bold')
                        plt.ylabel(num_col)
                        plt.xlabel(cat_col)
                        plt.xticks(rotation=30, ha='right')
                    
                    plt.grid(axis='both', linestyle='--', alpha=0.7)
                    plt.title(f"{num_col} by {cat_col} (Top {len(top_values)})", pad=20)
                    plt.tight_layout(pad=2)
                else:
                    # Improved bar chart for numeric data only
                    plt.figure(figsize=(12, 7))
                    top_values = df[numeric_cols[0]].nlargest(10)
                    ax = top_values.plot.bar(
                        color='#5975a4', 
                        edgecolor='#344e7a',
                        linewidth=1.5
                    )
                    
                    # Add value labels on top of bars
                    for i, v in enumerate(top_values):
                        ax.text(i, v * 1.01, f'{v:,.1f}', ha='center', fontweight='bold')
                        
                    plt.title(f"Top {len(top_values)} highest values of {numeric_cols[0]}", pad=20)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.xticks(rotation=30, ha='right')
                    plt.tight_layout(pad=2)
                    
            elif any(keyword in query.lower() for keyword in ["scatter", "dispersion"]):
                # Enhanced scatter plot
                if len(numeric_cols) >= 2:
                    plt.figure(figsize=(12, 8))
                    
                    # If we have a categorical column, use it for coloring
                    if len(categorical_cols) > 0:
                        # Limit to a reasonable number of categories for coloring
                        cat_col = categorical_cols[0]
                        top_cats = df[cat_col].value_counts().nlargest(8).index.tolist()
                        
                        # Create a color map
                        colormap = plt.cm.get_cmap('tab10', len(top_cats))
                        
                        # Plot each category with different color
                        for i, category in enumerate(top_cats):
                            subset = df[df[cat_col] == category]
                            plt.scatter(
                                subset[numeric_cols[0]], 
                                subset[numeric_cols[1]], 
                                alpha=0.7,
                                edgecolor='w', 
                                s=80, 
                                label=category,
                                color=colormap(i)
                            )
                        plt.legend(title=cat_col, loc='best')
                    else:
                        # Regular scatter plot with improved visibility
                        scatter = plt.scatter(
                            df[numeric_cols[0]], 
                            df[numeric_cols[1]], 
                            alpha=0.7,
                            edgecolor='w', 
                            s=80,
                            c=df[numeric_cols[0]],  # Color by x-axis value for visual enhancement
                            cmap='viridis'
                        )
                        plt.colorbar(scatter, label=numeric_cols[0])
                    
                    plt.xlabel(numeric_cols[0], fontweight='bold')
                    plt.ylabel(numeric_cols[1], fontweight='bold')
                    plt.title(f"Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}", pad=20)
                    
                    # Add trend line if there seems to be a correlation
                    if abs(df[numeric_cols[0]].corr(df[numeric_cols[1]])) > 0.3:
                        z = np.polyfit(df[numeric_cols[0]].dropna(), df[numeric_cols[1]].dropna(), 1)
                        p = np.poly1d(z)
                        plt.plot(
                            df[numeric_cols[0]].sort_values(), 
                            p(df[numeric_cols[0]].sort_values()),
                            "r--", 
                            linewidth=2,
                            label=f"Trend line (r={df[numeric_cols[0]].corr(df[numeric_cols[1]]):.2f})"
                        )
                        plt.legend()
                    
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout(pad=2)
                    
            elif any(keyword in query.lower() for keyword in ["histogram", "hist", "distribution"]):
                # Enhanced histogram
                plt.figure(figsize=(12, 7))
                # Calculate optimal number of bins
                data = df[numeric_cols[0]].dropna()
                
                # Better bin calculation based on data distribution
                iqr = np.percentile(data, 75) - np.percentile(data, 25)
                bin_width = 2 * iqr / (len(data) ** (1/3))  # Freedman-Diaconis rule
                if bin_width > 0:
                    bins = int((data.max() - data.min()) / bin_width)
                    bins = min(max(bins, 10), 50)  # Between 10 and 50 bins
                else:
                    bins = 15  # Default if calculation fails
                
                # Plot histogram with KDE
                ax = plt.subplot(111)
                n, bins_arr, patches = ax.hist(
                    data, 
                    bins=bins, 
                    alpha=0.7,
                    color='#5975a4', 
                    edgecolor='#344e7a',
                    linewidth=1.5,
                    density=True  # Normalize for KDE overlay
                )
                
                # Add KDE line for smoother visualization
                from scipy import stats
                kde_x = np.linspace(data.min(), data.max(), 1000)
                kde = stats.gaussian_kde(data)
                ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='Density')
                
                # Add vertical lines for key statistics
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='orange', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')
                
                plt.title(f"Distribution of {numeric_cols[0]}", pad=20)
                plt.xlabel(numeric_cols[0], fontweight='bold')
                plt.ylabel("Frequency", fontweight='bold')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend()
                plt.tight_layout(pad=2)
                
            elif len(date_cols) > 0 and len(numeric_cols) > 0:
                # Enhanced time series chart
                plt.figure(figsize=(14, 8))
                date_col = date_cols[0]
                num_col = numeric_cols[0]
                
                # Sort by date and plot
                temp_df = df[[date_col, num_col]].dropna().sort_values(date_col)
                
                # Limit number of points for readability if too many
                if len(temp_df) > 100:
                    # Resample to reduce point density
                    temp_df = temp_df.set_index(date_col)
                    # Determine appropriate frequency based on date range
                    date_range = (temp_df.index.max() - temp_df.index.min()).days
                    if date_range > 365*2:  # More than 2 years
                        freq = 'M'  # Monthly
                    elif date_range > 90:  # More than 3 months
                        freq = 'W'  # Weekly
                    else:
                        freq = 'D'  # Daily
                    
                    temp_df = temp_df.resample(freq).mean().reset_index()
                
                # Plot with enhanced styling
                plt.plot(
                    temp_df[date_col], 
                    temp_df[num_col], 
                    marker='o', 
                    markersize=6,
                    markerfacecolor='white',
                    markeredgecolor='#5975a4',
                    markeredgewidth=1.5,
                    linestyle='-', 
                    linewidth=2,
                    color='#5975a4'
                )
                
                plt.title(f"{num_col} over time", pad=20)
                plt.xlabel("Date", fontweight='bold')
                plt.ylabel(num_col, fontweight='bold')
                
                # Format x-axis date labels better
                plt.gcf().autofmt_xdate()
                plt.grid(True, alpha=0.3)
                
                # Add trend line
                try:
                    x = np.arange(len(temp_df))
                    z = np.polyfit(x, temp_df[num_col], 1)
                    p = np.poly1d(z)
                    plt.plot(temp_df[date_col], p(x), "r--", linewidth=2, 
                             label=f"Trend line (slope: {z[0]:.4f})")
                    plt.legend()
                except Exception:
                    pass  # Skip trend line if it fails
                
                plt.tight_layout(pad=2)
                
            else:
                # Default: enhanced line chart for numeric data
                if len(numeric_cols) > 0:
                    plt.figure(figsize=(14, 8))
                    
                    # Get the data
                    data = df[numeric_cols[0]]
                    
                    # If too many points, bin or resample
                    if len(data) > 100:
                        # Use rolling average for smoother line
                        window = max(5, len(data) // 50)  # Adaptive window size
                        rolling_data = data.rolling(window=window, center=True).mean()
                        
                        # Plot both original and smoothed data
                        plt.plot(data.index, data, 'o', markersize=4, alpha=0.4, label='Original data')
                        plt.plot(
                            rolling_data.index, 
                            rolling_data, 
                            linewidth=3, 
                            color='#d62728', 
                            label=f'Moving average (window={window})'
                        )
                        plt.legend()
                    else:
                        # For fewer points, use a more detailed visualization
                        plt.plot(
                            data.index, 
                            data, 
                            marker='o', 
                            markersize=6,
                            markerfacecolor='white',
                            markeredgecolor='#5975a4',
                            markeredgewidth=1.5,
                            linestyle='-', 
                            linewidth=2,
                            color='#5975a4'
                        )
                    
                    plt.title(f"Line chart for {numeric_cols[0]}", pad=20)
                    plt.ylabel(numeric_cols[0], fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout(pad=2)
            
            # Save chart to bytes buffer with higher DPI for better quality
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
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