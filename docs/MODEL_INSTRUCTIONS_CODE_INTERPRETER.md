# Model Instructions - Code Interpreter Usage

## 🎯 Overview

This document explains how the AI model should use the code interpreter tool to ensure packages are automatically installed and files are properly managed.

---

## 📦 **Package Auto-Installation**

### ✅ **What the Model SHOULD Do**

**Just import packages normally - they auto-install if missing!**

```python
# CORRECT - Just import what you need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Even specialized libraries
import tensorflow as tf
import torch
import geopandas as gpd
import opencv as cv2
```

### ❌ **What the Model SHOULD NOT Do**

**Don't check if packages are installed or ask users to install them:**

```python
# WRONG - Don't do this!
try:
    import seaborn
except ImportError:
    print("Please install seaborn")

# WRONG - Don't do this!
import subprocess
subprocess.run(['pip', 'install', 'seaborn'])

# WRONG - Don't do this!
print("First, install pandas: pip install pandas")
```

---

## 🔧 **How Auto-Install Works**

### **Behind the Scenes:**

1. Model writes code: `import seaborn as sns`
2. Code executes → ModuleNotFoundError detected
3. System auto-installs: `pip install seaborn`
4. Code re-executes automatically → Success!
5. User gets notification: "📦 Auto-installed: seaborn"

### **No Action Required from Model**

The model doesn't need to:
- Check if packages are installed
- Use `install_packages` parameter
- Handle installation errors
- Retry code execution

**Everything is automatic!**

---

## 📁 **File Management**

### **Loading User Files**

When users upload files, they get a `file_id`:

```python
# User uploaded "sales_data.csv" → file_id: "123456789_1696118400_abc123"

# Model's code:
import pandas as pd

# Load the file
df = load_file('123456789_1696118400_abc123')

print(f"Loaded {len(df)} rows")
print(df.head())
```

### **Creating Output Files**

**ANY file the model creates is captured and sent to the user:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import json

# Create CSV export
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
df.to_csv('results.csv', index=False)  # ✅ User gets this!

# Create visualization
plt.figure(figsize=(10, 6))
plt.plot(df['x'], df['y'])
plt.title('Results')
plt.savefig('plot.png')  # ✅ User gets this!

# Create JSON report
summary = {'total': 6, 'mean': 3.5}
with open('summary.json', 'w') as f:
    json.dump(summary, f, indent=2)  # ✅ User gets this!

# Create text report
with open('report.txt', 'w') as f:
    f.write('Analysis Results\n')
    f.write('================\n')
    f.write(f'Total: {summary["total"]}\n')  # ✅ User gets this!

print('Generated 4 files: CSV, PNG, JSON, TXT')
```

### **Supported Output Files (80+ formats)**

✅ **Data**: CSV, Excel, Parquet, JSON, XML, YAML  
✅ **Images**: PNG, JPEG, GIF, SVG, BMP, TIFF  
✅ **Text**: TXT, MD, LOG, HTML  
✅ **Code**: Python, JavaScript, SQL, R  
✅ **Scientific**: NumPy (.npy), Pickle, HDF5  
✅ **Archives**: ZIP, TAR, GZIP  

---

## 💡 **Best Practices for the Model**

### **1. Don't Over-Explain Package Installation**

❌ **BAD:**
```
I'll use seaborn for visualization. First, let me check if it's installed...
<execute code with try/except>
```

✅ **GOOD:**
```
I'll create a correlation heatmap using seaborn.
<execute code with import seaborn>
```

### **2. Create Files Instead of Printing Long Output**

❌ **BAD:**
```python
# Don't print entire dataframes
print(df.to_string())  # May get truncated!
```

✅ **GOOD:**
```python
# Save as file instead
df.to_csv('full_data.csv', index=False)
print(f"Saved {len(df)} rows to full_data.csv")
```

### **3. Use Descriptive Filenames**

❌ **BAD:**
```python
plt.savefig('1.png')
df.to_csv('output.csv')
```

✅ **GOOD:**
```python
plt.savefig('sales_trend_2024.png')
df.to_csv('cleaned_customer_data.csv')
```

### **4. Generate Multiple Output Types**

✅ **EXCELLENT:**
```python
# Give users data in multiple formats
df.to_csv('data.csv')           # For Excel users
df.to_json('data.json')         # For developers
df.to_parquet('data.parquet')   # For data scientists

# Create visualization
plt.savefig('chart.png')        # For viewing

# Create summary report
with open('summary.txt', 'w') as f:
    f.write('Analysis Summary\n')
    f.write(df.describe().to_string())
```

### **5. Handle Errors Gracefully**

✅ **GOOD:**
```python
try:
    df = load_file('file_id_here')
    # Process data
    df.to_csv('results.csv')
except Exception as e:
    print(f"Error: {e}")
    # Provide helpful message to user
```

---

## 📊 **Complete Example: Data Analysis**

### **User Request:**
"Analyze this CSV file and show me sales trends"

### **Model's Approach:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Auto-installs if needed!
import json

# Load user's uploaded file
df = load_file('user_file_id_123')

# 1. Basic exploration
print(f"Dataset: {len(df)} rows × {len(df.columns)} columns")
print(f"\nColumns: {', '.join(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# 2. Save summary statistics as JSON
summary = {
    'total_rows': len(df),
    'columns': df.columns.tolist(),
    'numeric_summary': df.describe().to_dict(),
    'date_range': {
        'start': df['date'].min(),
        'end': df['date'].max()
    } if 'date' in df.columns else None
}

with open('summary_statistics.json', 'w') as f:
    json.dump(summary, f, indent=2)

# 3. Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Sales trend over time
if 'date' in df.columns and 'sales' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    axes[0, 0].plot(df['date'], df['sales'])
    axes[0, 0].set_title('Sales Trend Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Sales ($)')
    axes[0, 0].grid(True)

# Distribution
df['sales'].hist(bins=30, ax=axes[0, 1])
axes[0, 1].set_title('Sales Distribution')
axes[0, 1].set_xlabel('Sales ($)')
axes[0, 1].set_ylabel('Frequency')

# Box plot
df.boxplot(column='sales', by='category', ax=axes[1, 0])
axes[1, 0].set_title('Sales by Category')
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Sales ($)')

# Top products
top_products = df.groupby('product')['sales'].sum().nlargest(10)
axes[1, 1].barh(top_products.index, top_products.values)
axes[1, 1].set_title('Top 10 Products by Sales')
axes[1, 1].set_xlabel('Total Sales ($)')

plt.tight_layout()
plt.savefig('sales_analysis.png', dpi=150)

# 4. Export cleaned data
df_cleaned = df.dropna()
df_cleaned.to_csv('cleaned_sales_data.csv', index=False)

# 5. Generate text report
with open('analysis_report.txt', 'w') as f:
    f.write('SALES ANALYSIS REPORT\n')
    f.write('=' * 70 + '\n\n')
    f.write(f'Dataset Size: {len(df)} rows × {len(df.columns)} columns\n')
    f.write(f'Date Range: {summary["date_range"]["start"]} to {summary["date_range"]["end"]}\n\n')
    f.write('Summary Statistics:\n')
    f.write('-' * 70 + '\n')
    f.write(df['sales'].describe().to_string())
    f.write('\n\n')
    f.write('Top 5 Products:\n')
    f.write('-' * 70 + '\n')
    f.write(top_products.head().to_string())

print("\n✅ Analysis complete! Generated 4 files:")
print("1. summary_statistics.json - Detailed statistics")
print("2. sales_analysis.png - Visualizations")
print("3. cleaned_sales_data.csv - Cleaned dataset")
print("4. analysis_report.txt - Full text report")
```

### **What the User Receives:**

```
✅ Execution succeeded!

Dataset: 365 rows × 5 columns
Columns: date, product, category, sales, quantity
[... output ...]

✅ Analysis complete! Generated 4 files:
1. summary_statistics.json - Detailed statistics
2. sales_analysis.png - Visualizations
3. cleaned_sales_data.csv - Cleaned dataset
4. analysis_report.txt - Full text report

📎 Generated 4 file(s):
• summary_statistics.json (structured, 2.1 KB)
• sales_analysis.png (image, 145.2 KB)
• cleaned_sales_data.csv (data, 45.6 KB)
• analysis_report.txt (text, 3.2 KB)

[4 downloadable file attachments in Discord]

⏱️ Executed in 3.45s
📦 Auto-installed: seaborn
```

---

## 🚫 **Common Model Mistakes**

### **Mistake #1: Checking Package Availability**

❌ **DON'T:**
```python
import sys
if 'seaborn' not in sys.modules:
    print("Seaborn is not installed")
```

✅ **DO:**
```python
import seaborn as sns  # Just import it!
```

### **Mistake #2: Using install_packages Parameter**

❌ **DON'T:**
```json
{
  "code": "import pandas as pd",
  "install_packages": ["pandas"]  // Unnecessary!
}
```

✅ **DO:**
```json
{
  "code": "import pandas as pd"  // That's it!
}
```

### **Mistake #3: Printing Instead of Saving**

❌ **DON'T:**
```python
print(df.to_string())  // Output gets truncated!
```

✅ **DO:**
```python
df.to_csv('data.csv')  // User gets full data!
```

### **Mistake #4: Not Using load_file()**

❌ **DON'T:**
```python
df = pd.read_csv('/path/to/file.csv')  // Won't work!
```

✅ **DO:**
```python
df = load_file('file_id_from_user')  // Correct!
```

---

## ✅ **Checklist for Model Developers**

When updating the model's behavior:

- [ ] Model knows packages auto-install (no manual checks)
- [ ] Model uses `load_file()` for user uploads
- [ ] Model creates files instead of printing long output
- [ ] Model uses descriptive filenames
- [ ] Model handles errors gracefully
- [ ] Model generates multiple output types when useful
- [ ] Tool description emphasizes auto-install feature
- [ ] System prompt includes code interpreter capabilities
- [ ] Examples show correct usage patterns

---

## 📚 **Related Documentation**

- **GENERATED_FILES_GUIDE.md** - Complete file handling guide
- **CODE_INTERPRETER_GUIDE.md** - Technical implementation details
- **NEW_FEATURES_GUIDE.md** - All new features overview
- **code_interpreter_prompts.py** - System prompt definitions

---

## 🎉 **Summary**

**Key Message to the Model:**

> "Just write Python code normally. Import any approved package - it auto-installs if missing. Create files (CSV, images, reports) - they're automatically sent to users. Use `load_file('file_id')` to access user uploads. That's it!"

**What the Model Should Remember:**

1. ✅ **Auto-install is automatic** - just import packages
2. ✅ **All files are captured** - create files, don't print
3. ✅ **Use load_file()** - for user uploads
4. ✅ **Be descriptive** - good filenames help users
5. ✅ **Handle errors** - gracefully inform users

The system handles everything else automatically! 🚀
