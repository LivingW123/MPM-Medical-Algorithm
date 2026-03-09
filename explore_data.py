import pandas as pd
import os
import sys

def process_excel(file_path, output_dir):
    try:
        excel_name = os.path.basename(file_path)
        print(f"--- Processing {excel_name} ---")
        xls = pd.ExcelFile(file_path)
        print(f"Sheets: {xls.sheet_names}")
        
        for sheet in xls.sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet)
                print(f"\nSheet: {sheet}")
                print(f"Shape: {df.shape}")
                print(f"Columns: {df.columns.tolist() if len(df.columns) < 20 else str(df.columns.tolist()[:10]) + '... [' + str(len(df.columns)) + ' total]'}")
                print(f"Sample data:")
                print(df.head(2).to_string())
                
                # Save to CSV
                safe_sheet = sheet.replace('/', '_').replace('\\', '_').replace(' ', '_')
                safe_name = excel_name.replace('.xlsx', '').replace('.xlsm', '')
                csv_path = os.path.join(output_dir, f"{safe_name}_{safe_sheet}.csv")
                df.to_csv(csv_path, index=False)
                print(f"Saved to {csv_path}")
            except Exception as e:
                print(f"Error processing sheet {sheet}: {e}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

if __name__ == '__main__':
    data_dir = r"c:\Users\oneal\VS Code Stuff\MPM-Medical-Algorithm\Manufacturing Data"
    output_dir = r"c:\Users\oneal\VS Code Stuff\MPM-Medical-Algorithm\Manufacturing Data CSVs"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for f in os.listdir(data_dir):
        if f.endswith('.xlsx') or f.endswith('.xlsm'):
            process_excel(os.path.join(data_dir, f), output_dir)
