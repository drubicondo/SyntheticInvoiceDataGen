import os
import pandas as pd
import argparse
import glob
from utils import check_write_permission


def csv_to_xlsx_sheets(csv_files: list[str], output_xlsx_file: str):
    """
    Converts a list of CSV files into a single XLSX file, with each CSV
    file becoming a separate sheet.

    Args:
        csv_files (list[str]): A list of paths to the input CSV files.
        output_xlsx_file (str): The path for the output XLSX file.
    """
    if not check_write_permission(output_xlsx_file):
        print(f"Cannot proceed: Output file '{output_xlsx_file}' is not writable.")
        return

    try:
        with pd.ExcelWriter(output_xlsx_file, engine='xlsxwriter') as writer:
            print(f"Starting conversion to '{output_xlsx_file}'...")
            processed_count = 0
            for csv_file in csv_files:
                if not os.path.exists(csv_file):
                    print(f"Warning: CSV file '{csv_file}' not found. Skipping.")
                    continue
                if not os.path.isfile(csv_file):
                    print(f"Warning: Path '{csv_file}' is not a file. Skipping.")
                    continue
                if not os.access(csv_file, os.R_OK):
                    print(f"Warning: CSV file '{csv_file}' is not readable. Skipping.")
                    continue

                try:

                    df = pd.read_csv(csv_file)
                    sheet_name = os.path.splitext(os.path.basename(csv_file))[0]
                    # Excel sheet names have restrictions. Simple sanitization:
                    sheet_name = sheet_name.replace('/', '_').replace('\\', '_').replace('*', '_')
                    sheet_name = sheet_name.replace('?', '_').replace('[', '_').replace(']', '_')
                    sheet_name = sheet_name.replace(':', '_') # Colons are invalid in sheet names
                    sheet_name = sheet_name[:31] # Max 31 characters for sheet name

                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"Successfully added '{csv_file}' as sheet '{sheet_name}'.")
                    processed_count += 1

                except pd.errors.EmptyDataError:
                    print(f"Warning: CSV file '{csv_file}' is empty. Skipping.")
                except Exception as e:
                    print(f"Error processing '{csv_file}': {e}")

            if processed_count == 0:
                print("No valid CSV files were processed. The output XLSX file might be empty or not created.")
            else:
                print(f"Conversion complete! {processed_count} CSV files converted to '{output_xlsx_file}'.")

    except Exception as e:
        print(f"An error occurred during Excel file creation: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert multiple CSV files into a single XLSX file, "
                    "with each CSV as a separate sheet."
    )
    parser.add_argument(
        "csv_files",
        nargs="+", # Accepts one or more arguments
        help="List of CSV files to concatenate (e.g., 'file1.csv' 'data/*.csv'). "
             "Use quotes for wildcards."
    )
    parser.add_argument(
        "output_xlsx_file",
        help="Name of the output XLSX file (e.g., 'report.xlsx')."
    )

    args = parser.parse_args()

    expanded_csv_files = []
    for pattern in args.csv_files:
        expanded_csv_files.extend(glob.glob(pattern))
    expanded_csv_files = sorted(list(set(expanded_csv_files)))

    if not expanded_csv_files:
        print("Error: No CSV files found matching the provided patterns.")
    else:
        print(f"Found {len(expanded_csv_files)} CSV files to process:")
        for f in expanded_csv_files:
            print(f"- {f}")
        csv_to_xlsx_sheets(expanded_csv_files, args.output_xlsx_file)

