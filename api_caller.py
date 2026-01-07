"""
CSV-Based Parts Processor for MySQL Import
Stores all results in a single CSV for easy database import
"""
# the descriptions of this are good, but the json format and csv file are not. need to simplify the csv files and improve the json format

import os
import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from openai import OpenAI
import pandas as pd


class CSVPartsProcessor:
    def __init__(self, base_path: str, output_dir: str = "parts_output"):
        """Initialize processor with CSV output"""
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.output_dir / f"parts_data_{timestamp}.csv"
        self.progress_file = self.output_dir / "progress.json"

        # Initialize OpenAI
        self.client = OpenAI()

        # Load or create progress tracking
        self.load_progress()

        # Initialize CSV if new
        if not self.csv_file.exists():
            self.init_csv()

    def init_csv(self):
        """Initialize CSV with headers"""
        headers = [
            'part_number',
            'category',
            'description',
            'manufacturer',
            'model',
            'json_specs',  # Full JSON as string
            # Flattened key specs for easier querying
            'interface',
            'speed',
            'ports',
            'power_max',
            'form_factor',
            'processing_date',
            'source_files',
            'extraction_status'
        ]

        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

        print(f"📄 Created CSV: {self.csv_file}")

    def load_progress(self):
        """Load processing progress"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "processed": [],
                "failed": [],
                "stats": {"total": 0, "completed": 0, "failed": 0, "tokens": 0}
            }

    def save_progress(self):
        """Save progress after each part"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def create_extraction_prompt(self, part_number: str, text: str, filename: str = "") -> str:
        """Create extraction prompt"""
        # Extract hints from filename
        model_hint = ""
        if filename:
            # Look for model patterns
            patterns = [
                r'(MCX\d+\w+)',  # NVIDIA ConnectX
                r'(MCP\d+\w+)',  # NVIDIA Cables
                r'([A-Z]{2,}\d+\w+)',  # General pattern
            ]
            for pattern in patterns:
                match = re.search(pattern, filename, re.I)
                if match:
                    model_hint = f"\nLIKELY MODEL NUMBER: {match.group(1)}"
                    break

        prompt = f"""Extract hardware specifications from this datasheet.
Part: {part_number}
Filename: {filename}{model_hint}

BE AGGRESSIVE - extract every specification you can find. Look in:
- Tables listing models and specs
- Bullet points with features
- Technical specifications sections
- Compatibility lists
- Even footnotes and headers

Return a JSON with these exact fields:
{{
    "category": "[CPU/RAM/NIC/STORAGE/GPU/PSU/CABLE/etc.]",
    "part_number": "{part_number}",
    "description": "[Create using appropriate template format]",
    "manufacturer": "[Company name]",
    "model": "[Model number - CHECK FILENAME]",
    "interface": "[PCIe/SATA/etc. with version]",
    "speed": "[Primary speed rating]",
    "ports": "[Number of ports if applicable]",
    "power_max": "[Maximum power in watts]",
    "form_factor": "[Physical form factor]",
    "json_specs": {{
        "physical_specs": {{}},
        "core_specs": {{}},
        "extended_specs": {{}},
        "power_specs": {{}}
    }}
}}

For descriptions use these formats:
- NIC: "NIC | [interface] | [manufacturer] | [model] | [speed_ports] | [connector] | [power]"
- CPU: "CPU | [model] | [cores/threads] | [clock] | [socket] | [power]"
- RAM: "RAM | [capacity] | [speed] | [rank] | [type] | [voltage] | [power]"
- STORAGE: "STORAGE | [form] | [manufacturer] | [model] | [capacity] | [interface] | [power]"

TEXT TO ANALYZE:
{text[:12000]}

Return ONLY the JSON object."""

        return prompt

    def process_part(self, folder_path: Path) -> Dict:
        """Process a single part folder"""
        part_number = folder_path.name

        # Skip if already processed
        if part_number in self.progress["processed"]:
            return {"skipped": True}

        print(f"\n📁 Processing: {part_number}")

        # Read text files
        txt_files = list(folder_path.glob("*.txt"))
        if not txt_files:
            self.mark_failed(part_number, "No text files")
            return {"success": False}

        # Combine text files
        combined_text = ""
        source_files = []

        for txt_file in txt_files[:3]:  # Max 3 files
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if content:
                        combined_text += f"\n=== {txt_file.name} ===\n{content}\n"
                        source_files.append(txt_file.name)
            except:
                continue

        if not combined_text:
            self.mark_failed(part_number, "Empty text files")
            return {"success": False}

        # Call OpenAI API
        try:
            prompt = self.create_extraction_prompt(part_number, combined_text,
                                                   source_files[0] if source_files else "")

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "Extract detailed hardware specifications."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            # Parse response
            data = json.loads(response.choices[0].message.content)

            # Write to CSV
            self.write_to_csv(data, source_files)

            # Update progress
            self.progress["processed"].append(part_number)
            self.progress["stats"]["completed"] += 1
            if hasattr(response, 'usage'):
                self.progress["stats"]["tokens"] += response.usage.total_tokens
            self.save_progress()

            print(f"  ✅ Success")
            return {"success": True, "data": data}

        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.mark_failed(part_number, str(e))
            return {"success": False, "error": str(e)}

    def write_to_csv(self, data: Dict, source_files: List[str]):
        """Append result to CSV"""
        row = {
            'part_number': data.get('part_number', ''),
            'category': data.get('category', ''),
            'description': data.get('description', ''),
            'manufacturer': data.get('manufacturer', ''),
            'model': data.get('model', ''),
            'json_specs': json.dumps(data.get('json_specs', {})),
            'interface': data.get('interface', ''),
            'speed': data.get('speed', ''),
            'ports': data.get('ports', ''),
            'power_max': data.get('power_max', ''),
            'form_factor': data.get('form_factor', ''),
            'processing_date': datetime.now().isoformat(),
            'source_files': '|'.join(source_files),
            'extraction_status': 'success'
        }

        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            headers = list(row.keys())
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(row)

    def mark_failed(self, part_number: str, error: str):
        """Mark a part as failed and write to CSV"""
        row = {
            'part_number': part_number,
            'category': '',
            'description': f'ERROR: {error}',
            'manufacturer': '',
            'model': '',
            'json_specs': '{}',
            'interface': '',
            'speed': '',
            'ports': '',
            'power_max': '',
            'form_factor': '',
            'processing_date': datetime.now().isoformat(),
            'source_files': '',
            'extraction_status': 'failed'
        }

        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            headers = list(row.keys())
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(row)

        self.progress["failed"].append(part_number)
        self.progress["stats"]["failed"] += 1
        self.save_progress()

    def run_batch(self, limit: Optional[int] = None):
        """Process parts in batch"""
        all_folders = sorted([d for d in self.base_path.iterdir() if d.is_dir()])

        if limit:
            all_folders = all_folders[:limit]

        total = len(all_folders)
        print(f"\n{'=' * 60}")
        print(f"Processing {total} parts")
        print(f"Output CSV: {self.csv_file}")
        print(f"{'=' * 60}")

        for i, folder in enumerate(all_folders, 1):
            print(f"\n[{i}/{total}]", end="")
            self.process_part(folder)

        self.print_summary()
        self.create_mysql_import_script()

    def print_summary(self):
        """Print processing summary"""
        stats = self.progress["stats"]
        print(f"\n{'=' * 60}")
        print("COMPLETE")
        print(f"✅ Successful: {stats['completed']}")
        print(f"❌ Failed: {stats['failed']}")
        print(f"📊 Total tokens: {stats['tokens']:,}")
        print(f"💾 Results in: {self.csv_file}")

    def create_mysql_import_script(self):
        """Create Python script for MySQL import"""
        script = f'''"""
Import parts data from CSV to MySQL
Auto-generated script
"""
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine

# Load CSV
df = pd.read_csv("{self.csv_file.name}")

# MySQL connection
engine = create_engine(
    "mysql+mysqlconnector://username:password@localhost/database",
    echo=True
)

# Create table and import
df.to_sql(
    name='parts',
    con=engine,
    if_exists='append',
    index=False,
    chunksize=100
)

print(f"Imported {{len(df)}} parts to MySQL")
'''

        script_file = self.output_dir / "mysql_import.py"
        with open(script_file, 'w') as f:
            f.write(script)

        print(f"📝 Created MySQL import script: {script_file}")


def main():
    """Main entry point"""
    print("CSV Parts Processor for MySQL")
    print("-" * 50)

    # Configuration
    BASE_PATH = r"C:\Users\jlundstedt\Desktop\Configurator_Project_Master\PDF_Scanner\Outputs\Test_Run_Txt_Processor"  # Update
    OUTPUT_DIR = r"C:\Users\jlundstedt\Desktop\Configurator_Project_Master\PDF_Scanner\Outputs\Test_Run_Txt_Processor_Results_3"  # Update

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        return

    processor = CSVPartsProcessor(BASE_PATH, OUTPUT_DIR)

    # Test with few parts
    processor.run_batch(limit=5)

    # For full processing:
    # processor.run_batch()


if __name__ == "__main__":
    main()