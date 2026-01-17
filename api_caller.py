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

Your goal is to provide a standardized, detailed description for every hardware part after reading its datasheet specs. This will be the foundation
of our database and allow for AI analysis/automation. Read the .txt files provided and extract every possible technical or form/fit/function specification you can find. 
The purpose of this is to create a standardized description and JSON format for every single part in our company's ERP database, with every technical spec from the datasheet
that will allow AI to analyze our the parts we select and determine if they are compatible with each other as part of a military grade rugged server build.

First begin with returning the JSON format.

Return a JSON with these exact fields, including all JSON specs in 1 csv cell to keep it compact. If the datasheet does not contain a value
for a certain key, list 'N/A' as the value:
In order to keep the JSON format standardized, while also accounting for different types of hardware, we will define a Hybrid JSON Strategy. This approach uses four consistent top-level blocks while allowing the core_specs to adapt based on the part type.
The Hybrid JSON Standard:
Objective: Extract hardware data into a structured JSON object that supports both universal system tracking (physical/power) and specific technical performance (core).
1. The Universal Four-Block Structure: Every JSON object must contain these four keys:
•	physical_specs: Dimensions, weight, and mounting details.
•	core_specs: Dynamic technical attributes specific to the part type (CPU, RAM, Cable, etc.).
•	extended_specs: Compliance, warranty, and an AI-generated technical summary.
•	power_specs: Detailed electrical requirements for system-wide power budgeting.

The Master Hybrid JSON Structure to be used for API calls that analyze our quotes and warn us of potential compatibility problems, fitment issues, provide recommendations, etc. 
This top-level schema remains identical for every part, ensuring our database queries are predictable.
JSON
{
  "physical_specs": {
    "dimensions": "L x W x H (units)",
    "weight": "Value (units)",
    "material": "e.g., Aluminum, Stainless Steel",
    "mounting": "e.g., VESA 75, M3 Screws, Tool-less"
  },
  "core_specs": {
    "//": "This section varies dynamically based on the Part Type."
  },
  "extended_specs": {
    "compliance": ["RoHS", "TAA", "MIL-STD-810H"],
    "warranty": "e.g., 3-Year Limited",
    "datasheet_summary": "Short AI-generated technical summary of the part."
  },
  "power_specs": {
    "voltage_range": "e.g., 100-240V AC or 12V DC",
    "max_power_draw": "Wattage Value (e.g., 270W)",
    "idle_power_draw": "Wattage Value",
    "connector_type": "e.g., 8-pin EPS, 12VHPWR"
  }
}

How core_specs varies by Part Type
While the other blocks stay the same, the core_specs block adapts to the specific technology of the part. Here are the most critical version:
1. CPU Core Specs
JSON
"core_specs": {
  "model": "Xeon Gold 6430",
  "cores_threads": "32C/64T",
  "base_clock": "2.1GHz",
  "socket": "LGA 4677",
  "tdp_thermal": "270W"
}
2. Storage Core Specs
JSON
"core_specs": {
  "form_factor": "U.2 (15mm)",
  "capacity_tb": 7.68,
  "interface": "NVMe PCIe Gen4 x4",
  "sequential_read_mbps": 7000,
  "endurance_dwpd": 1.0
}
3. Cable Core Specs
JSON
"core_specs": {
  "connector_a": "SlimSAS 8i",
  "connector_b": "2x U.2 SFF-8639",
  "length_mm": 500,
  "front_device": "RAID Controller",
  "back_device": "Drive Backplane",
  "speed_rating": "PCIe Gen5 (32Gbps/lane)"
}
4. Adapter Core Specs
JSON
"core_specs": {
  "input_connector": "USB 3.0 19-pin",
  "output_connector": "USB 2.0 9-pin",
  "gender": "Male to Female",
  "logic_type": "Passive",
  "max_bandwidth": "480Mbps"
}


Example:
{
  "physical_specs": {
    "dimensions": "167mm x 69mm",
    "weight": "250g",
    "material": "PCB / Aluminum Heatsink",
    "mounting": "HHHL (Half-Height, Half-Length) PCIe Slot"
  },
  "core_specs": {
    "bus_interface": "PCIe Gen6 x16",
    "ports": 2,
    "speed_per_port": "400GbE",
    "connector_type": "QSFP112",
    "controller": "ConnectX-8 C8240",
    "offloading_features": ["RDMA", "RoCE v2", "GPUDirect"]
  },
  "extended_specs": {
    "compliance": ["RoHS", "TAA", "CE"],
    "warranty": "3-Year Limited",
    "datasheet_summary": "High-performance dual-port 400GbE NIC designed for AI and hyperscale workloads, featuring PCIe Gen6 support."
  },
  "power_specs": {
    "voltage_range": "12V DC",
    "max_power_draw": "75W Max",
    "idle_power_draw": "18W",
    "connector_type": "PCIe Slot Power"
  }
}

For descriptions use these formats. If a specific data point is not found, put 'N/A':

0.	Motherboards:
•	Format: MB | [Form Factor / Socket] | [Manufacturer] | [Model] | [Chipset / Platform] | [Key I/O & Expansion] | [Max Power]
•	Example: MB | ATX / LGA4677 | Advantech | ASMB-817T2-00A1 | Intel C741 | Gen5 PCIe, 8x SATA3, M.2 NVMe, Dual 10GbE | 75W Max

1.	Compute & Memory
•	CPU
o	Format: CPU | [Model] | [Cores/Threads] | [Clock] | [Socket] | [Max Power]
o	Example: CPU | Xeon Gold 6430 | 32C/64T | 2.1GHz | LGA 4677 | 350W Max
•	MEMORY (RAM)
o	Format: RAM | [Capacity] | [Speed] | [Rank] | [Type/ECC] | [Manufacturer] | [Voltage] | [Max Power]
o	Example: RAM | 64GB | DDR5-4800 | 2Rx8 | ECC RDIMM | Samsung | 1.1V | 6W Max
2. Storage & RAID
•	STORAGE (SSD/HDD)
o	Format: SSD/HDD | [Model] |  [Form Factor] | [Manufacturer] | [Capacity] | [Interface] | [Max Power]
o	Example: SSD | U.2 (15mm) | Samsung | PM1733 | 7.68TB | NVMe Gen4 | 22W Max
•	RAID Controller
o	Format: RAID | [Profile] | [Manufacturer] | [Model] | [Cache] | [Interface] | [Max Power]
o	Example: RAID | Low-Profile | Broadcom | 9560-8i | 4GB | PCIe Gen4 | 15W Max
•	Drive CARRIER
o	Format: CARRIER | [Size/Height] | [Manufacturer] | [Model] | [Bay Count] | [Interface] | [0W]
•	RAID CABLES
o	Format: CABLE | [Conn A-B] | [Length] | [Manufacturer] | [Model] | [Protocol] | [0W]
3. Foundation & Mechanical
•	CHASSIS
o	Format: CHASSIS | [U/Depth] | [Manufacturer] | [Backplane] | [Fans] | [Material] | [Fan Power]
o	Example: CHASSIS | 2U/20" | Trenton | 5-Slot Gen5 | 3x 80mm | Aluminum | 45W Max
•	POWER SUPPLY (PSU)
o	Format: PSU | [Wattage] | [Efficiency] | [Form Factor] | [Voltage] | [Type] | [Input Amps]
o	Example: PSU | 1200W | Platinum | Slimline | 110/220V | Redundant | 12A @ 115V
•	SLIDE RAILS
o	Format: RAILS | [Travel] | [Load] | [Manufacturer] | [Compatibility] | [Type] | [0W]
4. Expansion Cards
•	NIC (Networking)
o	Format: NIC | [Interface] | [Manufacturer] | [Model] | [Speed/Ports] | [Connector] | [Max Power]
o	Example: NIC | PCIe Gen6 x16 | NVIDIA | ConnectX-8 | Dual 400GbE | QSFP112 | 75W Max
•	GPU / Accelerators
o	Format: GPU | [Width/TDP] | [Manufacturer] | [Model] | [VRAM] | [Architecture] | [Max Power]
o	Example: GPU | Dual-Slot/450W | NVIDIA | RTX 6000 | 48GB | Ada | 450W Max	
•	HBA
o	HBA | [Form Factor] | [Manufacturer] | [Model] | [Protocol / Ports] | [Interface] | [Max Power]
o	Example: HBA | FHHL | Inateck | KU8211-R-US | USB 3.2 Gen2 (8-Port) | PCIe Gen3 x4 | 25W Max
•	RISER CARDS
o	Format: RISER | [Elec/Mech] | [Manufacturer] | [Model] | [Orientation] | [Slots] | [Max Power]
o	Example: RISER | x16 to 2x8 | Trenton | RSR-2000 | Right-Angle | 2-Slot | 5W Max
5. Software & Autonomous
•	OPERATING SYSTEM (OS)
o	Format: OS | [Family/Version] | [Edition] | [Licensing] | [Cores] | [Arch] | [0W]
6. Other (for parts that do not fit the classification types above)
o	Format: [Type] | [Manufacturer] | [Model] | [Interface / Connector] | [Platform / Compatibility] | [Max Power]
o	Example 1: TPM | Infineon | SLB9670 | SPI Header | Intel Xeon Platforms | 0.5W Max
o	Example 2: FAN | Delta | PFR0912XHE | 4-Pin PWM | 1U–2U Server Chassis | 18W Max


TEXT TO ANALYZE:
{text[:12000]}

"""

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