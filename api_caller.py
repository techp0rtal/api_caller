"""
Enhanced Parts Processor with GSS Data Enrichment
Combines ERP descriptions with datasheet analysis for better accuracy
"""

import os
import csv
import json
import re
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI

class EnhancedPartsProcessor:
    def __init__(self, base_path: str, output_dir: str = "parts_output", gss_csv_path: str = None):
        """Initialize processor with GSS data enrichment"""
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = self.output_dir / f"parts_data_{timestamp}.csv"
        self.progress_file = self.output_dir / "progress.json"
        
        # Initialize OpenAI
        self.client = OpenAI()
        
        # Load GSS baseline data
        self.gss_data = {}
        if gss_csv_path and os.path.exists(gss_csv_path):
            self.load_gss_data(gss_csv_path)
            print(f"📊 Loaded {len(self.gss_data)} GSS parts from baseline")
        else:
            print("⚠️  No GSS baseline file provided - processing without ERP enrichment")
        
        # Load progress
        self.load_progress()
        
        # Initialize CSV
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['part_number', 'description', 'full_specs'])
            print(f"📄 Created CSV: {self.csv_file}")
    
    def load_gss_data(self, gss_csv_path: str):
        """Load GSS baseline data from CSV file"""
        try:
            # Load the CSV file
            df = pd.read_csv(gss_csv_path, dtype=str)
            
            # Ensure required columns exist
            required_cols = ['PART', 'GSS_Description', 'MPN']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Missing columns in GSS CSV: {missing_cols}")
                return
            
            # Create lookup dictionary indexed by base part number (without revision)
            for _, row in df.iterrows():
                full_part = str(row['PART']).strip()
                gss_desc = str(row['GSS_Description']).strip() if pd.notna(row['GSS_Description']) else "N/A"
                mpn = str(row['MPN']).strip() if pd.notna(row['MPN']) else "N/A"
                
                # Extract base part number (everything before first space)
                base_part = full_part.split()[0] if ' ' in full_part else full_part
                
                # Store data (first match wins, as requested)
                if base_part not in self.gss_data:
                    self.gss_data[base_part] = {
                        'full_part_number': full_part,  # With revision
                        'gss_description': gss_desc,
                        'mpn': mpn,
                        'base_part': base_part  # Without revision
                    }
            
        except Exception as e:
            print(f"❌ Error loading GSS data: {e}")
            self.gss_data = {}
    
    def get_gss_info(self, folder_part_number: str) -> Dict:
        """
        Get GSS information for a part number
        
        Args:
            folder_part_number: Part number from folder name (e.g., "264-2675")
            
        Returns:
            Dict with GSS info or default values
        """
        if folder_part_number in self.gss_data:
            return self.gss_data[folder_part_number]
        else:
            return {
                'full_part_number': folder_part_number,
                'gss_description': "N/A",
                'mpn': "N/A",
                'base_part': folder_part_number
            }
    
    def determine_category_from_gss(self, gss_description: str) -> str:
        """
        Determine hardware category from GSS description
        This helps avoid misclassification when datasheet text is minimal
        """
        desc_lower = gss_description.lower()
        
        # Map GSS keywords to categories
        category_patterns = {
            "NIC": ['network', 'ethernet', 'nic', 'connectx', 'mellanox', 'infiniband'],
            "CPU": ['processor', 'cpu', 'xeon', 'epyc', 'intel', 'amd'],
            "RAM": ['memory', 'ddr', 'dimm', 'ram', 'sodimm'],
            "STORAGE": ['ssd', 'hdd', 'drive', 'storage', 'disk', 'nvme', 'sata'],
            "GPU": ['graphics', 'gpu', 'video', 'geforce', 'quadro', 'rtx'],
            "PSU": ['power supply', 'psu', 'power unit'],
            "CHASSIS": ['chassis', 'enclosure', 'case'],
            "MOTHERBOARD": ['motherboard', 'mainboard', 'system board'],
            "CABLE": ['cable', 'cord', 'wire', 'harness'],
            "CARRIER": ['carrier', 'caddy', 'tray', 'bracket'],
            "RISER": ['riser', 'adapter card', 'extension'],
            "BOARD": ['board', 'pcb', 'front panel', 'backplane', 'controller'],
            "FAN": ['fan', 'blower', 'cooling'],
            "RAILS": ['rails', 'slide', 'mounting'],
            "OTHER": []  # Fallback
        }
        
        # Check each category
        for category, keywords in category_patterns.items():
            if any(keyword in desc_lower for keyword in keywords):
                return category
        
        return "OTHER"
    
    def create_enriched_prompt(self, folder_part: str, gss_info: Dict, datasheet_text: str, filename: str = "") -> str:
        """
        Create enriched prompt combining GSS data and datasheet analysis
        """
        full_part_number = gss_info['full_part_number']
        gss_description = gss_info['gss_description']
        mpn = gss_info['mpn']
        
        # Determine category from GSS description
        suggested_category = self.determine_category_from_gss(gss_description)
        
        prompt = f"""I am providing two sources for Part Number {full_part_number}.

1. ERP DESCRIPTION: {gss_description}
2. MANUFACTURER PART NUMBER (MPN): {mpn}
3. DATASHEET TEXT: [See below]

CRITICAL INSTRUCTIONS:
- Use the ERP Description to identify the Category and primary function (e.g., if GSS says 'Board, Front Panel', do NOT classify it as SSD/HDD)
- Suggested Category from ERP: {suggested_category}
- Use the ERP Description as the foundation for classification and basic info
- Use the Datasheet Text to fill in technical details like dimensions, connectors, speeds, and power specs
- If the Datasheet has minimal text (common with drawings), rely heavily on the ERP Description
- If there's conflicting info, prioritize ERP Description for category/function, datasheet for technical specs

DESCRIPTION TEMPLATES BY CATEGORY:
- NIC: "NIC | [Interface] | [Manufacturer] | [Model] | [Speed/Ports] | [Connector] | [Max Power]"
- CPU: "CPU | [Model] | [Cores/Threads] | [Clock] | [Socket] | [Max Power]"
- SSD/HDD: "SSD/HDD | [Form Factor] | [Manufacturer] | [Model] | [Capacity] | [Interface] | [Max Power]"
- RAM: "RAM | [Capacity] | [Speed] | [Rank] | [Type/ECC] | [Voltage] | [Max Power]"
- BOARD: "BOARD | [Type] | [Manufacturer] | [Model] | [Function] | [Interface] | [Max Power]"
- OTHER: "[Type] | [Manufacturer] | [Model] | [Function/Description] | [Interface] | [Max Power]"

EXAMPLE FOR FRONT PANEL BOARD:
- ERP: "Board, Front panel, x6 LED, UI"
- Description: "BOARD | Front Panel | SuperMicro | FPB-826-T | 6x LED, UID Button | Internal Connector | 5W Max"
- Category: "BOARD" (NOT "SSD/HDD")

REQUIRED JSON STRUCTURE:"""

        # JSON template (separate string to avoid formatting issues)
        json_template = '''{
  "part_number": "''' + full_part_number + '''",
  "description": "[Use template above based on ERP category]",
  "full_specs": {
    "physical_specs": {
      "dimensions": "[Extract from datasheet or estimate from category]",
      "weight": "[Extract from datasheet or estimate]",
      "material": "[PCB, aluminum, etc. - infer from ERP/datasheet]",
      "mounting": "[How it mounts - infer from category/function]"
    },
    "core_specs": {
      "category": "''' + suggested_category + '''",
      "manufacturer": "[Extract from ERP/datasheet]",
      "model": "[Use MPN if available: ''' + mpn + ''']",
      "function": "[Primary function from ERP description]",
      "interface": "[Connection type from datasheet]",
      "compatibility": "[What systems/chassis it works with]",
      "features": ["List key features from ERP + datasheet"]
    },
    "extended_specs": {
      "compliance": ["Extract certifications or use common ones"],
      "warranty": "[Standard warranty or N/A]",
      "datasheet_summary": "[Combine ERP function + any datasheet technical details]"
    },
    "power_specs": {
      "voltage_range": "[Extract from datasheet or estimate]",
      "max_power_draw": "[Extract or estimate based on category]",
      "idle_power_draw": "[Estimate if not in datasheet]",
      "connector_type": "[Power connection type]"
    }
  }
}'''

        # Add datasheet text section
        datasheet_section = f"""

DATASHEET TEXT TO ANALYZE:
{datasheet_text[:10000]}

Return ONLY the JSON object above, combining ERP foundation with datasheet technical details."""

        return prompt + json_template + datasheet_section

    def load_progress(self):
        """Load or initialize progress tracking"""
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

    def process_part(self, folder_path: Path) -> Dict:
        """Process a single part folder with GSS enrichment"""
        folder_part_number = folder_path.name

        # Skip if already processed
        if folder_part_number in self.progress["processed"]:
            return {"skipped": True}

        print(f"\n📁 Processing: {folder_part_number}")

        # Get GSS information
        gss_info = self.get_gss_info(folder_part_number)
        full_part_number = gss_info['full_part_number']

        print(f"  📋 GSS Info: {gss_info['gss_description'][:60]}...")
        print(f"  🏷️  Full Part: {full_part_number}")
        print(f"  🔧 MPN: {gss_info['mpn']}")

        # Find and read text files
        txt_files = list(folder_path.glob("*.txt"))
        if not txt_files:
            # Even without datasheet, we can still process using GSS info
            print("  ⚠️  No text files - using GSS data only")
            datasheet_text = "No datasheet text available - minimal PDF content or extraction failed."
        else:
            # Combine text content
            combined_text = ""
            for txt_file in txt_files[:2]:
                try:
                    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                        if content:
                            combined_text += f"\n=== {txt_file.name} ===\n{content}\n"
                except Exception as e:
                    print(f"    Error reading {txt_file.name}: {e}")
                    continue

            datasheet_text = combined_text if combined_text.strip() else "No readable datasheet text found."

        # Determine category from GSS
        category = self.determine_category_from_gss(gss_info['gss_description'])
        print(f"  🏷️  Category from GSS: {category}")

        # Create enriched prompt
        prompt = self.create_enriched_prompt(
            folder_part_number,
            gss_info,
            datasheet_text,
            txt_files[0].name if txt_files else ""
        )

        # Call OpenAI API
        try:
            print(f"  🤖 Calling OpenAI API with enriched data...")
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a hardware specification expert. Combine ERP descriptions with datasheet analysis to create accurate, standardized parts data. Prioritize ERP category identification over datasheet guessing."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            # Parse response
            content = response.choices[0].message.content
            data = json.loads(content)

            # Write to CSV using the full part number (with revision)
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    full_part_number,  # Use full part number with revision
                    data.get('description', 'N/A'),
                    json.dumps(data.get('full_specs', {}))
                ])

            # Update progress
            self.progress["processed"].append(folder_part_number)
            self.progress["stats"]["completed"] += 1

            # Track tokens
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
                self.progress["stats"]["tokens"] += tokens_used
                print(f"  ✅ Success ({tokens_used} tokens)")
            else:
                print(f"  ✅ Success")

            self.save_progress()
            return {"success": True, "data": data}

        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parsing error: {e}")
            self.mark_failed(full_part_number, f"JSON parsing error: {e}")
            return {"success": False}

        except Exception as e:
            print(f"  ❌ API error: {e}")
            self.mark_failed(full_part_number, f"API error: {e}")
            return {"success": False}

    def mark_failed(self, part_number: str, error: str):
        """Mark a part as failed and add to CSV"""
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                part_number,
                f"ERROR: {error}",
                "{}"
            ])

        # Use base part number for progress tracking
        base_part = part_number.split()[0] if ' ' in part_number else part_number
        self.progress["failed"].append(base_part)
        self.progress["stats"]["failed"] += 1
        self.save_progress()

    def run_batch(self, limit: Optional[int] = None):
        """Process parts in batches with GSS enrichment"""
        all_folders = sorted([d for d in self.base_path.iterdir() if d.is_dir()])

        if limit:
            all_folders = all_folders[:limit]

        total = len(all_folders)
        self.progress["stats"]["total"] = total

        print(f"\n{'='*60}")
        print(f"PROCESSING {total} PARTS WITH GSS ENRICHMENT")
        print(f"GSS Data Available: {'YES' if self.gss_data else 'NO'}")
        print(f"Output CSV: {self.csv_file}")
        print(f"{'='*60}")

        for i, folder in enumerate(all_folders, 1):
            print(f"\n[{i}/{total}]", end="")
            self.process_part(folder)

            if i % 10 == 0:
                print(f"\n💾 Checkpoint saved at part {i}")

        self.print_summary()

    def print_summary(self):
        """Print processing summary"""
        stats = self.progress["stats"]
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE - ENRICHED WITH GSS DATA")
        print(f"{'='*60}")
        print(f"Total parts: {stats.get('total', 0)}")
        print(f"✅ Successful: {stats['completed']}")
        print(f"❌ Failed: {stats['failed']}")
        print(f"📊 Total tokens used: {stats['tokens']:,}")

        if stats['tokens'] > 0:
            cost_estimate = (stats['tokens'] / 1000) * 0.002
            print(f"💰 Estimated cost: ${cost_estimate:.2f}")

        print(f"\n📁 Results: {self.csv_file}")
        print(f"📋 GSS Parts Loaded: {len(self.gss_data)}")


def main():
    """Main entry point with GSS enrichment"""
    print("Enhanced Parts Processor with GSS Data Enrichment")
    print("-" * 60)

    # ============================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ============================================
    BASE_PATH = r"C:\Users\jlundstedt\Desktop\Configurator_Project_Master\PDF_Scanner\Outputs\Test_Run_Txt_Processor_2"
    OUTPUT_DIR = r"C:\Users\jlundstedt\Desktop\Configurator_Project_Master\PDF_Scanner\Outputs\Test_Run_Txt_Processor_Results_ENRICHED_2"
    GSS_CSV_PATH = r"C:\Users\jlundstedt\Desktop\Configurator_Project_Master\PARTS_VIEWER\Database_Files\Base_DB_for_AI_Prompt\trenton_gss_baseline.csv"
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Check if GSS file exists
    if not os.path.exists(GSS_CSV_PATH):
        print(f"⚠️  Warning: GSS baseline file not found at: {GSS_CSV_PATH}")
        print("Processing will continue without ERP enrichment")
        GSS_CSV_PATH = None
    
    try:
        # Create enhanced processor
        processor = EnhancedPartsProcessor(BASE_PATH, OUTPUT_DIR, GSS_CSV_PATH)
        
        # Test with 5 parts first
        print(f"\n🧪 TEST MODE: Processing 5 parts with GSS enrichment")
        processor.run_batch(limit=5)
        
        print(f"\n✅ Test complete! Check enriched results in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
