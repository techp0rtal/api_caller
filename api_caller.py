from openai import OpenAI

# Here was my first test of the API call, calling a haiku for fun
# client = OpenAI()
#
# response = client.responses.create(
#   model="gpt-5-nano",
#   input="write a haiku about the power of AI to change a small company",
#   store=True,
# )
#
# print(response.output_text);

# Now for the actual program code

# client = OpenAI() # This is basically saying to the AI “Look in the environment for the key.” to the api
#
# response = client.responses.create(
#   model="gpt-5-mini",
#   input="Read the text and createa a part description and json format containing the full specs of this hardware part.",
#   store=True,
# )
"""
Hardware Parts Text Processor with Specific Templates
Processes text files to generate standardized descriptions and JSON specs
Following Trenton's exact format requirements
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import OpenAI


class HardwarePartsProcessor:
  def __init__(self, base_path: str, api_key: str = None, output_dir: str = "api_results"):
    """
    Initialize the processor with exact templates

    Args:
        base_path: Root directory containing part folders with text files
        api_key: OpenAI API key (optional, will check environment if not provided)
        output_dir: Where to save results
    """
    self.base_path = Path(base_path)
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(exist_ok=True)

    # Initialize OpenAI client
    # First check if API key was provided, then check environment
    if api_key and api_key != "sk-...":
      self.client = OpenAI(api_key=api_key)
    else:
      # This will automatically look for OPENAI_API_KEY in environment
      self.client = OpenAI()

    # Statistics
    self.stats = {
      "total_folders": 0,
      "processed": 0,
      "failed": 0,
      "api_calls": 0,
      "errors": [],
      "empty_files": 0,
      "missing_files": 0
    }

    # Initialize templates
    self.setup_templates()
    self.setup_filters()

  def setup_templates(self):
    """Define all description templates and core_specs structures"""

    # Description templates by category
    self.description_templates = {
      "CPU": "CPU | {model} | {cores_threads} | {clock} | {socket} | {max_power}",
      "RAM": "RAM | {capacity} | {speed} | {rank} | {type_ecc} | {voltage} | {max_power}",
      "STORAGE": "STORAGE | {form_factor} | {manufacturer} | {model} | {capacity} | {interface} | {max_power}",
      "RAID": "RAID | {profile} | {manufacturer} | {model} | {cache} | {interface} | {max_power}",
      "CARRIER": "CARRIER | {size_height} | {manufacturer} | {model} | {bay_count} | {interface} | 0W",
      "CABLE": "CABLE | {conn_a_b} | {length} | {manufacturer} | {model} | {protocol} | 0W",
      "CHASSIS": "CHASSIS | {u_depth} | {manufacturer} | {backplane} | {fans} | {material} | {fan_power}",
      "PSU": "PSU | {wattage} | {efficiency} | {form_factor} | {voltage} | {type} | {input_amps}",
      "RAILS": "RAILS | {travel} | {load} | {manufacturer} | {compatibility} | {type} | 0W",
      "NIC": "NIC | {interface} | {manufacturer} | {model} | {speed_ports} | {connector} | {max_power}",
      "GPU": "GPU | {width_tdp} | {manufacturer} | {model} | {vram} | {architecture} | {max_power}",
      "RISER": "RISER | {elec_mech} | {manufacturer} | {model} | {orientation} | {slots} | {max_power}",
      "OS": "OS | {family_version} | {edition} | {licensing} | {cores} | {arch} | 0W",
      "AUTONOMOUS": "{type} | {interface} | {manufacturer} | {model} | {primary_spec} | {constraints} | {max_power}",
      "DEFAULT": "PART | {manufacturer} | {model} | {specifications} | {interface} | {max_power}"
    }

    # Core specs templates by category
    self.core_specs_templates = {
      "CPU": {
        "model": "N/A",
        "cores": "N/A",
        "threads": "N/A",
        "base_clock": "N/A",
        "turbo_clock": "N/A",
        "socket": "N/A",
        "tdp": "N/A",
        "cache_l3": "N/A",
        "memory_support": "N/A"
      },
      "RAM": {
        "capacity": "N/A",
        "speed": "N/A",
        "type": "N/A",
        "ecc": "N/A",
        "rank": "N/A",
        "cas_latency": "N/A",
        "voltage": "N/A",
        "operating_temp": "N/A"
      },
      "STORAGE": {
        "form_factor": "N/A",
        "capacity": "N/A",
        "interface": "N/A",
        "protocol": "N/A",
        "sequential_read": "N/A",
        "sequential_write": "N/A",
        "random_read_iops": "N/A",
        "random_write_iops": "N/A",
        "endurance_tbw": "N/A"
      },
      "RAID": {
        "controller_model": "N/A",
        "cache_size": "N/A",
        "raid_levels": ["N/A"],
        "max_drives": "N/A",
        "interface": "N/A",
        "data_transfer_rate": "N/A",
        "ports": "N/A"
      },
      "NIC": {
        "bus_interface": "N/A",
        "ports": "N/A",
        "speed_per_port": "N/A",
        "connector_type": "N/A",
        "controller": "N/A",
        "offloading_features": []
      },
      "GPU": {
        "model": "N/A",
        "architecture": "N/A",
        "vram": "N/A",
        "vram_type": "N/A",
        "cuda_cores": "N/A",
        "tensor_cores": "N/A",
        "base_clock": "N/A",
        "boost_clock": "N/A",
        "memory_bandwidth": "N/A"
      },
      "PSU": {
        "wattage": "N/A",
        "efficiency_rating": "N/A",
        "form_factor": "N/A",
        "input_voltage": "N/A",
        "output_rails": {},
        "modular": "N/A",
        "fan_size": "N/A"
      },
      "CHASSIS": {
        "form_factor": "N/A",
        "rack_units": "N/A",
        "depth": "N/A",
        "backplane": "N/A",
        "drive_bays": "N/A",
        "expansion_slots": "N/A",
        "cooling_fans": "N/A",
        "material": "N/A"
      },
      "CABLE": {
        "connector_a": "N/A",
        "connector_b": "N/A",
        "length": "N/A",
        "protocol": "N/A",
        "speed": "N/A",
        "awg": "N/A",
        "shielding": "N/A"
      },
      "RISER": {
        "electrical_config": "N/A",
        "mechanical_config": "N/A",
        "orientation": "N/A",
        "slot_count": "N/A",
        "pcie_generation": "N/A",
        "compatible_slots": []
      },
      "DEFAULT": {
        "category": "N/A",
        "manufacturer": "N/A",
        "model": "N/A",
        "specifications": {}
      }
    }

  def setup_filters(self):
    """Setup filtering rules for identifying relevant text files"""

    # Keywords that indicate datasheet content
    self.positive_keywords = [
      'specification', 'datasheet', 'technical', 'electrical',
      'mechanical', 'environmental', 'operating', 'characteristics',
      'features', 'description', 'parameter', 'rating', 'performance',
      'dimension', 'pinout', 'voltage', 'current', 'temperature',
      'frequency', 'power', 'memory', 'speed', 'capacity',
      'interface', 'connector', 'compatibility'
    ]

    # Keywords indicating non-datasheet content
    self.negative_keywords = [
      'invoice', 'purchase order', 'price list', 'quote',
      'shipping', 'tracking', 'email', 'correspondence',
      'meeting notes', 'calendar', 'schedule', 'agenda'
    ]

  def identify_category(self, text: str, filename: str = "") -> str:
    """
    Identify the hardware category from text content
    """
    text_lower = text.lower()
    filename_lower = filename.lower()

    # Category detection patterns
    category_patterns = {
      "CPU": [r'\bcpu\b', r'\bprocessor\b', r'\bxeon\b', r'\bepyc\b', r'\bcore.*i\d', r'\bryzen\b'],
      "RAM": [r'\bmemory\b', r'\bddr\d', r'\bdimm\b', r'\bram\b', r'\bsdram\b', r'\budimm\b', r'\brdimm\b'],
      "STORAGE": [r'\bssd\b', r'\bhdd\b', r'\bnvme\b', r'\bsata\b', r'\bm\.2\b', r'\bu\.2\b', r'\bsolid state\b'],
      "RAID": [r'\braid\b', r'\bhba\b', r'\bsas\b', r'\bperc\b', r'\bmegraid\b'],
      "NIC": [r'\bnetwork\b', r'\bethernet\b', r'\bnic\b', r'\bconnectx\b', r'\bmellanox\b', r'\bgbe\b'],
      "GPU": [r'\bgpu\b', r'\bgraphics\b', r'\bgeforce\b', r'\bquadro\b', r'\brtx\b', r'\bgtx\b', r'\brada\b'],
      "PSU": [r'\bpower supply\b', r'\bpsu\b', r'\bpower\s+unit\b', r'\bwatt\b'],
      "CHASSIS": [r'\bchassis\b', r'\benclosure\b', r'\bcase\b', r'\brack\b', r'\bbackplane\b'],
      "CABLE": [r'\bcable\b', r'\bconnector\b', r'\bwire\b', r'\bharness\b'],
      "RISER": [r'\briser\b', r'\badapter card\b', r'\bextension\b'],
      "CARRIER": [r'\bcarrier\b', r'\bdrive carrier\b', r'\bcaddy\b', r'\btray\b'],
      "RAILS": [r'\brails?\b', r'\bslide\b', r'\bmounting\b', r'\bracket\b'],
      "OS": [r'\boperating system\b', r'\bwindows\b', r'\blinux\b', r'\bubuntu\b', r'\brhel\b', r'\bvmware\b']
    }

    # Count pattern matches for each category
    category_scores = {}
    for category, patterns in category_patterns.items():
      score = 0
      for pattern in patterns:
        if re.search(pattern, text_lower):
          score += 1
        if re.search(pattern, filename_lower):
          score += 2  # Filename matches count more
      if score > 0:
        category_scores[category] = score

    # Return category with highest score
    if category_scores:
      return max(category_scores, key=category_scores.get)

    return "DEFAULT"

  def analyze_text_relevance(self, text: str, filename: str = "") -> Dict:
    """
    Determine if a text file is likely a datasheet
    """
    result = {
      "is_relevant": False,
      "confidence": 0.0,
      "reasons": []
    }

    # Check minimum length
    if len(text.strip()) < 100:
      result["reasons"].append("Too short")
      return result

    text_lower = text.lower()

    # Count keyword matches
    positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
    negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)

    # Calculate confidence
    if positive_count > negative_count and positive_count >= 3:
      result["is_relevant"] = True
      result["confidence"] = min(positive_count / 10, 1.0)
      result["reasons"].append(f"{positive_count} technical keywords found")
    elif negative_count > positive_count:
      result["reasons"].append(f"{negative_count} non-technical keywords found")

    return result

  def combine_text_files(self, folder_path: Path) -> Tuple[str, List[str], List[str], str]:
    """
    Combine multiple text files from a folder intelligently

    Returns:
        - Combined text
        - List of used files
        - List of filtered files
        - Status message (missing, empty, or ok)
    """
    txt_files = list(folder_path.glob("*.txt"))

    if not txt_files:
      self.stats["missing_files"] += 1
      return "", [], [], "NO_FILES"

    # Check for empty files
    file_data = []
    empty_files = []

    for txt_file in txt_files:
      try:
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
          content = f.read()

        # Check if file is empty or just whitespace
        if not content or not content.strip():
          empty_files.append(txt_file.name)
          self.stats["empty_files"] += 1
          continue

        analysis = self.analyze_text_relevance(content, txt_file.name)
        file_data.append({
          "file": txt_file,
          "content": content,
          "is_relevant": analysis["is_relevant"],
          "confidence": analysis["confidence"]
        })
      except Exception as e:
        print(f"    ⚠️  Error reading {txt_file.name}: {e}")
        continue

    # If all files were empty
    if not file_data and empty_files:
      return "", [], empty_files, "EMPTY_FILES"

    # If no valid content found
    if not file_data:
      return "", [], [], "NO_VALID_CONTENT"

    # Single file case
    if len(file_data) == 1:
      return file_data[0]["content"], [file_data[0]["file"].name], empty_files, "OK"

    # Multiple files - filter and combine
    # Sort by confidence
    file_data.sort(key=lambda x: x["confidence"], reverse=True)

    # Separate relevant and filtered files
    relevant_files = [f for f in file_data if f["is_relevant"]]
    filtered_files = [f["file"].name for f in file_data if not f["is_relevant"]]
    filtered_files.extend(empty_files)  # Add empty files to filtered list

    if not relevant_files:
      # Use the best match if nothing passes filter
      relevant_files = [file_data[0]] if file_data else []

    # Combine relevant files (max 3)
    combined_parts = []
    used_files = []

    for i, f in enumerate(relevant_files[:3]):
      used_files.append(f["file"].name)
      if i == 0:
        combined_parts.append(f"=== PRIMARY DATASHEET: {f['file'].name} ===\n{f['content']}")
      else:
        content = f['content']
        if len(content) > 5000:
          content = content[:5000] + "\n[truncated]"
        combined_parts.append(f"\n=== ADDITIONAL: {f['file'].name} ===\n{content}")

    combined_text = "\n\n".join(combined_parts)

    # Truncate if too long
    if len(combined_text) > 12000:
      combined_text = combined_text[:12000] + "\n[content truncated]"

    return combined_text, used_files, filtered_files, "OK"

  def create_api_prompt(self, part_number: str, text: str, category: str) -> str:
    """
    Create the prompt for OpenAI API with exact format requirements
    """

    # Get the appropriate templates
    desc_template = self.description_templates.get(category, self.description_templates["DEFAULT"])
    core_template = self.core_specs_templates.get(category, self.core_specs_templates["DEFAULT"])

    prompt = f"""You are analyzing a hardware datasheet for part number: {part_number}
Category detected: {category}

Your task is to extract information and create TWO outputs:

1. DESCRIPTION: Create a standardized description using this EXACT template:
{desc_template}

Fill in the template with actual values. If a value is not found, use "N/A".

2. JSON SPECS: Create a JSON object with this EXACT structure:
{{
  "category": "{category}",
  "part_number": "{part_number}",
  "description": "[The description string from above]",
  "physical_specs": {{
    "dimensions": "[L x W x H or N/A]",
    "weight": "[Value with unit or N/A]",
    "material": "[Material type or N/A]",
    "mounting": "[Mounting type or N/A]"
  }},
  "core_specs": {json.dumps(core_template, indent=2)},
  "extended_specs": {{
    "compliance": ["List of certifications or empty array"],
    "warranty": "[Warranty period or N/A]",
    "datasheet_summary": "[One sentence technical summary]"
  }},
  "power_specs": {{
    "voltage_range": "[Voltage or N/A]",
    "max_power_draw": "[Value with W Max or N/A]",
    "idle_power_draw": "[Value with W or N/A]",
    "connector_type": "[Power connector or N/A]"
  }}
}}

IMPORTANT RULES:
- Use "N/A" for any missing values (never leave empty or null)
- For power values, always include "W Max" suffix (e.g., "75W Max")
- For dimensions, use format "L x W x H" with units
- Compliance must be an array (empty [] if none found)
- Extract actual values from the datasheet text below

DATASHEET TEXT:
{text}

Return ONLY the JSON object, no additional text."""

    return prompt

  def call_openai_api(self, prompt: str, part_number: str) -> Dict:
    """
    Call OpenAI API using the specified format
    """
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
      try:
        # Using the format specified in requirements
        response = self.client.chat.completions.create(
          model="gpt-3.5-turbo",  # Use gpt-4-turbo for better accuracy
          messages=[
            {"role": "system",
             "content": "You are a technical documentation specialist extracting structured hardware specifications. Always return valid JSON only."},
            {"role": "user", "content": prompt}
          ],
          temperature=0.1,
          max_tokens=2000,
          response_format={"type": "json_object"}
        )

        # Extract and parse response
        content = response.choices[0].message.content
        parsed = json.loads(content)

        self.stats["api_calls"] += 1

        return {
          "success": True,
          "data": parsed,
          "tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0
        }

      except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error: {e}")
        if attempt < max_retries - 1:
          time.sleep(retry_delay)
          continue
        return {"success": False, "error": f"JSON parse error: {e}"}

      except Exception as e:
        print(f"  ❌ API error: {e}")
        if attempt < max_retries - 1:
          time.sleep(retry_delay)
          continue
        return {"success": False, "error": str(e)}

    return {"success": False, "error": "Max retries exceeded"}

  def process_part_folder(self, folder_path: Path) -> Dict:
    """
    Process a single part folder
    """
    part_number = folder_path.name
    print(f"\n📁 Processing: {part_number}")

    # Combine text files
    combined_text, used_files, filtered_files, status = self.combine_text_files(folder_path)

    # Handle different statuses
    if status == "NO_FILES":
      print(f"  ❌ No text files found in folder")
      return {
        "part_number": part_number,
        "success": False,
        "error": "No text files found"
      }
    elif status == "EMPTY_FILES":
      print(f"  ⚠️  All text files are empty (likely scanned PDFs without OCR)")
      return {
        "part_number": part_number,
        "success": False,
        "error": "Empty text files - original PDFs may be scanned images",
        "empty_files": filtered_files
      }
    elif status == "NO_VALID_CONTENT":
      print(f"  ⚠️  No valid content found in text files")
      return {
        "part_number": part_number,
        "success": False,
        "error": "No valid content in text files"
      }

    print(f"  📄 Using {len(used_files)} files, filtered {len(filtered_files)}")

    # Identify category
    category = self.identify_category(combined_text, part_number)
    print(f"  🏷️  Category: {category}")

    # Create prompt
    prompt = self.create_api_prompt(part_number, combined_text, category)

    # Call API
    print(f"  🤖 Calling OpenAI API...")
    api_result = self.call_openai_api(prompt, part_number)

    if api_result["success"]:
      self.stats["processed"] += 1
      print(f"  ✅ Success ({api_result.get('tokens', 0)} tokens)")

      # Validate the result has both description and JSON
      data = api_result["data"]

      return {
        "part_number": part_number,
        "success": True,
        "category": category,
        "description": data.get("description", "N/A"),
        "json_specs": data,
        "files_used": used_files,
        "files_filtered": filtered_files,
        "tokens_used": api_result.get("tokens", 0)
      }
    else:
      self.stats["failed"] += 1
      self.stats["errors"].append({
        "part_number": part_number,
        "error": api_result.get("error")
      })
      print(f"  ❌ Failed: {api_result.get('error')}")
      return {
        "part_number": part_number,
        "success": False,
        "error": api_result.get("error")
      }

  def run(self, limit: Optional[int] = None, resume_from: Optional[str] = None):
    """
    Process all part folders
    """
    print("\n" + "=" * 60)
    print("HARDWARE PARTS PROCESSOR - Trenton Format")
    print("=" * 60)

    # Find all folders
    all_folders = sorted([d for d in self.base_path.iterdir() if d.is_dir()])
    self.stats["total_folders"] = len(all_folders)

    print(f"\n📊 Found {len(all_folders)} part folders")

    # Handle resume
    if resume_from:
      try:
        start_idx = next(i for i, f in enumerate(all_folders) if f.name == resume_from)
        all_folders = all_folders[start_idx:]
        print(f"📍 Resuming from: {resume_from}")
      except StopIteration:
        print(f"⚠️  Resume part {resume_from} not found, starting from beginning")

    # Apply limit
    if limit:
      all_folders = all_folders[:limit]
      print(f"🔢 Limited to {limit} parts")

    # Process
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, folder in enumerate(all_folders, 1):
      print(f"\n[{i}/{len(all_folders)}]", end="")
      result = self.process_part_folder(folder)
      results.append(result)

      # Save intermediate results
      if i % 10 == 0:
        self.save_results(results, timestamp, intermediate=True)
        print(f"\n💾 Saved intermediate results ({i} processed)")

    # Save final results
    self.save_results(results, timestamp, intermediate=False)

    # Print summary
    self.print_summary(results)

    return results

  def save_results(self, results: List[Dict], timestamp: str, intermediate: bool = False):
    """
    Save processing results in multiple formats
    """
    suffix = "_intermediate" if intermediate else "_final"

    # 1. Full JSON results
    json_file = self.output_dir / f"results_{timestamp}{suffix}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
      json.dump(results, f, indent=2, ensure_ascii=False)

    # 2. Descriptions only (for easy review)
    desc_file = self.output_dir / f"descriptions_{timestamp}{suffix}.txt"
    with open(desc_file, 'w', encoding='utf-8') as f:
      for r in results:
        if r["success"]:
          f.write(f"{r['part_number']}: {r.get('description', 'N/A')}\n")
        else:
          f.write(f"{r['part_number']}: ERROR - {r.get('error', 'Unknown')}\n")

    # 3. SQL Insert statements (for database import)
    sql_file = self.output_dir / f"sql_inserts_{timestamp}{suffix}.sql"
    with open(sql_file, 'w', encoding='utf-8') as f:
      f.write("-- SQL Insert Statements for Parts Database\n")
      f.write("-- Generated: {}\n\n".format(datetime.now().isoformat()))

      for r in results:
        if r["success"]:
          part_num = r['part_number'].replace("'", "''")
          desc = r.get('description', '').replace("'", "''")
          json_str = json.dumps(r.get('json_specs', {})).replace("'", "''")
          category = r.get('category', 'DEFAULT')

          f.write(f"INSERT INTO parts (part_number, category, description, full_specs) VALUES (\n")
          f.write(f"  '{part_num}',\n")
          f.write(f"  '{category}',\n")
          f.write(f"  '{desc}',\n")
          f.write(f"  '{json_str}'\n")
          f.write(f");\n\n")

    # 4. Summary CSV
    csv_file = self.output_dir / f"summary_{timestamp}{suffix}.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
      f.write("Part Number,Success,Category,Description,Files Used,Files Filtered,Error\n")
      for r in results:
        part = r['part_number']
        success = "TRUE" if r['success'] else "FALSE"
        category = r.get('category', '')
        desc = r.get('description', '').replace('"', '""')
        used = str(len(r.get('files_used', [])))
        filtered = str(len(r.get('files_filtered', [])))
        error = r.get('error', '').replace('"', '""') if not r['success'] else ''

        f.write(f'"{part}",{success},"{category}","{desc}",{used},{filtered},"{error}"\n')

    print(f"\n📁 Results saved to: {self.output_dir}")

  def print_summary(self, results: List[Dict]):
    """
    Print processing summary with statistics
    """
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE - SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Total processed: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"Success rate: {(len(successful) / max(len(results), 1) * 100):.1f}%")

    # File statistics
    if self.stats.get("empty_files", 0) > 0 or self.stats.get("missing_files", 0) > 0:
      print("\nFile Issues:")
      if self.stats.get("empty_files", 0) > 0:
        print(f"  📄 Empty text files: {self.stats['empty_files']} (PDFs may be scanned images)")
      if self.stats.get("missing_files", 0) > 0:
        print(f"  ❓ Folders with no text files: {self.stats['missing_files']}")

    # Category breakdown
    categories = {}
    for r in successful:
      cat = r.get('category', 'UNKNOWN')
      categories[cat] = categories.get(cat, 0) + 1

    if categories:
      print("\nCategories identified:")
      for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Token usage
    total_tokens = sum(r.get('tokens_used', 0) for r in successful)
    if total_tokens > 0:
      # Rough cost estimate
      cost_per_1k = 0.002  # Adjust based on current pricing
      estimated_cost = (total_tokens / 1000) * cost_per_1k
      print(f"\nTotal tokens used: {total_tokens:,}")
      print(f"Estimated cost: ${estimated_cost:.2f}")

    if failed:
      print(f"\n⚠️  {len(failed)} parts failed. Check summary CSV for details.")
      # Show first few errors
      print("\nFirst few errors:")
      for err in self.stats["errors"][:5]:
        print(f"  - {err['part_number']}: {err['error'][:50]}...")


def main():
  """
  Main entry point
  """
  import os

  print("Hardware Parts Processor - Trenton Format")
  print("-" * 50)

  # ============================================================
  # CONFIGURATION - UPDATE THESE VALUES
  # ============================================================

  BASE_PATH = r"C:\Users\jlundstedt\Desktop\Configurator_Project_Master\PDF_Scanner\Outputs\Test_Run_Txt_Processor"  # Path to folders with .txt files

  # API Key - Multiple options for security:
  # Option 1: Set as environment variable (recommended)
  #   Set OPENAI_API_KEY in your environment
  #   Then leave API_KEY as None
  # Option 2: Hardcode here (not recommended for production)
  API_KEY = None  # Will use environment variable OPENAI_API_KEY
  # API_KEY = "sk-..."  # Or hardcode if testing locally

  OUTPUT_DIR = r"C:\Users\jlundstedt\Desktop\Configurator_Project_Master\PDF_Scanner\Outputs\Test_Run_Txt_Processor_Results_2"  # Where to save results

  # Test mode settings
  TEST_MODE = True  # Set to False for production
  TEST_LIMIT = 5  # Number of parts to process in test mode

  # Resume from specific part (if interrupted)
  RESUME_FROM = None  # Set to part number like "264-001" to resume

  # ============================================================

  # Validate configuration
  if not os.path.exists(BASE_PATH):
    print(f"❌ Error: Base path does not exist: {BASE_PATH}")
    return

  # Check for API key in environment if not provided
  if not API_KEY:
    API_KEY = os.environ.get("OPENAI_API_KEY")
    if API_KEY:
      print("✅ Using OpenAI API key from environment variable")
    else:
      print("❌ Error: No OpenAI API key found!")
      print("   Set it as environment variable: OPENAI_API_KEY")
      print("   Or provide it in the script (not recommended)")
      return

  # Create processor
  processor = HardwarePartsProcessor(BASE_PATH, API_KEY, OUTPUT_DIR)

  if TEST_MODE:
    print(f"\n🧪 TEST MODE: Processing only {TEST_LIMIT} parts")
    print("Set TEST_MODE = False to process all parts")
    processor.run(limit=TEST_LIMIT, resume_from=RESUME_FROM)
  else:
    print(f"\n⚠️  PRODUCTION MODE: This will process ALL parts")
    confirm = input("Continue? (yes/no): ")
    if confirm.lower() == 'yes':
      processor.run(resume_from=RESUME_FROM)
    else:
      print("Cancelled.")


if __name__ == "__main__":
  main()
