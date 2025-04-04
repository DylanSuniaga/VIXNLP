#!/usr/bin/env python3
import os
import sys
import subprocess
import time
from datetime import datetime

def run_scraper(script_path):
    """Run a scraper script and return its output"""
    try:
        print(f"\nRunning: {script_path}")
        print("=" * 80)
        
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print the output
        print(result.stdout)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error running {script_path}: {e}")
        return False

def main():
    """Run all scrapers in the titles directory"""
    print(f"VIXNLP News Title Scrapers - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Get the path to the scrapers/titles directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scrapers_dir = os.path.join(base_dir, "scrapers", "titles")
    
    # Verify the directory exists
    if not os.path.isdir(scrapers_dir):
        print(f"Error: Directory not found: {scrapers_dir}")
        return
    
    # Get all Python files in the directory
    scraper_files = []
    for file in os.listdir(scrapers_dir):
        if file.endswith(".py"):
            scraper_files.append(os.path.join(scrapers_dir, file))
    
    if not scraper_files:
        print(f"No scraper scripts found in {scrapers_dir}")
        return
    
    print(f"Found {len(scraper_files)} scraper(s):")
    for i, scraper in enumerate(scraper_files, 1):
        print(f"{i}. {os.path.basename(scraper)}")
    
    # Run each scraper
    successful = 0
    for scraper in scraper_files:
        if run_scraper(scraper):
            successful += 1
        
        # Add a small delay between scrapers
        time.sleep(1)
    
    print("\nSummary:")
    print("=" * 80)
    print(f"Total scrapers: {len(scraper_files)}")
    print(f"Successfully run: {successful}")
    print(f"Failed: {len(scraper_files) - successful}")
    print("\nNote: Some scrapers may have run successfully but found no results.")
    print("This is normal, especially for specialized scrapers like VIX-related news.")

if __name__ == "__main__":
    main() 