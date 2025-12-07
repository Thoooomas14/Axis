import os
import sys
import re

def verify_imports(root_dir):
    """
    Scans python files in root_dir for forbidden imports (training.model)
    and verifies that src.models and imitation are used instead.
    """
    print(f"Scanning {root_dir}...")
    
    error_count = 0
    warning_count = 0
    
    # Patterns to catch
    forbidden_patterns = [
        (r"from training\.model", "ERROR: Found 'from training.model' - should be 'from src.models'"),
        (r"import training\.model", "ERROR: Found 'import training.model' - should be 'import src.models'"),
        (r"from training\.", "WARNING: Found 'from training.' - check if should be 'from imitation.'"),
        (r"import training", "WARNING: Found 'import training' - check if should be 'import imitation'")
    ]
    
    for dirpath, _, filenames in os.walk(root_dir):
        if "venv" in dirpath or ".git" in dirpath or "__pycache__" in dirpath:
            continue
            
        for filename in filenames:
            if filename.endswith(".py") or filename.endswith(".md"): # Check docs too
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        
                    for i, line in enumerate(lines):
                        for pattern, msg in forbidden_patterns:
                            if re.search(pattern, line):
                                # Ignore if it's the script itself or verified exemptions?
                                if filename == "verify_reorg.py": 
                                    continue
                                    
                                print(f"{filepath}:{i+1}: {msg}")
                                print(f"  Line: {line.strip()}")
                                if "ERROR" in msg:
                                    error_count += 1
                                else:
                                    warning_count += 1
                except Exception as e:
                    print(f"Could not read {filepath}: {e}")

    print("-" * 30)
    print(f"Found {error_count} errors and {warning_count} potential warnings.")
    if error_count > 0:
        print("FAIL: Please fix errors.")
        sys.exit(1)
    else:
        print("PASS: No critical broken imports found (manual check of warnings recommended).")
        sys.exit(0)

if __name__ == "__main__":
    verify_imports(os.getcwd())
