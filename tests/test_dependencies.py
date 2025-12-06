import os
import ast
import re
import yaml
import pytest
import sys

# Define project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_imports_from_file(filepath):
    """Extracts top-level imports from a Python file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def get_codebase_imports(root_dir, ignore_dirs=None):
    """Scans the codebase for external imports."""
    if ignore_dirs is None:
        ignore_dirs = ['.git', '.github', '__pycache__', 'venv', 'env', 'tests', 'simulation']
    
    unique_imports = set()
    std_lib = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else set() # Python 3.10+
    
    # Fallback for older python or if sys.stdlib_module_names is missing (unlikely in 3.10)
    if not std_lib:
        import sysconfig
        # This is a rough approximation
        std_lib = set(['os', 'sys', 'math', 'json', 're', 'time', 'datetime', 'argparse', 'shutil', 'pickle', 'traceback', 'typing', 'unittest', 'csv', 'random'])

    custom_modules = set(['training', 'scripts']) # Internal packages

    for subdir, dirs, files in os.walk(root_dir):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(subdir, file)
                file_imports = get_imports_from_file(filepath)
                
                for imp in file_imports:
                    if imp not in std_lib and imp not in custom_modules:
                        unique_imports.add(imp)

    return unique_imports

def parse_requirements(filepath):
    """Parses requirements.txt into a set of package names."""
    if not os.path.exists(filepath):
        return set()
    
    packages = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Remove version specifiers
            package = re.split(r'[<>=!]', line)[0].strip()
            # Handle package aliases (e.g. scikit-learn vs sklearn)
            if package == 'scikit-learn': package = 'sklearn'
            if package == 'Pillow': package = 'PIL'
            packages.add(package.lower())
    return packages

def parse_environment_yml(filepath):
    """Parses environment.yml into a set of package names."""
    if not os.path.exists(filepath):
        return set()
    
    with open(filepath, 'r') as f:
        env = yaml.safe_load(f)
    
    deps = env.get('dependencies', [])
    packages = set()
    
    for dep in deps:
        if isinstance(dep, str):
            package = re.split(r'[<>=!]', dep)[0].strip()
            if package != 'python' and package != 'pip':
                packages.add(package.lower())
        elif isinstance(dep, dict) and 'pip' in dep:
            for pip_dep in dep['pip']:
                package = re.split(r'[<>=!]', pip_dep)[0].strip()
                packages.add(package.lower())
                
    # Aliases
    if 'scikit-learn' in packages: 
        packages.remove('scikit-learn')
        packages.add('sklearn')
    if 'pytorch' in packages:
        packages.remove('pytorch')
        packages.add('torch')
        
    return packages

def test_requirements_vs_codebase():
    """Verifies that all imports in the codebase are in requirements.txt."""
    code_imports = get_codebase_imports(PROJECT_ROOT)
    reqs = parse_requirements(os.path.join(PROJECT_ROOT, 'requirements.txt'))
    
    # Map common import names to package names
    import_map = {
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'yaml': 'pyyaml',
        'skimage': 'scikit-image',
        'tqdm': 'tqdm',
        'bs4': 'beautifulsoup4'
    }
    
    # Invert requirements alias mapping for checking
    # requirements.txt usually has 'pillow', code has 'PIL'
    # strict check: name in code import -> name in requirements line
    
    missing_deps = []
    
    for imp in code_imports:
        # Check direct match
        if imp.lower() in reqs:
            continue
            
        # Check mapped match
        if imp in import_map:
            if import_map[imp] in reqs:
                continue
        
        # Check if it's a known non-pypi package or special case
        # e.g., 'training' is internal, but we filtered that.
        # 'tensorflow' -> 'tensorflow'
        
        missing_deps.append(imp)
    
    # Filter out some known loose ends if necessary, or fail
    # For now, we want stricness.
    # Note: 'tensorflow.keras' -> 'tensorflow' checked by logic 'tensorflow'
    
    assert not missing_deps, f"The following imports are used in code but missing from requirements.txt: {missing_deps}"

def test_environment_vs_requirements():
    """Verifies consistency between environment.yml and requirements.txt."""
    reqs = parse_requirements(os.path.join(PROJECT_ROOT, 'requirements.txt'))
    env_deps = parse_environment_yml(os.path.join(PROJECT_ROOT, 'environment.yml'))
    
    # Check for items in requirements that are missing from environment.yml
    # Note: environment.yml often has conda packages which might be named differently
    # But for now we assume mostly pip consistency or same names.
    
    missing_in_env = reqs - env_deps
    
    # Allow some differences (e.g. system libs)
    ignored = {'tensorflow-io'} # specific exclusion if needed, but let's try strict first
    missing_in_env = {d for d in missing_in_env if d not in ignored}

    # Fixup alias issues manual check
    # e.g. torch vs pytorch (handled in parser)
    
    assert not missing_in_env, f"packages in requirements.txt but missing from environment.yml: {missing_in_env}"

if __name__ == "__main__":
    # verification
    print("Imports in codebase:", get_codebase_imports(PROJECT_ROOT))
    print("Requirements:", parse_requirements(os.path.join(PROJECT_ROOT, 'requirements.txt')))
    print("Env:", parse_environment_yml(os.path.join(PROJECT_ROOT, 'environment.yml')))
