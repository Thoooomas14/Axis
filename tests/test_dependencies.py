import os
import ast
import re
import yaml
import pytest
import sys

# Define project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def normalize_package_name(name):
    """Normalizes package name to lowercase and hyphens instead of underscores."""
    return name.lower().replace('_', '-')

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
                # Handle relative imports (start with .)
                if not node.level:
                    imports.add(node.module.split('.')[0])
    return imports

def is_internal_module(module_name, current_dir, root_dir):
    """Checks if a module is likely internal (exists as file or dir in codebase)."""
    # 1. Check relative to current directory (for poor man's relative imports without dot)
    if os.path.exists(os.path.join(current_dir, f"{module_name}.py")):
        return True
    if os.path.isdir(os.path.join(current_dir, module_name)):
        return True
        
    # 2. Check relative to project root (absolute internal imports)
    if os.path.exists(os.path.join(root_dir, f"{module_name}.py")):
        return True
    if os.path.isdir(os.path.join(root_dir, module_name)):
        return True
    
    # 3. Known Internal prefixes
    if module_name in ['training', 'scripts', 'simulation', 'tests', 'data']:
        return True
        
    return False

def get_codebase_imports(root_dir, ignore_dirs=None):
    """Scans the codebase for external imports."""
    if ignore_dirs is None:
        ignore_dirs = ['.git', '.github', '__pycache__', 'venv', 'env', 'tests', 'simulation']
    
    unique_imports = set()
    std_lib = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else set()
    
    # Fallback
    if not std_lib:
        std_lib = set(['os', 'sys', 'math', 'json', 're', 'time', 'datetime', 'argparse', 'shutil', 'pickle', 'traceback', 'typing', 'unittest', 'csv', 'random', 'collections', 'functools', 'itertools'])

    for subdir, dirs, files in os.walk(root_dir):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(subdir, file)
                file_imports = get_imports_from_file(filepath)
                
                for imp in file_imports:
                    if imp not in std_lib:
                        # Check if internal
                        if not is_internal_module(imp, subdir, root_dir):
                            unique_imports.add(imp)

    return unique_imports

def parse_requirements(filepath):
    """Parses requirements.txt into a set of normalized package names."""
    if not os.path.exists(filepath):
        return set()
    
    packages = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            package = re.split(r'[<>=!]', line)[0].strip()
            packages.add(normalize_package_name(package))
    return packages

def parse_environment_yml(filepath):
    """Parses environment.yml into a set of normalized package names."""
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
                packages.add(normalize_package_name(package))
        elif isinstance(dep, dict) and 'pip' in dep:
            for pip_dep in dep['pip']:
                package = re.split(r'[<>=!]', pip_dep)[0].strip()
                packages.add(normalize_package_name(package))
                
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
        'cv2': 'opencv-python-headless',
        'yaml': 'pyyaml',
        'skimage': 'scikit-image',
        'tqdm': 'tqdm',
        'bs4': 'beautifulsoup4',
        'google.colab': None, # Ignore colab specific
        'mpl_toolkits': 'matplotlib',  # Part of matplotlib
        # Isaac Sim/Lab ecosystem (not pip-installable, installed via Isaac Sim)
        'omni': None,
        'isaacsim': None,
        'isaaclab': None,
        'my_robot_ext': None,  # Local Isaac Lab extension package
    }
    
    missing_deps = []
    
    for imp in code_imports:
        norm_imp = normalize_package_name(imp)
        
        # Check direct match
        if norm_imp in reqs:
            continue
            
        # Check mapped match
        if imp in import_map:
            target = import_map[imp]
            if target is None: continue # Ignored
            if normalize_package_name(target) in reqs:
                continue
        
        missing_deps.append(imp)
    
    assert not missing_deps, f"The following imports are used in code but missing from requirements.txt: {missing_deps}"

def test_environment_vs_requirements():
    """Verifies consistency between environment.yml and requirements.txt."""
    reqs = parse_requirements(os.path.join(PROJECT_ROOT, 'requirements.txt'))
    env_deps = parse_environment_yml(os.path.join(PROJECT_ROOT, 'environment.yml'))
    
    # Check for items in requirements that are missing from environment.yml
    missing_in_env = reqs - env_deps
    
    # Exclusions
    # 'pip' is implicit
    
    assert not missing_in_env, f"packages in requirements.txt but missing from environment.yml: {missing_in_env}"

