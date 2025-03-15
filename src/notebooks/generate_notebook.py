import os
import sys
import json

def get_project_python_files(base_dir):
	"""
	Get Python files only from allowed parts of the project:
	  - All Python files under the 'src' directory, excluding files from simulation and tests,
		and excluding __init__.py, setup.py, and generate_notebook.py.
	
	Returns:
		list: Sorted list of Python file paths relative to base_dir.
	"""
	py_files = set()
	
	# Add all Python files under the 'src' directory.
	src_dir = os.path.join(base_dir, 'src')
	if os.path.isdir(src_dir):
		for root, dirs, files in os.walk(src_dir):
			# Exclude hidden directories and common unwanted ones.
			dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', '__pycache__')]
			# Skip directories named 'simulation' or 'tests'
			if any(excluded in os.path.normpath(root).split(os.sep) for excluded in ('simulation', 'tests')):
				continue
			for file in files:
				if file.endswith('.py'):
					# Exclude specific files.
					if file in {"generate_notebook.py", "__init__.py", "setup.py"}:
						continue
					full_path = os.path.join(root, file)
					rel_path = os.path.relpath(full_path, base_dir)
					py_files.add(rel_path)
	
	return sorted(py_files)

def main():
	# Use the provided directory or assume the project root is three levels up.
	if len(sys.argv) > 1:
		directory = sys.argv[1]
	else:
		directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	
	print(f"Searching for Python files in: {directory}")
	
	py_files = get_project_python_files(directory)
	
	if not py_files:
		print("No Python files found!")
		return

	cells = []
	
	# 1. Insert a cell to set the working directory and adjust sys.path.
	env_cell = {
		"cell_type": "code",
		"execution_count": None,
		"metadata": {},
		"outputs": [],
		"source": [
			"import os, sys\n",
			"# Ensure the working directory is the project root\n",
			"project_root = os.getcwd()\n",
			"if project_root not in sys.path:\n",
			"    sys.path.insert(0, project_root)\n",
			"print('Project root added to sys.path:', project_root)\n"
		]
	}
	cells.append(env_cell)
	
	# 2. Insert a cell to create necessary directories.
	# Compute all unique directories from the file paths.
	dir_set = set()
	for rel_path in py_files:
		dirname = os.path.dirname(rel_path)
		if dirname:
			dir_set.add(dirname)
	dirs_list = sorted(dir_set)
	mkdir_cell = {
		"cell_type": "code",
		"execution_count": None,
		"metadata": {},
		"outputs": [],
		"source": [
			"import os\n",
			"dirs = " + str(dirs_list) + "\n",
			"for d in dirs:\n",
			"    os.makedirs(d, exist_ok=True)\n",
			"print('Created directories:', dirs)\n"
		]
	}
	cells.append(mkdir_cell)
	
	# 3. Add a markdown header cell.
	header_cell = {
		"cell_type": "markdown",
		"metadata": {},
		"source": [
			"# Auto-Generated Notebook\n",
			"\n",
			f"This notebook was automatically generated from Python files in the project: `{directory}`\n"
		]
	}
	cells.append(header_cell)
	
	# 4. Create a cell for each Python file using the %%writefile magic.
	for rel_path in py_files:
		filepath = os.path.join(directory, rel_path)
		try:
			with open(filepath, 'r', encoding='utf-8') as f:
				content = f.read()
		except Exception as e:
			print(f"Error reading {rel_path}: {e}")
			continue

		# Add a markdown cell with the file path.
		path_cell = {
			"cell_type": "markdown",
			"metadata": {},
			"source": [f"## File: {rel_path}\n"]
		}
		cells.append(path_cell)

		# Use %%writefile so that when the cell is executed, it writes the file.
		cell_source = [f"%%writefile {rel_path}\n", content]
		
		code_cell = {
			"cell_type": "code",
			"execution_count": None,
			"metadata": {},
			"outputs": [],
			"source": cell_source
		}
		cells.append(code_cell)
	
	# 5. Add a final cell to run train_diffusion.py (if it exists) using a shell command that sets PYTHONPATH.
	for file in py_files:
		if "train_diffusion.py" in file:
			run_cell = {
				"cell_type": "code",
				"execution_count": None,
				"metadata": {},
				"outputs": [],
				# Set PYTHONPATH so the subprocess can locate 'src'
				"source": [f"!PYTHONPATH=$(pwd) python {file}\n"]
			}
			cells.append(run_cell)
			break
	
	# Build the complete notebook structure.
	notebook = {
		"cells": cells,
		"metadata": {
			"kernelspec": {
				"display_name": "Python 3",
				"language": "python",
				"name": "python3"
			},
			"language_info": {
				"name": "python",
				"version": sys.version.split()[0]
			}
		},
		"nbformat": 4,
		"nbformat_minor": 5
	}
	
	output_filename = "project_notebook.ipynb"
	try:
		with open(output_filename, "w", encoding="utf-8") as f:
			json.dump(notebook, f, indent=1, ensure_ascii=False)
		print(f"Notebook generated: {output_filename}")
		print(f"Included {len(py_files)} Python files")
	except Exception as e:
		print("Error writing the notebook file:", e)

if __name__ == "__main__":
	main()
