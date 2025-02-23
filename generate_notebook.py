import os
import sys
import json

def main():
	# Use the provided directory or default to the current working directory.
	if len(sys.argv) > 1:
		directory = sys.argv[1]
	else:
		directory = os.getcwd()

	# List all Python files in the directory, excluding this generator script.
	this_script = os.path.basename(__file__)
	py_files = sorted([
		f for f in os.listdir(directory)
		if f.endswith('.py') and f != this_script
	])

	cells = []
	
	# Add a markdown header cell.
	header_cell = {
		"cell_type": "markdown",
		"metadata": {},
		"source": [
			"# Auto-Generated Notebook\n",
			"\n",
			"This notebook was automatically generated from Python files in the directory: `{}`".format(directory)
		]
	}
	cells.append(header_cell)
	
	# Create a code cell for each Python file using the %%writefile magic.
	for filename in py_files:
		filepath = os.path.join(directory, filename)
		try:
			with open(filepath, 'r', encoding='utf-8') as f:
				content = f.read()
		except Exception as e:
			print(f"Error reading {filename}: {e}")
			continue

		cell_source = []
		cell_source.append("%%writefile {}\n".format(filename))
		cell_source.append(content)
		
		code_cell = {
			"cell_type": "code",
			"execution_count": None,
			"metadata": {},
			"outputs": [],
			"source": cell_source
		}
		cells.append(code_cell)
	
	# Optionally, add a final cell to run the training script if it exists.
	if "train_diffusion.py" in py_files:
		run_cell = {
			"cell_type": "code",
			"execution_count": None,
			"metadata": {},
			"outputs": [],
			"source": ["!python train_diffusion.py\n"]
		}
		cells.append(run_cell)
	
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
	
	# Write the notebook JSON to a file.
	output_filename = "auto_generated_notebook.ipynb"
	try:
		with open(output_filename, "w", encoding='utf-8') as f:
			json.dump(notebook, f, indent=1)
		print("Notebook generated:", output_filename)
	except Exception as e:
		print("Error writing the notebook file:", e)

if __name__ == "__main__":
	main()
