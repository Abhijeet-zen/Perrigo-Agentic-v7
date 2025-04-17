"""
__init__.py

This package holds all prompt templates for the generative AI project.
It provides functions to load a single template or all templates in the folder.
"""

import os

def load_template(template_filename: str) -> str:
    """
    Loads and returns the content of a prompt template file.

    Args:
        template_filename (str): The filename of the prompt template (e.g., "BI_agent_prompt.txt").

    Returns:
        str: The content of the prompt template.
    """
    base_dir = os.path.dirname(__file__)
    template_path = os.path.join(base_dir, template_filename)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file '{template_filename}' not found in {base_dir}.")
    
    with open(template_path, "r", encoding="utf-8") as file:
        return file.read()

def load_all_templates() -> dict:
    """
    Loads all prompt template files (with .txt extension) in the prompt_templates folder.

    Returns:
        dict: A dictionary mapping the template name (filename without extension)
              to its content.
    """
    base_dir = os.path.dirname(__file__)
    templates = {}
    for filename in os.listdir(base_dir):
        if filename.endswith(".txt"):
            template_name = os.path.splitext(filename)[0]
            templates[template_name] = load_template(filename)
    return templates
