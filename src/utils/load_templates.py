import os

# Set the correct path to the prompt_templates directory
PROMPT_TEMPLATES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../prompt_templates"))


def load_template(template_filename: str) -> str:
    """
    Loads and returns the content of a prompt template file.

    Args:
        template_filename (str): The filename of the prompt template (e.g., "BI_agent_prompt.txt").

    Returns:
        str: The content of the prompt template.
    """
    template_path = os.path.join(PROMPT_TEMPLATES_DIR, template_filename)

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file '{template_filename}' not found in {PROMPT_TEMPLATES_DIR}.")

    with open(template_path, "r", encoding="utf-8") as file:
        return file.read()


def load_all_templates() -> dict:
    """
    Loads all prompt template files (with .txt extension) in the prompt_templates folder.

    Returns:
        dict: A dictionary mapping the template name (filename without extension)
              to its content.
    """
    templates = {}

    if not os.path.exists(PROMPT_TEMPLATES_DIR):
        raise FileNotFoundError(f"Prompt templates directory '{PROMPT_TEMPLATES_DIR}' not found.")

    for filename in os.listdir(PROMPT_TEMPLATES_DIR):
        if filename.endswith(".txt"):
            template_name = os.path.splitext(filename)[0]
            templates[template_name] = load_template(filename)

    return templates
