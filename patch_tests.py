import os
import glob

def apply_patch(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Replace print statements with logging
    if "import logging" not in content:
        content = content.replace("import os", "import os\nimport logging\n\nlogging.basicConfig(level=logging.INFO, format='%(message)s')\nlogger = logging.getLogger(__name__)")
    
    content = content.replace("print(", "logger.info(")
    
    with open(file_path, 'w') as f:
        f.write(content)

for file in glob.glob('ProjectManagement/starter/phase_1/*_agent.py'):
    apply_patch(file)
