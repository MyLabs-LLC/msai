import re

def apply_patch(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Add logging
    content = content.replace(
        "import os\nfrom pathlib import Path",
        "import os\nimport logging\nfrom pathlib import Path\n\nlogging.basicConfig(level=logging.INFO, format='%(message)s')\nlogger = logging.getLogger(__name__)"
    )

    # Use logger instead of print
    content = content.replace("print(\"\\n*** Workflow execution started ***\\n\")", "logger.info(\"\\n*** Workflow execution started ***\\n\")")
    content = content.replace("print(f\"Task to complete in this workflow, workflow prompt = {workflow_prompt}\")", "logger.info(f\"Task to complete in this workflow, workflow prompt = {workflow_prompt}\")")
    content = content.replace("print(\"\\nDefining workflow steps from the workflow prompt\")", "logger.info(\"\\nDefining workflow steps from the workflow prompt\")")
    content = content.replace("print(f\"Workflow steps: {workflow_steps}\\n\")", "logger.info(f\"Workflow steps: {workflow_steps}\\n\")")
    
    content = content.replace("print(f\"\\n{'='*60}\")", "logger.info(f\"\\n{'='*60}\")")
    content = content.replace("print(f\"Executing Step {i+1}: {step}\")", "logger.info(f\"Executing Step {i+1}: {step}\")")
    content = content.replace("print(f\"{'='*60}\")", "logger.info(f\"{'='*60}\")")
    content = content.replace("print(f\"\\nStep {i+1} Result:\\n{result}\")", "logger.info(f\"\\nStep {i+1} Result:\\n{result}\")")
    
    content = content.replace("print(\"*** Workflow execution completed ***\")", "logger.info(\"*** Workflow execution completed ***\")")
    content = content.replace("print(\"\\nFinal Development Plan:\")", "logger.info(\"\\nFinal Development Plan:\")")
    content = content.replace("print(f\"\\nUser Stories:\\n{workflow_context['user_stories']}\")", "logger.info(f\"\\nUser Stories:\\n{workflow_context['user_stories']}\")")
    content = content.replace("print(f\"\\nProduct Features:\\n{workflow_context['features']}\")", "logger.info(f\"\\nProduct Features:\\n{workflow_context['features']}\")")
    content = content.replace("print(f\"\\nEngineering Tasks:\\n{workflow_context['tasks']}\")", "logger.info(f\"\\nEngineering Tasks:\\n{workflow_context['tasks']}\")")

    # Add success checks
    content = content.replace(
        "final_response = normalize_section_output(result[\"final_response\"], \"User Stories:\")",
        "if not result.get('success', False):\n        logger.warning('Warning: Product Manager evaluation did not pass criteria.')\n    final_response = normalize_section_output(result[\"final_response\"], \"User Stories:\")"
    )
    content = content.replace(
        "final_response = normalize_section_output(result[\"final_response\"], \"Product Features:\")",
        "if not result.get('success', False):\n        logger.warning('Warning: Program Manager evaluation did not pass criteria.')\n    final_response = normalize_section_output(result[\"final_response\"], \"Product Features:\")"
    )
    content = content.replace(
        "final_response = normalize_section_output(result[\"final_response\"], \"Engineering Tasks:\")",
        "if not result.get('success', False):\n        logger.warning('Warning: Development Engineer evaluation did not pass criteria.')\n    final_response = normalize_section_output(result[\"final_response\"], \"Engineering Tasks:\")"
    )

    with open(file_path, 'w') as f:
        f.write(content)

apply_patch('ProjectManagement/starter/phase_2/agentic_workflow.py')
