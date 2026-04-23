#!/usr/bin/env python3
"""
Full project test runner for the AI-Powered Agentic Workflow.

Runs every Phase 1 agent test and the Phase 2 workflow step by step,
saves each output to its corresponding *_output.txt file, writes the
entire combined output to test_output.txt, and zips the ProjectManagement
folder with the current date and time on completion.
"""

import io
import os
import sys
import time
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent
PHASE1_DIR = PROJECT_DIR / "starter" / "phase_1"
PHASE2_DIR = PROJECT_DIR / "starter" / "phase_2"
TEST_OUTPUT = PROJECT_DIR / "test_output.txt"

PHASE1_SCRIPTS = [
    ("DirectPromptAgent",                "direct_prompt_agent.py",                "direct_prompt_agent_output.txt"),
    ("AugmentedPromptAgent",             "augmented_prompt_agent.py",             "augmented_prompt_agent_output.txt"),
    ("KnowledgeAugmentedPromptAgent",    "knowledge_augmented_prompt_agent.py",   "knowledge_augmented_prompt_agent_output.txt"),
    ("RAGKnowledgePromptAgent",          "rag_knowledge_prompt_agent.py",         "rag_knowledge_prompt_agent_output.txt"),
    ("EvaluationAgent",                  "evaluation_agent.py",                   "evaluation_agent_output.txt"),
    ("RoutingAgent",                     "routing_agent.py",                      "routing_agent_output.txt"),
    ("ActionPlanningAgent",              "action_planning_agent.py",              "action_planning_agent_output.txt"),
]

DIVIDER   = "=" * 70
THIN_LINE = "-" * 70


class TeeWriter:
    """Write to both the terminal and a string buffer simultaneously."""

    def __init__(self, original_stdout):
        self.terminal = original_stdout
        self.buffer = io.StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)

    def flush(self):
        self.terminal.flush()

    def get_value(self) -> str:
        return self.buffer.getvalue()


def banner(text: str) -> str:
    return f"\n{DIVIDER}\n  {text}\n{DIVIDER}"


def run_script(script_path: Path, output_path: Path, cwd: Path) -> tuple[bool, float]:
    """Run a Python script, capture combined stdout+stderr, overwrite the output file."""
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.time() - start
    combined = result.stdout
    if result.stderr:
        combined += "\n" + result.stderr

    output_path.write_text(combined, encoding="utf-8")
    return result.returncode == 0, elapsed


def main():
    tee = TeeWriter(sys.stdout)
    sys.stdout = tee

    overall_start = time.time()
    results = []

    print(banner("PROJECT TEST RUNNER"))
    print(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project dir: {PROJECT_DIR}")
    print(f"  Python     : {sys.executable}")
    print(f"  Full log   : {TEST_OUTPUT}")
    print(DIVIDER)

    # ------------------------------------------------------------------
    # Phase 1: Individual Agent Tests
    # ------------------------------------------------------------------
    print(banner("PHASE 1 — Individual Agent Tests"))
    phase1_pass = 0
    phase1_fail = 0

    for i, (name, script, output_file) in enumerate(PHASE1_SCRIPTS, 1):
        step_label = f"[Phase 1 — Step {i}/7] {name}"
        print(f"\n{THIN_LINE}")
        print(f"  {step_label}")
        print(f"  Script : {script}")
        print(f"  Output : {output_file}")
        print(THIN_LINE)

        script_path = PHASE1_DIR / script
        output_path = PHASE1_DIR / output_file

        try:
            ok, elapsed = run_script(script_path, output_path, cwd=PHASE1_DIR)
            status = "PASS" if ok else "FAIL"
            if ok:
                phase1_pass += 1
            else:
                phase1_fail += 1
            print(f"  Status : {status}  ({elapsed:.1f}s)")
            print(f"  Saved  : {output_path.relative_to(PROJECT_DIR)}")

            output_text = output_path.read_text(encoding="utf-8").strip()
            print(f"\n  --- Full Output ---\n{output_text}\n")

        except subprocess.TimeoutExpired:
            phase1_fail += 1
            status = "TIMEOUT"
            print(f"  Status : {status}  (exceeded 300s limit)")

        except Exception as e:
            phase1_fail += 1
            status = "ERROR"
            print(f"  Status : {status}  ({e})")

        results.append((step_label, status))

    print(f"\n{THIN_LINE}")
    print(f"  Phase 1 Summary: {phase1_pass} passed, {phase1_fail} failed out of 7")
    print(THIN_LINE)

    # ------------------------------------------------------------------
    # Phase 2: Full Agentic Workflow
    # ------------------------------------------------------------------
    print(banner("PHASE 2 — Full Agentic Workflow"))
    step_label = "[Phase 2] agentic_workflow.py"
    print(f"\n{THIN_LINE}")
    print(f"  {step_label}")
    print(f"  This runs the full PM -> PgM -> Dev pipeline with dual-model consensus.")
    print(f"  It may take several minutes depending on API response times.")
    print(THIN_LINE)

    workflow_script = PHASE2_DIR / "agentic_workflow.py"
    workflow_output = PHASE2_DIR / "workflow_output.txt"

    try:
        ok, elapsed = run_script(workflow_script, workflow_output, cwd=PHASE2_DIR)
        status = "PASS" if ok else "FAIL"
        print(f"  Status : {status}  ({elapsed:.1f}s)")
        print(f"  Saved  : {workflow_output.relative_to(PROJECT_DIR)}")

        output_text = workflow_output.read_text(encoding="utf-8").strip()
        print(f"\n  --- Full Output ---\n{output_text}\n")

    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
        print(f"  Status : {status}  (exceeded 300s limit)")

    except Exception as e:
        status = "ERROR"
        print(f"  Status : {status}  ({e})")

    results.append((step_label, status))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    overall_elapsed = time.time() - overall_start
    print(banner("TEST SUMMARY"))
    for label, status in results:
        icon = "+" if status == "PASS" else "X"
        print(f"  [{icon}] {label} — {status}")
    print(f"\n  Total time: {overall_elapsed:.1f}s")
    print(DIVIDER)

    # ------------------------------------------------------------------
    # Save combined test_output.txt
    # ------------------------------------------------------------------
    TEST_OUTPUT.write_text(tee.get_value(), encoding="utf-8")
    print(f"\n  Combined test log saved to: {TEST_OUTPUT}")

    # ------------------------------------------------------------------
    # Zip the project folder
    # ------------------------------------------------------------------
    print(banner("CREATING SUBMISSION ZIP"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    zip_name = f"ProjectManagement_{timestamp}"
    zip_path = PROJECT_DIR.parent / f"{zip_name}.zip"

    EXCLUDE = {".env"}

    print(f"  Source : {PROJECT_DIR}")
    print(f"  Target : {zip_path}")
    print(f"  Excluding: {', '.join(sorted(EXCLUDE))}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(PROJECT_DIR.rglob("*")):
            if not file.is_file():
                continue
            if file.name in EXCLUDE:
                continue
            arcname = Path(PROJECT_DIR.name) / file.relative_to(PROJECT_DIR)
            zf.write(file, arcname)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  Size   : {zip_size_mb:.2f} MB")
    print(f"\n  Zip created successfully: {zip_path}")
    print(DIVIDER)

    # Final write so the zip/summary lines are also captured
    sys.stdout = tee.terminal
    TEST_OUTPUT.write_text(tee.get_value(), encoding="utf-8")

    all_passed = all(s == "PASS" for _, s in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
