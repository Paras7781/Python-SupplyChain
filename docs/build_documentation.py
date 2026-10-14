"""
SupplyIQ Documentation & PDF Report Generator
Generates:
1. docs/Project_Change_Audit_and_File_Documentation.md (Complete editable Markdown report source)
2. docs/Project_Change_Audit_and_File_Documentation.pdf (Publication-quality PDF report with ReportLab)
"""

import os
import sys
from pathlib import Path

# Set up paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = WORKSPACE_DIR / "docs"
MD_REPORT_PATH = DOCS_DIR / "Project_Change_Audit_and_File_Documentation.md"
PDF_REPORT_PATH = DOCS_DIR / "Project_Change_Audit_and_File_Documentation.pdf"

print(f"Workspace: {WORKSPACE_DIR}")
print(f"Docs Directory: {DOCS_DIR}")
