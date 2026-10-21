"""
Professional PDF Generator for SupplyIQ Audit & Architecture Documentation
Uses ReportLab 4.2.5 to compile docs/Project_Change_Audit_and_File_Documentation.pdf
"""

import sys
import re
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Preformatted, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# Custom NumberedCanvas for professional "Page X of Y" and Running Headers
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Save each page until the total is known for the "Page X of Y" footer.
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Omit headers and footers on Cover Page (Page 1)
        if self._pageNumber > 1:
            self.saveState()
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#0891B2"))
            self.drawString(36, letter[1] - 28, "SUPPLYIQ")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawString(84, letter[1] - 28, "|   Project Change Audit & Technical Architecture Documentation")
            
            # Running header accent rule
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.75)
            self.line(36, letter[1] - 32, letter[0] - 36, letter[1] - 32)
            
            # Running footer rule
            self.line(36, 32, letter[0] - 36, 32)
            
            # Running footer text
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawString(36, 22, "Confidential · Internal Project Review & Technical Audit Report · Version 1.0")
            self.drawRightString(letter[0] - 36, 22, f"Page {self._pageNumber} of {page_count}")
            self.restoreState()

print("NumberedCanvas initialized successfully.")
