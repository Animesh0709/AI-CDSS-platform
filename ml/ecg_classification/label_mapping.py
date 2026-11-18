"""
PTB-XL Multi-Label Target Definition and SCP-ECG Statement Mapping.

Target Schema:
1. 5 Diagnostic Superclasses:
   - NORM: Normal ECG
   - MI: Myocardial Infarction
   - STTC: ST/T Change
   - CD: Conduction Disturbance
   - HYP: Hypertrophy

2. 23 Diagnostic Subclasses:
   - Nested under diagnostic superclasses per scp_statements.csv:
     - MI: AMI, IMI, LMI, PMI
     - STTC: ISCA, ISCI, ISC_, NST_, STTC
     - CD: _AVB, CLBBB, CRBBB, ILBBB, IRBBB, IVCD, LAFB/LPFB, WPW
     - HYP: LVH, RVH, LAO/LAE, RAO/RAE, SEHYP
     - NORM: NORM

3. Weight Inclusion Policy:
   - SCP statement weights range from 0.0 to 100.0 (representing clinical likelihood / percentage).
   - Zero-weight labels (weight == 0.0) indicate explicit ruled-out or unconfirmed statements and MUST NOT be treated as positive targets.
   - For benchmark consistency with PhysioNet PTB-XL literature:
     - Diagnostic superclass inclusion requires: code in diag_statements AND weight > 0.
     - Diagnostic subclass inclusion requires: code in diag_subclass_statements AND weight > 0.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

DIAGNOSTIC_SUPERCLASSES: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]

DIAGNOSTIC_SUBCLASSES: List[str] = [
    "NORM", "AMI", "IMI", "LMI", "PMI",
    "ISCA", "ISCI", "ISC_", "NST_", "STTC",
    "_AVB", "CLBBB", "CRBBB", "ILBBB", "IRBBB", "IVCD", "LAFB/LPFB", "WPW",
    "LVH", "RVH", "LAO/LAE", "RAO/RAE", "SEHYP",
]


class PTBXLLabelMapper:
    """
    Handles parsing, mapping, and multi-label binary vector generation
    for PTB-XL SCP-ECG statements.
    """

    def __init__(self, scp_statements_path: str | Path):
        self.scp_path = Path(scp_statements_path)
        if not self.scp_path.exists():
            raise FileNotFoundError(f"scp_statements.csv not found at {self.scp_path}")

        self.scp_df = pd.read_csv(self.scp_path, index_col=0)

        # Separate statement types
        self.diagnostic_df = self.scp_df[self.scp_df["diagnostic"] == 1.0]
        self.form_df = self.scp_df[self.scp_df["form"] == 1.0]
        self.rhythm_df = self.scp_df[self.scp_df["rhythm"] == 1.0]

        # Mappings
        self.diag_class_map = self.diagnostic_df["diagnostic_class"].dropna().to_dict()
        self.diag_subclass_map = self.diagnostic_df["diagnostic_subclass"].dropna().to_dict()

    def get_statement_category(self, code: str) -> Dict[str, bool]:
        """Classify a given SCP code into diagnostic, form, and/or rhythm category."""
        if code not in self.scp_df.index:
            return {"diagnostic": False, "form": False, "rhythm": False, "known": False}
        row = self.scp_df.loc[code]
        return {
            "diagnostic": bool(row.get("diagnostic") == 1.0),
            "form": bool(row.get("form") == 1.0),
            "rhythm": bool(row.get("rhythm") == 1.0),
            "known": True,
        }

    def map_record_statements(
        self,
        scp_codes_input: str | Dict[str, float],
        min_weight_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Parse scp_codes dictionary and return mapped superclasses, subclasses,
        and multi-hot binary vectors.

        Rule: Labels with weight <= min_weight_threshold are excluded.
        """
        if isinstance(scp_codes_input, str):
            scp_dict: Dict[str, float] = ast.literal_eval(scp_codes_input)
        elif isinstance(scp_codes_input, dict):
            scp_dict = scp_codes_input
        else:
            raise TypeError(f"Unsupported scp_codes format: {type(scp_codes_input)}")

        superclasses: Set[str] = set()
        subclasses: Set[str] = set()
        form_statements: List[str] = []
        rhythm_statements: List[str] = []
        zero_weight_statements: List[str] = []

        for code, weight in scp_dict.items():
            if weight <= min_weight_threshold:
                zero_weight_statements.append(code)
                continue

            if code in self.diag_class_map:
                superclasses.add(self.diag_class_map[code])
            if code in self.diag_subclass_map:
                subclasses.add(self.diag_subclass_map[code])

            if code in self.form_df.index:
                form_statements.append(code)
            if code in self.rhythm_df.index:
                rhythm_statements.append(code)

        superclasses_sorted = sorted(list(superclasses))
        subclasses_sorted = sorted(list(subclasses))

        # Binary multi-hot targets
        superclass_vector = {f"label_{sc}": (1 if sc in superclasses else 0) for sc in DIAGNOSTIC_SUPERCLASSES}
        subclass_vector = {f"sub_{sub}": (1 if sub in subclasses else 0) for sub in DIAGNOSTIC_SUBCLASSES}

        return {
            "superclasses": superclasses_sorted,
            "subclasses": subclasses_sorted,
            "superclass_vector": superclass_vector,
            "subclass_vector": subclass_vector,
            "form_statements": sorted(form_statements),
            "rhythm_statements": sorted(rhythm_statements),
            "zero_weight_statements": sorted(zero_weight_statements),
            "num_superclasses": len(superclasses_sorted),
            "num_subclasses": len(subclasses_sorted),
        }
