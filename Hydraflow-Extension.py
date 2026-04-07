"""
Storm Sewer Profile Generator and Hydraflow Extension
Parses Hydraflow .stm files and DOT tabulation PDFs to batch-generate
storm sewer deliverable packages as PDF output.

Deliverable structure per system:
  1. Plan view (generated from .stm)
  2. For each return period (sorted):
     a. Other report page(s) if included (inserted from Hydraflow PDF)
     b. DOT tabulation page(s) (inserted from Hydraflow PDF)
     c. Profile plots for each line designation
"""

import re
import os
import sys
import argparse
import tempfile
from collections import defaultdict

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject, DictionaryObject, FloatObject, NameObject, NumberObject
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# =============================================================================
# SECTION 1: STM FILE PARSER
# =============================================================================

def parse_stm(filepath):
    with open(filepath, 'r') as f:
        raw = f.read()
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    rows = raw.split('\n')
    header = {}
    lines = {}
    current_line = {}
    in_line_data = False

    for row in rows:
        row = row.strip()
        if not row:
            continue
        if 'LINE DATA' in row:
            in_line_data = True
            continue
        if row.startswith('"---'):
            if current_line and 'Line No.' in current_line:
                lines[current_line['Line No.']] = current_line
            current_line = {}
            continue
        if '"IDF Curves"' in row or '"Number of Parcel' in row:
            if current_line and 'Line No.' in current_line:
                lines[current_line['Line No.']] = current_line
            break
        # X,Y coordinate fields have comma in key name
        xy_match = re.match(r'^"X,Y Coord (Dn|Up)\s*=?\s*"\s*,\s*(\d+)\s*,\s*(\d+)', row)
        if xy_match and in_line_data:
            current_line[f'X Coord {xy_match.group(1)}'] = int(xy_match.group(2))
            current_line[f'Y Coord {xy_match.group(1)}'] = int(xy_match.group(3))
            continue
        if ',' in row and row.startswith('"'):
            parts = row.split(',', 1)
            key = parts[0].strip().strip('"').strip()
            val = parts[1].strip().strip('"').strip()
            key = re.sub(r'\s*=\s*$', '', key)
            val = _convert_value(val)
            if in_line_data:
                current_line[key] = val
            else:
                header[key] = val
    return header, lines

def _convert_value(val):
    if val == '#TRUE#': return True
    if val == '#FALSE#': return False
    try:
        return int(val)
    except (ValueError, AttributeError):
        pass
    try:
        return float(val)
    except (ValueError, AttributeError):
        pass
    return val


# =============================================================================
# SECTION 1B: STM EDITOR — MERGE LOGIC
# =============================================================================

import csv as _csv_mod

# ---- CSV column → .stm field key mappings --------------------------------

LINES_CSV_MAP = {
    'Line ID':            'Line ID',
    'Junction Type':      'Junction Type',
    'Known Q':            'Known Q',
    'Drainage Area':      'Drainage Area',
    'Runoff Coeff':       'Runoff Coeff.',
    'Inlet Time':         'Inlet Time',
    'Invert Elev Dn':     'Invert Elev Dn',
    'Line Slope':         'Line Slope',
    'Invert Elev Up':     'Invert Elev Up',
    'Rise':               'Rise',
    'Line Type':          'Line Type',
    'Span':               'Span',
    'N-Value':            'N-Value',
    'J-Loss Coeff':       'Junction Loss Coeff',
    'Ground/Rim Elev Dn': 'Ground / Rim Elev Dn',
    'Ground/Rim Elev Up': 'Ground / Rim Elev Up',
    'Inl/Rim Elev Dn':    'Ground / Rim Elev Dn',
    'Inl/Rim Elev Up':    'Ground / Rim Elev Up',
}

INLETS_CSV_MAP = {
    'Inlet ID':           'Inlet ID',
    'Inlet Type':         'Junction Type',
    'Location':           'Inlet Sag',
    'Bypass Line No.':    'Downstream Inlet No.',
    'Curb Length':         'Inlet Length',
    'Curb Tht. Ht.':      'Inlet throat height',
    'Grate Area':         'Grate Opening Area',
    'Grate Width':        'Grate Width',
    'Grate Length':        'Grate Length',
    'Generic Capac.':     'Known Capacity',
    'Rd. Cross Slope Sx': 'Inlet Cross Slope Sx',
    'Gtr. Cross Slope Sw':'Inlet Cross Slope Sw',
    'Local Depression':   'Local Inlet Depression',
    'Gutter Width':       'Gutter Width',
    'Long. Slope':        'Gutter Slope',
    'Gutter N-Value':     'Gutter N-Value',
}

# ---- Enumeration look-ups ------------------------------------------------

JUNCTION_TYPE_CODES = {
    'manhole':                    0,
    'curb':                       1,
    'grate':                      2,
    'combination curb and grate': 3,
    'combination':                3,
    'generic':                    4,
    'drop curb':                  5,
    'drop grate':                 6,
    'open headwall':              7,
    'headwall':                   7,
    'none':                       8,
    'no junction':                8,
}

LOCATION_CODES = {
    'on grade': 0,
    'grade':    0,
    'sag':      1,
}

# ---- .stm fields that require quoted-string values -----------------------

_STM_STRING_FIELDS = {'Line ID', 'Line Type', 'Inlet ID'}

# ---- Numeric formatter matching .stm style -------------------------------

def _fmt_stm_num(val):
    """Format a number for .stm output.
    Integers stay as integers; floats drop the leading zero for |val| < 1."""
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float) and val == int(val) and abs(val) < 1e15:
        return str(int(val))
    s = f"{val}"
    if s.startswith('0.'):
        s = s[1:]
    elif s.startswith('-0.'):
        s = '-' + s[2:]
    return s

def _format_stm_line(key, value):
    """Reconstruct a single .stm line for the given key and value."""
    if key in _STM_STRING_FIELDS:
        return f'"{key} = ","{value}"'
    return f'"{key} = ",{_fmt_stm_num(value)}'

# ---- Raw .stm reader (preserves every line for in-place editing) ---------

def _parse_stm_raw(filepath):
    """Return (raw_lines, line_blocks, line_lengths, line_ending).

    raw_lines    – list[str]  every text line in the file
    line_blocks  – {line_no: {stm_key: line_index}}
    line_lengths – {line_no: float}
    line_ending  – '\r\n' or '\n' (detected from original file)
    """
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()
    line_ending = '\r\n' if b'\r\n' in raw_bytes else '\n'
    raw = raw_bytes.decode('utf-8', errors='replace')
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    raw_lines = raw.split('\n')

    line_blocks  = {}
    line_lengths = {}
    current_no   = None
    current_map  = {}
    in_line_data = False

    for idx, row in enumerate(raw_lines):
        stripped = row.strip()
        if not stripped:
            continue
        if 'LINE DATA' in stripped:
            in_line_data = True
            continue
        if not in_line_data:
            continue

        # ---- separator: save current block ----
        if stripped.startswith('"---'):
            if current_no is not None:
                line_blocks[current_no] = current_map
            current_no  = None
            current_map = {}
            continue

        # ---- end of line data ----
        if '"IDF Curves"' in stripped or '"Number of Parcel' in stripped:
            if current_no is not None:
                line_blocks[current_no] = current_map
            break

        # ---- X,Y coordinates (always preserved — skip) ----
        if '"X,Y Coord' in stripped:
            continue

        # ---- normal key,value line ----
        if ',' in stripped and stripped.startswith('"'):
            parts = stripped.split(',', 1)
            key   = parts[0].strip().strip('"').strip()
            key   = re.sub(r'\s*=\s*$', '', key)
            val_s = parts[1].strip()

            if key == 'Line No.':
                try:
                    current_no = int(val_s)
                except ValueError:
                    pass
                continue

            current_map[key] = idx

            if key == 'Line Length':
                try:
                    line_lengths[current_no] = float(val_s)
                except (ValueError, TypeError):
                    pass

    return raw_lines, line_blocks, line_lengths, line_ending

# ---- CSV parser with auto-detection -------------------------------------

def _strip_csv_units(header):
    """'Known Q (cfs)' → 'Known Q'"""
    return re.sub(r'\s*\(.*?\)\s*$', '', header).strip()

def _parse_editor_csv(filepath):
    """Return (csv_type, headers, defaults, data).

    csv_type – 'lines' or 'inlets'
    headers  – list[str] stripped column names
    defaults – {col_name: str}
    data     – {line_no: {col_name: str}}
    """
    with open(filepath, 'r', newline='', encoding='utf-8-sig') as f:
        rows = list(_csv_mod.reader(f))

    if len(rows) < 4:
        raise ValueError(f"CSV has fewer than 4 rows: {filepath}")

    # Row 0  – "DEFAULT VALUES" label
    # Row 1  – default values
    # Row 2  – headers (with units)
    # Row 3+ – data
    raw_headers = rows[2]
    headers = [_strip_csv_units(h) for h in raw_headers]

    # Auto-detect type by header keywords
    header_set = set(headers)
    if header_set & {'Inlet ID', 'Inlet Type', 'Location',
                     'Curb Length', 'Grate Area'}:
        csv_type = 'inlets'
    else:
        csv_type = 'lines'

    # Defaults from row 1
    defaults = {}
    for i, val in enumerate(rows[1]):
        if i < len(headers) and val and val.strip():
            defaults[headers[i]] = val.strip()

    # Data rows
    data = {}
    for row in rows[3:]:
        if not row or not row[0].strip():
            continue
        try:
            line_no = int(float(row[0].strip()))
        except (ValueError, TypeError):
            continue
        row_data = {}
        for i in range(1, min(len(row), len(headers))):
            row_data[headers[i]] = row[i].strip() if row[i] else ''
        data[line_no] = row_data

    return csv_type, headers, defaults, data

# ---- Classify dropped files ----------------------------------------------

def _classify_editor_files(paths):
    """Sort paths into (stm_path, lines_csv, inlets_csv, unknown).
    CSVs are classified by reading their headers."""
    stm_path   = None
    lines_csv  = None
    inlets_csv = None
    unknown    = []

    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == '.stm':
            stm_path = p
        elif ext == '.csv':
            try:
                csv_type, *_ = _parse_editor_csv(p)
                if csv_type == 'lines':
                    lines_csv = p
                else:
                    inlets_csv = p
            except Exception:
                unknown.append(p)
        else:
            unknown.append(p)

    return stm_path, lines_csv, inlets_csv, unknown

# ---- Core merge ----------------------------------------------------------

def merge_stm(stm_path, lines_csv_path=None, inlets_csv_path=None):
    """Merge CSV data into .stm.  Returns (modified_text, warnings)."""
    raw_lines, line_blocks, line_lengths, line_ending = _parse_stm_raw(stm_path)
    warnings = []

    def _coerce_value(stm_key, raw_val, csv_label, line_no):
        """Convert a raw CSV string to the typed value for .stm output.
        Returns the value or None on failure (warning appended)."""
        # --- Junction Type: text → int ---
        if stm_key == 'Junction Type':
            code = JUNCTION_TYPE_CODES.get(raw_val.lower())
            if code is not None:
                return code
            try:
                return int(raw_val)
            except ValueError:
                warnings.append(
                    f"[{csv_label}] Line {line_no}: "
                    f"unknown Junction/Inlet Type '{raw_val}'"
                )
                return None

        # --- Inlet Sag (Location): text → int ---
        if stm_key == 'Inlet Sag':
            code = LOCATION_CODES.get(raw_val.lower())
            if code is not None:
                return code
            try:
                return int(raw_val)
            except ValueError:
                warnings.append(
                    f"[{csv_label}] Line {line_no}: "
                    f"unknown Location '{raw_val}'"
                )
                return None

        # --- String fields ---
        if stm_key in _STM_STRING_FIELDS:
            return raw_val

        # --- Numeric fields ---
        try:
            if '.' in raw_val:
                return float(raw_val)
            return int(raw_val)
        except ValueError:
            warnings.append(
                f"[{csv_label}] Line {line_no}: "
                f"bad numeric value for {stm_key}: '{raw_val}'"
            )
            return None

    def _apply_csv(csv_path, csv_map, csv_label):
        if csv_path is None:
            return
        csv_type, headers, defaults, data = _parse_editor_csv(csv_path)

        # Build active mapping (CSV col → STM key) for columns in this file
        active = {}
        for h in headers:
            if h == 'Line No.':
                continue
            if h in csv_map:
                active[h] = csv_map[h]
            else:
                warnings.append(f"[{csv_label}] Unrecognized column '{h}' — ignored")

        for line_no, row_data in data.items():
            if line_no not in line_blocks:
                warnings.append(
                    f"[{csv_label}] Line {line_no} not found in .stm — skipped"
                )
                continue

            block = line_blocks[line_no]

            # Track inverts / explicit slope for auto-calc
            inv_dn = None
            inv_up = None
            has_explicit_slope = False

            for csv_col, stm_key in active.items():
                # Resolve value: cell → default → skip
                val_raw = row_data.get(csv_col, '')
                if not val_raw:
                    val_raw = defaults.get(csv_col, '')
                if not val_raw:
                    continue

                val = _coerce_value(stm_key, val_raw, csv_label, line_no)
                if val is None:
                    continue

                # Track for slope auto-calc
                if stm_key == 'Invert Elev Dn':
                    inv_dn = float(val)
                elif stm_key == 'Invert Elev Up':
                    inv_up = float(val)
                elif stm_key == 'Line Slope':
                    has_explicit_slope = True

                # Write into raw lines
                if stm_key in block:
                    raw_lines[block[stm_key]] = _format_stm_line(stm_key, val)

            # ---- Auto-calculate slope (Lines CSV only) ----
            if csv_label == 'Lines':
                if (inv_dn is not None and inv_up is not None
                        and not has_explicit_slope):
                    length = line_lengths.get(line_no)
                    if length and length > 0:
                        slope = (inv_up - inv_dn) / length * 100
                        slope = round(slope, 6)
                        if 'Line Slope' in block:
                            raw_lines[block['Line Slope']] = \
                                _format_stm_line('Line Slope', slope)
                            print(f"  Line {line_no}: slope auto-calculated "
                                  f"→ {slope}%")

    # Apply Lines first, then Inlets (Inlets wins on conflicts)
    _apply_csv(lines_csv_path,  LINES_CSV_MAP,  'Lines')
    _apply_csv(inlets_csv_path, INLETS_CSV_MAP,  'Inlets')

    return line_ending.join(raw_lines), warnings


# ---- Branch tracing and auto-naming --------------------------------------

def _build_line_tree(lines):
    """Build upstream adjacency from parsed .stm lines.
    Returns (children, outfall) where children = {line_no: [upstream_line_nos]}
    sorted ascending, and outfall is the line with Downstream Line No. == 0."""
    children = defaultdict(list)
    outfall  = None
    for line_no, data in lines.items():
        ds = data.get('Downstream Line No.', 0)
        if ds == 0:
            outfall = line_no
        else:
            children[ds].append(line_no)
    for k in children:
        children[k].sort()
    return children, outfall


def _trace_branches(children, outfall):
    """Downstream-to-upstream traversal from outfall.
    Returns list of branches (each a list of line_nos, downstream → upstream).
    Lowest line number at each junction continues the current branch;
    higher numbers are queued as new branches in the order encountered,
    so letters increment as you walk away from the outfall."""
    branches = []
    branch_queue = []  # FIFO queue of starting line numbers

    def _walk_branch(start):
        branch = []
        line_no = start
        while True:
            branch.append(line_no)
            ups = children.get(line_no, [])
            if not ups:
                break
            # Lowest continues this branch; others queued in order
            for u in ups[1:]:
                branch_queue.append(u)
            line_no = ups[0]
        return branch

    if outfall is not None:
        branches.append(_walk_branch(outfall))

    while branch_queue:
        start = branch_queue.pop(0)
        branches.append(_walk_branch(start))

    return branches


def generate_names(lines, designation):
    """Generate Line IDs and Inlet IDs from .stm topology and designation.

    lines       – parsed line dict from parse_stm()
    designation – e.g. "EB-EC"  (start prefix is trunk)

    Returns ({line_no: {'Line ID': str, 'Inlet ID': str}}, warnings)
    """
    warnings = []
    children, outfall = _build_line_tree(lines)
    if outfall is None:
        return {}, ["No outfall found (no line with Downstream Line No. = 0)"]

    branches = _trace_branches(children, outfall)

    # Parse designation — take the first part as trunk prefix
    parts = designation.upper().strip().split('-')
    if len(parts) < 2 or len(parts[0]) < 2:
        return {}, [f"Invalid designation '{designation}' — expected format like 'EB-EC'"]

    base_char    = parts[0][0]               # 'E'
    start_offset = ord(parts[0][1]) - ord('A')  # 'B' → 1

    names = {}
    # Map from downstream line_no → list of (branch_prefix, 1) for dual naming
    branch_junctions = defaultdict(list)

    for branch_idx, branch in enumerate(branches):
        letter = chr(ord('A') + start_offset + branch_idx)
        prefix = f"{base_char}{letter}"

        for pos, line_no in enumerate(branch):
            line_id  = f"LINE {prefix}-{pos + 1}"
            inlet_id = f"{prefix}-{pos + 2}"
            names[line_no] = {'Line ID': line_id, 'Inlet ID': inlet_id}

        # Record branch junction for dual naming
        if branch_idx > 0 and branch:
            ds_no = lines[branch[0]].get('Downstream Line No.', 0)
            if ds_no in names:
                branch_junctions[ds_no].append(f"{prefix}-1")

    # Apply dual naming at branch points
    for ds_line_no, suffixes in branch_junctions.items():
        existing = names[ds_line_no]['Inlet ID']
        for s in suffixes:
            existing += f"/{s}"
        names[ds_line_no]['Inlet ID'] = existing

    return names, warnings


def apply_editor_defaults(stm_path, designation, defaults):
    """Apply auto-naming and UI defaults to a raw .stm file.

    defaults dict keys:
        inlet_type    – int (Junction Type code)
        location      – int (Inlet Sag code)
        runoff_coeff  – float or None
        sx            – float or None
        sw            – float or None
        curb_length   – float or None
        throat_height – float or None
        grate_width   – float or None
        grate_length  – float or None
        grate_area    – float or None

    Returns (modified_text, warnings)
    """
    raw_lines, line_blocks, line_lengths, line_ending = _parse_stm_raw(stm_path)
    _, lines = parse_stm(stm_path)
    warnings = []

    # Generate auto-names
    names, name_warns = generate_names(lines, designation)
    warnings.extend(name_warns)
    if not names:
        return line_ending.join(raw_lines), warnings

    # Find outfall line
    outfall_no = None
    for ln, data in lines.items():
        if data.get('Downstream Line No.', 0) == 0:
            outfall_no = ln
            break

    def _write_field(block, key, value):
        if key in block:
            raw_lines[block[key]] = _format_stm_line(key, value)

    for line_no, block in line_blocks.items():
        # ---- Auto-names (all lines) ----
        if line_no in names:
            _write_field(block, 'Line ID',  names[line_no]['Line ID'])
            _write_field(block, 'Inlet ID', names[line_no]['Inlet ID'])

        # ---- Outfall line: Open Headwall only ----
        if line_no == outfall_no:
            _write_field(block, 'Junction Type', 7)
            continue

        # ---- All other lines: apply defaults ----
        if defaults.get('inlet_type') is not None:
            _write_field(block, 'Junction Type', defaults['inlet_type'])

        if defaults.get('location') is not None:
            _write_field(block, 'Inlet Sag', defaults['location'])

        if defaults.get('runoff_coeff') is not None:
            _write_field(block, 'Runoff Coeff.', defaults['runoff_coeff'])

        if defaults.get('sx') is not None:
            _write_field(block, 'Inlet Cross Slope Sx', defaults['sx'])

        if defaults.get('sw') is not None:
            _write_field(block, 'Inlet Cross Slope Sw', defaults['sw'])

        if defaults.get('curb_length') is not None:
            _write_field(block, 'Inlet Length', defaults['curb_length'])

        if defaults.get('throat_height') is not None:
            _write_field(block, 'Inlet throat height', defaults['throat_height'])

        if defaults.get('grate_width') is not None:
            _write_field(block, 'Grate Width', defaults['grate_width'])

        if defaults.get('grate_length') is not None:
            _write_field(block, 'Grate Length', defaults['grate_length'])

        if defaults.get('grate_area') is not None:
            _write_field(block, 'Grate Opening Area', defaults['grate_area'])

    return line_ending.join(raw_lines), warnings


# =============================================================================
# SECTION 2: DOT TABULATION PDF PARSER
# =============================================================================

def _norm_cell(cell):
    """Normalize a cell string for header matching."""
    return ' '.join((cell or '').split()).strip().upper()

def _find_dot_columns_single(row):
    """Find DOT column indices from a single header row.
    Returns dict with possible keys: hgl_up, hgl_dn, total_flow, capacity, line_id."""
    cells = [_norm_cell(c) for c in row]
    cols = {}
    # HGL Up/Dn
    for j, c in enumerate(cells):
        if 'HGL' in c:
            for k in range(j, min(j + 4, len(cells) - 1)):
                if 'UP' in cells[k] and k + 1 < len(cells) and 'DN' in cells[k + 1]:
                    cols['hgl_up'] = k
                    cols['hgl_dn'] = k + 1
                    break
    # Total Flow — require both TOTAL and FLOW in same cell
    for j, c in enumerate(cells):
        if 'TOTAL' in c and 'FLOW' in c:
            cols['total_flow'] = j
            break
    # Capacity — require CAPACITY or (CAP + FULL) in same cell, exclude CAPTURED
    for j, c in enumerate(cells):
        if 'CAPTURED' in c:
            continue
        if 'CAPACITY' in c or ('CAP' in c and 'FULL' in c):
            cols['capacity'] = j
            break
    # Line ID
    for j, c in enumerate(cells):
        if 'LINE' in c and 'ID' in c:
            cols['line_id'] = j
            break
    return cols

def _find_dot_columns_pair(row1, row2):
    """Find DOT column indices from a two-row header.
    DOT tables split headers across two rows:
      Row 1: ... Total    Cap   ... HGL    EGL    Depth ...
      Row 2: ... flow     full  ... Up  Dn  Up  Dn  Up  Dn ..."""
    c1 = [_norm_cell(c) for c in row1]
    c2 = [_norm_cell(c) for c in row2]
    cols = {}
    # HGL: category in row1, Up/Dn in row2
    for j, c in enumerate(c1):
        if 'HGL' in c:
            for k in range(max(0, j - 1), min(j + 4, len(c2) - 1)):
                if 'UP' in c2[k] and k + 1 < len(c2) and 'DN' in c2[k + 1]:
                    cols['hgl_up'] = k
                    cols['hgl_dn'] = k + 1
                    break
    # Total Flow: "TOTAL" in row1 confirmed by "FLOW" in row2 at same column
    for j, c in enumerate(c1):
        if 'TOTAL' in c and 'FLOW' in c:
            cols['total_flow'] = j
            break
        if 'TOTAL' in c and j < len(c2) and 'FLOW' in c2[j]:
            cols['total_flow'] = j
            break
    # Capacity: "CAP" in row1 confirmed by "FULL" in row2
    for j, c in enumerate(c1):
        if 'CAPTURED' in c:
            continue
        if 'CAPACITY' in c or ('CAP' in c and 'FULL' in c):
            cols['capacity'] = j
            break
        if ('CAP' in c or 'CAPAC' in c) and j < len(c2) and 'FULL' in c2[j]:
            cols['capacity'] = j
            break
    # Line ID
    for j, c in enumerate(c1):
        if 'LINE' in c and 'ID' in c:
            cols['line_id'] = j
            break
    if 'line_id' not in cols:
        for j, c in enumerate(c2):
            if 'LINE' in c and 'ID' in c:
                cols['line_id'] = j
                break
    return cols

def _parse_dot_rows_table(table):
    """Extract per-line data from a pdfplumber table.
    Handles both single-row and two-row (split) header formats.
    Returns {line_no: {'HGL Up', 'HGL Dn', 'Line ID', 'Total Flow', 'Capacity'}}."""
    result = {}
    cols = {}
    data_start = 0

    # Scan for header — try single-row first, then consecutive row pairs
    for i, row in enumerate(table):
        if not row:
            continue
        cols = _find_dot_columns_single(row)
        if cols.get('hgl_up') is not None:
            data_start = i + 1
            break
        if i + 1 < len(table) and table[i + 1]:
            cols = _find_dot_columns_pair(row, table[i + 1])
            if cols.get('hgl_up') is not None:
                data_start = i + 2
                break

    if not cols.get('hgl_up'):
        return result

    hgl_up_col = cols['hgl_up']
    hgl_dn_col = cols['hgl_dn']
    flow_col = cols.get('total_flow')
    cap_col = cols.get('capacity')
    lid_col = cols.get('line_id')

    # Parse data rows
    for row in table[data_start:]:
        if not row:
            continue
        raw_ln = (row[0] or '').strip()
        try:
            line_no = int(raw_ln)
            if line_no <= 0:
                continue
        except (ValueError, AttributeError):
            continue
        # HGL (required)
        try:
            hgl_up = float(row[hgl_up_col]) if row[hgl_up_col] else None
            hgl_dn = float(row[hgl_dn_col]) if row[hgl_dn_col] else None
        except (ValueError, TypeError, IndexError):
            continue
        if hgl_up is None or hgl_dn is None:
            continue
        entry = {'HGL Up': hgl_up, 'HGL Dn': hgl_dn}
        # Line ID — use detected column, or last non-empty cell as fallback
        if lid_col is not None and lid_col < len(row) and row[lid_col]:
            entry['Line ID'] = row[lid_col].strip()
        else:
            for ci in range(len(row) - 1, 0, -1):
                if row[ci] and not row[ci].strip().replace('.','').replace('-','').isdigit():
                    entry['Line ID'] = row[ci].strip()
                    break
        # Total Flow (optional)
        if flow_col is not None:
            try:
                entry['Total Flow'] = float(row[flow_col]) if row[flow_col] else None
            except (ValueError, TypeError, IndexError):
                entry['Total Flow'] = None
        # Capacity (optional)
        if cap_col is not None:
            try:
                entry['Capacity'] = float(row[cap_col]) if row[cap_col] else None
            except (ValueError, TypeError, IndexError):
                entry['Capacity'] = None
        result[line_no] = entry
    return result

def _parse_dot_rows_regex(text):
    """Fallback: extract per-line data via regex.
    Returns {line_no: {'HGL Up', 'HGL Dn', 'Line ID', 'Total Flow', 'Capacity'}}.
    DOT columns: LineNo ToLine Len DrngI DrngT Rnoff AxCI AxCT TcIn TcSys Rain
                 TotalFlow Capacity Vel PipeSize Slope InvUp InvDn HGLUp HGLDn
                 GrdUp GrdDn LineID"""
    result = {}
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        dot_match = re.match(
            r'^(\d+)\s+(\S+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+'
            r'[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+'
            r'([\d.]+)\s+([\d.]+)\s+'
            r'[\d.]+\s+\d+\s+[\d.]+\s+'
            r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+'
            r'[\d.]+\s+[\d.]+\s+(.+)$', line)
        if dot_match:
            line_no = int(dot_match.group(1))
            result[line_no] = {
                'Line ID': dot_match.group(9).strip(),
                'Total Flow': float(dot_match.group(3)),
                'Capacity': float(dot_match.group(4)),
                'HGL Up': float(dot_match.group(7)),
                'HGL Dn': float(dot_match.group(8)),
            }
    return result

def parse_dot_pdf(filepath):
    """Parse DOT tabulation PDF. Returns LIST of dicts, one per return period found.
    Each dict has: project_file, return_period, return_period_num, data, dot_pages."""
    results = []
    current = None
    project_file = None

    with pdfplumber.open(filepath) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text or 'Storm Sewer Tabulation' not in text:
                continue

            # Check for return period on this page
            page_rp = None
            for raw_line in text.split('\n'):
                rp = re.search(r'Return period\s*=\s*(\d+)\s*Yrs', raw_line)
                if rp:
                    page_rp = int(rp.group(1))
                pf = re.search(r'Project [Ff]ile:\s*(.*?\.stm)', raw_line)
                if pf and not project_file:
                    project_file = os.path.splitext(pf.group(1).strip())[0]

            # Start new entry if return period changed
            if page_rp and (current is None or current['return_period_num'] != page_rp):
                current = {
                    'project_file': project_file,
                    'return_period': f"{page_rp} YEAR",
                    'return_period_num': page_rp,
                    'data': {},
                    'dot_pages': [],
                }
                results.append(current)

            if current is None:
                continue

            current['dot_pages'].append(page_idx)

            # Parse data rows via extract_tables(); fall back to regex if needed
            table_settings = {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
            }
            tables = page.extract_tables(table_settings)
            page_data = {}
            if tables:
                for table in tables:
                    parsed = _parse_dot_rows_table(table)
                    page_data.update(parsed)
            if not page_data:
                page_data = _parse_dot_rows_regex(text)
            current['data'].update(page_data)

    # Set project file on all entries
    for r in results:
        if not r['project_file']:
            r['project_file'] = project_file

    return results

def parse_other_report_pdf(filepath):
    """Parse a non-DOT report PDF for project file, return period, and report type.
    Uses the same footer patterns as parse_dot_pdf (Project File / Return period).
    Extracts report type from the first line of text on the first matched page.
    Returns a LIST of dicts, one per return period found.
    Each dict has: project_file, return_period, return_period_num, report_type, pages."""
    results = []
    current = None
    project_file = None
    report_type = None

    with pdfplumber.open(filepath) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue

            # Extract report type from the first line (header) on the first page
            if report_type is None:
                first_line = text.split('\n')[0].strip()
                if first_line:
                    report_type = first_line

            # Look for return period and project file using same patterns as DOT
            page_rp = None
            for raw_line in text.split('\n'):
                rp = re.search(r'Return period\s*=\s*(\d+)\s*Yrs', raw_line)
                if rp:
                    page_rp = int(rp.group(1))
                pf = re.search(r'Project [Ff]ile:\s*(.*?\.stm)', raw_line)
                if pf and not project_file:
                    project_file = os.path.splitext(pf.group(1).strip())[0]

            # Start new entry if return period changed
            if page_rp and (current is None or current['return_period_num'] != page_rp):
                current = {
                    'project_file': project_file,
                    'return_period': f"{page_rp} YEAR",
                    'return_period_num': page_rp,
                    'report_type': report_type,
                    'pages': [],
                }
                results.append(current)

            if current is None:
                continue

            current['pages'].append(page_idx)

    # Set project file on all entries
    for r in results:
        if not r['project_file']:
            r['project_file'] = project_file

    return results

def parse_plan_view_pdf(filepath):
    """Parse a Hydraflow plan view PDF.
    Identifies plan view pages by 'Plan View' in the page text header.
    Extracts project file and line count from footer.
    Returns a dict with project_file, num_lines, path, pages, or None if not a plan view."""
    project_file = None
    num_lines = None
    pages = []

    with pdfplumber.open(filepath) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue

            # Check for Plan View identifier in header
            is_plan = False
            for raw_line in text.split('\n'):
                if re.search(r'Plan\s*View', raw_line, re.IGNORECASE):
                    is_plan = True
                pf = re.search(r'Project [Ff]ile:\s*(.*?\.stm)', raw_line)
                if pf and not project_file:
                    project_file = os.path.splitext(pf.group(1).strip())[0]
                nl = re.search(r'No\.\s*Lines:\s*(\d+)', raw_line)
                if nl and num_lines is None:
                    num_lines = int(nl.group(1))

            if is_plan:
                pages.append(page_idx)

    if not pages:
        return None

    return {
        'project_file': project_file,
        'num_lines': num_lines,
        'pages': pages,
    }

def parse_report_pdf(filepath):
    """Parse custom report PDF (legacy). Returns dict keyed by Line No."""
    results = {}
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text: continue
            for raw_line in text.split('\n'):
                line = raw_line.strip()
                match = re.match(
                    r'^(\d+)\s+([Ll][Ii][Nn][Ee]\s+\S+)\s+'
                    r'([\d.]+\**)\s+([\d.]+\**)\s+([\d.]+\**)\s+([\d.]+\**)\s+'
                    r'([\d.]+)\s+([\d.]+)\s+([\d.]+\**)\s+([\d.]+\**)\s+'
                    r'([\d.]+)\s+([\d.]+)', line)
                if match:
                    results[int(match.group(1))] = {
                        'Line ID': match.group(2).strip(),
                        'HGL Up': float(match.group(3).replace('*','')),
                        'HGL Dn': float(match.group(4).replace('*','')),
                        'EGL Up': float(match.group(5).replace('*','')),
                        'EGL Dn': float(match.group(6).replace('*','')),
                        'Total Runoff': float(match.group(7)),
                        'Vel Ave': float(match.group(8)),
                        'Depth Up': float(match.group(9).replace('*','')),
                        'Depth Dn': float(match.group(10).replace('*','')),
                        'Capacity': float(match.group(11)),
                        'Tc': float(match.group(12)),
                    }
    return results


# =============================================================================
# SECTION 3: PATH TRACING
# =============================================================================

def group_paths_by_prefix(lines):
    pattern = re.compile(r'^[Ll][Ii][Nn][Ee]\s+([A-Za-z][A-Za-z0-9]*)-(\d+)$')
    groups = defaultdict(list)
    for line_no, data in lines.items():
        m = pattern.match(data.get('Line ID', '').strip())
        if m:
            groups[m.group(1).upper()].append((int(m.group(2)), line_no))
    result = {}
    for prefix, entries in sorted(groups.items()):
        entries.sort(key=lambda x: x[0])
        nums = [ln for _, ln in entries]
        subs = [[nums[0]]]
        for j in range(1, len(nums)):
            if lines[nums[j]].get('Downstream Line No.', 0) == nums[j-1]:
                subs[-1].append(nums[j])
            else:
                subs.append([nums[j]])
        if len(subs) == 1:
            result[prefix] = subs[0]
        else:
            for k, sg in enumerate(subs):
                result[f"{prefix}_{chr(97+k)}"] = sg
    return result


# =============================================================================
# SECTION 4: PROFILE DATA ASSEMBLY
# =============================================================================

def assemble_profile_data(path, lines, report_data=None):
    profile = {'reach': [], 'ground': [], 'invert': [], 'crown': [],
                'segments': [], 'line_numbers': path}
    cum = 0.0
    for i, ln in enumerate(path):
        d = lines[ln]
        length = d.get('Line Length', 0)
        inv_dn, inv_up = d.get('Invert Elev Dn', 0), d.get('Invert Elev Up', 0)
        grd_dn, grd_up = d.get('Ground / Rim Elev Dn', 0), d.get('Ground / Rim Elev Up', 0)
        rise = d.get('Rise', 0)
        si = rise * 12
        sl = f"{int(si)} (in)" if si == int(si) else f"{si:.1f} (in)"
        hdn = hup = None
        if report_data and ln in report_data:
            hdn, hup = report_data[ln].get('HGL Dn'), report_data[ln].get('HGL Up')
        if i == 0:
            profile['reach'].append(cum)
            profile['ground'].append(grd_dn)
            profile['invert'].append(inv_dn)
            profile['crown'].append(inv_dn + rise)
        cum += length
        profile['reach'].append(cum)
        profile['ground'].append(grd_up)
        profile['invert'].append(inv_up)
        profile['crown'].append(inv_up + rise)
        profile['segments'].append({'hgl_dn': hdn, 'hgl_up': hup, 'size_label': sl,
            'line_id': d.get('Line ID', f'Line {ln}'), 'line_no': ln, 'rise': rise, 'length': length})
    return profile


# =============================================================================
# SECTION 5: PROFILE PLOTTING
# =============================================================================

COLOR_GROUND = '#228B22'; COLOR_INVERT = '#000080'; COLOR_CROWN = '#000080'
COLOR_HGL = '#CC0000'; COLOR_STRUCTURE = '#000080'
COLOR_GRID = '#C0C0C0'; COLOR_TITLE_RP = '#CC0000'
LW_GND = 1.5; LW_INV = 1.2; LW_CRN = 1.2; LW_HGL = 1.5

def plot_profile(profile, project_name='', return_period='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    else:
        fig = ax.get_figure()
    R, G, I, S = profile['reach'], profile['ground'], profile['invert'], profile['segments']
    np_ = len(S); nn = len(R)
    vg = [g for g in G if g > 0]; vi = [v for v in I if v > 0]
    ae = vg + vi
    for s in S:
        if s['hgl_dn'] is not None: ae.append(s['hgl_dn'])
        if s['hgl_up'] is not None: ae.append(s['hgl_up'])
    if not ae: return fig
    emin, emax = min(ae) - 5.0, max(ae) + 8.0
    rmax = max(R)
    sw = max(3.0, rmax * 0.012); hw = sw / 2.0
    xmax = rmax + sw * 2.0
    ax.set_axisbelow(True)
    ax.grid(True, which='both', color=COLOR_GRID, linewidth=0.5, linestyle='-', zorder=0)
    # Ground
    gr, ge = [], []
    for r, g in zip(R, G):
        if g > 0: gr.append(r); ge.append(g)
    if gr:
        ax.plot(gr, ge, color=COLOR_GROUND, linewidth=LW_GND, zorder=5)
    # Pipes
    for i in range(np_):
        x_start = R[i] + hw
        x_end = R[i+1] - hw
        rise = S[i]['rise']
        cr_start = I[i] + rise
        cr_end   = I[i+1] + rise
        ax.plot([x_start, x_end], [I[i], I[i+1]], color=COLOR_INVERT, linewidth=LW_INV, zorder=4)
        ax.plot([x_start, x_end], [cr_start, cr_end], color=COLOR_CROWN, linewidth=LW_CRN, zorder=4)
        # Pipe barrel hatch (disabled)
        # ax.fill_between(
        #     [x_start, x_end],
        #     [I[i], I[i+1]],
        #     [C[i], C[i+1]],
        #     facecolor='none',
        #     edgecolor=COLOR_INVERT,
        #     alpha=0.3,
        #     hatch='///',
        #     zorder=3,
        # )
    # HGL
    if any(s['hgl_dn'] is not None for s in S):
        for i, s in enumerate(S):
            if s['hgl_dn'] is not None and s['hgl_up'] is not None:
                ax.plot([R[i], R[i+1]], [s['hgl_dn'], s['hgl_up']], color=COLOR_HGL, linewidth=LW_HGL, zorder=6)
            if i < np_ - 1:
                nd = S[i+1]['hgl_dn']
                if s['hgl_up'] is not None and nd is not None:
                    ax.plot([R[i+1], R[i+1]], [s['hgl_up'], nd], color=COLOR_HGL, linewidth=LW_HGL, zorder=6)
    # Structures
    drawn = set()
    for i in range(nn):
        r, inv, grd = R[i], I[i], G[i]
        if grd <= 0 or r in drawn: continue
        drawn.add(r)
        bl = r - hw
        ax.add_patch(patches.Rectangle(
            (bl, inv), sw, grd - inv,
            lw=1.2, ec=COLOR_STRUCTURE, fc='none', zorder=7, clip_on=False,
            # hatch='///', alpha=0.4,  # structure box hatch (disabled)
        ))
        sq = (emax - emin) * 0.012
        ax.add_patch(patches.Rectangle((bl, inv - sq), sw, sq, lw=0.5, ec=COLOR_STRUCTURE, fc=COLOR_STRUCTURE, zorder=8, clip_on=False))
    # Error annotation
    errors = []
    for i in range(nn):
        inv, grd = I[i], G[i]
        if grd > 0 and grd <= inv:
            label = S[i]['line_id'] if i < np_ else S[i-1]['line_id']
            errors.append(f"Ln {label}: Rim ({grd:.2f}) \u2264 Invert ({inv:.2f})")
    if errors:
        msg = '\u26A0  DATA ERROR  \u26A0\n' + '\n'.join(errors)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color='#CC0000',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFFF99',
                          edgecolor='#CC0000', linewidth=3, alpha=0.95),
                zorder=20)
    # Labels
    lo = (emax - emin) * 0.06
    for i, s in enumerate(S):
        rm = (R[i] + R[i+1]) / 2.0; cm = (I[i] + I[i+1]) / 2.0 + s['rise']; ly = cm + lo
        ax.plot([rm, rm], [cm, ly], color='#999999', lw=0.6, zorder=3)
        ax.plot(rm, cm, marker='_', color='#999999', ms=4, zorder=3)
        ax.text(rm, ly, f"Ln: {s['line_no']}\n{s['size_label']}", ha='center', va='bottom', fontsize=9, color='#333333', zorder=10)
    ax.set_xlim(0, xmax); ax.set_ylim(emin, emax)
    axes_width_points = fig.get_figwidth() * (0.95 - 0.08) * 72
    points_per_data_unit = axes_width_points / xmax
    overhang_pad = hw * points_per_data_unit
    ax.set_xlabel('Reach (ft)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Elev. (ft)', fontsize=11, fontweight='bold', labelpad=4 + overhang_pad )
    _auto_ticks(ax, xmax, emin, emax)
    ax.tick_params(axis='y', labelsize=9, pad=10 + overhang_pad); ax.tick_params(axis='x', labelsize=9)
    ax.set_title('Storm Sewer Profile', fontsize=14, fontweight='bold', loc='left', pad=15)
    if project_name: ax.text(0.99, 1.02, f'Proj. file: {project_name}', transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color='#555555')
    if return_period: ax.text(0.99, 1.06, return_period, transform=ax.transAxes, ha='right', va='bottom', fontsize=14, fontweight='bold', color=COLOR_TITLE_RP)
    ax.text(0.99, -0.08, 'Storm Sewer Profile Generator', transform=ax.transAxes, ha='right', va='top', fontsize=7, color='#999999')
    fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.10)
    return fig

def _auto_ticks(ax, rmax, emin, emax):
    rr = rmax
    rs = 10 if rr <= 200 else 25 if rr <= 500 else 50 if rr <= 1000 else 100 if rr <= 2000 else 200
    ax.set_xticks(np.arange(0, rmax + rs, rs))
    er = emax - emin
    es = 1 if er <= 20 else 2 if er <= 50 else 5 if er <= 100 else 10
    ax.set_yticks(np.arange(int(emin / es) * es, emax + es, es))


# =============================================================================
# SECTION 5b: PLAN VIEW
# =============================================================================

def plot_plan_view(lines, project_name=''):
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    nodes = {}; edges = []
    for ln, d in lines.items():
        xd = d.get('X Coord Dn')
        yd = d.get('Y Coord Dn')
        xu = d.get('X Coord Up')
        yu = d.get('Y Coord Up')
        if xd is None or yd is None or xu is None or yu is None:
            continue
        try:
            xd, yd = float(xd), float(yd)
            xu, yu = float(xu), float(yu)
        except (ValueError, TypeError):
            continue
        dk, uk = (xd, yd), (xu, yu)
        if dk not in nodes:
            nodes[dk] = ['Outfall'] if d.get('Downstream Line No.', 0) == 0 else []
        if uk not in nodes: nodes[uk] = []
        iid = d.get('Inlet ID', '')
        if iid and iid not in nodes[uk]: nodes[uk].append(iid)
        edges.append((dk, uk, ln, d.get('Line ID', '')))
    if not edges:
        ax.text(0.5, 0.5, 'No coordinate data in .stm', ha='center', va='center', fontsize=14, transform=ax.transAxes)
        return fig
    ax_vals = [n[0] for n in nodes]; ay_vals = [n[1] for n in nodes]
    xmn, xmx = min(ax_vals), max(ax_vals); ymn, ymx = min(ay_vals), max(ay_vals)
    xp = max((xmx - xmn) * 0.1, 50); yp = max((ymx - ymn) * 0.1, 50)
    for (x1,y1),(x2,y2),ln,lid in edges:
        ax.plot([x1,x2],[y1,y2], color='#4AA8C0', lw=1.2, zorder=2)
    # Build neighbor map for label offset direction
    neighbors = defaultdict(list)
    for (x1,y1),(x2,y2),ln,lid in edges:
        neighbors[(x1,y1)].append((x2,y2))
        neighbors[(x2,y2)].append((x1,y1))
    # Pipe number labels at midpoint
    for (x1,y1),(x2,y2),ln,lid in edges:
        ax.text((x1+x2)/2, (y1+y2)/2, str(ln), fontsize=7,
                ha='center', va='center', color='#555555', zorder=4,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.6))
    ms = max(4, min(8, (xmx - xmn) / 200))
    for (x,y), labels in nodes.items():
        # Compute offset direction away from neighbors
        nx = [n[0] for n in neighbors[(x,y)]]
        ny = [n[1] for n in neighbors[(x,y)]]
        if nx:
            dx = x - sum(nx)/len(nx); dy = y - sum(ny)/len(ny)
            mag = max((dx**2+dy**2)**0.5, 1)
            ox = (dx/mag) * xp * 0.28; oy = (dy/mag) * yp * 0.28
        else:
            ox = xp * 0.10; oy = 0
        if 'Outfall' in labels:
            ax.plot(x, y, 'o', color='black', ms=ms+2, zorder=5)
            ax.annotate('Outfall', xy=(x, y), xytext=(x+ox, y+oy),
                        fontsize=7, ha='center', va='center', color='black', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85),
                        arrowprops=dict(arrowstyle='-', color='#888888', lw=0.8))
        else:
            ax.plot(x, y, 's', color='#2a5d9f', ms=ms, zorder=5)
            lt = '/'.join([l for l in labels if l])
            if lt:
                ax.annotate(lt, xy=(x, y), xytext=(x+ox, y+oy),
                            fontsize=6, ha='center', va='center', color='#333333', zorder=6,
                            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85),
                            arrowprops=dict(arrowstyle='-', color='#888888', lw=0.8))
    # Match frame size to profile plots by filling the axes box exactly.
    # Instead of set_aspect('equal') (which shrinks the axes), expand whichever
    # data dimension is too narrow so the coordinate ratio equals the axes ratio.
    _l, _r, _t, _b = 0.08, 0.95, 0.88, 0.10
    axes_aspect = (_r - _l) * 11.0 / ((_t - _b) * 8.5)  # width/height in inches
    data_w = (xmx + xp) - (xmn - xp)
    data_h = (ymx + yp) - (ymn - yp)
    if data_w / data_h < axes_aspect:
        extra = (axes_aspect * data_h - data_w) / 2
        ax.set_xlim(xmn - xp - extra, xmx + xp + extra)
        ax.set_ylim(ymn - yp, ymx + yp)
    else:
        extra = (data_w / axes_aspect - data_h) / 2
        ax.set_xlim(xmn - xp, xmx + xp)
        ax.set_ylim(ymn - yp - extra, ymx + yp + extra)
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    ax.set_title(f'Plan View\n{project_name}', fontsize=14, fontweight='bold', loc='left', pad=10)
    for sp in ax.spines.values(): sp.set_linewidth(1.5)
    fig.subplots_adjust(left=_l, right=_r, top=_t, bottom=_b)
    return fig


# =============================================================================
# SECTION 6: DELIVERABLE ASSEMBLY
# =============================================================================

def classify_files(file_paths):
    stm_files = {}; dot_files = []; other_files = []; plan_files = []
    for fp in file_paths:
        ext = os.path.splitext(fp)[1].lower()
        if ext == '.stm':
            stm_files[os.path.splitext(os.path.basename(fp))[0]] = fp
        elif ext == '.pdf':
            # Try DOT first
            entries = parse_dot_pdf(fp)
            is_dot = False
            for info in entries:
                if info['data'] and info['project_file'] and info['return_period']:
                    dot_files.append({**info, 'path': fp})
                    print(f"  DOT: {os.path.basename(fp)} -> {info['project_file']} / {info['return_period']} ({len(info['data'])} lines)")
                    is_dot = True
            if not is_dot:
                # Try as plan view
                plan_info = parse_plan_view_pdf(fp)
                if plan_info and plan_info['project_file']:
                    plan_files.append({**plan_info, 'path': fp})
                    print(f"  Plan View: {os.path.basename(fp)} -> {plan_info['project_file']} ({len(plan_info['pages'])} pages)")
                else:
                    # Try as other report (non-DOT)
                    other_entries = parse_other_report_pdf(fp)
                    for info in other_entries:
                        if info['project_file'] and info['return_period']:
                            other_files.append({**info, 'path': fp})
                            print(f"  Report: {os.path.basename(fp)} -> {info['project_file']} / {info['return_period']} ({len(info['pages'])} pages)")
                    if not other_entries:
                        print(f"  Skipped: {os.path.basename(fp)} (no DOT tabulations, plan view, or report data found)")
    return stm_files, dot_files, other_files, plan_files

def _fig_to_temp_pdf(fig, prefix='temp'):
    fd, path = tempfile.mkstemp(suffix='.pdf', prefix=f'ssp_{prefix}_')
    os.close(fd)
    fig.savefig(path, dpi=150, format='pdf')
    return path

def _append_pdf(writer, path, page_nums=None):
    reader = PdfReader(path)
    if page_nums is None:
        for p in reader.pages: writer.add_page(p)
    else:
        for pn in page_nums: writer.add_page(reader.pages[pn])

def generate_deliverable(stm_files, dot_files, output_path='deliverable.pdf',
                         other_files=None, plan_files=None,
                         missing_plan_cb=None, warning_cb=None):
    """Generate the full deliverable PDF.

    Args:
        stm_files: dict of {system_name: stm_path}
        dot_files: list of DOT report dicts
        output_path: output PDF path
        other_files: list of other report dicts (optional)
        plan_files: list of plan view dicts (optional)
        missing_plan_cb: callback(system_name) -> bool; called when a system has
                         no Hydraflow plan view. Return True to continue, False to abort.
        warning_cb: callback(message) -> bool; called for warnings (e.g. duplicate
                    reports, plan view mismatch). Return True to continue, False to abort.

    Returns:
        True if the deliverable was written successfully, False if aborted.
    """
    # Default warning_cb: print and continue
    if warning_cb is None:
        def warning_cb(msg):
            print(f"  WARNING: {msg}")
            return True

    writer = PdfWriter()

    # --- Duplicate report detection ---
    # Check DOT reports: keyed by (project_file, return_period_num)
    dot_seen = {}
    deduped_dot = []
    for d in dot_files:
        key = (d['project_file'], d['return_period_num'], 'Storm Sewer Tabulation')
        if key in dot_seen:
            msg = (f"Duplicate DOT report for {d['project_file']} / {d['return_period']}\n"
                   f"  File 1: {os.path.basename(dot_seen[key])}\n"
                   f"  File 2: {os.path.basename(d['path'])}\n"
                   f"Skip the duplicate?")
            if not warning_cb(msg):
                print("  Aborted by user."); return False
            # User chose to skip — don't add to deduped list
            print(f"  Skipping duplicate: {os.path.basename(d['path'])} (DOT)")
            continue
        dot_seen[key] = d['path']
        deduped_dot.append(d)
    # Rebuild dot_by_proj from deduped list
    dot_by_proj = defaultdict(list)
    for d in deduped_dot: dot_by_proj[d['project_file']].append(d)
    for pf in dot_by_proj: dot_by_proj[pf].sort(key=lambda x: x['return_period_num'])

    # Check other reports: keyed by (project_file, return_period_num, report_type)
    other_by_proj_rp = defaultdict(list)
    other_seen = {}
    if other_files:
        for o in other_files:
            rt = o.get('report_type', 'Unknown')
            key = (o['project_file'], o['return_period_num'], rt)
            if key in other_seen:
                msg = (f"Duplicate report: '{rt}' for {o['project_file']} / {o['return_period']}\n"
                       f"  File 1: {os.path.basename(other_seen[key])}\n"
                       f"  File 2: {os.path.basename(o['path'])}\n"
                       f"Skip the duplicate?")
                if not warning_cb(msg):
                    print("  Aborted by user."); return False
                # User chose to skip — don't add
                print(f"  Skipping duplicate: {os.path.basename(o['path'])} ({rt})")
                continue
            other_seen[key] = o['path']
            key2 = (o['project_file'], o['return_period_num'])
            other_by_proj_rp[key2].append(o)

    # Group plan views by project file
    plan_by_proj = {}
    if plan_files:
        for pv in plan_files:
            plan_by_proj[pv['project_file']] = pv

    systems = []
    for sn, sp in sorted(stm_files.items()):
        systems.append((sn, sp, dot_by_proj.get(sn, [])))
    if not systems:
        print("ERROR: No systems to process."); return False

    pc = 0
    for sn, sp, dots in systems:
        print(f"\n{'='*60}\nSystem: {sn}\n{'='*60}")
        header, lines = parse_stm(sp)
        stm_line_count = len(lines)
        print(f"  {stm_line_count} lines")
        pgroups = group_paths_by_prefix(lines)
        print(f"  Designations: {sorted(pgroups.keys())}")

        # Plan view — use Hydraflow plan view if available
        if sn in plan_by_proj:
            pv = plan_by_proj[sn]
            # Validate line count against STM
            pv_lines = pv.get('num_lines')
            if pv_lines is not None and pv_lines != stm_line_count:
                msg = (f"Plan view line count mismatch for {sn}:\n"
                       f"  Hydraflow plan view: {pv_lines} lines\n"
                       f"  STM file: {stm_line_count} lines\n"
                       f"Continue anyway?")
                if not warning_cb(msg):
                    print("  Aborted by user."); return False
            reader = PdfReader(pv['path'])
            for pg_idx in pv['pages']:
                writer.add_page(reader.pages[pg_idx]); pc += 1
                print(f"  Page {pc}: Plan view (Hydraflow)")
        else:
            print(f"  WARNING: No Hydraflow plan view found for {sn}")
            if missing_plan_cb:
                if not missing_plan_cb(sn):
                    print("  Aborted by user."); return False
            # No plan view included for this system

        if dots:
            for di in dots:
                rl = di['return_period']; rd = di['data']; dp = di['path']
                rp_num = di['return_period_num']
                # Other report pages (before DOT)
                for oi in other_by_proj_rp.get((sn, rp_num), []):
                    reader = PdfReader(oi['path'])
                    for pg_idx in oi['pages']:
                        writer.add_page(reader.pages[pg_idx]); pc += 1
                        print(f"  Page {pc}: Report - {os.path.basename(oi['path'])} - {rl}")
                # DOT pages
                for dpi in di.get('dot_pages', []):
                    _append_pdf(writer, dp, [dpi]); pc += 1
                    print(f"  Page {pc}: DOT - {rl}")
                # Profiles
                for pfx, path in sorted(pgroups.items()):
                    prof = assemble_profile_data(path, lines, rd)
                    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
                    plot_profile(prof, sn, rl, ax=ax)
                    tp = _fig_to_temp_pdf(fig, f'p_{pfx}'); plt.close(fig)
                    _append_pdf(writer, tp); os.unlink(tp); pc += 1
                    fid = lines[path[0]].get('Line ID','')
                    lid = lines[path[-1]].get('Line ID','')
                    print(f"  Page {pc}: {fid} to {lid} - {rl}")
        else:
            for pfx, path in sorted(pgroups.items()):
                prof = assemble_profile_data(path, lines, None)
                fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
                plot_profile(prof, sn, '', ax=ax)
                tp = _fig_to_temp_pdf(fig, f'p_{pfx}'); plt.close(fig)
                _append_pdf(writer, tp); os.unlink(tp); pc += 1
                print(f"  Page {pc}: {pfx} (geometry only)")

    with open(output_path, 'wb') as f: writer.write(f)
    print(f"\nDone! {pc} pages written to {output_path}")
    return True

def compile_reports(dot_files, output_path='reports.pdf', other_files=None):
    """Compile report PDFs only (no STM/profiles required).
    Page order per project file / return period: other reports -> DOT pages."""
    writer = PdfWriter()
    # Group DOT files by project, sorted by return period
    dot_by_proj = defaultdict(list)
    for d in dot_files:
        dot_by_proj[d['project_file']].append(d)
    for pf in dot_by_proj:
        dot_by_proj[pf].sort(key=lambda x: x['return_period_num'])
    # Group other reports by (project_file, return_period_num)
    other_by_proj_rp = defaultdict(list)
    if other_files:
        for o in other_files:
            key = (o['project_file'], o['return_period_num'])
            other_by_proj_rp[key].append(o)
    # Collect all project files from both DOT and other reports
    all_projects = sorted(set(
        list(dot_by_proj.keys()) +
        [k[0] for k in other_by_proj_rp.keys()]
    ))
    if not all_projects:
        print("ERROR: No reports to compile."); return
    # Collect all return periods per project (from both sources)
    rp_by_proj = defaultdict(set)
    for pf in dot_by_proj:
        for d in dot_by_proj[pf]:
            rp_by_proj[pf].add(d['return_period_num'])
    for (pf, rp_num) in other_by_proj_rp:
        rp_by_proj[pf].add(rp_num)
    pc = 0
    for proj in all_projects:
        print(f"\n{'='*60}\nProject: {proj}\n{'='*60}")
        for rp_num in sorted(rp_by_proj[proj]):
            rl = f"{rp_num} YEAR"
            # Other report pages
            for oi in other_by_proj_rp.get((proj, rp_num), []):
                reader = PdfReader(oi['path'])
                for pg_idx in oi['pages']:
                    writer.add_page(reader.pages[pg_idx]); pc += 1
                    print(f"  Page {pc}: Report - {os.path.basename(oi['path'])} - {rl}")
            # DOT pages
            for di in dot_by_proj.get(proj, []):
                if di['return_period_num'] != rp_num:
                    continue
                for dpi in di.get('dot_pages', []):
                    _append_pdf(writer, di['path'], [dpi]); pc += 1
                    print(f"  Page {pc}: DOT - {rl}")
    if pc == 0:
        print("ERROR: No report pages found."); return
    with open(output_path, 'wb') as f: writer.write(f)
    print(f"\nDone! {pc} pages written to {output_path}")

# Legacy single-system
def generate_profiles_pdf(stm_path, report_paths=None, output_path='profiles.pdf', manual_paths=None, debug=False):
    header, lines = parse_stm(stm_path)
    pn = os.path.basename(stm_path)
    print(f"Parsing STM: {pn} ({len(lines)} lines)")
    reports = {}
    if report_paths:
        for rl, rp in report_paths.items():
            reports[rl] = parse_report_pdf(rp)
            print(f"  Report {rl}: {len(reports[rl])} lines")
    pp = []
    if manual_paths:
        for mp in manual_paths:
            p = []; id2n = {d.get('Line ID','').strip().upper(): ln for ln, d in lines.items()}
            for lid in mp:
                n = id2n.get(lid.strip().upper())
                if n: p.append(n)
            if p: pp.append(('manual', p))
    else:
        pg = group_paths_by_prefix(lines)
        for pfx, path in pg.items(): pp.append((pfx, path))
    if not pp: print("ERROR: No paths."); return
    with PdfPages(output_path) as pdf:
        pc = 0
        for pl, path in pp:
            if reports:
                for rl, rd in reports.items():
                    prof = assemble_profile_data(path, lines, rd)
                    if debug: _print_debug_table(prof, pl, rl)
                    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
                    plot_profile(prof, pn, rl, ax=ax)
                    pdf.savefig(fig, dpi=150); plt.close(fig); pc += 1
            else:
                prof = assemble_profile_data(path, lines, None)
                if debug: _print_debug_table(prof, pl, 'geometry')
                fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
                plot_profile(prof, pn, '', ax=ax)
                pdf.savefig(fig, dpi=150); plt.close(fig); pc += 1
    print(f"Done! {pc} pages -> {output_path}")

def _print_debug_table(profile, pl, rl):
    R, C, S = profile['reach'], profile['crown'], profile['segments']
    print(f"\n{'='*80}\nDEBUG | {pl} | {rl}\n{'='*80}")
    print(f"{'Line ID':<20} {'StaDn':>8} {'StaUp':>8} {'CrnDn':>9} {'CrnUp':>9} {'HGLDn':>8} {'HGLUp':>8}  Status")
    for i, s in enumerate(S):
        cd, cu = C[i], C[i+1]; hd, hu = s['hgl_dn'], s['hgl_up']
        ds = f"{hd:8.2f}" if hd else f"{'--':>8}"; us = f"{hu:8.2f}" if hu else f"{'--':>8}"
        sc = (hd and hd > cd) or (hu and hu > cu)
        print(f"{s['line_id']:<20} {R[i]:8.1f} {R[i+1]:8.1f} {cd:9.2f} {cu:9.2f} {ds} {us}  {'SURCHARGED' if sc else 'ok'}")


# =============================================================================
# SECTION 6b: SYSTEM CHECK
# =============================================================================

def _add_highlight_rect(page, x1, y1, x2, y2):
    """Add a translucent red rectangle annotation to a pypdf page."""
    annot = DictionaryObject({
        NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/Square"),
        NameObject("/Rect"): ArrayObject([
            FloatObject(x1), FloatObject(y1),
            FloatObject(x2), FloatObject(y2),
        ]),
        NameObject("/IC"): ArrayObject([
            FloatObject(0.929), FloatObject(0.631), FloatObject(0.631),
        ]),
        NameObject("/C"): ArrayObject([
            FloatObject(1.0), FloatObject(0.0), FloatObject(0.0),
        ]),
        NameObject("/CA"): FloatObject(0.4),
        NameObject("/BS"): DictionaryObject({
            NameObject("/Type"): NameObject("/Border"),
            NameObject("/W"): FloatObject(0),
        }),
        NameObject("/F"): NumberObject(4),
    })
    if NameObject("/Annots") not in page:
        page[NameObject("/Annots")] = ArrayObject()
    page[NameObject("/Annots")].append(annot)

def _highlight_dot_rows(writer_page, plumber_page, failing_lines):
    """Add red highlight rectangles to rows for lines where Total Flow > Capacity."""
    words = plumber_page.extract_words()
    if not words:
        return
    page_height = float(plumber_page.height)
    page_width = float(plumber_page.width)

    # Group words into rows by vertical position
    rows = defaultdict(list)
    for w in words:
        row_key = round(w['top'] / 2) * 2
        rows[row_key].append(w)

    for row_key in sorted(rows):
        row_words = rows[row_key]
        leftmost = min(row_words, key=lambda w: w['x0'])
        # Only consider words near the left margin (first column)
        if leftmost['x0'] > page_width * 0.08:
            continue
        try:
            ln = int(leftmost['text'].strip())
        except (ValueError, AttributeError):
            continue
        if ln not in failing_lines:
            continue
        # Row bounds with padding
        top = min(w['top'] for w in row_words) - 1
        bottom = max(w['bottom'] for w in row_words) + 1
        # Convert pdfplumber coords (top-down) to PDF coords (bottom-up)
        pdf_y1 = page_height - bottom
        pdf_y2 = page_height - top
        _add_highlight_rect(writer_page, 0, pdf_y1, page_width, pdf_y2)

def _create_check_summary_pages(all_failures):
    """Create matplotlib summary pages listing all capacity exceedances.
    Returns list of figures (multiple pages if >30 rows)."""
    figures = []
    max_rows = 30

    if not all_failures:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.text(0.5, 0.55, 'System Check \u2014 Summary',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=16, fontweight='bold')
        ax.text(0.5, 0.45, 'All lines OK \u2014 No lines exceed capacity.',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='#228B22')
        ax.text(0.99, 0.01, 'Capacity check',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7, color='#999999')
        fig.subplots_adjust(left=0.05, right=0.98, top=1.0, bottom=0.05)
        figures.append(fig)
        return figures

    col_labels = ['File', 'Return Period', 'Line No', 'Line ID',
                  'Total Flow', 'Capacity', 'Flow/Cap']
    all_rows = []
    for f in all_failures:
        ratio = f['total_flow'] / f['capacity'] if f['capacity'] > 0 else float('inf')
        all_rows.append([
            f['file'], f['return_period'], str(f['line_no']),
            f.get('line_id', ''),
            f'{f["total_flow"]:.2f}', f'{f["capacity"]:.2f}',
            f'{ratio:.2f}',
        ])

    n_pages = max(1, (len(all_rows) + max_rows - 1) // max_rows)
    for pg in range(n_pages):
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        start = pg * max_rows
        end = min(start + max_rows, len(all_rows))
        page_rows = all_rows[start:end]

        page_label = f'  (Page {pg+1} of {n_pages})' if n_pages > 1 else ''
        ax.text(0.0, 1.0, f'System Check \u2014 Summary{page_label}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=14, fontweight='bold')
        count_text = f'{len(all_failures)} line{"s" if len(all_failures) != 1 else ""} exceed capacity'
        ax.text(0.0, 0.97, count_text, transform=ax.transAxes, ha='left', va='top',
                fontsize=10, color='#CC0000')

        tbl = ax.table(
            cellText=page_rows, colLabels=col_labels,
            loc='upper center', cellLoc='center',
            bbox=[0.0, 0.02, 1.0, 0.92],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        # Header style
        for j in range(len(col_labels)):
            cell = tbl[0, j]
            cell.set_facecolor('#2a5d9f')
            cell.set_text_props(color='white', fontweight='bold')
        # Row shading by severity
        for i in range(len(page_rows)):
            idx = start + i
            ratio = all_failures[idx]['total_flow'] / all_failures[idx]['capacity'] if all_failures[idx]['capacity'] > 0 else float('inf')
            if ratio >= 1.5:
                bg = '#FFFFFF'
            elif ratio >= 1.2:
                bg = '#FFFFFF'
            else:
                bg = '#FFFFFF'
            for j in range(len(col_labels)):
                tbl[i + 1, j].set_facecolor(bg)

        ax.text(0.99, 0.005, 'System Capacity Check',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7, color='#999999')
        fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.03)
        figures.append(fig)

    return figures
    
def generate_system_check(dot_paths, output_path='system_check.pdf'):
    """Generate a system check PDF highlighting lines where Total Flow > Capacity.
    Produces a summary page followed by annotated copies of all DOT pages,
    sorted by system name then return period."""
    writer = PdfWriter()
    all_failures = []

    # Parse and classify all PDFs, then sort by system name and return period
    print("Classifying files...")
    _, dot_files, _, _ = classify_files(dot_paths)
    if not dot_files:
        print("No DOT tabulations found in any of the provided files.")
        return

    dot_files.sort(key=lambda e: (
        (e['project_file'] or '').lower(),
        e['return_period_num'],
    ))

    # Group entries by source PDF path so each file is only opened once
    entries_by_path = defaultdict(list)
    for entry in dot_files:
        entries_by_path[entry['path']].append(entry)

    for entry in dot_files:
        pf = entry['project_file'] or os.path.splitext(os.path.basename(entry['path']))[0]
        rp = entry['return_period']
        data = entry['data']
        dot_pages = entry['dot_pages']
        fp = entry['path']

        print(f"\n{pf} — {rp}")

        # Identify failing lines
        failing_lines = set()
        for ln, d in data.items():
            tf = d.get('Total Flow')
            cap = d.get('Capacity')
            if tf is not None and cap is not None and tf > cap:
                failing_lines.add(ln)
                all_failures.append({
                    'file': pf, 'return_period': rp,
                    'line_no': ln, 'line_id': d.get('Line ID', ''),
                    'total_flow': tf, 'capacity': cap,
                })

        n_fail = len(failing_lines)
        n_total = len(data)
        print(f"  {n_fail} of {n_total} lines exceed capacity")

        # Copy DOT pages and overlay highlights
        with pdfplumber.open(fp) as plumber_pdf:
            reader = PdfReader(fp)
            for page_idx in dot_pages:
                writer.add_page(reader.pages[page_idx])
                if failing_lines:
                    _highlight_dot_rows(
                        writer.pages[-1],
                        plumber_pdf.pages[page_idx],
                        failing_lines,
                    )

    # Build summary page(s) — failures are already in sorted order
    summary_figs = _create_check_summary_pages(all_failures)

    # Insert summary at front of output
    final = PdfWriter()
    for fig in summary_figs:
        tp = _fig_to_temp_pdf(fig, 'check_summary')
        plt.close(fig)
        summary_reader = PdfReader(tp)
        for p in summary_reader.pages:
            final.add_page(p)
        os.unlink(tp)
    for p in writer.pages:
        final.add_page(p)

    with open(output_path, 'wb') as f:
        final.write(f)
    print(f"\nSystem check complete: {len(all_failures)} exceedances found")
    print(f"{len(final.pages)} pages written to {output_path}")


# =============================================================================
# SECTION 7: CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Storm Sewer Profile Generator')
    parser.add_argument('stm', nargs='?', default=None)
    parser.add_argument('--files', '-f', nargs='+')
    parser.add_argument('--report', '-r', action='append', default=[])
    parser.add_argument('--output', '-o', default='deliverable.pdf')
    parser.add_argument('--path', '-p', action='append', default=[])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()
    if args.gui or (args.stm is None and not args.files and not args.report):
        launch_gui(); return
    if args.files:
        sf, df, of, pf = classify_files(args.files)
        if not sf: print("ERROR: No .stm files."); return
        def _cli_missing_plan(sn):
            resp = input(f"WARNING: No Hydraflow plan view for '{sn}'. Continue? [y/N] ")
            return resp.strip().lower() in ('y', 'yes')
        def _cli_warning(msg):
            resp = input(f"WARNING: {msg} [y/N] ")
            return resp.strip().lower() in ('y', 'yes')
        success = generate_deliverable(
            sf, df, args.output, of, pf,
            missing_plan_cb=_cli_missing_plan,
            warning_cb=_cli_warning)
        if not success:
            print("Generation cancelled.")
        return
    if not args.stm: parser.error('Provide .stm or --files')
    rp = {}
    for r in args.report:
        if ':' in r: l, f = r.split(':', 1); rp[l.strip()] = f.strip()
    mp = [[x.strip() for x in p.split(',')] for p in args.path] if args.path else None
    generate_profiles_pdf(args.stm, rp or None, args.output, mp, args.debug)


# =============================================================================
# SECTION 8: GUI
# =============================================================================

def launch_gui():
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
    from tkinterdnd2 import TkinterDnD, DND_FILES
    import threading
    import re as _re

    # Subclass that wires TkinterDnD into the CustomTkinter window
    class DnDCTk(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.TkdndVersion = TkinterDnD._require(self)

    # -------------------------------------------------------------------------
    # Constants
    # -------------------------------------------------------------------------
    SIDEBAR_BG      = "#152641"
    SIDEBAR_WIDTH   = 195
    NAV_ACTIVE_BG   = "#223d63"
    NAV_FG_MUTED    = "#7a9abf"
    NAV_FG_ACTIVE   = "#ffffff"
    COLOR_GENERATE  = "#1a3a6b"
    COLOR_COMPILE   = "#1e5d3a"
    COLOR_CHECK     = "#7c2020"
    COLOR_EDITOR    = "#5c4a8a"
    FONT_NAV        = ("Segoe UI", 12)
    FONT_LABEL      = ("Segoe UI", 10, "bold")
    FONT_BODY       = ("Segoe UI", 10)
    FONT_MONO       = ("Consolas", 9)

    # -------------------------------------------------------------------------
    # App setup
    # -------------------------------------------------------------------------
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    root = DnDCTk()
    root.title("Hydraflow Extension")
    root.geometry("860x640")
    root.minsize(720, 520)

    # Two columns: sidebar | main
    root.grid_columnconfigure(0, weight=0, minsize=SIDEBAR_WIDTH)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)

    # -------------------------------------------------------------------------
    # Log redirector — routes print() output to a CTkTextbox
    # -------------------------------------------------------------------------
    class LogRedirector:
        def __init__(self, widget):
            self.widget = widget

        def write(self, text):
            self.widget.configure(state="normal")
            self.widget.insert("end", text)
            self.widget.see("end")
            self.widget.configure(state="disabled")

        def flush(self):
            pass

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    sidebar = ctk.CTkFrame(root, fg_color=SIDEBAR_BG, corner_radius=0, width=SIDEBAR_WIDTH)
    sidebar.grid(row=0, column=0, sticky="nsew")
    sidebar.grid_propagate(False)
    sidebar.grid_columnconfigure(0, weight=1)
    sidebar.grid_rowconfigure(7, weight=1)  # pushes bottom items down

    # App title
    title_outer = ctk.CTkFrame(sidebar, fg_color="transparent")
    title_outer.grid(row=0, column=0, sticky="ew", padx=16, pady=(20, 16))

    ctk.CTkLabel(
        title_outer,
        text="Hydraflow Extension",
        font=("Segoe UI", 13, "bold"),
        text_color="#ffffff",
        justify="left",
        anchor="w",
    ).pack(anchor="w")

    ctk.CTkLabel(
        title_outer,
        text="v2.0",
        font=("Segoe UI", 10),
        text_color="#4a6a8a",
        justify="left",
        anchor="w",
    ).pack(anchor="w")

    # Divider
    ctk.CTkFrame(sidebar, fg_color="#243f63", height=1, corner_radius=0).grid(
        row=1, column=0, sticky="ew"
    )

    # Section label
    ctk.CTkLabel(
        sidebar,
        text="WORKSPACE",
        font=("Segoe UI", 9),
        text_color="#3a5a7a",
        anchor="w",
    ).grid(row=2, column=0, sticky="ew", padx=16, pady=(14, 4))

    # Nav buttons
    NAV_ITEMS = [
        ("Generate Profiles",   "generate"),
        ("Compile Reports",     "compile"),
        ("Line Capacity Check", "check"),
        ("STM Editor",          "editor"),
    ]
    nav_buttons = {}
    pages       = {}

    def switch_page(page_id):
        for pid, btn in nav_buttons.items():
            if pid == page_id:
                btn.configure(fg_color=NAV_ACTIVE_BG, text_color=NAV_FG_ACTIVE)
            else:
                btn.configure(fg_color="transparent", text_color=NAV_FG_MUTED)
        for pid, frame in pages.items():
            if pid == page_id:
                frame.grid()
            else:
                frame.grid_remove()

    for row_idx, (label, page_id) in enumerate(NAV_ITEMS):
        btn = ctk.CTkButton(
            sidebar,
            text=label,
            font=FONT_NAV,
            fg_color="transparent",
            text_color=NAV_FG_MUTED,
            hover_color="#1e3a5f",
            anchor="w",
            height=38,
            corner_radius=6,
            command=lambda pid=page_id: switch_page(pid),
        )
        btn.grid(row=3 + row_idx, column=0, sticky="ew", padx=8, pady=2)
        nav_buttons[page_id] = btn

    # =========================================================================
    # MAIN AREA
    # =========================================================================
    main_area = ctk.CTkFrame(root, fg_color="transparent", corner_radius=0)
    main_area.grid(row=0, column=1, sticky="nsew")
    main_area.grid_rowconfigure(0, weight=1)
    main_area.grid_columnconfigure(0, weight=1)

    # -------------------------------------------------------------------------
    # Helper: darken a hex color for hover states
    # -------------------------------------------------------------------------
    def _darken(hex_color):
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
        return "#{:02x}{:02x}{:02x}".format(
            max(0, int(r * 0.80)),
            max(0, int(g * 0.80)),
            max(0, int(b * 0.80)),
        )

    # -------------------------------------------------------------------------
    # Helper: build a standard page
    #   Returns (page_frame, file_list, output_var, log_widget, action_btn)
    # -------------------------------------------------------------------------
    def build_page(parent, action_label, action_color, action_fn, default_output_name,
                   accepted_extensions=None, sort_alpha=False):

        file_list  = []

        # -- Outer frame --
        frame = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=0)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=2)  # files card  ~2/3
        frame.grid_rowconfigure(2, weight=1)  # log card    ~1/3

        # ---- Files card ----
        files_card = ctk.CTkFrame(frame, corner_radius=10)
        files_card.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 6))
        files_card.grid_columnconfigure(0, weight=1)
        files_card.grid_rowconfigure(1, weight=1)

        # Card header row
        card_header = ctk.CTkFrame(files_card, fg_color="transparent")
        card_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        card_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card_header,
            text="PROJECT FILES",
            font=("Segoe UI", 10, "bold"),
            text_color="gray",
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        btn_frame = ctk.CTkFrame(card_header, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e")

        # Scrollable file list — no fixed height, expands with card
        file_scroll = ctk.CTkScrollableFrame(files_card, corner_radius=6)
        file_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        file_scroll.grid_columnconfigure(0, weight=1)

        # -- File row management --
        def refresh_file_rows():
            for widget in file_scroll.winfo_children():
                widget.destroy()
            for i, fp in enumerate(file_list):
                ext = os.path.splitext(fp)[1].lower()
                if ext == '.stm':
                    badge_text, badge_fg, badge_bg = "STM", "#0c447c", "#d6e8fb"
                elif ext == '.csv':
                    badge_text, badge_fg, badge_bg = "CSV", "#1a5c3a", "#d6f5e3"
                else:
                    badge_text, badge_fg, badge_bg = "PDF", "#633806", "#faeeda"

                row = ctk.CTkFrame(
                    file_scroll,
                    fg_color=("gray88", "gray22"),
                    corner_radius=6,
                    height=30,
                )
                row.grid(row=i, column=0, sticky="ew", pady=2, padx=2)
                row.grid_columnconfigure(1, weight=1)
                row.grid_propagate(False)

                ctk.CTkLabel(
                    row,
                    text=badge_text,
                    font=("Segoe UI", 9, "bold"),
                    text_color=badge_fg,
                    fg_color=badge_bg,
                    corner_radius=4,
                    width=34,
                    height=18,
                ).grid(row=0, column=0, padx=(6, 4), pady=6)

                ctk.CTkLabel(
                    row,
                    text=os.path.basename(fp),
                    font=FONT_BODY,
                    anchor="w",
                ).grid(row=0, column=1, sticky="ew", padx=2)

                ctk.CTkButton(
                    row,
                    text="×",
                    font=("Segoe UI", 13),
                    width=24,
                    height=24,
                    fg_color="transparent",
                    hover_color=("gray78", "gray32"),
                    text_color="gray",
                    corner_radius=4,
                    command=lambda idx=i: _remove_file(idx),
                ).grid(row=0, column=2, padx=(0, 4))

        def _remove_file(idx):
            if 0 <= idx < len(file_list):
                file_list.pop(idx)
                refresh_file_rows()

        def _add_files():
            paths = filedialog.askopenfilenames(
                title="Select Files",
                filetypes=[("Hydraflow files", "*.stm *.pdf *.csv"), ("All files", "*.*")],
            )
            for p in paths:
                if accepted_extensions and os.path.splitext(p)[1].lower() not in accepted_extensions:
                    continue
                if p not in file_list:
                    file_list.append(p)
            if sort_alpha:
                file_list.sort(key=lambda p: os.path.basename(p).lower())
            refresh_file_rows()

        def _clear_files():
            file_list.clear()
            refresh_file_rows()

        def _on_drop(data):
            # tkinterdnd2 wraps paths with spaces in {braces}; others are space-separated
            paths = [a or b for a, b in _re.findall(r'\{([^}]+)\}|(\S+)', data)]
            for p in paths:
                if accepted_extensions and os.path.splitext(p)[1].lower() not in accepted_extensions:
                    continue
                if p not in file_list:
                    file_list.append(p)
            if sort_alpha:
                file_list.sort(key=lambda p: os.path.basename(p).lower())
            refresh_file_rows()

        ctk.CTkButton(
            btn_frame, text="Add files",
            font=FONT_BODY, width=82, height=26,
            command=_add_files,
        ).grid(row=0, column=0, padx=(0, 4))

        ctk.CTkButton(
            btn_frame, text="Clear",
            font=FONT_BODY, width=56, height=26,
            fg_color="transparent", border_width=1,
            command=_clear_files,
        ).grid(row=0, column=1)

        # ---- Action button ----
        # Use a list so the lambda can reference the button after it is created.
        btn_ref = [None]

        def _on_action():
            action_fn(file_list, default_output_name, log_widget, btn_ref[0])

        action_btn = ctk.CTkButton(
            frame,
            text=action_label,
            font=("Segoe UI", 12, "bold"),
            fg_color=action_color,
            hover_color=_darken(action_color),
            height=42,
            corner_radius=8,
            command=_on_action,
        )
        action_btn.grid(row=1, column=0, sticky="ew", padx=16, pady=8)
        btn_ref[0] = action_btn

        # ---- Log card ----
        log_card = ctk.CTkFrame(frame, corner_radius=10)
        log_card.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))
        log_card.grid_rowconfigure(1, weight=1)
        log_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            log_card,
            text="LOG",
            font=("Segoe UI", 10, "bold"),
            text_color="gray",
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        log_widget = ctk.CTkTextbox(
            log_card,
            font=FONT_MONO,
            state="disabled",
            wrap="word",
            corner_radius=6,
        )
        log_widget.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        return frame, file_list, log_widget, action_btn, _on_drop

    # =========================================================================
    # PAGE RUNNERS
    # =========================================================================

    # ---- Generate profiles ----
    def run_generate(file_list, default_output_name, log_widget, action_btn):
        if not file_list:
            messagebox.showwarning("No Files", "Add .stm and DOT PDF files.")
            return
        op = filedialog.asksaveasfilename(
            title="Save output as",
            defaultextension=".pdf",
            initialfile=default_output_name,
            filetypes=[("PDF files", "*.pdf")],
        )
        if not op:
            return
        for fp in file_list:
            if not os.path.isfile(fp):
                messagebox.showerror("Not Found", f"Missing:\n{fp}")
                return

        log_widget.configure(state="normal")
        log_widget.delete("1.0", "end")
        log_widget.configure(state="disabled")
        action_btn.configure(state="disabled", text="Generating...")
        old_stdout = sys.stdout
        sys.stdout = LogRedirector(log_widget)

        def _worker():
            try:
                print("Classifying files...")
                sf, df, of, pf = classify_files(file_list)
                if not sf:
                    root.after(0, lambda: messagebox.showerror("Error", "No .stm files found."))
                    return

                import threading as _thr
                _result = [None]
                _event  = _thr.Event()

                def _gui_prompt(title, msg):
                    def _do():
                        _result[0] = messagebox.askyesno(title, msg)
                        _event.set()
                    root.after(0, _do)
                    _event.wait()
                    _event.clear()
                    return _result[0]

                def _ask_plan(sn):
                    return _gui_prompt(
                        "Missing Plan View",
                        f"No Hydraflow plan view found for system '{sn}'.\n\n"
                        "Continue without a plan view for this system?",
                    )

                def _ask_warning(msg):
                    return _gui_prompt("Warning", msg)

                success = generate_deliverable(
                    sf, df, op, of, pf,
                    missing_plan_cb=_ask_plan,
                    warning_cb=_ask_warning,
                )
                if success:
                    root.after(0, lambda: messagebox.showinfo("Complete", f"Saved to:\n{op}"))
                else:
                    root.after(0, lambda: messagebox.showwarning("Cancelled", "Generation was cancelled."))
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("Error", str(e)))
                import traceback; traceback.print_exc()
            finally:
                sys.stdout = old_stdout
                root.after(0, lambda: action_btn.configure(state="normal", text="Generate profiles"))

        threading.Thread(target=_worker, daemon=True).start()

    # ---- Compile reports ----
    def run_compile(file_list, default_output_name, log_widget, action_btn):
        pdfs = [f for f in file_list if f.lower().endswith(".pdf")]
        if not pdfs:
            messagebox.showwarning("No PDFs", "Add report PDF files.")
            return
        op = filedialog.asksaveasfilename(
            title="Save output as",
            defaultextension=".pdf",
            initialfile=default_output_name,
            filetypes=[("PDF files", "*.pdf")],
        )
        if not op:
            return
        for fp in pdfs:
            if not os.path.isfile(fp):
                messagebox.showerror("Not Found", f"Missing:\n{fp}")
                return

        log_widget.configure(state="normal")
        log_widget.delete("1.0", "end")
        log_widget.configure(state="disabled")
        action_btn.configure(state="disabled", text="Compiling...")
        old_stdout = sys.stdout
        sys.stdout = LogRedirector(log_widget)

        def _worker():
            try:
                print("Classifying files...")
                _, df, of, _pf = classify_files(pdfs)
                if not df and not of:
                    root.after(0, lambda: messagebox.showerror("Error", "No report data found in PDFs."))
                    return
                compile_reports(df, op, of)
                root.after(0, lambda: messagebox.showinfo("Complete", f"Saved to:\n{op}"))
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("Error", str(e)))
                import traceback; traceback.print_exc()
            finally:
                sys.stdout = old_stdout
                root.after(0, lambda: action_btn.configure(state="normal", text="Compile reports"))

        threading.Thread(target=_worker, daemon=True).start()

    # ---- Line capacity check ----
    def run_check(file_list, default_output_name, log_widget, action_btn):
        pdfs = [f for f in file_list if f.lower().endswith(".pdf")]
        if not pdfs:
            messagebox.showwarning("No PDFs", "Add DOT PDF files.")
            return
        op = filedialog.asksaveasfilename(
            title="Save output as",
            defaultextension=".pdf",
            initialfile=default_output_name,
            filetypes=[("PDF files", "*.pdf")],
        )
        if not op:
            return
        base, ext = os.path.splitext(op)
        sc_out = f"{base}_check{ext}"
        for fp in pdfs:
            if not os.path.isfile(fp):
                messagebox.showerror("Not Found", f"Missing:\n{fp}")
                return

        log_widget.configure(state="normal")
        log_widget.delete("1.0", "end")
        log_widget.configure(state="disabled")
        action_btn.configure(state="disabled", text="Checking...")
        old_stdout = sys.stdout
        sys.stdout = LogRedirector(log_widget)

        def _worker():
            try:
                generate_system_check(pdfs, sc_out)
                root.after(0, lambda: messagebox.showinfo("Complete", f"Saved to:\n{sc_out}"))
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("Error", str(e)))
                import traceback; traceback.print_exc()
            finally:
                sys.stdout = old_stdout
                root.after(0, lambda: action_btn.configure(state="normal", text="Run line capacity check"))

        threading.Thread(target=_worker, daemon=True).start()

    # =========================================================================
    # BUILD PAGES
    # =========================================================================
    page_gen, _, _, _, drop_generate = build_page(
        main_area,
        action_label="Generate profiles",
        action_color=COLOR_GENERATE,
        action_fn=run_generate,
        default_output_name=".pdf",
    )
    pages["generate"] = page_gen

    page_compile, _, _, _, drop_compile = build_page(
        main_area,
        action_label="Compile reports",
        action_color=COLOR_COMPILE,
        action_fn=run_compile,
        default_output_name="compiled.pdf",
    )
    pages["compile"] = page_compile

    page_check, _, _, _, drop_check = build_page(
        main_area,
        action_label="Run line capacity check",
        action_color=COLOR_CHECK,
        action_fn=run_check,
        default_output_name="check.pdf",
        accepted_extensions=[".pdf"],
        sort_alpha=True,
    )
    pages["check"] = page_check

    # ---- STM Editor (custom page) -------------------------------------------

    # -- State --
    editor_stm_path = [None]

    # -- Outer frame --
    editor_frame = ctk.CTkFrame(main_area, fg_color="transparent", corner_radius=0)
    editor_frame.grid(row=0, column=0, sticky="nsew")
    editor_frame.grid_columnconfigure(0, weight=1)
    editor_frame.grid_rowconfigure(0, weight=2)  # STM card  ~2/3
    editor_frame.grid_rowconfigure(2, weight=1)  # defaults  ~1/3

    # ---- STM file card ----
    stm_card = ctk.CTkFrame(editor_frame, corner_radius=10)
    stm_card.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 6))
    stm_card.grid_columnconfigure(1, weight=1)

    ctk.CTkLabel(
        stm_card, text="STM FILE",
        font=("Segoe UI", 10, "bold"), text_color="gray", anchor="w",
    ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 8))

    stm_file_label = ctk.CTkLabel(
        stm_card, text="No file loaded",
        font=FONT_BODY, text_color="#7a9abf", anchor="w",
    )
    stm_file_label.grid(row=0, column=1, sticky="ew", padx=4, pady=(10, 8))

    def _browse_stm():
        p = filedialog.askopenfilename(
            title="Select STM File",
            filetypes=[("Hydraflow STM", "*.stm"), ("All files", "*.*")],
        )
        if p:
            editor_stm_path[0] = p
            stm_file_label.configure(text=os.path.basename(p))

    def _drop_editor_handler(data):
        paths = [a or b for a, b in _re.findall(r'\{([^}]+)\}|(\S+)', data)]
        for p in paths:
            if p.lower().endswith('.stm'):
                editor_stm_path[0] = p
                stm_file_label.configure(text=os.path.basename(p))
                return

    ctk.CTkButton(
        stm_card, text="Browse",
        font=FONT_BODY, width=70, height=26,
        command=_browse_stm,
    ).grid(row=0, column=2, padx=(4, 10), pady=(10, 8))

    # ---- Defaults card ----
    defaults_card = ctk.CTkFrame(editor_frame, corner_radius=10)
    defaults_card.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))
    defaults_card.grid_columnconfigure(1, weight=1)
    defaults_card.grid_columnconfigure(3, weight=1)
    defaults_card.grid_columnconfigure(5, weight=1)

    ctk.CTkLabel(
        defaults_card, text="DEFAULTS",
        font=("Segoe UI", 10, "bold"), text_color="gray", anchor="w",
    ).grid(row=0, column=0, columnspan=6, sticky="w", padx=12, pady=(10, 6))

    # Helper to add a labeled entry
    _ed_row = [1]
    def _add_field(label_text, col_pair=0, width=None):
        r = _ed_row[0]
        c = col_pair * 2
        ctk.CTkLabel(
            defaults_card, text=label_text,
            font=FONT_BODY, anchor="e",
        ).grid(row=r, column=c, sticky="e", padx=(12, 4), pady=3)
        entry = ctk.CTkEntry(defaults_card, font=FONT_BODY, height=28,
                             width=width or 0)
        entry.grid(row=r, column=c + 1, sticky="ew", padx=(0, 12), pady=3)
        return entry

    def _next_row():
        _ed_row[0] += 1

    # Row 1: Designation
    ent_designation = _add_field("Designation", col_pair=0)
    _next_row()

    # Row 2: Inlet Type + Location
    ctk.CTkLabel(
        defaults_card, text="Inlet Type",
        font=FONT_BODY, anchor="e",
    ).grid(row=_ed_row[0], column=0, sticky="e", padx=(12, 4), pady=3)

    inlet_type_var = ctk.StringVar(value="Combination Curb and Grate")
    inlet_type_menu = ctk.CTkOptionMenu(
        defaults_card,
        variable=inlet_type_var,
        values=["Manhole", "Curb", "Grate", "Combination Curb and Grate",
                "Generic", "Drop Curb", "Drop Grate"],
        font=FONT_BODY, height=28,
    )
    inlet_type_menu.grid(row=_ed_row[0], column=1, sticky="ew",
                         padx=(0, 12), pady=3)

    ctk.CTkLabel(
        defaults_card, text="Location",
        font=FONT_BODY, anchor="e",
    ).grid(row=_ed_row[0], column=2, sticky="e", padx=(12, 4), pady=3)

    location_var = ctk.StringVar(value="On Grade")
    location_menu = ctk.CTkOptionMenu(
        defaults_card,
        variable=location_var,
        values=["On Grade", "Sag"],
        font=FONT_BODY, height=28, width=100,
    )
    location_menu.grid(row=_ed_row[0], column=3, sticky="w",
                       padx=(0, 12), pady=3)
    _next_row()

    # Row 3: Runoff Coeff
    ent_runoff = _add_field("Runoff Coeff (C)", col_pair=0)
    _next_row()

    # Row 4: Sx + Sw
    ent_sx = _add_field("Cross Slope Sx", col_pair=0)
    ent_sw = _add_field("Cross Slope Sw", col_pair=1)
    _next_row()

    # Row 5: Curb Length + Throat Height (conditional)
    lbl_curb_len = ctk.CTkLabel(
        defaults_card, text="Curb Length (ft)",
        font=FONT_BODY, anchor="e",
    )
    ent_curb_len = ctk.CTkEntry(defaults_card, font=FONT_BODY, height=28)

    lbl_throat = ctk.CTkLabel(
        defaults_card, text="Throat Ht. (in)",
        font=FONT_BODY, anchor="e",
    )
    ent_throat = ctk.CTkEntry(defaults_card, font=FONT_BODY, height=28)

    curb_row = _ed_row[0]
    _next_row()

    # Row 6: Grate Width + Grate Length + Grate Area (conditional)
    lbl_grate_w = ctk.CTkLabel(
        defaults_card, text="Grate Width (ft)",
        font=FONT_BODY, anchor="e",
    )
    ent_grate_w = ctk.CTkEntry(defaults_card, font=FONT_BODY, height=28)

    lbl_grate_l = ctk.CTkLabel(
        defaults_card, text="Grate Length (ft)",
        font=FONT_BODY, anchor="e",
    )
    ent_grate_l = ctk.CTkEntry(defaults_card, font=FONT_BODY, height=28)

    lbl_grate_a = ctk.CTkLabel(
        defaults_card, text="Grate Area (sqft)",
        font=FONT_BODY, anchor="e",
    )
    ent_grate_a = ctk.CTkEntry(defaults_card, font=FONT_BODY, height=28)

    grate_row = _ed_row[0]
    _next_row()

    # ---- Conditional visibility logic ----
    def _update_conditional_fields(*_args):
        itype = inlet_type_var.get().lower()
        show_curb  = 'curb' in itype or 'combination' in itype
        show_grate = 'grate' in itype or 'combination' in itype

        if show_curb:
            lbl_curb_len.grid(row=curb_row, column=0, sticky="e",
                              padx=(12, 4), pady=3)
            ent_curb_len.grid(row=curb_row, column=1, sticky="ew",
                              padx=(0, 12), pady=3)
            lbl_throat.grid(row=curb_row, column=2, sticky="e",
                            padx=(12, 4), pady=3)
            ent_throat.grid(row=curb_row, column=3, sticky="ew",
                            padx=(0, 12), pady=3)
        else:
            lbl_curb_len.grid_remove()
            ent_curb_len.grid_remove()
            lbl_throat.grid_remove()
            ent_throat.grid_remove()

        if show_grate:
            lbl_grate_w.grid(row=grate_row, column=0, sticky="e",
                             padx=(12, 4), pady=3)
            ent_grate_w.grid(row=grate_row, column=1, sticky="ew",
                             padx=(0, 12), pady=3)
            lbl_grate_l.grid(row=grate_row, column=2, sticky="e",
                             padx=(12, 4), pady=3)
            ent_grate_l.grid(row=grate_row, column=3, sticky="ew",
                             padx=(0, 12), pady=3)
            lbl_grate_a.grid(row=grate_row, column=4, sticky="e",
                             padx=(12, 4), pady=3)
            ent_grate_a.grid(row=grate_row, column=5, sticky="ew",
                             padx=(0, 12), pady=3)
        else:
            lbl_grate_w.grid_remove()
            ent_grate_w.grid_remove()
            lbl_grate_l.grid_remove()
            ent_grate_l.grid_remove()
            lbl_grate_a.grid_remove()
            ent_grate_a.grid_remove()

    inlet_type_var.trace_add("write", _update_conditional_fields)
    _update_conditional_fields()  # initial state

    # ---- Save As button ----
    def _safe_float(entry):
        v = entry.get().strip()
        if not v:
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def _run_editor_save():
        stm = editor_stm_path[0]
        if not stm or not os.path.isfile(stm):
            messagebox.showwarning("No STM", "Load an .stm file first.")
            return
        desig = ent_designation.get().strip()
        if not desig:
            messagebox.showwarning("No Designation",
                                   "Enter a designation (e.g. EB-EC).")
            return

        base = os.path.splitext(os.path.basename(stm))[0]
        op = filedialog.asksaveasfilename(
            title="Save edited .stm as",
            defaultextension=".stm",
            initialfile=f"{base}_edited.stm",
            filetypes=[("Hydraflow STM", "*.stm"), ("All files", "*.*")],
        )
        if not op:
            return

        itype_code = JUNCTION_TYPE_CODES.get(
            inlet_type_var.get().lower(), 0)
        loc_code = LOCATION_CODES.get(
            location_var.get().lower(), 0)

        defaults = {
            'inlet_type':    itype_code,
            'location':      loc_code,
            'runoff_coeff':  _safe_float(ent_runoff),
            'sx':            _safe_float(ent_sx),
            'sw':            _safe_float(ent_sw),
            'curb_length':   _safe_float(ent_curb_len),
            'throat_height': _safe_float(ent_throat),
            'grate_width':   _safe_float(ent_grate_w),
            'grate_length':  _safe_float(ent_grate_l),
            'grate_area':    _safe_float(ent_grate_a),
        }

        editor_save_btn.configure(state="disabled", text="Saving...")

        def _worker():
            try:
                merged, warns = apply_editor_defaults(stm, desig, defaults)
                with open(op, 'wb') as f:
                    f.write(merged.encode('utf-8'))

                msg = f"Saved to:\n{op}"
                if warns:
                    msg += f"\n\n{len(warns)} warning(s):\n"
                    msg += "\n".join(warns[:10])
                root.after(0, lambda: messagebox.showinfo("Complete", msg))
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                root.after(0, lambda: editor_save_btn.configure(
                    state="normal", text="Save As .stm"))

        threading.Thread(target=_worker, daemon=True).start()

    editor_save_btn = ctk.CTkButton(
        editor_frame,
        text="Save As .stm",
        font=("Segoe UI", 12, "bold"),
        fg_color=COLOR_EDITOR,
        hover_color=_darken(COLOR_EDITOR),
        height=42,
        corner_radius=8,
        command=_run_editor_save,
    )
    editor_save_btn.grid(row=1, column=0, sticky="ew", padx=16, pady=8)

    pages["editor"] = editor_frame

    # Route drops to whichever page is currently active.
    # Registering on root covers all child widgets with no propagation issues.
    _drop_handlers = {
        "generate": drop_generate,
        "compile":  drop_compile,
        "check":    drop_check,
        "editor":   _drop_editor_handler,
    }
    _active_page = ["generate"]

    _original_switch = switch_page
    def switch_page(page_id):
        _active_page[0] = page_id
        _original_switch(page_id)

    def _root_on_drop(event):
        _drop_handlers[_active_page[0]](event.data)

    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', _root_on_drop)

    # Start on Generate page
    switch_page("generate")
    root.mainloop()


if __name__ == '__main__':
    main()
