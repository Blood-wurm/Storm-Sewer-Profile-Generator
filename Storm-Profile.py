"""
Storm Sewer Profile Generator
Parses Hydraflow .stm files and DOT tabulation PDFs to batch-generate
storm sewer deliverable packages as PDF output.

Deliverable structure per system:
  1. Plan view (generated from .stm)
  2. For each return period (sorted):
     a. DOT tabulation page(s) (inserted from Hydraflow PDF)
     b. Profile plots for each line designation
"""

import re
import os
import sys
import argparse
import tempfile
from collections import defaultdict

import pdfplumber
from pypdf import PdfReader, PdfWriter
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
# SECTION 2: DOT TABULATION PDF PARSER
# =============================================================================

_HEADER_FRAGMENTS = {"Line", "HGL", "EGL", "Up", "Dn"}

def _find_dot_header_cols(row):
    """Return (hgl_up_col, hgl_dn_col) from a header row, or (None, None)."""
    cells = [c.strip() if c else '' for c in row]
    matches = sum(1 for c in cells if any(f in c for f in _HEADER_FRAGMENTS))
    if matches < 3:
        return None, None
    # Find the first column whose header contains 'HGL' and whose next column
    # header contains 'Up', with the column after that containing 'Dn'.
    for j, c in enumerate(cells):
        if 'HGL' in c:
            # Look for adjacent Up/Dn pair
            for k in range(j, min(j + 4, len(cells) - 1)):
                if 'Up' in cells[k] and k + 1 < len(cells) and 'Dn' in cells[k + 1]:
                    return k, k + 1
    return None, None

def _parse_dot_rows_table(table):
    """Extract {line_no: {'HGL Up': float, 'HGL Dn': float}} from a pdfplumber table."""
    result = {}
    hgl_up_col = hgl_dn_col = None
    for row in table:
        if not row:
            continue
        # Try to identify header row
        if hgl_up_col is None:
            u, d = _find_dot_header_cols(row)
            if u is not None:
                hgl_up_col, hgl_dn_col = u, d
            continue
        # Data row: col 0 must be a positive integer line number
        raw_ln = row[0].strip() if row[0] else ''
        try:
            line_no = int(raw_ln)
            if line_no <= 0:
                continue
        except (ValueError, AttributeError):
            continue
        try:
            hgl_up = float(row[hgl_up_col]) if row[hgl_up_col] else None
            hgl_dn = float(row[hgl_dn_col]) if row[hgl_dn_col] else None
        except (ValueError, TypeError, IndexError):
            continue
        if hgl_up is None or hgl_dn is None:
            continue
        result[line_no] = {'HGL Up': hgl_up, 'HGL Dn': hgl_dn}
    return result

def _parse_dot_rows_regex(text):
    """Fallback: extract {line_no: {'HGL Up': float, 'HGL Dn': float}} via regex."""
    result = {}
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        dot_match = re.match(
            r'^(\d+)\s+\S+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+'
            r'[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+'
            r'[\d.]+\s+[\d.]+\s+\d+\s+[\d.]+\s+'
            r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+'
            r'[\d.]+\s+[\d.]+\s+(.+)$', line)
        if dot_match:
            line_no = int(dot_match.group(1))
            result[line_no] = {
                'HGL Up': float(dot_match.group(4)),
                'HGL Dn': float(dot_match.group(5)),
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
                pf = re.search(r'Project File:\s*(.*?\.stm)', raw_line)
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
            if tables:
                for table in tables:
                    parsed = _parse_dot_rows_table(table)
                    current['data'].update(parsed)
            else:
                parsed = _parse_dot_rows_regex(text)
                current['data'].update(parsed)

    # Set project file on all entries
    for r in results:
        if not r['project_file']:
            r['project_file'] = project_file

    return results

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
    R, G, I, C, S = profile['reach'], profile['ground'], profile['invert'], profile['crown'], profile['segments']
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
        x_start = sw if i == 0 else R[i] + hw
        x_end = R[i+1] - hw
        ax.plot([x_start, x_end], [I[i], I[i+1]], color=COLOR_INVERT, linewidth=LW_INV, zorder=4)
        ax.plot([x_start, x_end], [C[i], C[i+1]], color=COLOR_CROWN, linewidth=LW_CRN, zorder=4)
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
                x_start = sw if i == 0 else R[i] + hw
                x_end = R[i+1] - hw
                ax.plot([x_start, x_end], [s['hgl_dn'], s['hgl_up']], color=COLOR_HGL, linewidth=LW_HGL, zorder=6)
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
        bl = 0 if r == 0 else r - hw
        ax.add_patch(patches.Rectangle(
            (bl, inv), sw, grd - inv,
            lw=1.2, ec=COLOR_STRUCTURE, fc='none', zorder=7,
            # hatch='///', alpha=0.4,  # structure box hatch (disabled)
        ))
        sq = (emax - emin) * 0.012
        ax.add_patch(patches.Rectangle((bl + hw*0.2, inv), sw*0.6, sq, lw=0.5, ec=COLOR_STRUCTURE, fc=COLOR_STRUCTURE, zorder=8))
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
        rm = (R[i] + R[i+1]) / 2.0; cm = (C[i] + C[i+1]) / 2.0; ly = cm + lo
        ax.plot([rm, rm], [cm, ly], color='#999999', lw=0.6, zorder=3)
        ax.plot(rm, cm, marker='_', color='#999999', ms=4, zorder=3)
        ax.text(rm, ly, f"Ln: {s['line_no']}\n{s['size_label']}", ha='center', va='bottom', fontsize=9, color='#333333', zorder=10)
    ax.set_xlabel('Reach (ft)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Elev. (ft)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, xmax); ax.set_ylim(emin, emax)
    _auto_ticks(ax, xmax, emin, emax)
    ax.tick_params(axis='y', labelsize=9, pad=10); ax.tick_params(axis='x', labelsize=9)
    ax.set_title('Storm Sewer Profile', fontsize=14, fontweight='bold', loc='left', pad=15)
    if project_name: ax.text(0.99, 1.02, f'Proj. file: {project_name}', transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color='#555555')
    if return_period: ax.text(0.99, 1.06, return_period, transform=ax.transAxes, ha='right', va='bottom', fontsize=14, fontweight='bold', color=COLOR_TITLE_RP)
    ax.text(0.99, -0.08, 'Storm Sewer Profile Generator', transform=ax.transAxes, ha='right', va='top', fontsize=7, color='#999999')
    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.10)
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
    ax.set_title(f'Hydraflow Plan View\n{project_name}', fontsize=14, fontweight='bold', loc='left', pad=10)
    for sp in ax.spines.values(): sp.set_linewidth(1.5)
    fig.subplots_adjust(left=_l, right=_r, top=_t, bottom=_b)
    return fig


# =============================================================================
# SECTION 6: DELIVERABLE ASSEMBLY
# =============================================================================

def classify_files(file_paths):
    stm_files = {}; dot_files = []
    for fp in file_paths:
        ext = os.path.splitext(fp)[1].lower()
        if ext == '.stm':
            stm_files[os.path.splitext(os.path.basename(fp))[0]] = fp
        elif ext == '.pdf':
            entries = parse_dot_pdf(fp)
            for info in entries:
                if info['data'] and info['project_file'] and info['return_period']:
                    dot_files.append({**info, 'path': fp})
                    print(f"  DOT: {os.path.basename(fp)} -> {info['project_file']} / {info['return_period']} ({len(info['data'])} lines)")
            if not entries:
                print(f"  Skipped: {os.path.basename(fp)} (no DOT tabulations found)")
    return stm_files, dot_files

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

def generate_deliverable(stm_files, dot_files, output_path='deliverable.pdf'):
    writer = PdfWriter()
    dot_by_proj = defaultdict(list)
    for d in dot_files: dot_by_proj[d['project_file']].append(d)
    for pf in dot_by_proj: dot_by_proj[pf].sort(key=lambda x: x['return_period_num'])
    systems = []
    for sn, sp in sorted(stm_files.items()):
        systems.append((sn, sp, dot_by_proj.get(sn, [])))
    if not systems:
        print("ERROR: No systems to process."); return
    pc = 0
    for sn, sp, dots in systems:
        print(f"\n{'='*60}\nSystem: {sn}\n{'='*60}")
        header, lines = parse_stm(sp)
        print(f"  {len(lines)} lines")
        pgroups = group_paths_by_prefix(lines)
        print(f"  Designations: {sorted(pgroups.keys())}")
        # Plan view
        fig = plot_plan_view(lines, sn)
        tp = _fig_to_temp_pdf(fig, 'plan'); plt.close(fig)
        _append_pdf(writer, tp); os.unlink(tp); pc += 1
        print(f"  Page {pc}: Plan view")
        if dots:
            for di in dots:
                rl = di['return_period']; rd = di['data']; dp = di['path']
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
        sf, df = classify_files(args.files)
        if not sf: print("ERROR: No .stm files."); return
        generate_deliverable(sf, df, args.output); return
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
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import threading
    root = tk.Tk(); root.title("Storm Sewer Profile Generator")
    root.geometry("720x580"); root.minsize(620, 480)
    BG="#f5f5f5"; FBG="#ffffff"; AC="#2a5d9f"; BBG="#2a5d9f"; BFG="#ffffff"
    FN=("Segoe UI",10); FS=("Segoe UI",9); FL=("Segoe UI",10,"bold"); FM=("Consolas",9)
    root.configure(bg=BG); file_list=[]
    class LR:
        def __init__(s,w): s.w=w
        def write(s,t): s.w.configure(state='normal'); s.w.insert(tk.END,t); s.w.see(tk.END); s.w.configure(state='disabled')
        def flush(s): pass
    tf=tk.Frame(root,bg=AC,height=44); tf.pack(fill='x'); tf.pack_propagate(False)
    tk.Label(tf,text="Storm Sewer Profile Generator",font=("Segoe UI",14,"bold"),fg="white",bg=AC).pack(side='left',padx=14,pady=8)
    ct=tk.Frame(root,bg=BG,padx=14,pady=10); ct.pack(fill='both',expand=True)
    tk.Label(ct,text="Project Files (.stm and DOT PDFs):",font=FL,bg=BG,anchor='w').pack(fill='x',pady=(0,2))
    ff=tk.Frame(ct,bg=FBG,relief='sunken',bd=1); ff.pack(fill='x',pady=(0,4))
    fl=tk.Listbox(ff,font=FS,height=6,selectmode='extended',bg=FBG,bd=0)
    fsc=tk.Scrollbar(ff,command=fl.yview); fl.configure(yscrollcommand=fsc.set)
    fsc.pack(side='right',fill='y'); fl.pack(fill='both',expand=True,padx=2,pady=2)
    bf=tk.Frame(ct,bg=BG); bf.pack(fill='x',pady=(0,6))
    ov=tk.StringVar()
    def af():
        ps=filedialog.askopenfilenames(title="Select Files",filetypes=[("Hydraflow","*.stm *.pdf"),("All","*.*")])
        for p in ps:
            if p not in file_list:
                file_list.append(p); e=os.path.splitext(p)[1].lower()
                fl.insert(tk.END,f"  {'[STM]' if e=='.stm' else '[PDF]'}  {os.path.basename(p)}")
        if not ov.get():
            for p in file_list:
                if p.lower().endswith('.stm'):
                    ov.set(os.path.join(os.path.dirname(p),'deliverable.pdf')); break
    def rf():
        for i in sorted(fl.curselection(),reverse=True): fl.delete(i); file_list.pop(i)
    def cf(): fl.delete(0,tk.END); file_list.clear()
    tk.Button(bf,text="Add Files...",font=FS,command=af,padx=8).pack(side='left',padx=(0,6))
    tk.Button(bf,text="Remove",font=FS,command=rf,padx=8).pack(side='left',padx=(0,6))
    tk.Button(bf,text="Clear",font=FS,command=cf,padx=8).pack(side='left')
    ro=tk.Frame(ct,bg=BG); ro.pack(fill='x',pady=(6,6))
    tk.Label(ro,text="Output PDF:",font=FL,bg=BG,width=12,anchor='w').pack(side='left')
    oe=tk.Entry(ro,textvariable=ov,font=FN); oe.pack(side='left',fill='x',expand=True,padx=(0,6))
    def bo():
        p=filedialog.asksaveasfilename(title="Save As",defaultextension=".pdf",filetypes=[("PDF","*.pdf")])
        if p: ov.set(p)
    tk.Button(ro,text="Browse...",font=FS,command=bo,padx=10).pack(side='left')
    bfr=tk.Frame(ct,bg=BG); bfr.pack(fill='x',pady=(8,6))
    gb=tk.Button(bfr,text="Generate Deliverable",font=("Segoe UI",11,"bold"),bg=BBG,fg=BFG,activebackground="#1e4a80",activeforeground="white",padx=20,pady=6,cursor="hand2")
    gb.pack()
    tk.Label(ct,text="Log:",font=FL,bg=BG,anchor='w').pack(fill='x',pady=(8,2))
    lf=tk.Frame(ct,bg=FBG,relief='sunken',bd=1); lf.pack(fill='both',expand=True)
    lt=tk.Text(lf,font=FM,wrap='word',state='disabled',bg=FBG,bd=0,padx=6,pady=4)
    ls=tk.Scrollbar(lf,command=lt.yview); lt.configure(yscrollcommand=ls.set)
    ls.pack(side='right',fill='y'); lt.pack(fill='both',expand=True)
    def rg():
        if not file_list: messagebox.showwarning("No Files","Add .stm and DOT PDF files."); return
        op=ov.get().strip()
        if not op: messagebox.showwarning("No Output","Specify output path."); return
        for fp in file_list:
            if not os.path.isfile(fp): messagebox.showerror("Not Found",f"Missing:\n{fp}"); return
        lt.configure(state='normal'); lt.delete('1.0',tk.END); lt.configure(state='disabled')
        gb.configure(state='disabled',text="Generating...")
        old=sys.stdout; sys.stdout=LR(lt)
        def dw():
            try:
                print("Classifying files...")
                sf,df=classify_files(file_list)
                if not sf: root.after(0,lambda:messagebox.showerror("Error","No .stm files found.")); return
                generate_deliverable(sf,df,op)
                root.after(0,lambda:messagebox.showinfo("Complete",f"Saved to:\n{op}"))
            except Exception as e:
                root.after(0,lambda:messagebox.showerror("Error",str(e)))
                import traceback; traceback.print_exc()
            finally:
                sys.stdout=old; root.after(0,lambda:gb.configure(state='normal',text="Generate Deliverable"))
        threading.Thread(target=dw,daemon=True).start()
    gb.configure(command=rg); root.mainloop()


if __name__ == '__main__':
    main()