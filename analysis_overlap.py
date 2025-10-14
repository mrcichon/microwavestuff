import re
from itertools import combinations

def parse_frequency_file(filepath):
    """Parse frequency range file."""
    ranges = []
    with open(filepath, 'r') as file:
        for line in file:
            match = re.match(r"^\s*(\d+\.?\d*)\s+(\d+\.?\d*)", line)
            if match:
                start = float(match.group(1)) / 1e9
                end = float(match.group(2)) / 1e9
                ranges.append((start, end))
    return ranges


def find_overlaps(r1, r2):
    """Find overlapping regions between two range lists."""
    overlaps = []
    for s1, e1 in r1:
        for s2, e2 in r2:
            start = max(s1, s2)
            end = min(e1, e2)
            if start <= end:
                overlaps.append((start, end))
    return overlaps


def format_overlap_text(overlap_data):
    """Format overlap analysis results."""
    lines = []
    lines.append("Frequency ranges (GHz):")
    lines.append("-" * 40)
    
    for name, ranges in sorted(overlap_data.items()):
        formatted = [(round(s, 3), round(e, 3)) for s, e in ranges]
        lines.append(f"{name}: {formatted}")
    
    lines.append("\nOverlapping ranges:")
    lines.append("-" * 40)
    
    for (n1, r1), (n2, r2) in combinations(overlap_data.items(), 2):
        overlaps = find_overlaps(r1, r2)
        if overlaps:
            formatted = [(round(s, 3), round(e, 3)) for s, e in overlaps]
            lines.append(f"{n1} <-> {n2}: {formatted}")
        else:
            lines.append(f"{n1} <-> {n2}: no overlaps")
    
    return "\n".join(lines)
