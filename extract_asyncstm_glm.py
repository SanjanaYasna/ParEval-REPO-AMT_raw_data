import json
import os
import re
import shutil
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional


def normalize_indentation(text: str) -> str:
    """Remove common leading whitespace from all lines.
    
    Handles cases where the first line has different indentation
    than the rest of the block.
    """
    if not text or not text.strip():
        return text
        
    # Use textwrap.dedent to remove common leading whitespace
    dedented = textwrap.dedent(text)
    
    # If the result still has issues, try a more aggressive approach
    lines = dedented.split('\n')
    
    # Find minimum indentation of non-empty lines (excluding first line)
    indents = []
    for i, line in enumerate(lines[1:], 1):  # Skip first line
        if line.strip():  # Non-empty line
            # Count leading spaces
            indent = len(line) - len(line.lstrip())
            indents.append(indent)
    
    if indents:
        min_indent = min(indents)
        # If there's common indentation after the first line, remove it
        if min_indent > 0:
            normalized_lines = [lines[0]]  # Keep first line as-is
            for line in lines[1:]:
                if line.strip():  # Non-empty
                    normalized_lines.append(line[min_indent:] if len(line) > min_indent else line)
                else:  # Empty line
                    normalized_lines.append(line)
            return '\n'.join(normalized_lines)
    
    return dedented


def extract_astm_config_hpp(full_text: str) -> str:
    """Extract astm_config.hpp - last block (single ` or triple ```) containing legion.h."""
    # Check triple backticks first (from the bottom up)
    triple_blocks = re.findall(r'```[^\n]*\n(.*?)```', full_text, re.DOTALL)
    for block in reversed(triple_blocks):
        if 'legion.h' in block:
            return normalize_indentation(block)
            
    # Check single backticks (from the bottom up)
    single_blocks = re.findall(r'`([^`]+)`', full_text)
    for block in reversed(single_blocks):
        if 'legion.h' in block:
            return normalize_indentation(block)
            
    return "// ERROR: Could not extract astm_config.hpp (legion header not found in any block)"


def extract_astm_hpp(full_text: str) -> str:
    """Extract astm.hpp - combine specific marked sections or fallback to last 3 blocks."""
    parts = []
    
    # Regexes for the specific headers requested, made slightly flexible for minor LLM variations
    headers = [
        r'\*?Code Structure:?\*?',
        r'\*?Transaction Class Implementation:?\*?',
        r'\*?Async Logic:?\*?'
    ]
    
    for header in headers:
        pattern = header + r'[\s\n]*```(?:cpp|c\+\+)?[^\n]*\n(.*?)```'
        match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
        if match:
            parts.append(normalize_indentation(match.group(1)))
            
    if len(parts) == 3:
        return "\n\n".join(parts)
        
    # Fallback to taking the last 3 cpp blocks if the exact headers weren't used
    cpp_blocks = re.findall(r'```(?:cpp|c\+\+)?[^\n]*\n(.*?)```', full_text, re.DOTALL | re.IGNORECASE)
    if len(cpp_blocks) >= 3:
        return "\n\n".join(normalize_indentation(b) for b in cpp_blocks[-3:])
    elif cpp_blocks:
        return "\n\n".join(normalize_indentation(b) for b in cpp_blocks)
        
    return "// ERROR: Could not extract astm.hpp components"


def extract_generic_cpp(full_text: str) -> str:
    """Extract general cpp files - take the last cpp code block, allowing for unclosed blocks."""
    # Find all starts of cpp blocks (handles optional trailing spaces before newline)
    pattern = r'```(?:cpp|c\+\+)(?:[^\n]*\n|$)'
    matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
    
    if not matches:
        # Fallback: check if there are ANY code blocks without the 'cpp' tag
        pattern = r'```(?:[^\n]*\n|$)'
        matches = list(re.finditer(pattern, full_text))
        
    if not matches:
        return "// ERROR: Could not extract generic cpp (no code blocks found)"
        
    # Take the last match (ignores earlier small closed blocks)
    last_match = matches[-1]
    content = full_text[last_match.end():]
    
    # If there's a closing backtick sequence, stop there (block was closed)
    closing_idx = content.find('```')
    if closing_idx != -1:
        content = content[:closing_idx]
        
    # Otherwise, it runs to the end of the string (unclosed block handled safely)
    return normalize_indentation(content)


def extract_makefile(full_text: str) -> str:
    """Extract Makefile - last block (handles unclosed blocks), removing exact duplicate lines."""
    pattern = r'```(?:makefile|make)(?:[^\n]*\n|$)'
    matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
    
    if not matches:
        # Fallback: ANY code blocks
        pattern = r'```(?:[^\n]*\n|$)'
        matches = list(re.finditer(pattern, full_text))
        
    if not matches:
        return "# ERROR: Could not extract makefile (no code blocks found)"
        
    last_match = matches[-1]
    content = full_text[last_match.end():]
    
    closing_idx = content.find('```')
    if closing_idx != -1:
        content = content[:closing_idx]
        
    # Deduplicate lines while preserving order and blank lines
    lines = content.split('\n')
    seen = set()
    dedup_lines = []
    
    for line in lines:
        if not line.strip():
            # Always keep empty lines to preserve formatting
            dedup_lines.append(line)
        elif line not in seen:
            # Add unique lines and track them
            dedup_lines.append(line)
            seen.add(line)
            
    return normalize_indentation('\n'.join(dedup_lines))


def extract_readme(full_text: str) -> str:
    """Extract ReadMe.md - take the last markdown or cmake block, handling unclosed blocks."""
    # Find all starts of markdown/cmake blocks
    pattern = r'```(?:markdown|md|cmake)(?:[^\n]*\n|$)'
    matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
    
    if not matches:
        # Fallback: ANY code blocks
        pattern = r'```(?:[^\n]*\n|$)'
        matches = list(re.finditer(pattern, full_text))
    
    if not matches:
        return "# ERROR: Could not extract ReadMe.md (no code blocks found)"
    
    # Take the last match
    last_match = matches[-1]
    content = full_text[last_match.end():]
    
    # If there's a closing backtick sequence, stop there (block was closed)
    closing_idx = content.find('```')
    if closing_idx != -1:
        content = content[:closing_idx]
    
    # Otherwise, it runs to the end of the string (unclosed block handled safely)
    return normalize_indentation(content)


def extract_by_filename(filename: str, full_text: str) -> str:
    """Route to appropriate extraction function based on filename and catch blank output."""
    filename_lower = filename.lower()
    
    extracted = ""
    if 'astm_config.hpp' in filename_lower:
        extracted = extract_astm_config_hpp(full_text)
    elif 'astm.hpp' in filename_lower:
        extracted = extract_astm_hpp(full_text)
    elif filename_lower in ['binary_tree.cpp', 'concurrency_tests.cpp', 'unit_tests.cpp']:
        extracted = extract_generic_cpp(full_text)
    elif 'makefile' in filename_lower:
        extracted = extract_makefile(full_text)
    elif 'readme' in filename_lower:
        extracted = extract_readme(full_text)
    else:
        # Generic fallback for an unknown file type
        extracted = extract_generic_cpp(full_text)
        
    # ---------------------------------------------------------
    # GLOBAL SAFEGUARD: Ensure we never write a purely blank file
    # ---------------------------------------------------------
    if not extracted.strip():
        if filename_lower.endswith('.md') or 'makefile' in filename_lower:
            return f"# ERROR: Extracted content was completely empty for {filename}"
        else:
            return f"// ERROR: Extracted content was completely empty for {filename}"
            
    return extracted


def process_output_directory(output_dir: Path, target_dir: Path) -> Dict[str, Any]:
    """Process a single output-N directory with pattern-based extraction."""
    metadata_path = output_dir / "experiment_metadata.json"
    
    if not metadata_path.exists():
        print(f"  ✗ No experiment_metadata.json found in {output_dir}")
        return {'success': False, 'files_written': 0, 'files_failed': 0}
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"  ✗ Failed to load metadata: {e}")
        return {'success': False, 'files_written': 0, 'files_failed': 0}
    
    reasoning_texts = metadata.get("reasoning_texts", {})
    
    if not reasoning_texts:
        print(f"  ⚠ No reasoning_texts in {output_dir.name}")
        return {'success': False, 'files_written': 0, 'files_failed': 0}
    
    # Create target directory structure
    target_dir.mkdir(parents=True, exist_ok=True)
    target_repo = target_dir / "repo"
    target_repo.mkdir(exist_ok=True)
    
    # Copy experiment_metadata.json
    shutil.copy2(metadata_path, target_dir / "experiment_metadata.json")
    
    files_written = 0
    files_failed = 0
    
    for filename, full_text in reasoning_texts.items():
        extracted_code = extract_by_filename(filename, full_text)
        
        if extracted_code.startswith("// ERROR:") or extracted_code.startswith("# ERROR:"):
            files_failed += 1
        
        output_file = target_repo / filename
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_code)
            
            files_written += 1
            
            # Print success messaging based on the routing logic
            fname_lower = filename.lower()
            if extracted_code.startswith("// ERROR:") or extracted_code.startswith("# ERROR:"):
                print(f"    ✗ {filename}: Extracted as ERROR block")
            elif 'astm_config.hpp' in fname_lower:
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [last block containing legion.h]")
            elif 'astm.hpp' in fname_lower:
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [combined sections]")
            elif fname_lower in ['binary_tree.cpp', 'concurrency_tests.cpp', 'unit_tests.cpp']:
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [last cpp block, allows unclosed]")
            elif 'makefile' in fname_lower:
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [last makefile block (allows unclosed), deduped]")
            elif 'readme' in fname_lower:
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [last markdown block]")
            else:
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [generic pattern]")
                
        except Exception as e:
            print(f"    ✗ {filename}: Failed to write - {e}")
            files_failed += 1
    
    if files_written > 0:
        status = f"{files_written} written"
        if files_failed > 0:
            status += f", {files_failed} failed"
        print(f"  → {status} to {target_dir.name}/repo/")
    
    return {
        'success': files_written > 0,
        'files_written': files_written,
        'files_failed': files_failed
    }


def save_experiment_directory(root_dir: str, output_dir: Optional[str] = None):
    """Save pattern-extracted code from an experiment directory."""
    root_path = Path(root_dir).resolve()
    
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    if output_dir is None:
        output_path = root_path.parent / f"{root_path.name}_pattern_parsed"
    else:
        output_path = Path(output_dir).resolve()
    
    print(f"\n{'='*80}")
    print(f"PATTERN-BASED CODE EXTRACTION")
    print(f"{'='*80}")
    print(f"Source:      {root_path}")
    print(f"Destination: {output_path}")
    print(f"{'='*80}\n")
    
    output_dirs = sorted([
        d for d in root_path.iterdir()
        if d.is_dir() and d.name.startswith('output-')
    ], key=lambda x: int(x.name.split('-')[1]) if len(x.name.split('-')) > 1 and x.name.split('-')[1].isdigit() else 0)
    
    if not output_dirs:
        print(f"⚠ No output-N directories found")
        return
    
    print(f"Found {len(output_dirs)} output directories\n")
    
    success_count = 0
    total_failed = 0
    
    for output_dir in output_dirs:
        print(f"Processing {output_dir.name}...")
        target_dir = output_path / output_dir.name
        
        result = process_output_directory(output_dir, target_dir)
        if result['success']:
            success_count += 1
        total_failed += result['files_failed']
        print()
    
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully saved: {success_count}/{len(output_dirs)} outputs")
    if total_failed > 0:
        print(f"Files that failed extraction: {total_failed}")
    print(f"Output location: {output_path}")
    print(f"{'='*80}\n")


def save_multiple_experiments(parent_dir: str, output_parent: Optional[str] = None):
    """Save pattern-extracted code from multiple experiments."""
    parent_path = Path(parent_dir).resolve()
    
    if not parent_path.exists():
        raise FileNotFoundError(f"Directory not found: {parent_dir}")
    
    if output_parent is None:
        output_parent_path = parent_path.parent / f"{parent_path.name}_pattern_parsed"
    else:
        output_parent_path = Path(output_parent).resolve()
    
    print(f"\n{'='*80}")
    print(f"BATCH PATTERN-BASED CODE EXTRACTION")
    print(f"{'='*80}")
    print(f"Source parent:      {parent_path}")
    print(f"Destination parent: {output_parent_path}")
    print(f"{'='*80}\n")
    
    experiment_dirs = []
    for item in parent_path.iterdir():
        if item.is_dir():
            has_outputs = any(
                d.is_dir() and d.name.startswith('output-')
                for d in item.iterdir()
            )
            if has_outputs:
                experiment_dirs.append(item)
    
    if not experiment_dirs:
        print(f"⚠ No experiment directories found")
        return
    
    print(f"Found {len(experiment_dirs)} experiments:\n")
    for exp in sorted(experiment_dirs):
        print(f"  - {exp.name}")
    print()
    
    total_success = 0
    total_outputs = 0
    total_failed = 0
    
    for exp_dir in sorted(experiment_dirs):
        print(f"\n{'─'*80}")
        print(f"EXPERIMENT: {exp_dir.name}")
        print(f"{'─'*80}")
        
        target_exp_dir = output_parent_path / exp_dir.name
        
        output_dirs = sorted([
            d for d in exp_dir.iterdir()
            if d.is_dir() and d.name.startswith('output-')
        ], key=lambda x: int(x.name.split('-')[1]) if len(x.name.split('-')) > 1 and x.name.split('-')[1].isdigit() else 0)
        
        total_outputs += len(output_dirs)
        print(f"Processing {len(output_dirs)} outputs...\n")
        
        success = 0
        for output_dir in output_dirs:
            target_dir = target_exp_dir / output_dir.name
            result = process_output_directory(output_dir, target_dir)
            if result['success']:
                success += 1
            total_failed += result['files_failed']
        
        total_success += success
        print(f"✓ {exp_dir.name}: {success}/{len(output_dirs)} saved")
    
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Total: {total_success}/{total_outputs} outputs saved")
    if total_failed > 0:
        print(f"Files that failed extraction: {total_failed}")
    print(f"Output location: {output_parent_path}")
    print(f"{'='*80}\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract AsyncSTM code using file-specific patterns and save with normalized indentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Extraction patterns used for AsyncSTM:
  astm_config.hpp:        Last block (triple or single backtick) containing legion.h
  astm.hpp:               Combined sections (Code Structure, Transaction Class, Async Logic)
  *.cpp files:            Last ```cpp block (handles cutoff/unclosed outputs safely)
  makefile:               Last ```makefile block (handles cutoff safely), with duplicate lines removed
  ReadMe.md:              Last ```markdown block

All extracted code has indentation normalized (common leading whitespace removed).
Empty files will safely fallback to printing a commented error message.

Examples:
  # Single experiment
  python extract_pattern.py AsyncSTM --save-dir asyncstm_parsed
  
  # Batch all experiments
  python extract_pattern.py hpx_to_legion_repo --batch --save-dir all_parsed
        """
    )
    
    parser.add_argument(
        "input_dir",
        help="Input directory (experiment dir or parent of multiple experiments)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Output directory for extracted code (creates output-N/repo/ structure)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple experiment directories"
    )
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            save_multiple_experiments(args.input_dir, args.save_dir)
        else:
            save_experiment_directory(args.input_dir, args.save_dir)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)