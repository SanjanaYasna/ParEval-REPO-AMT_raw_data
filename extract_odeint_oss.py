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


def extract_last_cmake_block(full_text: str) -> str:
    """Extract CMakeLists.txt - last cmake code block, allowing for unclosed blocks."""
    # Find all starts of cmake blocks (handles optional trailing stuff and missing newline)
    pattern = r'```cmake(?:[^\n]*\n|$)'
    matches = list(re.finditer(pattern, full_text, re.IGNORECASE))

    if not matches:
        # Fallback: last code block of any kind, allowing for unclosed
        pattern = r'```(?:[^\n]*\n|$)'
        matches = list(re.finditer(pattern, full_text))
        if not matches:
            return "# ERROR: Could not extract CMakeLists.txt (no code blocks found)"

    # Take the last match
    last_match = matches[-1]
    content = full_text[last_match.end():]

    # If there's another fence, stop there (closed block); otherwise it's unclosed and runs to EOF
    closing_idx = content.find('```')
    if closing_idx != -1:
        content = content[:closing_idx]

    return normalize_indentation(content)


def extract_last_cpp_block_deduped(full_text: str) -> str:
    """Extract odeint.cpp - last .cpp code block, deduplicated."""
    # Find all cpp-tagged blocks
    blocks = re.findall(r'```(?:cpp|c\+\+)[^\n]*\n(.*?)```', full_text, re.DOTALL | re.IGNORECASE)
    
    if not blocks:
        # Fallback: any code block
        blocks = re.findall(r'```[^\n]*\n(.*?)```', full_text, re.DOTALL)
    
    if not blocks:
        # Handle unclosed last block
        pattern = r'```(?:cpp|c\+\+)(?:[^\n]*\n|$)'
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            content = full_text[last_match.end():]
            closing_idx = content.find('```')
            if closing_idx != -1:
                content = content[:closing_idx]
            return _deduplicate_lines(normalize_indentation(content))
        return "// ERROR: Could not extract odeint.cpp (no code blocks found)"
    
    content = blocks[-1]
    return _deduplicate_lines(normalize_indentation(content))


def _deduplicate_lines(text: str) -> str:
    """Remove exact duplicate lines while preserving order and blank lines."""
    lines = text.split('\n')
    seen = set()
    dedup_lines = []
    
    for line in lines:
        if not line.strip():
            dedup_lines.append(line)
        elif line not in seen:
            dedup_lines.append(line)
            seen.add(line)
    
    return '\n'.join(dedup_lines)

def extract_last_markdown_block(full_text: str) -> str:
    """Extract MakeFile/Readme - take everything after the last ```markdown/```md tag."""
    # Find all starts of markdown/md blocks
    pattern = r'```(?:markdown|md)(?:[^\n]*\n|$)'
    matches = list(re.finditer(pattern, full_text, re.IGNORECASE))

    if not matches:
        return "# ERROR: Could not extract MakeFile (no markdown blocks found)"

    # Take everything after the last markdown/md fence to EOF
    last_match = matches[-1]
    content = full_text[last_match.end():]

    return normalize_indentation(content)


def _find_cpp_block_after_header(full_text: str, header_pattern: str) -> Optional[str]:
    """Find the first cpp code block that appears after a given header pattern."""
    match = re.search(header_pattern, full_text, re.IGNORECASE)
    if not match:
        return None
    
    # Search for the next cpp block after this header
    remaining = full_text[match.end():]
    
    # Try explicit cpp-tagged block
    block_match = re.search(r'```(?:cpp|c\+\+)[^\n]*\n(.*?)```', remaining, re.DOTALL | re.IGNORECASE)
    if block_match:
        return normalize_indentation(block_match.group(1))
    
    # Try any code block
    block_match = re.search(r'```[^\n]*\n(.*?)```', remaining, re.DOTALL)
    if block_match:
        return normalize_indentation(block_match.group(1))
    
    # Handle unclosed block
    block_match = re.search(r'```(?:cpp|c\+\+)(?:[^\n]*\n|$)', remaining, re.IGNORECASE)
    if block_match:
        content = remaining[block_match.end():]
        closing_idx = content.find('```')
        if closing_idx != -1:
            content = content[:closing_idx]
        return normalize_indentation(content)
    
    return None


def extract_shared_operations_hpp(full_text: str) -> str:
    """Extract shared_operations.hpp - cpp code block after 'Code Structure:'."""
    result = _find_cpp_block_after_header(full_text, r'\*?Code Structure:?\*?')
    if result:
        return result
    
    # Fallback: last cpp block
    blocks = re.findall(r'```(?:cpp|c\+\+)[^\n]*\n(.*?)```', full_text, re.DOTALL | re.IGNORECASE)
    if blocks:
        return normalize_indentation(blocks[-1])
    
    return "// ERROR: Could not extract shared_operations.hpp (no 'Code Structure:' section found)"


def extract_shared_resize_hpp(full_text: str) -> str:
    """Extract shared_resize.hpp - combine cpp blocks after:
    1. 'Code Structure:'
    2. 'Task Registration:'
    3. 'The resize_impl implementation:'
    """
    parts = []
    
    headers = [
        r'\*?Code Structure:?\*?',
        r'\*?Task Registration:?\*?',
        r'\*?The resize_impl implementation:?\*?',
    ]
    
    labels = [
        "Code Structure",
        "Task Registration",
        "The resize_impl implementation",
    ]
    
    for header, label in zip(headers, labels):
        block = _find_cpp_block_after_header(full_text, header)
        if block:
            parts.append(block)
        else:
            parts.append(f"// WARNING: Could not find cpp block after '{label}'")
    
    if any(not p.startswith("// WARNING:") for p in parts):
        return "\n\n".join(parts)
    
    # Full fallback: last cpp block
    blocks = re.findall(r'```(?:cpp|c\+\+)[^\n]*\n(.*?)```', full_text, re.DOTALL | re.IGNORECASE)
    if blocks:
        return normalize_indentation(blocks[-1])
    
    return "// ERROR: Could not extract shared_resize.hpp"


def extract_generic_cpp(full_text: str) -> str:
    """Extract general cpp files - take the last cpp code block, allowing for unclosed blocks."""
    # Find all starts of cpp blocks
    pattern = r'```(?:cpp|c\+\+)(?:[^\n]*\n|$)'
    matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
    
    if not matches:
        # Fallback: ANY code blocks
        pattern = r'```(?:[^\n]*\n|$)'
        matches = list(re.finditer(pattern, full_text))
        
    if not matches:
        return "// ERROR: Could not extract (no code blocks found)"
        
    # Take the last match
    last_match = matches[-1]
    content = full_text[last_match.end():]
    
    # If there's a closing backtick sequence, stop there
    closing_idx = content.find('```')
    if closing_idx != -1:
        content = content[:closing_idx]
        
    return normalize_indentation(content)


def extract_by_filename(filename: str, full_text: str) -> str:
    """Route to appropriate extraction function based on filename and catch blank output."""
    filename_lower = filename.lower()
    
    extracted = ""
    if filename_lower == 'cmakelists.txt':
        extracted = extract_last_cmake_block(full_text)
    elif filename_lower == 'odeint.cpp':
        extracted = extract_last_cpp_block_deduped(full_text)
    elif filename_lower == 'makefile':
        extracted = extract_last_markdown_block(full_text)
    elif filename_lower == 'shared_operations.hpp':
        extracted = extract_shared_operations_hpp(full_text)
    elif filename_lower == 'shared_resize.hpp':
        extracted = extract_shared_resize_hpp(full_text)
    elif filename_lower in ['algebra.hpp', 'system']:
        extracted = extract_generic_cpp(full_text)
    else:
        # Generic fallback for unknown file types
        extracted = extract_generic_cpp(full_text)
        
    # GLOBAL SAFEGUARD: Ensure we never write a purely blank file
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
            elif fname_lower == 'cmakelists.txt':
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [last cmake block]")
            elif fname_lower == 'odeint.cpp':
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [last cpp block, deduplicated]")
            elif fname_lower == 'makefile':
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [last markdown block]")
            elif fname_lower == 'shared_operations.hpp':
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [cpp block after Code Structure:]")
            elif fname_lower == 'shared_resize.hpp':
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [combined: Code Structure + Task Registration + resize_impl]")
            elif fname_lower in ['algebra.hpp', 'system']:
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [last cpp block]")
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
        description="Extract code using file-specific patterns and save with normalized indentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Extraction patterns used:
  CMakeLists.txt:         Last ```cmake code block
  odeint.cpp:             Last ```cpp block, deduplicated
  MakeFile:               Last ```markdown block
  shared_operations.hpp:  cpp code block after "Code Structure:"
  shared_resize.hpp:      Combined cpp blocks after "Code Structure:", "Task Registration:",
                          and "The resize_impl implementation:"
  algebra.hpp, system:    Last ```cpp code block

All extracted code has indentation normalized (common leading whitespace removed).
Empty files will safely fallback to printing a commented error message.

Examples:
  # Single experiment
  python extract_pattern.py my_experiment --save-dir parsed_output
  
  # Batch all experiments
  python extract_pattern.py all_experiments --batch --save-dir all_parsed
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