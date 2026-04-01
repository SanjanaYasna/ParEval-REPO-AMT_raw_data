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
    
    Args:
        text: Text with potentially awkward indentation
        
    Returns:
        Text with normalized indentation
    """
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


def extract_heaps_cpp(full_text: str) -> str:
    """Extract heaps.cpp code after the specific pattern.
    
    Pattern: *Let's write the code.*</think>```cpp
    Takes everything after this marker until the closing ```
    """
    # Look for the pattern
    pattern = r'\*Let\'s write the code\.\*</think>```cpp\s*\n(.*?)```'
    match = re.search(pattern, full_text, re.DOTALL)
    
    if match:
        return normalize_indentation(match.group(1))
    
    # Fallback: look for just </think>```cpp
    pattern2 = r'</think>```cpp\s*\n(.*?)```'
    match2 = re.search(pattern2, full_text, re.DOTALL)
    if match2:
        return normalize_indentation(match2.group(1))
    
    # Last resort: find any ```cpp block
    cpp_blocks = re.findall(r'```cpp\s*\n(.*?)```', full_text, re.DOTALL)
    if cpp_blocks:
        return normalize_indentation(cpp_blocks[-1])
    
    return f"// ERROR: Could not extract heaps.cpp using pattern"


def extract_readme(full_text: str) -> str:
    """Extract ReadMe.md text after the specific pattern.
    
    Pattern: </think>```markdown
    Takes everything after this marker until the closing ```
    """
    pattern = r'</think>```markdown\s*\n(.*?)```'
    match = re.search(pattern, full_text, re.DOTALL)
    
    if match:
        return normalize_indentation(match.group(1))
    
    # Fallback: look for any ```markdown block
    md_blocks = re.findall(r'```markdown\s*\n(.*?)```', full_text, re.DOTALL)
    if md_blocks:
        return normalize_indentation(md_blocks[-1])
    
    # Fallback: look for ```md block
    md_blocks2 = re.findall(r'```md\s*\n(.*?)```', full_text, re.DOTALL)
    if md_blocks2:
        return normalize_indentation(md_blocks2[-1])
    
    return f"# ERROR: Could not extract ReadMe.md using pattern"


def extract_sift_hpp(full_text: str) -> str:
    """Extract sift.hpp code - last block after ```cpp (may be unclosed).
    
    Pattern: Takes the last ```cpp block (which may not have closing ```)
    """
    # First try: find all closed ```cpp blocks
    cpp_blocks = re.findall(r'```cpp\s*\n(.*?)```', full_text, re.DOTALL)
    if cpp_blocks:
        return normalize_indentation(cpp_blocks[-1])
    
    # Second try: find unclosed ```cpp block (take until end or next ```)
    pattern = r'```cpp\s*\n(.*)$'
    match = re.search(pattern, full_text, re.DOTALL)
    if match:
        content = match.group(1)
        # If there's a closing ```, stop there
        if '```' in content:
            content = content.split('```')[0]
        return normalize_indentation(content)
    
    return f"// ERROR: Could not extract sift.hpp using pattern"


def extract_cmake(full_text: str) -> str:
    """Extract CMakeLists.txt - last block of ```cmake.
    
    Pattern: Takes the last ```cmake block
    """
    cmake_blocks = re.findall(r'```cmake\s*\n(.*?)```', full_text, re.DOTALL)
    if cmake_blocks:
        return normalize_indentation(cmake_blocks[-1])
    
    return f"# ERROR: Could not extract CMakeLists.txt using pattern"


def extract_by_filename(filename: str, full_text: str) -> str:
    """Route to appropriate extraction function based on filename.
    
    Args:
        filename: Name of the file (e.g., "heaps.cpp")
        full_text: Full reasoning text
        
    Returns:
        Extracted code with normalized indentation
    """
    filename_lower = filename.lower()
    
    if 'heaps.cpp' in filename_lower:
        return extract_heaps_cpp(full_text)
    elif 'readme' in filename_lower or 'readme.md' in filename_lower:
        return extract_readme(full_text)
    elif 'sift.hpp' in filename_lower:
        return extract_sift_hpp(full_text)
    elif 'cmakelists' in filename_lower:
        return extract_cmake(full_text)
    else:
        # Generic: try to find last code block of any type
        all_blocks = re.findall(r'```[^\n]*?\n(.*?)```', full_text, re.DOTALL)
        if all_blocks:
            return normalize_indentation(all_blocks[-1])
        return f"// ERROR: Could not extract {filename} (no pattern matched)"


def process_output_directory(output_dir: Path, target_dir: Path) -> Dict[str, Any]:
    """Process a single output-N directory with pattern-based extraction.
    
    Args:
        output_dir: Source output-N directory containing experiment_metadata.json
        target_dir: Target directory (will create repo/ subdirectory)
        
    Returns:
        Dict with success status and stats
    """
    metadata_path = output_dir / "experiment_metadata.json"
    
    if not metadata_path.exists():
        print(f"  ✗ No experiment_metadata.json found in {output_dir}")
        return {'success': False, 'files_written': 0, 'files_failed': 0}
    
    # Load metadata
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
    
    # Extract and save each file using pattern matching
    files_written = 0
    files_failed = 0
    
    for filename, full_text in reasoning_texts.items():
        extracted_code = extract_by_filename(filename, full_text)
        
        # Check if extraction failed
        if extracted_code.startswith("// ERROR:") or extracted_code.startswith("# ERROR:"):
            files_failed += 1
        
        # Write to repo/ directory
        output_file = target_repo / filename
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_code)
            
            files_written += 1
            
            # Show what pattern was used
            if 'heaps.cpp' in filename.lower():
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [after '*Let's write*</think>```cpp']")
            elif 'readme' in filename.lower():
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [after '</think>```markdown']")
            elif 'sift.hpp' in filename.lower():
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [last ```cpp, unclosed ok]")
            elif 'cmakelists' in filename.lower():
                print(f"    ✓ {filename} ({len(extracted_code):,} chars) [last ```cmake]")
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
    """Save pattern-extracted code from an experiment directory.
    
    Args:
        root_dir: Path to experiment root directory (e.g., "heapify")
        output_dir: Optional output directory name.
                    If None, appends "_pattern_parsed" to root_dir name.
    """
    root_path = Path(root_dir).resolve()
    
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    # Determine output directory name
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
    
    # Find all output directories
    output_dirs = sorted([
        d for d in root_path.iterdir()
        if d.is_dir() and d.name.startswith('output-')
    ], key=lambda x: int(x.name.split('-')[1]) if len(x.name.split('-')) > 1 and x.name.split('-')[1].isdigit() else 0)
    
    if not output_dirs:
        print(f"⚠ No output-N directories found")
        return
    
    print(f"Found {len(output_dirs)} output directories\n")
    
    # Process each
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
    
    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully saved: {success_count}/{len(output_dirs)} outputs")
    if total_failed > 0:
        print(f"Files that failed extraction: {total_failed}")
    print(f"Output location: {output_path}")
    print(f"{'='*80}\n")


def save_multiple_experiments(parent_dir: str, output_parent: Optional[str] = None):
    """Save pattern-extracted code from multiple experiments.
    
    Args:
        parent_dir: Path to parent containing multiple experiment dirs
        output_parent: Optional output parent directory.
                       If None, appends "_pattern_parsed" to parent_dir name.
    """
    parent_path = Path(parent_dir).resolve()
    
    if not parent_path.exists():
        raise FileNotFoundError(f"Directory not found: {parent_dir}")
    
    # Determine output parent directory
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
    
    # Find all experiment directories
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
    
    # Process each experiment
    total_success = 0
    total_outputs = 0
    total_failed = 0
    
    for exp_dir in sorted(experiment_dirs):
        print(f"\n{'─'*80}")
        print(f"EXPERIMENT: {exp_dir.name}")
        print(f"{'─'*80}")
        
        target_exp_dir = output_parent_path / exp_dir.name
        
        # Find output directories
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
  heaps.cpp:        After "*Let's write the code.*</think>```cpp"
  ReadMe.md:        After "</think>```markdown"
  sift.hpp:         Last ```cpp block (may be unclosed)
  CMakeLists.txt:   Last ```cmake block

All extracted code has indentation normalized (common leading whitespace removed).

Examples:
  # Single experiment
  python extract_pattern.py heapify --save-dir heapify_parsed
  
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