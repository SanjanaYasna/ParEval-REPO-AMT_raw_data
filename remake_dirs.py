import re
import json
import os
import shutil
from typing import Dict, Any, Optional
from pathlib import Path


def extract_code_blocks_from_text(full_text: str) -> str:
    """Extract the last code block from reasoning text.
    
    Args:
        full_text: Full LLM output with prompt repetition and reasoning
        
    Returns:
        Extracted code string (last code block content)
    """
    # Find all fenced code blocks (```...```)
    code_blocks = re.findall(
        r'```(?:[a-zA-Z+#]*)\n(.*?)\n```',
        full_text,
        re.DOTALL
    )
    
    if code_blocks:
        # Take the LAST code block (the final generated code after reasoning)
        return code_blocks[-1].strip()
    
    # Fallback: try to find content after "assistantfinal"
    final_match = re.search(r'assistantfinal(.+)', full_text, re.DOTALL)
    if final_match:
        final_section = final_match.group(1)
        final_blocks = re.findall(
            r'```(?:[a-zA-Z+#]*)\n(.*?)\n```',
            final_section,
            re.DOTALL
        )
        if final_blocks:
            return final_blocks[-1].strip()
        return final_section.strip()
    
    # Last resort: return full text
    return full_text


def process_output_directory(output_dir: Path, target_dir: Path) -> bool:
    """Process a single output-N directory.
    
    Args:
        output_dir: Source output-N directory containing experiment_metadata.json
        target_dir: Target output-N directory to create with extracted code
        
    Returns:
        True if successful, False otherwise
    """
    metadata_path = output_dir / "experiment_metadata.json"
    
    if not metadata_path.exists():
        print(f"  ✗ No experiment_metadata.json found in {output_dir}")
        return False
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    reasoning_texts = metadata.get("reasoning_texts", {})
    
    if not reasoning_texts:
        print(f"  ⚠ No reasoning_texts in {output_dir.name}")
        return False
    
    # Create target directory structure
    target_dir.mkdir(parents=True, exist_ok=True)
    target_repo = target_dir / "repo"
    target_repo.mkdir(exist_ok=True)
    
    # Copy experiment_metadata.json to target
    shutil.copy2(metadata_path, target_dir / "experiment_metadata.json")
    
    # Extract and write each file
    files_written = 0
    for filename, full_text in reasoning_texts.items():
        extracted_code = extract_code_blocks_from_text(full_text)
        
        # Write to repo/ directory
        output_file = target_repo / filename
        with open(output_file, 'w') as f:
            f.write(extracted_code)
        
        files_written += 1
        print(f"    ✓ {filename} ({len(extracted_code):,} chars)")
    
    print(f"  → Wrote {files_written} file(s) to {target_dir.name}/repo/")
    return True


def reparse_experiment_directory(root_dir: str, output_dir: Optional[str] = None):
    """Reparse an entire experiment directory, extracting generated code.
    
    Takes a root directory containing output-0, output-1, ... subdirectories,
    extracts the generated code from each experiment_metadata.json, and creates
    a parallel directory structure with the extracted code files.
    
    Args:
        root_dir: Path to root directory (e.g., "osc_chain_1d")
        output_dir: Optional custom output directory name. 
                    If None, appends "_reparsed" to root_dir name.
    
    Example:
        Input:  osc_chain_1d/output-0/experiment_metadata.json
                osc_chain_1d/output-0/repo/
                osc_chain_1d/output-1/experiment_metadata.json
                ...
        
        Output: osc_chain_1d_reparsed/output-0/experiment_metadata.json (copy)
                osc_chain_1d_reparsed/output-0/repo/CMakeLists.txt (extracted)
                osc_chain_1d_reparsed/output-0/repo/odeint.cpp (extracted)
                ...
    """
    root_path = Path(root_dir).resolve()
    
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    # Determine output directory name
    if output_dir is None:
        output_path = root_path.parent / f"{root_path.name}_reparsed"
    else:
        output_path = Path(output_dir).resolve()
    
    print(f"\n{'='*80}")
    print(f"REPARSING EXPERIMENT DIRECTORY")
    print(f"{'='*80}")
    print(f"Source:      {root_path}")
    print(f"Destination: {output_path}")
    print(f"{'='*80}\n")
    
    # Find all output-N directories
    output_dirs = sorted([
        d for d in root_path.iterdir()
        if d.is_dir() and d.name.startswith('output-')
    ])
    
    if not output_dirs:
        print(f"⚠ No output-N directories found in {root_dir}")
        return
    
    print(f"Found {len(output_dirs)} output directories\n")
    
    # Process each output directory
    success_count = 0
    for output_dir in output_dirs:
        print(f"Processing {output_dir.name}...")
        
        target_dir = output_path / output_dir.name
        
        if process_output_directory(output_dir, target_dir):
            success_count += 1
        
        print()
    
    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed: {success_count}/{len(output_dirs)} output directories")
    print(f"Output location: {output_path}")
    print(f"{'='*80}\n")


def reparse_multiple_experiments(parent_dir: str, output_parent: Optional[str] = None):
    """Reparse multiple experiment directories in a parent directory.
    
    Args:
        parent_dir: Path to parent containing multiple experiment dirs
                    (e.g., "hpx_to_legion_repo" containing "osc_chain_1d", "heapify", etc.)
        output_parent: Optional custom output parent directory.
                       If None, appends "_reparsed" to parent_dir name.
    
    Example:
        Input:  hpx_to_legion_repo/osc_chain_1d/output-0/...
                hpx_to_legion_repo/heapify/output-0/...
        
        Output: hpx_to_legion_repo_reparsed/osc_chain_1d/output-0/...
                hpx_to_legion_repo_reparsed/heapify/output-0/...
    """
    parent_path = Path(parent_dir).resolve()
    
    if not parent_path.exists():
        raise FileNotFoundError(f"Directory not found: {parent_dir}")
    
    # Determine output parent directory
    if output_parent is None:
        output_parent_path = parent_path.parent / f"{parent_path.name}_reparsed"
    else:
        output_parent_path = Path(output_parent).resolve()
    
    print(f"\n{'='*80}")
    print(f"BATCH REPARSING MULTIPLE EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Source parent:      {parent_path}")
    print(f"Destination parent: {output_parent_path}")
    print(f"{'='*80}\n")
    
    # Find all experiment directories (those containing output-N subdirs)
    experiment_dirs = []
    for item in parent_path.iterdir():
        if item.is_dir():
            # Check if it contains output-N directories
            has_outputs = any(
                d.is_dir() and d.name.startswith('output-')
                for d in item.iterdir()
            )
            if has_outputs:
                experiment_dirs.append(item)
    
    if not experiment_dirs:
        print(f"⚠ No experiment directories found in {parent_dir}")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directories:\n")
    for exp_dir in experiment_dirs:
        print(f"  - {exp_dir.name}")
    print()
    
    # Process each experiment directory
    for exp_dir in experiment_dirs:
        print(f"\n{'─'*80}")
        print(f"EXPERIMENT: {exp_dir.name}")
        print(f"{'─'*80}")
        
        target_exp_dir = output_parent_path / exp_dir.name
        
        # Find output directories
        output_dirs = sorted([
            d for d in exp_dir.iterdir()
            if d.is_dir() and d.name.startswith('output-')
        ])
        
        print(f"Processing {len(output_dirs)} outputs...")
        
        success = 0
        for output_dir in output_dirs:
            target_dir = target_exp_dir / output_dir.name
            if process_output_directory(output_dir, target_dir):
                success += 1
        
        print(f"✓ {exp_dir.name}: {success}/{len(output_dirs)} successful\n")
    
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"Output location: {output_parent_path}")
    print(f"{'='*80}\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract generated code from experiment reasoning texts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment directory
  python extract_code.py osc_chain_1d
  python extract_code.py osc_chain_1d -o osc_chain_1d_clean
  
  # Batch process multiple experiments
  python extract_code.py hpx_to_legion_repo --batch
  python extract_code.py hpx_to_legion_repo --batch -o hpx_to_legion_clean
        """
    )
    
    parser.add_argument(
        "input_dir",
        help="Input directory (experiment dir or parent of multiple experiments)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: input_dir + '_reparsed')",
        default=None
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple experiment directories in the input parent directory"
    )
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            reparse_multiple_experiments(args.input_dir, args.output)
        else:
            reparse_experiment_directory(args.input_dir, args.output)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)