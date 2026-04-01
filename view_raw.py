import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


def display_reasoning_texts(json_path: str):
    """Display all reasoning texts from an experiment_metadata.json file.
    
    Args:
        json_path: Path to experiment_metadata.json file
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    reasoning_texts = metadata.get("reasoning_texts", {})
    
    if not reasoning_texts:
        print("⚠ No reasoning_texts found in metadata")
        return
    
    # Display metadata summary
    print(f"\n{'='*80}")
    print(f"METADATA")
    print(f"{'='*80}")
    print(f"App:            {metadata.get('app', 'N/A')}")
    print(f"Model:          {metadata.get('llm_name', 'N/A')}")
    print(f"Source:         {metadata.get('source_model', 'N/A')}")
    print(f"Dest:           {metadata.get('dest_model', 'N/A')}")
    print(f"Output #:       {metadata.get('output_number', 'N/A')}")
    print(f"Prompt strat:   {metadata.get('prompt_strategy', 'N/A')}")
    print(f"Files:          {len(reasoning_texts)}")
    print(f"{'='*80}\n")
    
    # Display each file's reasoning text
    for filename, full_text in reasoning_texts.items():
        print(f"\n{'╔'*80}")
        print(f"║ FILE: {filename}")
        print(f"║ Length: {len(full_text):,} characters")
        print(f"╚{'═'*79}")
        print()
        print(full_text)
        print()
        print(f"{'─'*80}")
        print(f"END OF {filename}")
        print(f"{'─'*80}\n")


def save_output_directory(output_dir: Path, target_dir: Path) -> Dict[str, Any]:
    """Save all reasoning texts (FULL, unparsed) from an output directory.
    
    Args:
        output_dir: Source output-N directory containing experiment_metadata.json
        target_dir: Target directory (will create repo/ subdirectory)
        
    Returns:
        Dict with success status and stats
    """
    metadata_path = output_dir / "experiment_metadata.json"
    
    if not metadata_path.exists():
        print(f"  ✗ No experiment_metadata.json found in {output_dir}")
        return {'success': False, 'files_written': 0}
    
    # Load metadata
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"  ✗ Failed to load metadata: {e}")
        return {'success': False, 'files_written': 0}
    
    reasoning_texts = metadata.get("reasoning_texts", {})
    
    if not reasoning_texts:
        print(f"  ⚠ No reasoning_texts in {output_dir.name}")
        return {'success': False, 'files_written': 0}
    
    # Create target directory structure
    target_dir.mkdir(parents=True, exist_ok=True)
    target_repo = target_dir / "repo"
    target_repo.mkdir(exist_ok=True)
    
    # Copy experiment_metadata.json
    shutil.copy2(metadata_path, target_dir / "experiment_metadata.json")
    
    # Save each reasoning text AS-IS (no parsing)
    files_written = 0
    
    for filename, full_text in reasoning_texts.items():
        output_file = target_repo / filename
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            files_written += 1
            print(f"    ✓ {filename} ({len(full_text):,} chars)")
                
        except Exception as e:
            print(f"    ✗ {filename}: Failed to write - {e}")
    
    if files_written > 0:
        print(f"  → Saved {files_written} file(s) to {target_dir.name}/repo/")
    
    return {
        'success': files_written > 0,
        'files_written': files_written
    }


def save_experiment_directory(root_dir: str, output_dir: Optional[str] = None):
    """Save all reasoning texts (FULL, unparsed) from an experiment directory.
    
    Args:
        root_dir: Path to experiment root directory (e.g., "heapify")
        output_dir: Optional output directory name.
                    If None, appends "_full" to root_dir name.
    
    Example:
        Input:  heapify/output-0/experiment_metadata.json
        Output: heapify_full/output-0/experiment_metadata.json (copy)
                heapify_full/output-0/repo/ReadME.md (FULL reasoning text)
                heapify_full/output-0/repo/heaps.cpp (FULL reasoning text)
    """
    root_path = Path(root_dir).resolve()
    
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    # Determine output directory name
    if output_dir is None:
        output_path = root_path.parent / f"{root_path.name}_full"
    else:
        output_path = Path(output_dir).resolve()
    
    print(f"\n{'='*80}")
    print(f"SAVING FULL REASONING TEXTS (NO PARSING)")
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
    
    for output_dir in output_dirs:
        print(f"Processing {output_dir.name}...")
        target_dir = output_path / output_dir.name
        
        result = save_output_directory(output_dir, target_dir)
        if result['success']:
            success_count += 1
        print()
    
    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully saved: {success_count}/{len(output_dirs)} outputs")
    print(f"Output location: {output_path}")
    print(f"{'='*80}\n")


def save_multiple_experiments(parent_dir: str, output_parent: Optional[str] = None):
    """Save full reasoning texts from multiple experiments.
    
    Args:
        parent_dir: Path to parent containing multiple experiment dirs
        output_parent: Optional output parent directory.
                       If None, appends "_full" to parent_dir name.
    
    Example:
        Input:  hpx_to_legion_repo/heapify/output-0/experiment_metadata.json
                hpx_to_legion_repo/osc_chain_1d/output-0/experiment_metadata.json
        Output: hpx_to_legion_repo_full/heapify/output-0/repo/heaps.cpp (FULL text)
                hpx_to_legion_repo_full/osc_chain_1d/output-0/repo/odeint.cpp (FULL text)
    """
    parent_path = Path(parent_dir).resolve()
    
    if not parent_path.exists():
        raise FileNotFoundError(f"Directory not found: {parent_dir}")
    
    # Determine output parent directory
    if output_parent is None:
        output_parent_path = parent_path.parent / f"{parent_path.name}_full"
    else:
        output_parent_path = Path(output_parent).resolve()
    
    print(f"\n{'='*80}")
    print(f"BATCH SAVING FULL REASONING TEXTS (NO PARSING)")
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
            result = save_output_directory(output_dir, target_dir)
            if result['success']:
                success += 1
        
        total_success += success
        print(f"✓ {exp_dir.name}: {success}/{len(output_dirs)} saved")
    
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Total: {total_success}/{total_outputs} outputs saved")
    print(f"Output location: {output_parent_path}")
    print(f"{'='*80}\n")


def display_output_directory(output_dir: Path):
    """Display reasoning texts from a single output-N directory.
    
    Args:
        output_dir: Path to output-N directory
    """
    metadata_path = output_dir / "experiment_metadata.json"
    
    if not metadata_path.exists():
        print(f"✗ No experiment_metadata.json found in {output_dir}")
        return
    
    print(f"\n{'█'*80}")
    print(f"█ OUTPUT DIRECTORY: {output_dir.name}")
    print(f"{'█'*80}")
    
    display_reasoning_texts(str(metadata_path))


def display_experiment_directory(root_dir: str, output_number: Optional[int] = None):
    """Display reasoning texts from experiment directory.
    
    Args:
        root_dir: Path to experiment root directory
        output_number: Optional specific output number to display
    """
    root_path = Path(root_dir).resolve()
    
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    print(f"\n{'█'*80}")
    print(f"█ EXPERIMENT: {root_path.name}")
    print(f"█ Path: {root_path}")
    print(f"{'█'*80}")
    
    # Find output directories
    if output_number is not None:
        output_dir = root_path / f"output-{output_number}"
        if not output_dir.exists():
            print(f"✗ output-{output_number} not found")
            return
        display_output_directory(output_dir)
    else:
        output_dirs = sorted([
            d for d in root_path.iterdir()
            if d.is_dir() and d.name.startswith('output-')
        ], key=lambda x: int(x.name.split('-')[1]) if len(x.name.split('-')) > 1 and x.name.split('-')[1].isdigit() else 0)
        
        if not output_dirs:
            print(f"⚠ No output-N directories found")
            return
        
        print(f"\nFound {len(output_dirs)} output directories")
        
        for output_dir in output_dirs:
            display_output_directory(output_dir)


def display_multiple_experiments(parent_dir: str):
    """Display reasoning texts from multiple experiments.
    
    Args:
        parent_dir: Path to parent directory
    """
    parent_path = Path(parent_dir).resolve()
    
    if not parent_path.exists():
        raise FileNotFoundError(f"Directory not found: {parent_dir}")
    
    print(f"\n{'█'*80}")
    print(f"█ BATCH DISPLAY")
    print(f"█ Parent: {parent_path}")
    print(f"{'█'*80}")
    
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
    
    print(f"\nFound {len(experiment_dirs)} experiments:")
    for exp in sorted(experiment_dirs):
        print(f"  - {exp.name}")
    
    for exp_dir in sorted(experiment_dirs):
        display_experiment_directory(str(exp_dir))


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Display or save full reasoning texts (NO PARSING) from experiment metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DISPLAY MODE (shows full reasoning text to console):
  python script.py heapify                      # Display all outputs
  python script.py heapify --output 0           # Display only output-0
  python script.py hpx_to_legion_repo --batch   # Display all experiments
  
  # SAVE MODE (saves FULL reasoning text to repo/ subdirs):
  python script.py heapify --save-dir heapify_full
  python script.py hpx_to_legion_repo --batch --save-dir all_full_texts
  
Note: NO parsing or extraction is performed. All text is saved/displayed as-is.
        """
    )
    
    parser.add_argument(
        "input_dir",
        help="Input directory (experiment dir or parent of multiple experiments)"
    )
    parser.add_argument(
        "-n", "--output",
        type=int,
        help="Specific output number to display (e.g., 0 for output-0)",
        default=None
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Save FULL reasoning texts to directory (creates output-N/repo/ structure)",
        default=None
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple experiment directories"
    )
    
    args = parser.parse_args()
    
    try:
        if args.save_dir:
            # SAVE MODE: Save full reasoning text to repo/ subdirectories
            if args.batch:
                save_multiple_experiments(args.input_dir, args.save_dir)
            else:
                save_experiment_directory(args.input_dir, args.save_dir)
        else:
            # DISPLAY MODE: Show full reasoning text to console
            if args.batch:
                display_multiple_experiments(args.input_dir)
            else:
                display_experiment_directory(args.input_dir, args.output)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)