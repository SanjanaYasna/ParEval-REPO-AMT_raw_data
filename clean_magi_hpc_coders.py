import json
import os
import re
import shutil
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional


def normalize_indentation(text: str) -> str:
    """Remove common leading whitespace from all lines."""
    if not text or not text.strip():
        return ""
        
    dedented = textwrap.dedent(text)
    lines = dedented.split('\n')
    
    indents = []
    for i, line in enumerate(lines[1:], 1):
        if line.strip():
            indents.append(len(line) - len(line.lstrip()))
    
    if indents:
        min_indent = min(indents)
        if min_indent > 0:
            normalized_lines = [lines[0]]
            for line in lines[1:]:
                if line.strip():
                    normalized_lines.append(line[min_indent:] if len(line) > min_indent else line)
                else:
                    normalized_lines.append(line)
            return '\n'.join(normalized_lines)
    
    return dedented


def extract_code_block(filename: str, full_text: str) -> str:
    """Extract code based on @@ Response or Assistant: triggers."""
    content = ""
    trigger_found = False
    
    # 1. Try "@@ Response" (case insensitive, optional space)
    resp_match = re.search(r'@@\s*Response', full_text, re.IGNORECASE)
    if resp_match:
        trigger_found = True
        post_text = full_text[resp_match.end():]
        # Find the FIRST code block after @@ Response
        block_match = re.search(r'```[^\n]*\n(.*)', post_text, re.DOTALL)
        if block_match:
            content = block_match.group(1)
            closing_idx = content.find('```')
            if closing_idx != -1:
                content = content[:closing_idx]
                
    # 2. Try "Assistant:" if @@ Response wasn't found
    elif "Assistant:" in full_text or "Assistant :" in full_text:
        trigger_found = True
        idx1 = full_text.rfind("Assistant:")
        idx2 = full_text.rfind("Assistant :")
        idx = max(idx1, idx2)
        
        post_text = full_text[idx:]
        # Find ALL code blocks after Assistant, take the LAST one
        matches = list(re.finditer(r'```[^\n]*\n', post_text))
        if matches:
            last_match = matches[-1]
            content = post_text[last_match.end():]
            closing_idx = content.find('```')
            if closing_idx != -1:
                content = content[:closing_idx]
                
    # 3. Generic Fallback: ONLY if no triggers were found at all
    # (Prevents accidentally grabbing prompt code if the LLM output is missing code blocks)
    if not trigger_found and not content:
        matches = list(re.finditer(r'```[^\n]*\n', full_text))
        if matches:
            last_match = matches[-1]
            content = full_text[last_match.end():]
            closing_idx = content.find('```')
            if closing_idx != -1:
                content = content[:closing_idx]
                
    # 4. Handle empty results (return empty string, no error comments)
    if not content.strip():
        return ""
        
    # 5. File-specific post-processing (Makefile deduplication)
    if 'makefile' in filename.lower():
        lines = content.split('\n')
        seen = set()
        dedup_lines = []
        for line in lines:
            if not line.strip():
                dedup_lines.append(line)
            elif line not in seen:
                dedup_lines.append(line)
                seen.add(line)
        content = '\n'.join(dedup_lines)
        
    return normalize_indentation(content)


def process_output_directory(output_dir: Path, target_dir: Path) -> Dict[str, Any]:
    """Process a single output-N directory with pattern-based extraction."""
    metadata_path = output_dir / "experiment_metadata.json"
    
    if not metadata_path.exists():
        print(f"  ✗ No experiment_metadata.json found in {output_dir}")
        return {'success': False, 'files_written': 0, 'files_empty': 0}
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"  ✗ Failed to load metadata: {e}")
        return {'success': False, 'files_written': 0, 'files_empty': 0}
    
    reasoning_texts = metadata.get("reasoning_texts", {})
    
    if not reasoning_texts:
        print(f"  ⚠ No reasoning_texts in {output_dir.name}")
        return {'success': False, 'files_written': 0, 'files_empty': 0}
    
    # Create target directory structure
    target_dir.mkdir(parents=True, exist_ok=True)
    target_repo = target_dir / "repo"
    target_repo.mkdir(exist_ok=True)
    
    # Copy experiment_metadata.json
    shutil.copy2(metadata_path, target_dir / "experiment_metadata.json")
    
    files_written = 0
    files_empty = 0
    
    for filename, full_text in reasoning_texts.items():
        extracted_code = extract_code_block(filename, full_text)
        
        output_file = target_repo / filename
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_code)
            
            files_written += 1
            
            if not extracted_code.strip():
                print(f"    ⚠ {filename}: No code block found after trigger, wrote empty file")
                files_empty += 1
            else:
                print(f"    ✓ {filename} ({len(extracted_code):,} chars)")
                
        except Exception as e:
            print(f"    ✗ {filename}: Failed to write - {e}")
            files_empty += 1
    
    if files_written > 0:
        status = f"{files_written} files processed"
        if files_empty > 0:
            status += f" ({files_empty} empty)"
        print(f"  → {status} to {target_dir.name}/repo/")
    
    return {
        'success': files_written > 0,
        'files_written': files_written,
        'files_empty': files_empty
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
    print(f"TRIGGER-BASED CODE EXTRACTION (@@ Response / Assistant:)")
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
    total_empty = 0
    
    for output_dir in output_dirs:
        print(f"Processing {output_dir.name}...")
        target_dir = output_path / output_dir.name
        
        result = process_output_directory(output_dir, target_dir)
        if result['success']:
            success_count += 1
        total_empty += result['files_empty']
        print()
    
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully saved: {success_count}/{len(output_dirs)} outputs")
    if total_empty > 0:
        print(f"Files resulting in empty extraction: {total_empty}")
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
    print(f"BATCH TRIGGER-BASED CODE EXTRACTION")
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
    total_empty = 0
    
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
            total_empty += result['files_empty']
        
        total_success += success
        print(f"✓ {exp_dir.name}: {success}/{len(output_dirs)} saved")
    
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Total: {total_success}/{total_outputs} outputs saved")
    if total_empty > 0:
        print(f"Files resulting in empty extraction: {total_empty}")
    print(f"Output location: {output_parent_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract code blocks following '@@ Response' or 'Assistant:' markers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Extraction logic:
  1. Searches for "@@ Response" -> Grabs the FIRST code block after it.
  2. Searches for "Assistant:" -> Grabs the LAST code block after it.
  3. Fallback: Grabs the last code block ONLY if no triggers were found.
  4. If no code blocks are found post-trigger, outputs a completely empty file.
  
Special Cases:
  makefile: Deduplicates exact line matches while retaining structure.

Examples:
  # Single experiment
  python extract_response.py my_experiment --save-dir my_parsed
  
  # Batch all experiments
  python extract_response.py parent_dir --batch --save-dir all_parsed
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