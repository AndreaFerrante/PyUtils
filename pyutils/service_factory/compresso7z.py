import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


def check_7zip_installation() -> str:
    """
    Check if 7zip is installed and return the command to use.
    
    Returns:
        Command string for 7zip
        
    Raises:
        FileNotFoundError: If 7zip is not found
    """
    # Try different possible 7zip commands
    possible_commands = ['7z', '7za', '7zz', 'p7zip']
    
    for cmd in possible_commands:
        if shutil.which(cmd):
            return cmd
    
    raise FileNotFoundError(
        "7zip not found. Please install 7zip:\n"
        "  Ubuntu/Debian: sudo apt install p7zip-full\n"
        "  CentOS/RHEL: sudo yum install p7zip\n"
        "  macOS: brew install p7zip\n"
        "  Windows: Download from https://7-zip.org/"
    )


def get_txt_files(directory: Path) -> List[Path]:
    """
    Get all .txt files from the specified directory.
    
    Args:
        directory: Path to the directory to scan
        
    Returns:
        List of Path objects for .txt files
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a directory")
    
    txt_files = list(directory.glob("*.txt"))
    return txt_files


def get_compression_args(level: int) -> List[str]:
    """
    Get 7zip compression arguments based on level.
    
    Args:
        level: Compression level (1-9)
        
    Returns:
        List of 7zip arguments for the specified compression level
    """
    # 7zip compression level mapping
    compression_configs = {
        1: ['-mx1', '-mfb=32', '-md=1m'],      # Ultra fast
        2: ['-mx3', '-mfb=32', '-md=2m'],      # Fast
        3: ['-mx5', '-mfb=32', '-md=4m'],      # Normal
        4: ['-mx5', '-mfb=64', '-md=8m'],      # Normal+
        5: ['-mx7', '-mfb=64', '-md=16m'],     # Maximum
        6: ['-mx7', '-mfb=128', '-md=32m'],    # Maximum+
        7: ['-mx9', '-mfb=128', '-md=64m'],    # Ultra
        8: ['-mx9', '-mfb=273', '-md=128m'],   # Ultra+
        9: ['-mx9', '-mfb=273', '-md=512m']    # Ultra maximum
    }
    
    return compression_configs.get(level, compression_configs[6])


def compress_file(input_path: Path, compression_level: int, output_dir: Path, 
                 sevenz_cmd: str, archive_format: str = '7z') -> Tuple[Path, bool]:
    """
    Compress a single text file using 7zip.
    
    Args:
        input_path: Path to the input .txt file
        compression_level: Compression level (1-9)
        output_dir: Output directory
        sevenz_cmd: 7zip command to use
        archive_format: Archive format (7z, zip, tar, etc.)
        
    Returns:
        Tuple of (output_path, success_status)
    """
    # Create output filename
    if archive_format == '7z':
        output_filename = input_path.stem + ".txt.7z"
    elif archive_format == 'zip':
        output_filename = input_path.stem + ".txt.zip"
    else:
        output_filename = input_path.stem + f".txt.{archive_format}"
    
    output_path = output_dir / output_filename
    
    try:
        # Build 7zip command
        compression_args = get_compression_args(compression_level)
        
        cmd = [
            sevenz_cmd,
            'a',  # Add to archive
            f'-t{archive_format}',  # Archive type
            *compression_args,
            '-y',  # Assume yes for all queries
            str(output_path),
            str(input_path)
        ]
        
        # Execute compression
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"✗ Failed to compress {input_path.name}")
            print(f"  Error: {result.stderr.strip()}")
            return output_path, False
        
        # Get file sizes for reporting
        original_size = input_path.stat().st_size
        compressed_size = output_path.stat().st_size if output_path.exists() else 0
        compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        print(f"✓ Compressed: {input_path.name}")
        print(f"  Original size: {original_size:,} bytes")
        print(f"  Compressed size: {compressed_size:,} bytes")
        print(f"  Compression ratio: {compression_ratio:.1f}%")
        print()
        
        return output_path, True
        
    except Exception as e:
        print(f"✗ Failed to compress {input_path.name}: {str(e)}", file=sys.stderr)
        return output_path, False


def compress_multiple_files(input_files: List[Path], compression_level: int, 
                          output_dir: Path, sevenz_cmd: str, 
                          archive_name: str = None, archive_format: str = '7z') -> Tuple[Path, bool]:
    """
    Compress multiple files into a single archive.
    
    Args:
        input_files: List of files to compress
        compression_level: Compression level (1-9)
        output_dir: Output directory
        sevenz_cmd: 7zip command to use
        archive_name: Name for the archive (without extension)
        archive_format: Archive format
        
    Returns:
        Tuple of (output_path, success_status)
    """
    if archive_name is None:
        archive_name = "compressed_texts"
    
    output_filename = f"{archive_name}.{archive_format}"
    output_path = output_dir / output_filename
    
    try:
        # Build 7zip command
        compression_args = get_compression_args(compression_level)
        
        cmd = [
            sevenz_cmd,
            'a',  # Add to archive
            f'-t{archive_format}',  # Archive type
            *compression_args,
            '-y',  # Assume yes for all queries
            str(output_path)
        ] + [str(f) for f in input_files]
        
        # Execute compression
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"✗ Failed to create archive {output_filename}")
            print(f"  Error: {result.stderr.strip()}")
            return output_path, False
        
        # Calculate compression statistics
        total_original_size = sum(f.stat().st_size for f in input_files)
        compressed_size = output_path.stat().st_size if output_path.exists() else 0
        compression_ratio = (1 - compressed_size / total_original_size) * 100 if total_original_size > 0 else 0
        
        print(f"✓ Created archive: {output_filename}")
        print(f"  Files compressed: {len(input_files)}")
        print(f"  Total original size: {total_original_size:,} bytes")
        print(f"  Compressed size: {compressed_size:,} bytes")
        print(f"  Compression ratio: {compression_ratio:.1f}%")
        print()
        
        return output_path, True
        
    except Exception as e:
        print(f"✗ Failed to create archive: {str(e)}", file=sys.stderr)
        return output_path, False


def main():
    """Main function to handle CLI arguments and coordinate compression."""
    
    parser = argparse.ArgumentParser(
        description="Compress all .txt files in a directory using 7zip compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Compression Levels:
  1-2: Fast compression (lower ratios, faster)
  3-5: Balanced compression (good ratio vs speed)
  6-7: High compression (better ratios, slower)
  8-9: Ultra compression (maximum ratios, slowest)

Examples:
  %(prog)s /path/to/txt/files --level 6
  %(prog)s ./documents --level 9 --output ./compressed --format zip
  %(prog)s ~/texts -l 1 --remove-original --single-archive
  %(prog)s ./files --level 8 --archive-name "my_texts"
        """
    )
    
    parser.add_argument(
        'directory',
        type=Path,
        help='Directory containing .txt files to compress'
    )
    
    parser.add_argument(
        '-l', '--level',
        type=int,
        choices=range(1, 10),
        default=6,
        help='Compression level (1=fastest, 9=maximum compression, default=6)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output directory for compressed files (default: same as input directory)'
    )
    
    parser.add_argument(
        '-f', '--format',
        choices=['7z', 'zip', 'tar', 'gzip'],
        default='7z',
        help='Archive format (default: 7z)'
    )
    
    parser.add_argument(
        '--single-archive',
        action='store_true',
        help='Compress all files into a single archive instead of individual files'
    )
    
    parser.add_argument(
        '--archive-name',
        type=str,
        help='Name for single archive (used with --single-archive)'
    )
    
    parser.add_argument(
        '--remove-original',
        action='store_true',
        help='Remove original .txt files after successful compression'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what files would be compressed without actually doing it'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Check 7zip installation
    try:
        sevenz_cmd = check_7zip_installation()
        if args.verbose:
            print(f"Using 7zip command: {sevenz_cmd}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Validate and prepare directories
    try:
        input_dir = args.directory.resolve()
        txt_files = get_txt_files(input_dir)
        
        if not txt_files:
            print(f"No .txt files found in '{input_dir}'")
            return 0
        
        if args.output:
            output_dir = args.output.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = input_dir
            
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error preparing directories: {e}", file=sys.stderr)
        return 1
    
    # Display summary
    print(f"Found {len(txt_files)} .txt file(s) in '{input_dir}'")
    print(f"Compression level: {args.level}")
    print(f"Archive format: {args.format}")
    print(f"Output directory: '{output_dir}'")
    
    if args.single_archive:
        archive_name = args.archive_name or "compressed_texts"
        print(f"Mode: Single archive ({archive_name}.{args.format})")
    else:
        print("Mode: Individual file compression")
    
    if args.dry_run:
        print("\n--- DRY RUN MODE ---")
        print("Files that would be compressed:")
        for txt_file in txt_files:
            print(f"  - {txt_file.name}")
        return 0
    
    if args.verbose:
        print(f"Remove original files: {'Yes' if args.remove_original else 'No'}")
    
    print()
    
    # Process files
    if args.single_archive:
        # Compress all files into a single archive
        output_path, success = compress_multiple_files(
            txt_files, args.level, output_dir, sevenz_cmd, 
            args.archive_name, args.format
        )
        
        if success and args.remove_original:
            for txt_file in txt_files:
                txt_file.unlink()
                if args.verbose:
                    print(f"Removed original: {txt_file.name}")
        
        successful_compressions = 1 if success else 0
        failed_compressions = [] if success else [("Archive creation", "Failed")]
        
    else:
        # Compress files individually
        successful_compressions = []
        failed_compressions = []
        
        for txt_file in txt_files:
            try:
                if args.verbose:
                    print(f"Processing: {txt_file.name}")
                
                compressed_path, success = compress_file(
                    txt_file, args.level, output_dir, sevenz_cmd, args.format
                )
                
                if success:
                    successful_compressions.append((txt_file, compressed_path))
                    
                    # Remove original file if requested
                    if args.remove_original:
                        txt_file.unlink()
                        if args.verbose:
                            print(f"  Removed original: {txt_file.name}")
                else:
                    failed_compressions.append((txt_file, "Compression failed"))
                    
            except Exception as e:
                failed_compressions.append((txt_file, str(e)))
                if args.verbose:
                    print(f"  Error: {e}")
                continue
    
    # Final summary
    print("=" * 50)
    print(f"7zip compression completed!")
    
    if args.single_archive:
        print(f"Archive created: {'Yes' if successful_compressions else 'No'}")
        if successful_compressions:
            print(f"Files in archive: {len(txt_files)}")
    else:
        print(f"Successfully compressed: {len(successful_compressions)} files")
    
    if failed_compressions:
        print(f"Failed operations: {len(failed_compressions)}")
        print("\nFailed items:")
        for failed_item, error in failed_compressions:
            item_name = failed_item.name if hasattr(failed_item, 'name') else str(failed_item)
            print(f"  - {item_name}: {error}")
    
    return 1 if failed_compressions else 0


if __name__ == "__main__":
    sys.exit(main())