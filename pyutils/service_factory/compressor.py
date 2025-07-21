import os
import sys
import gzip
import argparse
from pathlib import Path
from typing import List, Optional


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


def compress_file(input_path: Path, compression_level: int, output_dir: Optional[Path] = None) -> Path:
    """
    Compress a single text file using gzip compression.
    
    Args:
        input_path: Path to the input .txt file
        compression_level: Compression level (1-9, where 9 is maximum compression)
        output_dir: Optional output directory (defaults to same as input)
        
    Returns:
        Path to the compressed file
    """
    if output_dir is None:
        output_dir = input_path.parent
    
    # Create output filename with .gz extension
    output_filename = input_path.stem + ".txt.gz"
    output_path = output_dir / output_filename
    
    try:
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb', compresslevel=compression_level) as f_out:
                f_out.write(f_in.read())
        
        # Get file sizes for reporting
        original_size = input_path.stat().st_size
        compressed_size = output_path.stat().st_size
        compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        print(f"Compressed: {input_path.name}")
        print(f"Original size: {original_size:,} bytes")
        print(f"Compressed size: {compressed_size:,} bytes")
        print(f"Compression ratio: {compression_ratio:.1f}%")
        print()
        
        return output_path
        
    except Exception as e:
        print(f"Failed to compress {input_path.name}: {str(e)}", file=sys.stderr)
        raise


def main():
    
    parser = argparse.ArgumentParser(
        description="Compress all .txt files in a directory using gzip compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            %(prog)s /path/to/txt/files --level 6
            %(prog)s ./documents --level 9 --output ./compressed
            %(prog)s ~/texts -l 1 --remove-original
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
        help='Compression level (1=fastest, 9=best compression, default=6)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output directory for compressed files (default: same as input directory)'
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
    print(f"Output directory: '{output_dir}'")
    
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
    successful_compressions = []
    failed_compressions = []
    
    for txt_file in txt_files:
        try:
            if args.verbose:
                print(f"Processing: {txt_file.name}")
            
            compressed_path = compress_file(txt_file, args.level, output_dir)
            successful_compressions.append((txt_file, compressed_path))
            
            # Remove original file if requested
            if args.remove_original:
                txt_file.unlink()
                if args.verbose:
                    print(f"  Removed original: {txt_file.name}")
                    
        except Exception as e:
            failed_compressions.append((txt_file, str(e)))
            if args.verbose:
                print(f"  Error: {e}")
            continue
    
    # Final summary
    print("=" * 50)
    print(f"Compression completed!")
    print(f"Successfully compressed: {len(successful_compressions)} files")
    
    if failed_compressions:
        print(f"Failed compressions: {len(failed_compressions)} files")
        print("\nFailed files:")
        for failed_file, error in failed_compressions:
            print(f"  - {failed_file.name}: {error}")
    
    return 1 if failed_compressions else 0


if __name__ == "__main__":
    sys.exit(main())
    