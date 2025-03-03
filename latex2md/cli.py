import argparse
import os
import logging
from pathlib import Path
from .converter import Latex2Md

def main():
    """Command line interface for latex2md."""
    parser = argparse.ArgumentParser(description='Convert LaTeX to Markdown')
    parser.add_argument('input', help='Input LaTeX file (main file)')
    parser.add_argument('-o', '--output', help='Output Markdown file')
    parser.add_argument('-i', '--image-dir', help='Directory to save images', default='images')
    parser.add_argument('-p', '--project-dir', help='Root directory of the LaTeX project')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Set default output folder structure
    if args.input:
        input_path = Path(args.input)
        input_dir = input_path.parent
        input_stem = input_path.stem

        # Create output directory - for single file it's filename_md, for project it's project_dir_md
        if args.project_dir:
            project_path = Path(args.project_dir)
            output_dir = project_path.parent / f"{project_path.name}_md"
        else:
            output_dir = input_dir / f"{input_stem}_md"

        # Create images directory inside output directory
        images_dir = output_dir / "images"
        os.makedirs(images_dir, exist_ok=True)

        # Set default output file
        if not args.output:
            args.output = str(output_dir / f"{input_stem}.md")

        # Update image directory
        args.image_dir = str(images_dir)

    # Set default project directory if not provided
    if not args.project_dir:
        args.project_dir = os.path.dirname(os.path.abspath(args.input))

    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Create converter and process
    converter = Latex2Md(args.input, args.output, args.image_dir, args.project_dir)

    try:
        # Consolidate project files
        converter.consolidate_project()

        # Load and process the main file
        if converter.load_file():
            converter.convert()
            converter.save_file()
            print(f"Conversion complete: {args.input} -> {args.output}")
        else:
            print(f"Failed to convert {args.input}")
    finally:
        # Clean up temporary files
        converter.cleanup_temp_dir()
