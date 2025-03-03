
import re
import os
import sys
import shutil
from pathlib import Path
import argparse
import logging
import tempfile
import glob


class Latex2Md:
    """
    LaTeX to Markdown converter that handles various LaTeX elements including
    figures, tables, equations, references, and multi-column layouts.
    Supports academic paper project structures with multiple files.
    """

    def __init__(self, input_file=None, output_file=None, image_dir=None, project_dir=None):
        """
        Initialize the converter with input and output files.

        Args:
            input_file (str): Path to the main LaTeX file to convert
            output_file (str): Path to save the Markdown output
            image_dir (str): Directory to save extracted images
            project_dir (str): Root directory of the LaTeX project
        """
        self.input_file = input_file
        self.output_file = output_file
        self.image_dir = image_dir or 'images'
        self.project_dir = project_dir or (os.path.dirname(input_file) if input_file else '.')
        self.content = ""
        self.md_content = ""
        self.references = {}
        self.figure_counter = 0
        self.table_counter = 0
        self.equation_counter = 0
        self.processed_files = set()  # Track processed files to avoid circular references
        self.temp_dir = None  # For consolidated project files

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('latex2md')

    def load_file(self, file_path=None):
        """Load LaTeX content from file and process includes/inputs."""
        path = file_path or self.input_file
        if not path:
            raise ValueError("No input file specified")

        # Convert to absolute path
        abs_path = os.path.abspath(path)

        # Check if already processed to avoid circular references
        if abs_path in self.processed_files:
            self.logger.warning(f"Skipping already processed file: {abs_path}")
            return True

        self.processed_files.add(abs_path)

        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # If this is the main file, store it
            if path == self.input_file:
                self.content = content

            # Process \input and \include commands
            self.process_input_commands(content, os.path.dirname(abs_path))

            self.logger.info(f"Loaded LaTeX file: {abs_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load file {abs_path}: {str(e)}")
            return False

    def process_input_commands(self, content, base_dir):
        """
        Process \input and \include commands to load referenced files.

        Args:
            content (str): LaTeX content to process
            base_dir (str): Base directory for resolving relative paths
        """
        # Find all \input{...} commands
        input_pattern = r'\\input{([^}]+)}'
        input_matches = re.finditer(input_pattern, content)

        for match in input_matches:
            input_file = match.group(1)
            # Add .tex extension if missing
            if not input_file.endswith('.tex'):
                input_file += '.tex'

            # Resolve path relative to the current file
            input_path = os.path.join(base_dir, input_file)

            # Load the referenced file
            self.load_file(input_path)

        # Find all \include{...} commands
        include_pattern = r'\\include{([^}]+)}'
        include_matches = re.finditer(include_pattern, content)

        for match in include_matches:
            include_file = match.group(1)
            # Add .tex extension if missing
            if not include_file.endswith('.tex'):
                include_file += '.tex'

            # Resolve path relative to the current file
            include_path = os.path.join(base_dir, include_file)

            # Load the referenced file
            self.load_file(include_path)

    def consolidate_project(self):
        """
        Consolidate all project files into a single temporary directory.
        This helps with processing relative paths and includes.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.logger.info(f"Created temporary directory for project: {self.temp_dir}")

        # Copy all .tex files from project directory
        for tex_file in glob.glob(os.path.join(self.project_dir, "**/*.tex"), recursive=True):
            rel_path = os.path.relpath(tex_file, self.project_dir)
            target_path = os.path.join(self.temp_dir, rel_path)

            # Create directory structure
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Copy file
            shutil.copy2(tex_file, target_path)
            self.logger.debug(f"Copied {tex_file} to {target_path}")

        # Copy all image files
        for img_ext in ['*.png', '*.jpg', '*.jpeg', '*.pdf', '*.eps', '*.svg']:
            for img_file in glob.glob(os.path.join(self.project_dir, "**/" + img_ext), recursive=True):
                rel_path = os.path.relpath(img_file, self.project_dir)
                target_path = os.path.join(self.temp_dir, rel_path)

                # Create directory structure
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # Copy file
                shutil.copy2(img_file, target_path)
                self.logger.debug(f"Copied {img_file} to {target_path}")

        # Copy .bib files for bibliography
        for bib_file in glob.glob(os.path.join(self.project_dir, "**/*.bib"), recursive=True):
            rel_path = os.path.relpath(bib_file, self.project_dir)
            target_path = os.path.join(self.temp_dir, rel_path)

            # Create directory structure
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Copy file
            shutil.copy2(bib_file, target_path)
            self.logger.debug(f"Copied {bib_file} to {target_path}")

        # Update input file path to point to the temporary directory
        if self.input_file:
            rel_input = os.path.relpath(self.input_file, self.project_dir)
            self.input_file = os.path.join(self.temp_dir, rel_input)
            self.logger.info(f"Updated input file path to: {self.input_file}")

    def cleanup_temp_dir(self):
        """Clean up temporary directory after processing."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Removed temporary directory: {self.temp_dir}")

    def save_file(self, file_path=None):
        """Save converted Markdown content to file."""
        path = file_path or self.output_file
        if not path:
            raise ValueError("No output file specified")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.md_content)
            self.logger.info(f"Saved Markdown file: {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save file {path}: {str(e)}")
            return False

    def convert(self, content=None):
        """
        Convert LaTeX content to Markdown.

        Args:
            content (str): LaTeX content to convert. If None, uses self.content

        Returns:
            str: Converted Markdown content
        """
        if content is not None:
            self.content = content

        if not self.content:
            self.logger.warning("No content to convert")
            return ""

        # Preprocessing
        self.preprocess()

        # Extract document structure
        self.extract_document_structure()

        # Process content
        self.process_content()

        return self.md_content

    def preprocess(self):
        """Preprocess LaTeX content before conversion."""
        # Remove comments
        self.content = re.sub(r'%.*?\n', '\n', self.content)

        # Extract references for later use
        self.extract_references()

        # Create image directory if it doesn't exist
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
            self.logger.info(f"Created image directory: {self.image_dir}")

    def extract_references(self):
        """Extract all \label commands to build reference dictionary."""
        label_pattern = r'\\label{([^}]+)}'
        labels = re.findall(label_pattern, self.content)

        for label in labels:
            # Find the context of this label (figure, table, section, etc.)
            context = self.find_label_context(label)
            self.references[label] = context

    def find_label_context(self, label):
        """Find the context (number, title) for a given label."""
        # This is a simplified implementation
        # In a complete version, we would track figure/table/section numbers
        label_pattern = r'\\label{' + re.escape(label) + r'}'
        label_pos = re.search(label_pattern, self.content)

        if not label_pos:
            return {"type": "unknown", "number": 0, "title": ""}

        # Check if it's in a figure environment
        figure_pattern = r'\\begin{figure}.*?' + label_pattern + r'.*?\\end{figure}'
        if re.search(figure_pattern, self.content, re.DOTALL):
            self.figure_counter += 1
            return {"type": "figure", "number": self.figure_counter, "title": self.extract_caption(label)}

        # Check if it's in a table environment
        table_pattern = r'\\begin{table}.*?' + label_pattern + r'.*?\\end{table}'
        if re.search(table_pattern, self.content, re.DOTALL):
            self.table_counter += 1
            return {"type": "table", "number": self.table_counter, "title": self.extract_caption(label)}

        # Check if it's an equation
        equation_pattern = r'\\begin{equation}.*?' + label_pattern + r'.*?\\end{equation}'
        if re.search(equation_pattern, self.content, re.DOTALL):
            self.equation_counter += 1
            return {"type": "equation", "number": self.equation_counter, "title": ""}

        # Default to section
        return {"type": "section", "number": 0, "title": ""}

    def extract_caption(self, label):
        """Extract caption for a given label."""
        label_pattern = r'\\label{' + re.escape(label) + r'}'
        caption_pattern = r'\\caption{(.*?)}.*?' + label_pattern
        alt_caption_pattern = label_pattern + r'.*?\\caption{(.*?)}'

        caption_match = re.search(caption_pattern, self.content, re.DOTALL)
        if not caption_match:
            caption_match = re.search(alt_caption_pattern, self.content, re.DOTALL)

        if caption_match:
            return self.clean_latex_formatting(caption_match.group(1))
        return ""

    def extract_document_structure(self):
        """Extract document structure (sections, subsections, etc.)."""
        # This would be implemented in a full version
        pass

    def process_content(self):
        """Process the main content conversion from LaTeX to Markdown."""
        content = self.content

        # Process document class and preamble
        content = self.remove_preamble(content)

        # Handle multi-column layouts
        content = self.process_multicol(content)

        # Convert sections and subsections
        content = self.convert_sections(content)

        # Convert figures and tables
        content = self.convert_figures(content)
        content = self.convert_tables(content)

        # Convert math environments
        content = self.convert_math(content)

        # Convert references
        content = self.convert_references(content)

        # Convert formatting
        content = self.convert_formatting(content)

        # Convert lists
        content = self.convert_lists(content)

        # Final cleanup
        content = self.final_cleanup(content)

        self.md_content = content

    def remove_preamble(self, content):
        """Remove LaTeX preamble and document environment tags."""
        # Remove everything before \begin{document}
        content = re.sub(r'.*?\\begin{document}', '', content, flags=re.DOTALL)

        # Remove \end{document} and everything after
        content = re.sub(r'\\end{document}.*', '', content, flags=re.DOTALL)

        return content

    def process_multicol(self, content):
        """
        Process multi-column layouts in LaTeX.
        Improved to better handle academic paper layouts.
        """
        # Find all multicol environments
        multicol_pattern = r'\\begin{multicols}{(\d+)}(.*?)\\end{multicols}'
        multicols = re.finditer(multicol_pattern, content, re.DOTALL)

        for match in multicols:
            num_cols = int(match.group(1))
            multicol_content = match.group(2)

            # If it's a 2-column layout
            if num_cols == 2:
                # Try to split the content into columns
                columns = self.split_multicol_content(multicol_content, num_cols)

                # Process each column separately
                processed_columns = [self.process_column_content(col) for col in columns]

                # Combine columns in the specified order (left then right for 2 columns)
                new_content = "\n\n".join(processed_columns)

                # Replace the original multicol environment
                content = content.replace(match.group(0), new_content)
            else:
                # For other column counts, process differently
                columns = self.split_multicol_content(multicol_content, num_cols)
                processed_columns = [self.process_column_content(col) for col in columns]

                # For more than 2 columns, we'll join them in reading order
                new_content = "\n\n".join(processed_columns)

                # Replace the original multicol environment
                content = content.replace(match.group(0), new_content)

        # Also handle two-column document class
        if re.search(r'\\documentclass(\[.*?\])?\{.*?twocolumn.*?\}', self.content, re.DOTALL) or \
                re.search(r'\\documentclass(\[.*?twocolumn.*?\])\{', self.content):
            # For two-column document class, we need to handle the entire content
            # This is a simplified approach - in reality, we'd need more sophisticated parsing

            # Try to identify column breaks (often indicated by page breaks or specific markers)
            # For simplicity, we'll just split the content in half
            lines = content.split('\n')
            mid_point = len(lines) // 2

            left_col = '\n'.join(lines[:mid_point])
            right_col = '\n'.join(lines[mid_point:])

            # Process each column
            left_col = self.process_column_content(left_col)
            right_col = self.process_column_content(right_col)

            # Combine columns
            content = f"{left_col}\n\n{right_col}"

        return content

    def split_multicol_content(self, content, num_cols):
        """
        Split multi-column content into separate columns.

        Args:
            content (str): Content within multicols environment
            num_cols (int): Number of columns

        Returns:
            list: List of column contents
        """
        columns = []

        # First check for explicit column breaks
        if '\\columnbreak' in content:
            # Split by column breaks
            col_parts = re.split(r'\\columnbreak', content)

            # If we have the right number of parts, use them
            if len(col_parts) == num_cols:
                return col_parts

            # If we have more parts than columns, combine extras
            if len(col_parts) > num_cols:
                columns = col_parts[:num_cols - 1]
                columns.append('\n'.join(col_parts[num_cols - 1:]))
                return columns

            # If we have fewer parts than columns, add empty columns
            columns = col_parts
            while len(columns) < num_cols:
                columns.append('')
            return columns

        # If no explicit breaks, try to identify logical breaks
        # This is a complex problem, but we'll use a simple approach

        # Split by paragraphs
        paragraphs = re.split(r'\n\s*\n', content)

        # Calculate paragraphs per column
        paras_per_col = len(paragraphs) // num_cols
        if paras_per_col == 0:
            paras_per_col = 1

        # Distribute paragraphs among columns
        for i in range(num_cols):
            start_idx = i * paras_per_col
            end_idx = (i + 1) * paras_per_col if i < num_cols - 1 else len(paragraphs)
            col_content = '\n\n'.join(paragraphs[start_idx:end_idx])
            columns.append(col_content)

        return columns

    def process_column_content(self, column_content):
        """Process the content of a single column."""
        # Apply all the same conversions we would to regular content
        column_content = self.convert_sections(column_content)
        column_content = self.convert_figures(column_content)
        column_content = self.convert_tables(column_content)
        column_content = self.convert_math(column_content)
        column_content = self.convert_references(column_content)
        column_content = self.convert_formatting(column_content)
        column_content = self.convert_lists(column_content)

        return column_content

    def convert_sections(self, content):
        """Convert LaTeX sections to Markdown headings."""
        # Convert \chapter
        content = re.sub(r'\\chapter{(.*?)}', r'# \1', content)

        # Convert \section
        content = re.sub(r'\\section{(.*?)}', r'## \1', content)

        # Convert \subsection
        content = re.sub(r'\\subsection{(.*?)}', r'### \1', content)

        # Convert \subsubsection
        content = re.sub(r'\\subsubsection{(.*?)}', r'#### \1', content)

        return content

    def convert_figures(self, content):
        """Convert LaTeX figures to Markdown image syntax."""
        # Find all figure environments
        figure_pattern = r'\\begin{figure}(.*?)\\end{figure}'
        figures = re.finditer(figure_pattern, content, re.DOTALL)

        for match in figures:
            figure_content = match.group(1)

            # Extract image path
            includegraphics_pattern = r'\\includegraphics(?:$$.*?$$)?{(.*?)}'
            img_match = re.search(includegraphics_pattern, figure_content)

            if img_match:
                img_path = img_match.group(1)
                # Handle relative paths and extensions
                img_filename = os.path.basename(img_path)
                target_path = os.path.join(self.image_dir, img_filename)

                # Extract caption if available
                caption = ""
                caption_match = re.search(r'\\caption{(.*?)}', figure_content)
                if caption_match:
                    caption = self.clean_latex_formatting(caption_match.group(1))

                # Extract label if available
                label = ""
                label_match = re.search(r'\\label{(.*?)}', figure_content)
                if label_match:
                    label = label_match.group(1)

                # Create markdown image with caption
                md_image = f"![{caption}]({target_path})\n\n*{caption}*"

                if label:
                    # Add an anchor for references
                    md_image = f'<a id="{label}"></a>\n\n{md_image}'

                # Replace the original figure environment
                content = content.replace(match.group(0), md_image)

                # Copy the image file if it exists
                self.copy_image_file(img_path, target_path)

        return content

    def copy_image_file(self, source_path, target_path):
        """
        Copy image file to the target directory.
        Handles various file paths and formats including PDF.
        """
        # Try to find the image in the project directory or temp directory
        source_paths_to_try = []

        # Try with original path
        source_paths_to_try.append(source_path)

        # Try with absolute path from project directory
        source_paths_to_try.append(os.path.join(self.project_dir, source_path))

        # Try with common image extensions if no extension is provided
        if not os.path.splitext(source_path)[1]:
            for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.eps', '.svg']:
                source_paths_to_try.append(source_path + ext)
                source_paths_to_try.append(os.path.join(self.project_dir, source_path + ext))

        # If we're using a temp directory, also look there
        if self.temp_dir:
            source_paths_to_try.append(os.path.join(self.temp_dir, source_path))
            if not os.path.splitext(source_path)[1]:
                for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.eps', '.svg']:
                    source_paths_to_try.append(os.path.join(self.temp_dir, source_path + ext))

        # Try to find the image file
        for src_path in source_paths_to_try:
            if os.path.exists(src_path):
                # Create target directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

                # Check if it's a PDF that needs conversion
                if src_path.lower().endswith('.pdf'):
                    # In a real implementation, we would convert PDF to PNG
                    # For this example, we'll just copy and change the extension
                    target_path = target_path.rsplit('.', 1)[0] + '.png'
                    self.logger.info(f"Would convert PDF {src_path} to PNG {target_path}")
                    shutil.copy2(src_path, target_path)
                else:
                    # Copy the file
                    shutil.copy2(src_path, target_path)

                self.logger.info(f"Copied image from {src_path} to {target_path}")
                return

        self.logger.warning(f"Could not find image file: {source_path}")

    def convert_tables(self, content):
        """Convert LaTeX tables to Markdown tables."""
        # Find all table environments
        table_pattern = r'\\begin{table}(.*?)\\end{table}'
        tables = re.finditer(table_pattern, content, re.DOTALL)

        for match in tables:
            table_content = match.group(1)

            # Extract caption if available
            caption = ""
            caption_match = re.search(r'\\caption{(.*?)}', table_content)
            if caption_match:
                caption = self.clean_latex_formatting(caption_match.group(1))

            # Extract label if available
            label = ""
            label_match = re.search(r'\\label{(.*?)}', table_content)
            if label_match:
                label = label_match.group(1)

            # Extract tabular environment
            tabular_pattern = r'\\begin{tabular}{(.*?)}(.*?)\\end{tabular}'
            tabular_match = re.search(tabular_pattern, table_content, re.DOTALL)

            if tabular_match:
                col_spec = tabular_match.group(1)
                rows_content = tabular_match.group(2)

                # Count columns
                num_cols = len(re.sub(r'[^lcr|]', '', col_spec))

                # Split into rows
                rows = rows_content.split('\\\\')

                # Create markdown table
                md_table = []

                # Add header row (first row)
                if rows:
                    header = rows[0].strip()
                    header_cells = re.split(r'&', header)
                    md_table.append('| ' + ' | '.join(
                        [self.clean_latex_formatting(cell.strip()) for cell in header_cells]) + ' |')

                    # Add separator row
                    md_table.append('| ' + ' | '.join(['---'] * len(header_cells)) + ' |')

                    # Add data rows
                    for row in rows[1:]:
                        row = row.strip()
                        if not row:
                            continue
                        cells = re.split(r'&', row)
                        md_table.append('| ' + ' | '.join(
                            [self.clean_latex_formatting(cell.strip()) for cell in cells]) + ' |')

                # Combine with caption
                md_table_str = '\n'.join(md_table)
                if caption:
                    md_table_str += f"\n\n*{caption}*"

                if label:
                    # Add an anchor for references
                    md_table_str = f'<a id="{label}"></a>\n\n{md_table_str}'

                # Replace the original table environment
                content = content.replace(match.group(0), md_table_str)

        return content

    def convert_math(self, content):
        """Convert LaTeX math environments to Markdown math syntax."""
        # Convert inline math: $...$ or $...$
        content = re.sub(r'\$([^$]+?)\$', r'$\1$', content)
        content = re.sub(r'\\[(](.*?)\\[)]', r'$\1$', content)

        # Convert display math environments

        # equation environment
        equation_pattern = r'\\begin{equation}(.*?)\\end{equation}'
        equations = re.finditer(equation_pattern, content, re.DOTALL)

        for match in equations:
            eq_content = match.group(1)

            # Extract label if available
            label = ""
            label_match = re.search(r'\\label{(.*?)}', eq_content)
            if label_match:
                label = label_match.group(1)
                eq_content = eq_content.replace(label_match.group(0), '')

            # Clean up the equation content
            eq_content = eq_content.strip()

            # Create markdown equation
            md_equation = f"$$\n{eq_content}\n$$"

            if label:
                # Add an anchor for references
                md_equation = f'<a id="{label}"></a>\n\n{md_equation}'

            # Replace the original equation environment
            content = content.replace(match.group(0), md_equation)

        # align environment
        align_pattern = r'\\begin{align}(.*?)\\end{align}'
        aligns = re.finditer(align_pattern, content, re.DOTALL)

        for match in aligns:
            align_content = match.group(1)

            # Extract label if available
            label = ""
            label_match = re.search(r'\\label{(.*?)}', align_content)
            if label_match:
                label = label_match.group(1)
                align_content = align_content.replace(label_match.group(0), '')

            # Clean up the align content
            align_content = align_content.strip()

            # Create markdown align
            md_align = f"$$\n\\begin{{align}}\n{align_content}\n\\end{{align}}\n$$"

            if label:
                # Add an anchor for references
                md_align = f'<a id="{label}"></a>\n\n{md_align}'

            # Replace the original align environment
            content = content.replace(match.group(0), md_align)

        return content

    def convert_references(self, content):
        """Convert LaTeX references to Markdown links."""
        # Convert \ref{...} to links
        ref_pattern = r'\\ref{([^}]+)}'
        refs = re.finditer(ref_pattern, content)

        for match in refs:
            label = match.group(1)

            if label in self.references:
                ref_info = self.references[label]
                ref_type = ref_info["type"]
                ref_number = ref_info["number"]

                if ref_type == "figure":
                    replacement = f"[Figure {ref_number}](#{label})"
                elif ref_type == "table":
                    replacement = f"[Table {ref_number}](#{label})"
                elif ref_type == "equation":
                    replacement = f"[Equation {ref_number}](#{label})"
                else:
                    replacement = f"[{ref_type.capitalize()} {ref_number}](#{label})"

                content = content.replace(match.group(0), replacement)
            else:
                # If reference not found, just use the label
                content = content.replace(match.group(0), f"[{label}](#{label})")

        # Convert \cite{...} to links or footnotes
        cite_pattern = r'\\cite{([^}]+)}'
        cites = re.finditer(cite_pattern, content)

        for match in cites:
            citation_keys = match.group(1).split(',')
            citations = []

            for key in citation_keys:
                key = key.strip()
                citations.append(f"[{key}]")

            replacement = ', '.join(citations)
            content = content.replace(match.group(0), replacement)

        # Process bibliography if present
        bib_pattern = r'\\bibliography{([^}]+)}'
        bib_match = re.search(bib_pattern, content)

        if bib_match:
            bib_file = bib_match.group(1)
            if not bib_file.endswith('.bib'):
                bib_file += '.bib'

            # Try to find the bib file
            bib_paths_to_try = [
                bib_file,
                os.path.join(self.project_dir, bib_file),
                os.path.join(os.path.dirname(self.input_file), bib_file)
            ]

            if self.temp_dir:
                bib_paths_to_try.append(os.path.join(self.temp_dir, bib_file))

            bib_content = None
            for bib_path in bib_paths_to_try:
                if os.path.exists(bib_path):
                    try:
                        with open(bib_path, 'r', encoding='utf-8') as f:
                            bib_content = f.read()
                        self.logger.info(f"Loaded bibliography file: {bib_path}")
                        break
                    except Exception as e:
                        self.logger.error(f"Failed to load bibliography file {bib_path}: {str(e)}")

            if bib_content:
                # Extract entries from bib file
                bib_entries = self.extract_bib_entries(bib_content)

                # Create bibliography section
                bib_section = "## References\n\n"
                for key, entry in bib_entries.items():
                    bib_section += f"[{key}]: {entry}\n\n"

                # Replace bibliography command with the section
                content = content.replace(bib_match.group(0), bib_section)
            else:
                # If bib file not found, just remove the command
                content = content.replace(bib_match.group(0), "")

        return content

    def extract_bib_entries(self, bib_content):
        """
        Extract entries from a BibTeX file.

        Args:
            bib_content (str): Content of the BibTeX file

        Returns:
            dict: Dictionary of BibTeX entries with keys as citation keys
        """
        entries = {}

        # Find all BibTeX entries
        entry_pattern = r'@(\w+)\{(\w+),\s*(.*?)\s*\}'
        entry_matches = re.finditer(entry_pattern, bib_content, re.DOTALL)

        for match in entry_matches:
            entry_type = match.group(1)
            entry_key = match.group(2)
            entry_content = match.group(3)

            # Extract fields
            fields = {}
            field_pattern = r'(\w+)\s*=\s*[{"](.*?)[}"],'
            field_matches = re.finditer(field_pattern, entry_content, re.DOTALL)

            for field_match in field_matches:
                field_name = field_match.group(1).lower()
                field_value = field_match.group(2)
                fields[field_name] = field_value

            # Format entry based on type
            if entry_type.lower() == 'article':
                formatted_entry = self.format_article_entry(fields)
            elif entry_type.lower() == 'book':
                formatted_entry = self.format_book_entry(fields)
            elif entry_type.lower() == 'inproceedings':
                formatted_entry = self.format_inproceedings_entry(fields)
            else:
                # Generic format for other types
                formatted_entry = self.format_generic_entry(fields)

            entries[entry_key] = formatted_entry

        return entries

    def format_article_entry(self, fields):
        """Format an article BibTeX entry for Markdown."""
        authors = fields.get('author', 'Unknown')
        title = fields.get('title', 'Untitled')
        journal = fields.get('journal', '')
        year = fields.get('year', '')
        volume = fields.get('volume', '')
        number = fields.get('number', '')
        pages = fields.get('pages', '')

        return f"{authors}. \"{title}\". *{journal}*, {volume}{f'({number})' if number else ''}, {pages}, {year}."

    def format_book_entry(self, fields):
        """Format a book BibTeX entry for Markdown."""
        authors = fields.get('author', fields.get('editor', 'Unknown'))
        title = fields.get('title', 'Untitled')
        publisher = fields.get('publisher', '')
        year = fields.get('year', '')

        return f"{authors}. *{title}*. {publisher}, {year}."

    def format_inproceedings_entry(self, fields):
        """Format an inproceedings BibTeX entry for Markdown."""
        authors = fields.get('author', 'Unknown')
        title = fields.get('title', 'Untitled')
        booktitle = fields.get('booktitle', '')
        year = fields.get('year', '')
        pages = fields.get('pages', '')

        return f"{authors}. \"{title}\". In *{booktitle}*, {pages}, {year}."

    def format_generic_entry(self, fields):
        """Format a generic BibTeX entry for Markdown."""
        authors = fields.get('author', fields.get('editor', 'Unknown'))
        title = fields.get('title', 'Untitled')
        year = fields.get('year', '')

        return f"{authors}. \"{title}\". {year}."

    def convert_formatting(self, content):
        """Convert LaTeX formatting to Markdown formatting."""
        # Convert bold: \textbf{...} to **...**
        content = re.sub(r'\\textbf{(.*?)}', r'**\1**', content)

        # Convert italic: \textit{...} or \emph{...} to *...*
        content = re.sub(r'\\textit{(.*?)}', r'*\1*', content)
        content = re.sub(r'\\emph{(.*?)}', r'*\1*', content)

        # Convert underline: \underline{...} to <u>...</u>
        content = re.sub(r'\\underline{(.*?)}', r'<u>\1</u>', content)

        # Convert strikethrough: \sout{...} to ~~...~~
        content = re.sub(r'\\sout{(.*?)}', r'~~\1~~', content)

        # Convert code: \texttt{...} to `...`
        content = re.sub(r'\\texttt{(.*?)}', r'`\1`', content)

        # Convert URLs: \url{...} to [...](...) or just the URL
        content = re.sub(r'\\url{(.*?)}', r'[\1](\1)', content)

        # Convert \textcolor{color}{text} to <span style="color:color">text</span>
        content = re.sub(r'\\textcolor{(.*?)}{(.*?)}', r'<span style="color:\1">\2</span>', content)

        return content

    def convert_lists(self, content):
        """Convert LaTeX lists to Markdown lists."""
        # Find all itemize environments
        itemize_pattern = r'\\begin{itemize}(.*?)\\end{itemize}'
        itemizes = re.finditer(itemize_pattern, content, re.DOTALL)

        for match in itemizes:
            itemize_content = match.group(1)

            # Split into items
            items = re.split(r'\\item\s+', itemize_content)

            # Create markdown list
            md_list = []
            for item in items:
                if item.strip():
                    md_list.append(f"- {item.strip()}")

            # Replace the original itemize environment
            md_list_str = '\n'.join(md_list)
            content = content.replace(match.group(0), md_list_str)

        # Find all enumerate environments
        enumerate_pattern = r'\\begin{enumerate}(.*?)\\end{enumerate}'
        enumerates = re.finditer(enumerate_pattern, content, re.DOTALL)

        for match in enumerates:
            enumerate_content = match.group(1)

            # Split into items
            items = re.split(r'\\item\s+', enumerate_content)

            # Create markdown list
            md_list = []
            item_num = 1
            for item in items:
                if item.strip():
                    md_list.append(f"{item_num}. {item.strip()}")
                    item_num += 1

            # Replace the original enumerate environment
            md_list_str = '\n'.join(md_list)
            content = content.replace(match.group(0), md_list_str)

        return content

    def final_cleanup(self, content):
        """Perform final cleanup on the converted content."""
        # Remove any remaining LaTeX commands
        content = re.sub(r'\\[a-zA-Z]+(\[.*?\])?{.*?}', '', content)

        # Fix multiple blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Fix spacing around headers
        content = re.sub(r'([^\n])(#+ )', r'\1\n\n\2', content)

        # Fix spacing after headers
        content = re.sub(r'(#+ .*?)\n([^\n])', r'\1\n\n\2', content)

        return content

    def clean_latex_formatting(self, text):
        """Clean LaTeX formatting from text."""
        # Remove LaTeX formatting commands
        text = re.sub(r'\\[a-zA-Z]+(\[.*?\])?{(.*?)}', r'\2', text)

        # Replace LaTeX special characters
        replacements = {
            '~': ' ',
            '\\&': '&',
            '\\%': '%',
            '\\$': '$',
            '\\#': '#',
            '\\_': '_',
            '\\{': '{',
            '\\}': '}',
            '\\textbackslash': '\\',
            '\\textasciitilde': '~',
            '\\textasciicircum': '^',
            '\\LaTeX': 'LaTeX',
            '\\TeX': 'TeX'
        }

        for latex, replacement in replacements.items():
            text = text.replace(latex, replacement)

        return text


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
        logging.getLogger('latex2md').setLevel(logging.DEBUG)

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
            sys.exit(1)
    finally:
        # Clean up temporary files
        converter.cleanup_temp_dir()


if __name__ == "__main__":
    main()