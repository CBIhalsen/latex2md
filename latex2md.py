
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

        # Copy all image files - scan deeply for figures folder
        for img_ext in ['*.png', '*.jpg', '*.jpeg', '*.pdf', '*.eps', '*.svg']:
            # Specifically look in figures directories
            for img_file in glob.glob(os.path.join(self.project_dir, "**/figures/" + img_ext), recursive=True):
                rel_path = os.path.relpath(img_file, self.project_dir)
                target_path = os.path.join(self.temp_dir, rel_path)

                # Create directory structure
                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                # Copy file
                shutil.copy2(img_file, target_path)
                self.logger.debug(f"Copied {img_file} to {target_path}")

            # Also look in all directories for images
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

        # Convert sections and subsections (including \maketitle)
        content = self.convert_sections(content)

        # Handle multi-column layouts
        content = self.process_multicol(content)

        # Convert figures and tables
        content = self.convert_figures(content)
        content = self.convert_tables(content)

        # Convert math environments
        content = self.convert_math(content)

        # Convert references
        content = self.convert_references(content)

        # Convert lists
        content = self.convert_lists(content)

        # Convert formatting
        content = self.convert_formatting(content)

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
        Process multi-column layouts in LaTeX. Improved to better handle academic paper layouts,
        without做无差别 're.sub(r"\\.*", "")' 的操作，避免误删表格文本。
        """
        # 匹配多列环境
        multicol_pattern = r'\\begin{multicols}{(\d+)}(.*?)\\end{multicols}'
        multicols = re.finditer(multicol_pattern, content, re.DOTALL)

        for match in multicols:
            num_cols = int(match.group(1))
            multicol_content = match.group(2)

            # 拆分多列内容
            columns = self.split_multicol_content(multicol_content, num_cols)
            # 处理每一列
            processed_columns = [self.process_column_content(col) for col in columns]
            # 将各列在逻辑顺序上拼回
            new_content = "\n\n".join(processed_columns)
            content = content.replace(match.group(0), new_content)

            # 如果是双栏文档类型（twocolumn), 简化处理为将整篇文本按行数一分为二
        if re.search(r'\\documentclass(\[.*?\])?\{.*?twocolumn.*?\}', self.content, re.DOTALL) or \
                re.search(r'\\documentclass(\[.*?twocolumn.*?\])\{', self.content):
            lines = content.split('\n')
            mid_point = len(lines) // 2
            left_col = '\n'.join(lines[:mid_point])
            right_col = '\n'.join(lines[mid_point:])

            left_col = self.process_column_content(left_col)
            right_col = self.process_column_content(right_col)

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
        content = re.sub(r'\\chapter\*{(.*?)}', r'# \1', content)
        content = re.sub(r'\\chapter{(.*?)}', r'# \1', content)

        # Convert \section
        content = re.sub(r'\\section\*{(.*?)}', r'## \1', content)
        content = re.sub(r'\\section{(.*?)}', r'## \1', content)

        # Convert \subsection
        content = re.sub(r'\\subsection\*{(.*?)}', r'### \1', content)
        content = re.sub(r'\\subsection{(.*?)}', r'### \1', content)

        # Convert \subsubsection
        content = re.sub(r'\\subsubsection\*{(.*?)}', r'#### \1', content)
        content = re.sub(r'\\subsubsection{(.*?)}', r'#### \1', content)

        # Convert \maketitle to a simple title
        if '\\maketitle' in content:
            title = ""
            author = ""
            date = ""
            # 从原始 self.content 中获取标题、作者、日期
            title_match = re.search(r'\\title{(.*?)}', self.content, re.DOTALL)
            if title_match:
                title = self.clean_latex_formatting(title_match.group(1))

            author_match = re.search(r'\\author{(.*?)}', self.content, re.DOTALL)
            if author_match:
                author = self.clean_latex_formatting(author_match.group(1))

            date_match = re.search(r'\\date{(.*?)}', self.content, re.DOTALL)
            if date_match:
                date_content = date_match.group(1).strip()
                if date_content == '\\today':
                    from datetime import datetime
                    date = datetime.now().strftime("%B %d, %Y")
                else:
                    date = self.clean_latex_formatting(date_content)
            elif '\\date{\\today}' in self.content or '\\today' in self.content:
                from datetime import datetime
                date = datetime.now().strftime("%B %d, %Y")

            md_title = f"# {title}\n\n"
            if author:
                md_title += f"**Author:** {author}\n\n"
            if date:
                md_title += f"**Date:** {date}\n\n"

            content = content.replace('\\maketitle', md_title)

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
        """
        将 LaTeX 中的长表(longtable)、常规 table/tabular 环境转换为带冒号对齐符号的 Markdown 表格。
        在第一行(表头)和数据行之间自动插入对齐分隔行 (---、:---、:---:) 等，以便 Markdown 正确渲染。
        """

        import re

        # ─────────────────────────────────────────────────────────────
        # 辅助函数：解析列格式，生成 ["l","c","r"] 这样的对齐列表
        # ─────────────────────────────────────────────────────────────
        def parse_column_spec(col_spec):
            # 去除竖线 | 和空白
            col_spec_clean = col_spec.replace('|', '').strip()
            # 匹配其中的 l / c / r
            alignments = re.findall(r'[lcr]', col_spec_clean)
            return alignments

        # ─────────────────────────────────────────────────────────────
        # 辅助函数：根据对齐信息生成“分隔线”行
        # 其中:
        #   l => :---
        #   c => :---:
        #   r => ---:
        # 如果不匹配，就用默认 '---'
        # ─────────────────────────────────────────────────────────────
        def build_alignment_row(alignments):
            row_parts = []
            for a in alignments:
                if a == 'l':
                    row_parts.append(':---')
                elif a == 'c':
                    row_parts.append(':---:')
                elif a == 'r':
                    row_parts.append('---:')
                else:
                    row_parts.append('---')
            return '| ' + ' | '.join(row_parts) + ' |'

        # ─────────────────────────────────────────────────────────────
        # 辅助函数：将二维数组(rows)转为 Markdown 表格文本
        # 第一行视为“表头”，紧跟一行对齐分隔行，其后是表内容
        # ─────────────────────────────────────────────────────────────
        def rows_to_md_table(rows, col_spec):
            if not rows:
                return ''

            # 解析到的对齐方式
            alignments = parse_column_spec(col_spec)

            # 计算所有行的最大列数
            max_cols = max(len(r) for r in rows)

            # 如果 alignments 长度比列数要短，就补齐；比列数长就截断
            if len(alignments) < max_cols:
                alignments += ['l'] * (max_cols - len(alignments))
            alignments = alignments[:max_cols]

            # 补空单元格
            for r in rows:
                while len(r) < max_cols:
                    r.append('')

            # 第一行视为表头
            header_row = rows[0]

            # 生成 Markdown 表格文本
            md_lines = []
            # 1) 表头行
            md_lines.append('| ' + ' | '.join(self.clean_latex_formatting(cell) for cell in header_row) + ' |')
            # 2) 分隔行
            md_lines.append(build_alignment_row(alignments))
            # 3) 数据行
            for row in rows[1:]:
                line = '| ' + ' | '.join(self.clean_latex_formatting(cell) for cell in row) + ' |'
                md_lines.append(line)

            return '\n'.join(md_lines)

        # ─────────────────────────────────────────────────────────────
        # 处理 longtable 环境
        # ─────────────────────────────────────────────────────────────
        longtable_pattern = r'\\begin{longtable}{(.*?)}(.*?)\\end{longtable}'
        longtables = re.finditer(longtable_pattern, content, re.DOTALL)
        for match in longtables:
            col_spec = match.group(1)
            table_content = match.group(2)

            # 提取 caption
            caption = ""
            caption_match = re.search(r'\\caption{(.*?)}', table_content, re.DOTALL)
            if caption_match:
                caption = self.clean_latex_formatting(caption_match.group(1))
                table_content = table_content.replace(caption_match.group(0), '')

            # 提取 label
            label = ""
            label_match = re.search(r'\\label{(.*?)}', table_content)
            if label_match:
                label = label_match.group(1)
                table_content = table_content.replace(label_match.group(0), '')

            # 去掉 longtable 中的一些控制命令
            for cmd in ['\\endhead', '\\endfirsthead', '\\endfoot', '\\endlastfoot', '\\hline']:
                table_content = table_content.replace(cmd, '')

            # 拆分成多行，然后按 "&" 分割列
            rows_raw = re.split(r'\\\\', table_content)
            rows = []
            for row in rows_raw:
                row = row.strip()
                if row:
                    cells = [cell.strip() for cell in row.split('&')]
                    rows.append(cells)

            # 转换为 Markdown 表格
            md_table = rows_to_md_table(rows, col_spec)
            if caption:
                md_table += f"\n\n*{caption}*"
            # 如果有 label，就加上锚点
            if label:
                md_table = f'<a id="{label}"></a>\n\n{md_table}'

            # 替换整个 longtable 环境
            content = content.replace(match.group(0), md_table)

        # ─────────────────────────────────────────────────────────────
        # 处理 \begin{table}...\end{table} 环境
        # ─────────────────────────────────────────────────────────────
        table_pattern = r'\\begin{table}(.*?)\\end{table}'
        tables = re.finditer(table_pattern, content, re.DOTALL)
        for match in tables:
            entire_table_env = match.group(0)
            table_content = match.group(1)

            # caption
            caption = ""
            caption_match = re.search(r'\\caption{(.*?)}', table_content, re.DOTALL)
            if caption_match:
                caption = self.clean_latex_formatting(caption_match.group(1))

            # label
            label = ""
            label_match = re.search(r'\\label{(.*?)}', table_content)
            if label_match:
                label = label_match.group(1)

            # 搜索 tabular
            tabular_pattern = r'\\begin{tabular}{(.*?)}(.*?)\\end{tabular}'
            tabular_match = re.search(tabular_pattern, table_content, re.DOTALL)
            if tabular_match:
                col_spec = tabular_match.group(1)
                rows_content = tabular_match.group(2)

                # 去掉 \hline
                rows_content = rows_content.replace('\\hline', '')

                # 根据 '\\\\' 分隔行，再根据 '&' 分隔列
                rows_raw = re.split(r'\\\\', rows_content)
                rows = []
                for row in rows_raw:
                    row = row.strip()
                    if row:
                        cells = [cell.strip() for cell in row.split('&')]
                        rows.append(cells)

                md_table = rows_to_md_table(rows, col_spec)
                if caption:
                    md_table += f"\n\n*{caption}*"
                if label:
                    md_table = f'<a id="{label}"></a>\n\n{md_table}'

                # 替换整个 \begin{table}...\end{table} 内容
                content = content.replace(entire_table_env, md_table)

        # ─────────────────────────────────────────────────────────────
        # 处理不在 table 环境里的“裸” tabular
        # ─────────────────────────────────────────────────────────────
        standalone_tabular_pattern = r'\\begin{tabular}{(.*?)}(.*?)\\end{tabular}'
        standalone_tabulars = re.finditer(standalone_tabular_pattern, content, re.DOTALL)
        for match in standalone_tabulars:
            # 如果它已经被上面 table 环境处理过，就跳过
            if re.search(r'\\begin{table}.*?' + re.escape(match.group(0)) + r'.*?\\end{table}', content, re.DOTALL):
                continue

            col_spec = match.group(1)
            rows_content = match.group(2)
            rows_content = rows_content.replace('\\hline', '')

            rows_raw = re.split(r'\\\\', rows_content)
            rows = []
            for row in rows_raw:
                row = row.strip()
                if row:
                    cells = [cell.strip() for cell in row.split('&')]
                    rows.append(cells)

            md_table = rows_to_md_table(rows, col_spec)
            # 直接替换这个 tabular
            content = content.replace(match.group(0), md_table)

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
                    # Clean the item text
                    clean_item = self.clean_latex_formatting(item.strip())
                    md_list.append(f"- {clean_item}")

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
                    # Clean the item text
                    clean_item = self.clean_latex_formatting(item.strip())
                    md_list.append(f"{item_num}. {clean_item}")
                    item_num += 1

            # Replace the original enumerate environment
            md_list_str = '\n'.join(md_list)
            content = content.replace(match.group(0), md_list_str)

        # Find all description environments
        description_pattern = r'\\begin{description}(.*?)\\end{description}'
        descriptions = re.finditer(description_pattern, content, re.DOTALL)

        for match in descriptions:
            description_content = match.group(1)

            # Split into items
            items = re.split(r'\\item\s+', description_content)

            # Create markdown list
            md_list = []
            for item in items:
                if item.strip():
                    # Check if item has a label
                    item_match = re.match(r'\[(.*?)\](.*)', item.strip(), re.DOTALL)
                    if item_match:
                        label = item_match.group(1)
                        desc = item_match.group(2)
                        # Clean the text
                        clean_label = self.clean_latex_formatting(label)
                        clean_desc = self.clean_latex_formatting(desc)
                        md_list.append(f"**{clean_label}**: {clean_desc}")
                    else:
                        # Clean the item text
                        clean_item = self.clean_latex_formatting(item.strip())
                        md_list.append(f"- {clean_item}")

            # Replace the original description environment
            md_list_str = '\n'.join(md_list)
            content = content.replace(match.group(0), md_list_str)

        return content

    def final_cleanup(self, content):
        """
        对转成 Markdown 的文本做最后的清理。尽量别大幅改动表格行
        """
        import re

        # 1) 去除形如 \centering, \vspace 等无参数命令（不跟 {…} 的）
        content = re.sub(r'\\[a-zA-Z]+(\*)?(?!\{)', '', content)

        # 2) 合并多余空行（3 行以上合并为 2 行）
        content = re.sub(r'\n{3,}', '\n\n', content)

        # -- 如果觉得表格和正文太挤，可以考虑只在“表格上方/下方没有空行”时插 1 行，
        #    而不是盲目地多插空行。以下示例是更温和的处理：只做一次性插行，确保
        #    表格行 |xxx|xxx| 之前后至少有一个空行。
        #
        # 3) 确保表格块段落前后有空行 (可选)
        content = re.sub(r'([^|\n])\n(\|)', r'\1\n\n\2', content)
        content = re.sub(r'(\|.*?\n)([^|\n])', r'\1\n\2', content)

        # 4) 标题行与正文间插空行（可选）
        content = re.sub(r'([^\n])(#+ )', r'\1\n\n\2', content)
        content = re.sub(r'(#+ .*?)\n([^-\n])', r'\1\n\n\2', content)

        # 5) 数学公式上下加空行（可选）
        content = re.sub(r'([^\n])(\n\$\$)', r'\1\n\n\2', content)
        content = re.sub(r'(\$\$\n)([^\n])', r'\1\n\2', content)

        return content

    def clean_latex_formatting(self, text):
        """
        尽量干净地去除或简化常见的 LaTeX 命令，同时保留常见数学公式的可读性。
        """
        import re

        # 1) 先特殊处理 \frac{...}{...}
        #    这里将它保留在数学环境，比如 "$\\frac{x}{y}$"；也可以改成 "(x/y)" 文本形式。
        text = re.sub(
            r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}',
            r'$\frac{\1}{\2}$',
            text
        )

        # 2) 处理 LaTeX 内联公式 ( \(...\) → $...$ )，以及行间公式 ( \[...\] → $$...$$ )
        text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
        text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)

        # 3) 常见的文字加粗、斜体、下划线等
        text = re.sub(r'\\textbf{(.*?)}', r'**\1**', text, flags=re.DOTALL)
        text = re.sub(r'\\textit{(.*?)}', r'*\1*', text, flags=re.DOTALL)
        text = re.sub(r'\\emph{(.*?)}', r'*\1*', text, flags=re.DOTALL)
        text = re.sub(r'\\underline{(.*?)}', r'<u>\1</u>', text, flags=re.DOTALL)
        text = re.sub(r'\\sout{(.*?)}', r'~~\1~~', text, flags=re.DOTALL)
        text = re.sub(r'\\texttt{(.*?)}', r'`\1`', text, flags=re.DOTALL)

        # 4) 处理 \url{...} 为 Markdown 链接
        text = re.sub(r'\\url{(.*?)}', r'[\1](\1)', text, flags=re.DOTALL)

        # 5) 处理 \textcolor{color}{text} 等类似命令，这里直接简化成 <span style="color:xxx">...</span>
        #    注意，如果 color 里有数字、空格等，需要自己再做更严格的过滤
        text = re.sub(
            r'\\textcolor\{([^\}]+)\}\{(.*?)\}',
            r'<span style="color:\1">\2</span>',
            text,
            flags=re.DOTALL
        )

        # 6) 对于 \mathbf{...}、\mathit{...} 等常见数学命令，可根据需要做简单处理
        text = re.sub(r'\\mathbf{(.*?)}', r'**\1**', text, flags=re.DOTALL)
        text = re.sub(r'\\mathit{(.*?)}', r'*\1*', text, flags=re.DOTALL)

        # 7) 再做一个通用处理：去掉大部分 “\cmd{…}” 类型的命令，只保留花括号内容
        #    但排除前面已经处理过的 \frac 之类。如果还想排除其他命令，也可加 (?!xxx)
        text = re.sub(
            r'\\(?!frac)[a-zA-Z]+(\[.*?\])?\s*\{(.*?)}',
            r'\2',
            text,
            flags=re.DOTALL
        )

        # 8) 去除多余花括号：有时嵌套花括号已经被前面替换大部分，这里做一次“傻瓜型”去壳
        #    注意要多次应用，直到再也没有类似 {…}，以免一次无法去干净
        old_text = ""
        while old_text != text:
            old_text = text
            text = re.sub(r'(?<!\\)\{([^{}]*)\}', r'\1', text)

        # 9) 替换一些 LaTeX 特殊转义字符：~、\%, \_ 等
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
        for latex_cmd, rep in replacements.items():
            text = text.replace(latex_cmd, rep)

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
