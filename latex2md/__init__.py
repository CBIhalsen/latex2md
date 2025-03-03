from .converter import Latex2Md as _Latex2Md  # 将原类改为内部实现

def latex2md(input_file, output_file=None, image_dir="images", project_dir=None):
    """
    Create a LaTeX to Markdown converter.

    Args:
        input_file (str): Path to the input LaTeX file.
        output_file (str): Path to the output Markdown file (optional).
        image_dir (str): Directory to save images (default: "images").
        project_dir (str): Directory of the LaTeX project (optional).

    Returns:
        A converter instance that can convert LaTeX to Markdown.
    """
    return _Latex2Md(input_file, output_file, image_dir, project_dir)

__all__ = ["latex2md"]  # 只导出 latex2md 函数
