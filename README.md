# LaTeX2Md - LaTeX to Markdown 转换器

latex to md 初始化版本，针对复杂学术论文project，处理仍有问题欢迎贡献

LaTeX2Md 是一个强大的 Python 工具，可将 LaTeX 文档转换为 Markdown 格式，同时保留结构、格式、图片、表格、公式和引用。它特别适用于学术论文和复杂文档的转换。

## 特性

- 将 LaTeX 章节、小节和格式转换为 Markdown
- 处理图片，并将其复制到输出目录
- 将 LaTeX 表格转换为 Markdown 表格
- 使用 Markdown 数学语法保留数学公式
- 处理引用和参考文献
- 支持多列布局
- 处理包含多个文件的学术论文项目结构
- 保留文档结构和交叉引用

## 安装

```bash
pip install latex2md
```

## 使用方法

### 基本用法

转换单个 LaTeX 文件到 Markdown：

```bash
latex2md path/to/your/document.tex
```

这将创建一个名为 `document_md` 的文件夹，其中包含：
- `document.md` - 转换后的 Markdown 文件
- `images/` - 包含所有提取的图片的目录

### 高级选项

```bash
latex2md path/to/your/document.tex -o custom_output.md -i custom_images_dir -p project_directory -v
```

选项：
- `-o, --output`：指定输出 Markdown 文件路径
- `-i, --image-dir`：指定保存图片的目录（默认为 `images`）
- `-p, --project-dir`：指定 LaTeX 项目的根目录（用于处理多文件项目）
- `-v, --verbose`：显示详细输出信息

### 转换学术论文项目

对于包含多个文件的学术论文项目：

```bash
latex2md path/to/main.tex -p path/to/project_directory
```

这将处理主文件及其引用的所有文件，并创建一个名为 `project_directory_md` 的文件夹。

## 输出结构

- 对于单个文件转换：创建 `filename_md` 文件夹，包含 Markdown 文件和 `images` 子文件夹
- 对于项目转换：创建 `project_name_md` 文件夹，包含转换后的内容和 `images` 子文件夹

## 中文支持

LaTeX2Md 完全支持中文内容的转换。对于包含中文的 LaTeX 文档，转换后的 Markdown 将正确保留所有中文字符。


## 常见问题

### 图片转换问题

如果 LaTeX 文档中包含 PDF 图片，LaTeX2Md 会尝试将其复制为 PNG 文件。对于更好的转换效果，建议使用图像处理工具（如 pymupdf）将 PDF 转换为 PNG。

### 复杂数学公式

对于非常复杂的数学公式，可能需要在转换后手动调整。LaTeX2Md 使用 Markdown 的数学语法（通常是 MathJax 或 KaTeX 兼容的语法）来表示公式。

### 表格处理

复杂的 LaTeX 表格（如合并单元格或特殊格式）可能需要在转换后手动调整。

## 贡献

欢迎贡献！请随时提交问题报告或拉取请求。

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

希望这个工具能帮助您轻松将 LaTeX 文档转换为 Markdown 格式！如有任何问题或建议，请通过 GitHub Issues 联系我们。
