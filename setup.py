from setuptools import setup, find_packages

setup(
    name="latex2md",
    version="0.1.0",
    author="ccchen",
    author_email="1421243966@qq.com",
    description="A tool to convert LaTeX documents to Markdown format",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/latex2md",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "latex2md=latex2md.cli:main",
        ],
    },
    install_requires=[],
)
