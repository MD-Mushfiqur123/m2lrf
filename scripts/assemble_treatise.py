# -*- coding: utf-8 -*-
"""
Assembler script for M-2LRF Volume 1 Treatise.
Combines all modular chapters into docs/VOLUME_1_MATHEMATICAL_FOUNDATIONS.md.
"""

import os
import sys

# Add scripts directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from treatise_frontmatter import FRONTMATTER
from treatise_ch1_ch2 import CHAPTER_1_2
from treatise_ch3_ch4 import CHAPTER_3_4
from treatise_ch5_ch6 import CHAPTER_5_6
from treatise_ch7_ch8 import CHAPTER_7_8
from treatise_ch9_ch10 import CHAPTER_9_10
from treatise_appendix import APPENDIX

def clean_section(text):
    lines = text.strip().splitlines()
    while lines and lines[-1].strip() == "---":
        lines.pop()
    while lines and lines[0].strip() == "---":
        lines.pop(0)
    return "\n".join(lines).strip()

def assemble():
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    docs_dir = os.path.join(repo_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    target_path = os.path.join(docs_dir, "VOLUME_1_MATHEMATICAL_FOUNDATIONS.md")
    
    raw_sections = [
        FRONTMATTER,
        CHAPTER_1_2,
        CHAPTER_3_4,
        CHAPTER_5_6,
        CHAPTER_7_8,
        CHAPTER_9_10,
        APPENDIX
    ]
    
    cleaned_sections = [clean_section(s) for s in raw_sections]
    full_content = "\n\n---\n\n".join(cleaned_sections) + "\n"
    
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(full_content)
        
    file_size = os.path.getsize(target_path)
    line_count = len(full_content.splitlines())
    
    print(f"Successfully assembled: {target_path}")
    print(f"Total Lines: {line_count}")
    print(f"Total Bytes: {file_size} ({file_size / 1024:.2f} KB)")
    
    return target_path, line_count, file_size

if __name__ == "__main__":
    assemble()
