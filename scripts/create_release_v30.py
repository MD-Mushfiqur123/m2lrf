import os
import zipfile

def create_release():
    src_dir = r"c:\Users\mushfiqur\Desktop\agent\projects\m2lrf-clean"
    out_zip = r"C:\Users\mushfiqur\Desktop\M2LRF_Master_Production_Release_v3.0_1M_Lines.zip"
    
    exclude_dirs = {".git", "__pycache__", ".pytest_cache", ".gemini", "scratch"}
    
    file_count = 0
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for f in files:
                if f.endswith('.pyc') or f.endswith('.zip'):
                    continue
                fp = os.path.join(root, f)
                rel_path = os.path.relpath(fp, src_dir)
                zf.write(fp, rel_path)
                file_count += 1

    size_mb = os.path.getsize(out_zip) / (1024 * 1024)
    print(f"Release v3.0 created: {out_zip}")
    print(f"Files packed: {file_count}, Archive size: {size_mb:.2f} MB")

if __name__ == "__main__":
    create_release()
