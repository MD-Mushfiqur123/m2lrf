import os

def analyze(root_dir):
    total_files = 0
    total_lines = 0
    total_words = 0
    by_ext = {}
    for root, dirs, files in os.walk(root_dir):
        if any(x in root for x in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            fp = os.path.join(root, f)
            try:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as fh:
                    content = fh.read()
                    lines = len(content.splitlines())
                    words = len(content.split())
                    total_files += 1
                    total_lines += lines
                    total_words += words
                    if ext not in by_ext:
                        by_ext[ext] = [0, 0, 0]
                    by_ext[ext][0] += 1
                    by_ext[ext][1] += lines
                    by_ext[ext][2] += words
            except Exception:
                pass
    return total_files, total_lines, total_words, by_ext

tf, tl, tw, by_ext = analyze(r'c:\Users\mushfiqur\Desktop\agent\projects\m2lrf-clean')
print(f"=== M2LRF-CLEAN AUDIT ===")
print(f"Total Files: {tf:,}")
print(f"Total Lines: {tl:,}")
print(f"Total Words: {tw:,}")
print("Top Extensions by Lines:")
for ext, (cnt, lns, wds) in sorted(by_ext.items(), key=lambda x: -x[1][1])[:10]:
    ext_name = ext if ext else "[no ext]"
    print(f"  {ext_name:10s}: {cnt:4d} files | {lns:8,d} lines | {wds:10,d} words")

# Also measure python files in m2lrf core vs tests
py_m2lrf = [0, 0, 0]
py_tests = [0, 0, 0]
py_bench = [0, 0, 0]
for root, dirs, files in os.walk(r'c:\Users\mushfiqur\Desktop\agent\projects\m2lrf-clean'):
    if '__pycache__' in root or '.git' in root: continue
    for f in files:
        if f.endswith('.py'):
            fp = os.path.join(root, f)
            with open(fp, 'r', encoding='utf-8', errors='ignore') as fh:
                c = fh.read()
                l = len(c.splitlines())
                w = len(c.split())
                if '\\tests' in root or '/tests' in root:
                    py_tests[0] += 1; py_tests[1] += l; py_tests[2] += w
                elif '\\benchmarks' in root or '/benchmarks' in root:
                    py_bench[0] += 1; py_bench[1] += l; py_bench[2] += w
                else:
                    py_m2lrf[0] += 1; py_m2lrf[1] += l; py_m2lrf[2] += w

print(f"\nCore Python Code (m2lrf/):   {py_m2lrf[0]} files | {py_m2lrf[1]:,d} lines | {py_m2lrf[2]:,d} words")
print(f"Unit Tests (tests/):         {py_tests[0]} files | {py_tests[1]:,d} lines | {py_tests[2]:,d} words")
print(f"Benchmarks (benchmarks/):    {py_bench[0]} files | {py_bench[1]:,d} lines | {py_bench[2]:,d} words")

ws_files, ws_lines, ws_words, _ = analyze(r'c:\Users\mushfiqur\Desktop\agent')
print(f"\n=== OVERALL WORKSPACE AUDIT (c:\\Users\\mushfiqur\\Desktop\\agent) ===")
print(f"Total Workspace Files: {ws_files:,}")
print(f"Total Workspace Lines: {ws_lines:,}")
print(f"Total Workspace Words: {ws_words:,}")
