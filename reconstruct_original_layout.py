"""Copy the archive files into the ORIGINAL project layout (experiments/, results/, data/ ...) expected by the frozen scripts.
usage: python reconstruct_original_layout.py <target_dir>
Afterwards edit the path constants listed in README.md ("Running the code") in the copied scripts before executing them."""
import csv, os, shutil, sys
here = os.path.dirname(os.path.abspath(__file__)); tgt = sys.argv[1]
for r in csv.DictReader(open(os.path.join(here, 'PATH_MAP.csv'))):
    dst = os.path.join(tgt, *r['original_relative_path_under_project_R2_tree'].split('/')); os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(os.path.join(here, *r['archive_path'].split('/')), dst)
print('done')
