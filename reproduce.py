"""One-click reproduction of the PRiCE results in the paper.

For California, the dataset is automatically downloaded from Hugging Face.
For Michigan, the dataset is not redistributable; the user must place the
NetCDF file at the path given by --michigan-nc (see README.md).

Both 24h and 48h lookaheads are run, with cluster files generated as needed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / 'data'

# CA dataset on Hugging Face. The hourly file is downloaded and renamed.
HF_REPO = 'shourya01/ca_power_outages'
HF_FILE = 'california_hourly.nc'
CA_LOCAL = DATA_DIR / 'california.nc'


def download_california() -> Path:
    """Download the hourly California dataset from Hugging Face into ../data/."""
    if CA_LOCAL.exists():
        print(f'[reproduce] California file already present at {CA_LOCAL}')
        return CA_LOCAL
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[reproduce] Downloading {HF_FILE} from huggingface.co/datasets/{HF_REPO}...')
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit('huggingface_hub is required for the California download. '
                 'Install it with: pip install huggingface_hub')
    src = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type='dataset',
                          local_dir=str(DATA_DIR))
    src = Path(src)
    if src != CA_LOCAL:
        shutil.move(src, CA_LOCAL)
    print(f'[reproduce] Saved to {CA_LOCAL}')
    return CA_LOCAL


def run(cmd: list[str]) -> None:
    print('[reproduce] $', ' '.join(cmd), flush=True)
    res = subprocess.run(cmd, cwd=str(HERE))
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=['california', 'michigan'], required=True,
                    help='Which dataset to run. California is auto-downloaded; '
                         'Michigan must be provided via --michigan-nc.')
    ap.add_argument('--michigan-nc', type=str, default=None,
                    help='Path to the user-supplied michigan.nc file '
                         '(required when --dataset michigan).')
    ap.add_argument('--lookaheads', type=int, nargs='+', default=[24, 48],
                    help='Lookahead horizons to run. Default: 24 48.')
    args = ap.parse_args()

    if args.dataset == 'california':
        nc_path = download_california()
        m_val = m_test = 30
        train_stride = 8
        num_leaves = 63
        tag = 'ca'
    else:
        if not args.michigan_nc:
            sys.exit('Michigan is not redistributable. Place michigan.nc on disk '
                     'and pass --michigan-nc /path/to/michigan.nc. See README.md.')
        nc_path = Path(args.michigan_nc).resolve()
        if not nc_path.exists():
            sys.exit(f'File not found: {nc_path}')
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        m_val = m_test = 5
        train_stride = 4
        num_leaves = 31
        tag = 'mi'

    py = sys.executable

    for la in args.lookaheads:
        clusters = DATA_DIR / f'clusters_{tag}_{la}.json'
        if not clusters.exists():
            run([py, str(HERE / 'cluster_counties.py'),
                 '--nc', str(nc_path), '--lookahead', str(la),
                 '--m-val', str(m_val), '--m-test', str(m_test),
                 '--out', str(clusters)])

        out_csv = DATA_DIR / f'{tag}{la}_price.csv'
        run([py, str(HERE / 'price_minimal.py'),
             '--nc', str(nc_path), '--clusters', str(clusters),
             '--lookahead', str(la),
             '--m-val', str(m_val), '--m-test', str(m_test),
             '--train-stride', str(train_stride),
             '--top-weather', '24', '--n-rounds', '400',
             '--num-leaves', str(num_leaves), '--ar-order', 'ar2',
             '--out-csv', str(out_csv)])


if __name__ == '__main__':
    main()
