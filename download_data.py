"""Fetch CAFA5 training inputs, retaining checksums and download provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

FILES = ('train_sequences.fasta', 'train_terms.tsv', 'go-basic.obo')
COMPETITION = 'cafa-5-protein-function-prediction'
MIRROR = 'https://zenodo.org/records/10951709'
URL = MIRROR + '/files/cafa-5-protein-function-prediction.zip?download=1'
EXPECTED_MD5 = '6647edcf2106cfdee537e3fbf6e51dfe'


def checksum(path, algorithm='sha256'):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['auto', 'kaggle', 'zenodo'], default='auto')
    parser.add_argument('--output', type=Path, default=Path('data/cafa5'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if all((args.output / f).is_file() for f in FILES):
        if not (args.output / 'download_provenance.json').is_file():
            raise SystemExit('Existing inputs have no download provenance; use a new output directory.')
        print(f'Inputs already exist: {args.output.resolve()}', flush=True)
        return
    provenance = {'downloaded_at_utc': datetime.now(timezone.utc).isoformat()}
    if args.source != 'zenodo':
        try:
            os.environ.setdefault('KAGGLEHUB_CACHE', str((args.output.parent / 'kaggle_cache').resolve()))
            import kagglehub
            print('Trying official Kaggle competition download...', flush=True)
            root = Path(kagglehub.competition_download(COMPETITION))
            for filename in FILES:
                matches = list(root.rglob(filename))
                if len(matches) != 1:
                    raise RuntimeError(f'Expected one {filename}, found {len(matches)}')
                shutil.copy2(matches[0], args.output / filename)
            provenance.update(source='official Kaggle competition',
                              url=f'https://www.kaggle.com/competitions/{COMPETITION}/data')
        except Exception as error:
            # Do not print exceptions that might contain credential-bearing URLs.
            print(f'Official download unavailable ({type(error).__name__}).', flush=True)
            if args.source == 'kaggle':
                raise SystemExit('Kaggle access failed. Sign in and accept the competition rules, or use --source zenodo.')
            provenance['kaggle_attempt'] = type(error).__name__
    if 'source' not in provenance:
        import requests
        archive = args.output.parent / 'cafa5_zenodo.zip'
        if not archive.is_file() or checksum(archive, 'md5') != EXPECTED_MD5:
            print(f'Downloading public mirror: {MIRROR}', flush=True)
            temporary = archive.with_suffix('.zip.part')
            with requests.get(URL, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            if checksum(temporary, 'md5') != EXPECTED_MD5:
                raise SystemExit('Mirror checksum mismatch; downloaded archive was not used.')
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as bundle:
            for filename in FILES:
                matches = [n for n in bundle.namelist() if Path(n).name == filename]
                if len(matches) != 1:
                    raise SystemExit(f'Expected one {filename} in mirror, found {len(matches)}')
                # Copy only selected training files, without trusting archive paths.
                with bundle.open(matches[0]) as source, (args.output / filename).open('wb') as target:
                    shutil.copyfileobj(source, target)
        provenance.update(source='third-party Zenodo mirror of Kaggle CAFA5 data',
                          url=MIRROR, download_url=URL, archive_md5=EXPECTED_MD5,
                          archive_md5_verified=True,
                          note='Publisher archive checksum verified; equivalence to official Kaggle files is not independently verified.')
    provenance['files'] = {f: {'sha256': checksum(args.output / f),
                              'bytes': (args.output / f).stat().st_size} for f in FILES}
    (args.output / 'download_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Training inputs ready: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
