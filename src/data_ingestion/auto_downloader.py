"""Public acquisition with bounded downloads, checksums and local fallbacks.

UCI official URL below. No guessed mirror DOI or image-to-trajectory conversion.
Operator mirror manifest specifies explicit open-access DOI, URL and SHA-256.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

UCI_URL='https://archive.ics.uci.edu/static/public/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet.zip'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def download(url,destination,expected_sha256=None,max_bytes=1024**3,timeout=30):
    """HTTPS only; atomic file replacement; never expose partially fetched data."""
    if urlparse(url).scheme!='https': raise ValueError('Only explicit HTTPS sources allowed')
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix(destination.suffix+'.partial')
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'NeuroTrace-V6-research/1'})
        with urllib.request.urlopen(request,timeout=timeout) as response,tmp.open('wb') as out:
            if urlparse(response.geturl()).scheme!='https': raise ValueError('Insecure redirect')
            total=0
            while True:
                block=response.read(1024*1024)
                if not block: break
                total+=len(block)
                if total>max_bytes: raise ValueError('Download size limit exceeded')
                out.write(block)
        digest=sha256(tmp)
        if expected_sha256 and digest.lower()!=expected_sha256.lower(): raise ValueError('SHA-256 mismatch')
        tmp.replace(destination)
        return digest
    finally:
        tmp.unlink(missing_ok=True)


def extract_zip(archive,destination,max_uncompressed=2*1024**3):
    """Reject traversal, duplicate paths, symlinks and oversized archives."""
    destination=Path(destination)
    if destination.exists() and any(destination.iterdir()): raise FileExistsError('Nonempty extraction target')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        stage=Path(tmp)
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist())>max_uncompressed: raise ValueError('Expanded archive too large')
            seen=set()
            for item in z.infolist():
                name=item.filename.replace('\\','/')
                target=stage/name
                if name in seen or name.startswith('/') or not target.resolve().is_relative_to(stage.resolve()):
                    raise ValueError('Unsafe/duplicate archive member')
                if (item.external_attr>>16)&0o170000==0o120000: raise ValueError('Symlink archive member')
                seen.add(name)
            z.extractall(stage)
        if destination.exists(): destination.rmdir()
        shutil.copytree(stage,destination)


def fetch_open_mirror(entry,output):
    """DOI-tagged direct files, not a DOI-resolution guess or access-gate bypass.

    Example manifest fields: name,doi,url,sha256,license,modality. Service-specific
    public APIs should resolve the file URL before supplying this manifest.
    """
    if not all(entry.get(k) for k in ('name','doi','url','sha256','license','modality')):
        raise ValueError('Mirror requires DOI, file URL, checksum, license and modality')
    host=urlparse(entry['url']).hostname or ''
    if not any(host==h or host.endswith('.'+h) for h in ('zenodo.org','figshare.com','figshare.s3.amazonaws.com','ndownloader.figshare.com','physionet.org')):
        raise ValueError('Unsupported public mirror host')
    if entry['modality']!='dynamic_trajectory': raise ValueError('Static images are not digitizer streams')
    if not entry['doi'].startswith('10.'): raise ValueError('Explicit DOI required')
    name=entry['name']
    if not name.replace('_','').replace('-','').isalnum(): raise ValueError('Unsafe dataset name')
    archive=Path(output)/f'{name}.zip'; digest=download(entry['url'],archive,entry['sha256'])
    extract_zip(archive,Path(output)/name)
    return dict(dataset=name,status='DOWNLOADED_CHECKSUM_VERIFIED',sha256=digest,
                license=entry['license'],doi=entry['doi'],format_review='REQUIRED before parser use')


def acquire(output='data/raw',local_uci=None,mirrors=(),report='reports/v6_ingested_datasets_summary.md'):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rows=[]; target=output/'uci_395'
    if target.exists() and any(target.rglob('*.txt')):
        rows.append(dict(dataset='UCI395',status='LOCAL_UNVERIFIED',path=str(target),
                         detail='Existing raw files; parser/QC and official release checksum review required'))
    else:
        archive=Path(local_uci) if local_uci else output/'uci_395.zip'
        try:
            if local_uci and archive.is_dir():
                if target.exists(): raise FileExistsError('Refuse overwrite; review existing target')
                shutil.copytree(archive,target)
                rows.append(dict(dataset='UCI395',status='LOCAL_FALLBACK_UNVERIFIED',path=str(target)))
            else:
                digest=sha256(archive) if archive.is_file() else download(UCI_URL,archive)
                extract_zip(archive,target)
                rows.append(dict(dataset='UCI395',status='EXTRACTED_NOT_CLINICALLY_VALIDATED',path=str(target),
                                 sha256=digest,detail='Computed hash is not an independent official checksum verification'))
        except Exception as exc:
            rows.append(dict(dataset='UCI395',status='BLOCKED',detail=f'{type(exc).__name__}: {exc}',
                             fallback='--local-uci /authorized/uci.zip OR /authorized/raw_dir'))
    for entry in mirrors:
        try: rows.append(fetch_open_mirror(entry,output))
        except Exception as exc:
            rows.append(dict(dataset=entry.get('name','unknown'),status='BLOCKED',detail=f'{type(exc).__name__}: {exc}'))
    if not mirrors:
        rows.append(dict(dataset='Zenodo/Figshare/PhysioNet',status='NO_VERIFIED_DOI_SUPPLIED',
                         detail='No automatic search result is assumed to be licensed dynamic handwriting. Supply reviewed mirror manifest.'))
    text='# V6 public dataset acquisition summary\n\nMachine-recorded acquisition outcomes; no real sample counts inferred from failed requests.\n\n'
    for row in rows: text+='## '+row['dataset']+'\n\n```json\n'+json.dumps(row,indent=2)+'\n```\n\n'
    text+='Only acquisition is reported here. Raw units, participant IDs, labels and licensing still require source review.\n'
    report=Path(report); report.parent.mkdir(parents=True,exist_ok=True); report.write_text(text)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',default='data/raw')
    p.add_argument('--local-uci'); p.add_argument('--mirrors',help='Reviewed JSON list of DOI/direct-file metadata')
    p.add_argument('--report',default='reports/v6_ingested_datasets_summary.md')
    a=p.parse_args(); rows=acquire(a.output,a.local_uci,json.loads(Path(a.mirrors).read_text()) if a.mirrors else (),a.report)
    print(json.dumps(rows,indent=2))
    # Blocked network is a recoverable acquisition outcome, not a success dataset.
    raise SystemExit(2 if all(r['status'] in ('BLOCKED','NO_VERIFIED_DOI_SUPPLIED') for r in rows) else 0)



def resolve_zenodo_files(record_id):
    """Public API discovery only. Operator still reviews license/modality/checksum."""
    record_id=str(record_id)
    if not record_id.isdigit(): raise ValueError('Numeric Zenodo record ID required')
    with urllib.request.urlopen(f'https://zenodo.org/api/records/{record_id}',timeout=30) as response:
        metadata=json.load(response)
    return dict(doi=metadata.get('doi'),license=metadata.get('metadata',{}).get('license'),
                files=[dict(name=f['key'],url=f['links']['self'],checksum=f.get('checksum')) for f in metadata.get('files',[])])


def resolve_figshare_files(article_id):
    article_id=str(article_id)
    if not article_id.isdigit(): raise ValueError('Numeric Figshare article ID required')
    with urllib.request.urlopen(f'https://api.figshare.com/v2/articles/{article_id}',timeout=30) as response:
        metadata=json.load(response)
    return dict(doi=metadata.get('doi'),license=metadata.get('license'),files=metadata.get('files',[]))

if __name__=='__main__': main()
