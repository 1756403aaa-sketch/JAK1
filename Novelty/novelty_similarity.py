import argparse
import concurrent.futures as cf
import csv
import gzip
import hashlib
import itertools
import json
import os
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator

CANDIDATES = {
    'NPBS_111591': 'C/C(C[C@@H](O)[C@@H](C)[C@H]1CC[C@H]2[C@@H]3C[C@@H]4O[C@@H]5CC(=O)[C@](C)([C@H]3CC[C@]12C)[C@@]4(O)[C@@H]5O)=C1/COC1=O',
    'NPBS_210368': 'C[C@@H]1CC[C@H]2[C@H](C(=O)c3c4c(c[nH]c3=O)[C@@]3(O)CC[C@@H](O)[C@@H](O)[C@@H]3O4)[C@H](C)C=C[C@@H]2C1',
    'NPBS_106117': 'C[C@H]([C@@H]1NC(=O)[C@@H](C[S@+](C)[O-])NC1=O)[C@@]1(O)C(=O)N(C)c2ccccc21',
    '3DSMILES_217': 'O=C(NC[C@@H]1CN(C(=O)c2cccc(O)c2)C[C@H]1O)c1ncc[nH]1',
    '3DSMILES_17365': 'O=C1CCc2cc(N3C[C@H](NC(=O)C4CCC4)[C@@H](O)C3)cc(F)c2N1',
    '3DSMILES_38767': 'O=C(Nc1cc(-c2cc(CO)c(CO)c(CO)c2O)ccn1)C1CC1',
}
PROTOCOL = 'RDKit2026.09.1-AsSuppliedSMILES-MorganR2B2048Chiral-v2'
GENERATOR = QUERY_FPS = None
STRUCTURE_TIMEOUT = 30

def parse_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES')
    return mol

def initialize(timeout=30):
    global GENERATOR, QUERY_FPS, STRUCTURE_TIMEOUT
    STRUCTURE_TIMEOUT = timeout
    RDLogger.DisableLog('rdApp.*')
    GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    QUERY_FPS = [GENERATOR.GetFingerprint(parse_structure(s)) for s in CANDIDATES.values()]

def _calculate_batch(job, skipped=None, current=None):
    index, records = job
    fps, valid, errors = [], [], []
    skipped = skipped or set()
    for position, (identifier, smiles) in enumerate(records):
        if current is not None:
            current.value = position
        if position in skipped:
            errors.append([identifier, 'Structure runtime timeout'])
            continue
        try:
            mol = parse_structure(smiles)
            fp = GENERATOR.GetFingerprint(mol)
            fps.append(fp)
            valid.append((identifier, smiles))
        except Exception as exc:
            errors.append([identifier, str(exc)])
    if current is not None:
        current.value = -2
    best = []
    for q in QUERY_FPS:
        if not fps:
            best.append(None)
            continue
        values = DataStructs.BulkTanimotoSimilarity(q, fps)
        j = max(range(len(values)), key=values.__getitem__)
        best.append({'Tanimoto': values[j], 'ID': valid[j][0], 'SMILES': valid[j][1]})
    return {'batch': index, 'records': len(records), 'valid': len(valid), 'errors': errors, 'best': best}

def _batch_child(job, skipped, current, connection):
    try:
        initialize()
        connection.send(('ok', _calculate_batch(job, skipped, current)))
    except BaseException as exc:
        connection.send(('error', repr(exc)))
    finally:
        connection.close()

def process_batch(job):
    context = mp.get_context('spawn')
    skipped = set()
    while True:
        current = context.RawValue('q', -1)
        receiver, sender = context.Pipe(duplex=False)
        child = context.Process(target=_batch_child, args=(job, skipped, current, sender))
        child.start()
        sender.close()
        previous, since = -1, time.monotonic()
        timed_out = False
        try:
            while True:
                if receiver.poll(0.2):
                    status, result = receiver.recv()
                    child.join()
                    if status != 'ok':
                        raise RuntimeError(result)
                    return result
                position = current.value
                if position != previous:
                    previous, since = position, time.monotonic()
                limit = STRUCTURE_TIMEOUT if position >= 0 else 300
                if time.monotonic() - since > limit:
                    if position < 0:
                        raise RuntimeError('Batch initialization or result transfer timed out')
                    skipped.add(position)
                    timed_out = True
                    break
                if not child.is_alive():
                    raise RuntimeError(f'Batch child exited unexpectedly: {child.exitcode}')
        finally:
            if child.is_alive():
                child.terminate()
            child.join()
            receiver.close()
        if not timed_out:
            raise RuntimeError('Unfinished batch')

def records_for(database, path):
    if database == 'JAK1':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for i, row in enumerate(csv.DictReader(stream), 2):
                yield str(i), row.get('Processed_SMILES') or row.get('SMILES', '')
    else:
        with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
            if database == 'ChEMBL':
                for row in csv.DictReader(stream, delimiter='\t'):
                    yield row['chembl_id'], row['canonical_smiles']
            else:
                for line in stream:
                    identifier, smiles = line.rstrip('\r\n').split('\t', 1)
                    yield identifier, smiles

def batches(records, size):
    iterator = iter(records)
    while batch := list(itertools.islice(iterator, size)):
        yield batch

def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def scan(database, path, output_dir, workers, batch_size, timeout):
    start = time.monotonic()
    metadata = {'database': database, 'input': str(path.resolve()), 'bytes': path.stat().st_size,
                'sha256': file_sha256(path),
                'protocol': PROTOCOL, 'batch_size': batch_size, 'rdkit_version': rdBase.rdkitVersion,
                'candidates': CANDIDATES, 'structure_timeout': timeout}
    log = output_dir / (database + '_batches.jsonl')
    meta_file = output_dir / (database + '_scan_metadata.json')
    if meta_file.exists():
        old = json.loads(meta_file.read_text(encoding='utf-8'))
        if old != metadata:
            raise ValueError(f'Input/protocol changed for {database}; use a fresh work directory')
    else:
        meta_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    complete_file = output_dir / (database + '_complete.json')
    if complete_file.exists():
        cached = json.loads(complete_file.read_text(encoding='utf-8'))
        if cached.get('complete') and all(cached.get(k) == v for k, v in metadata.items()):
            if cached.get('runtime_timeout_seconds') not in (None, timeout):
                raise ValueError('Timeout setting changed; use fresh source-specific checkpoint files to retry exceptions')
            print(database, 'verified complete-scan cache', cached['records'], flush=True)
            return cached
    done, totals, best = set(), {'records': 0, 'valid': 0, 'invalid': 0}, [None] * 6
    def merge(result):
        done.add(result['batch'])
        totals['records'] += result['records']
        totals['valid'] += result['valid']
        totals['invalid'] += len(result['errors'])
        for i, entry in enumerate(result['best']):
            if entry and (best[i] is None or (-entry['Tanimoto'], entry['ID']) < (-best[i]['Tanimoto'], best[i]['ID'])):
                best[i] = entry
    if log.exists():
        with log.open(encoding='utf-8') as stream:
            for line in stream:
                result = json.loads(line)
                if result['batch'] in done:
                    raise ValueError('Duplicate batch in checkpoint')
                merge(result)
    pending = iter((i, b) for i, b in enumerate(batches(records_for(database, path), batch_size)) if i not in done)
    last = 0
    with log.open('a', encoding='utf-8') as stream, cf.ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(timeout,)) as pool:
        futures = {}
        exhausted = False
        while futures or not exhausted:
            while len(futures) < workers * 2 and not exhausted:
                try:
                    job = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                futures[pool.submit(process_batch, job)] = job[0]
            if not futures:
                break
            completed, _ = cf.wait(futures, timeout=10, return_when=cf.FIRST_COMPLETED)
            for future in completed:
                result = future.result()
                stream.write(json.dumps(result, ensure_ascii=True) + '\n')
                stream.flush()
                merge(result)
                del futures[future]
            elapsed = time.monotonic() - start
            if elapsed - last >= 30:
                progress = {'database': database, **totals, 'elapsed_seconds': round(elapsed, 1),
                            'best': dict(zip(CANDIDATES, best)), 'complete': False}
                (output_dir / (database + '_progress.json')).write_text(json.dumps(progress, indent=2), encoding='utf-8')
                print(database, totals, 'seconds', round(elapsed), flush=True)
                last = elapsed
    if not done or any(v is None for v in best):
        raise ValueError('No valid reference structures')
    if done != set(range(max(done) + 1)):
        raise ValueError('Missing batches')
    result = {**metadata, **totals, 'complete': True, 'elapsed_seconds': time.monotonic()-start,
              'runtime_timeout_seconds': timeout,
              'best': dict(zip(CANDIDATES, best))}
    complete_file.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(database, 'COMPLETE', totals, flush=True)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=HERE/'data')
    parser.add_argument('--chembl', type=Path, help='Path to the complete ChEMBL chemreps gzip export')
    parser.add_argument('--pubchem', type=Path, help='Path to the complete PubChem CID-SMILES gzip export')
    parser.add_argument('--jak1', type=Path, help='Path to the unfiltered JAK1 reference CSV')
    parser.add_argument('--output-dir', type=Path, default=HERE/'results')
    parser.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2)-2)))
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--structure-timeout', type=float, default=30)
    parser.add_argument('--only', choices=['ChEMBL', 'PubChem', 'JAK1'])
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1 or args.structure_timeout <= 0:
        parser.error('Workers, batch size, and structure timeout must be positive.')
    sources = {'ChEMBL': args.chembl or args.data_dir/'chembl_37_chemreps.txt.gz',
               'PubChem': args.pubchem or args.data_dir/'CID-SMILES.gz',
               'JAK1': args.jak1 or args.data_dir/'jak1_processed.csv'}
    if args.only:
        sources = {args.only: sources[args.only]}
    for database, path in sources.items():
        if not path.is_file():
            parser.error(f'{database} reference file does not exist: {path}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    results = {}
    for database, path in sources.items():
        results[database] = scan(database, path, checkpoints, args.workers,
                                 args.batch_size, args.structure_timeout)
    with (args.output_dir/'similarity_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        columns = ['Candidate']
        for database in sources:
            columns.extend([database, {'ChEMBL': 'ChEMBL_ID', 'PubChem': 'PubChem_CID',
                                       'JAK1': 'JAK1_CSV_Line'}[database]])
        writer.writerow(columns)
        for name in CANDIDATES:
            row = [name]
            for database in sources:
                nearest = results[database]['best'][name]
                row.extend([format(nearest['Tanimoto'], '.6f'), nearest['ID']])
            writer.writerow(row)
    with (args.output_dir/'processing_exceptions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Database', 'Identifier', 'Reason'])
        for database in sources:
            with (checkpoints/(database+'_batches.jsonl')).open(encoding='utf-8') as log:
                for line in log:
                    for identifier, reason in json.loads(line)['errors']:
                        writer.writerow([database, identifier, reason])
    (args.output_dir/'processing_summary.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=True), encoding='utf-8')
    print('Outputs written to', args.output_dir.resolve(), flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()