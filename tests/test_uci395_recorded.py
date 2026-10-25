"""Primary inputs are committed genuine UCI excerpts, NOT formula trajectories.

Corruption/renaming tests intentionally modify only a temporary copy to exercise
failure handling; they do not create subjects or new clinical training examples.
"""
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
from src.data_loaders.uci395_loader import load,parse_file,subject_code,build,partition_subjects,to_record
from src.data_loaders.schema import Trajectory

ROOT=Path(__file__).parent/'fixtures/uci395'
PD=ROOT/'parkinson/P_26060001.txt'


def numeric_rows(path):
    return [[float(v) for v in line.split(';')] for line in path.read_text().splitlines() if line and line[0].isdigit()]


def test_real_fixture_provenance_and_canonical_columns():
    provenance=json.loads((ROOT/'PROVENANCE.json').read_text())
    for source in provenance['fixtures']:
        assert hashlib.sha256((ROOT/source['fixture']).read_bytes()).hexdigest()==source['fixture_sha256']
    records=load(ROOT,time_unit='milliseconds')
    assert len(records)==4
    raw=np.array(numeric_rows(PD))
    tr=next(t for t in records if t.subject_id=='uci::P_26060001')
    assert tr.samples.shape==(len(raw),5)
    np.testing.assert_array_equal(tr.samples[:,0],raw[:,0])
    np.testing.assert_array_equal(tr.samples[:,1],raw[:,1])
    np.testing.assert_allclose(tr.samples[:,2],(raw[:,5]-raw[0,5])*1e-3)
    np.testing.assert_array_equal(tr.samples[:,3],raw[:,3]) # NOT z at raw[:,2]!
    assert np.isnan(tr.samples[:,4]).all() # GripAngle is not azimuth.
    assert np.diff(tr.samples[:,2]).min()>0
    assert tr.domain=='wacom' and tr.source=='uci395' and not tr.synthetic
    assert tr.age_bucket==-1 and np.isnan(tr.updrs) and tr.coordinate_units=='unknown'


def test_documented_header_is_parsed_not_misread_as_samples():
    audit=Counter(); records=load(ROOT,audit=audit)
    assert audit['layout_named_header']==1 and audit['txt_files_parsed']==4
    assert audit['metadata_txt_files_ignored']==1
    assert all(Trajectory.from_dict(to_record(t)).samples.shape[1]==5 for t in records)


def test_clock_is_explicit_no_rate_inference():
    ms=parse_file(PD,'same_recorded_parent',1,time_unit='milliseconds')[0]
    sec=parse_file(PD,'same_recorded_parent',1,time_unit='seconds')[0]
    np.testing.assert_allclose(sec.samples[:,2],ms.samples[:,2]*1000)
    with pytest.raises(ValueError): parse_file(PD,'same_recorded_parent',1,time_unit='guess')


def test_duplicate_clock_keeps_last_actual_row(tmp_path):
    rows=numeric_rows(PD); duplicate=rows[40].copy(); duplicate[5]=rows[10][5]
    rows.insert(11,duplicate)
    path=tmp_path/'P_26060001.txt'; np.savetxt(path,rows,delimiter=';')
    audit=Counter(); tr=parse_file(path,'uci::P_26060001',1,audit=audit)[0]
    assert len(tr.samples)==700 and audit['duplicate_timestamp_rows_removed']==1
    np.testing.assert_array_equal(tr.samples[10,[0,1,3]],np.array(duplicate)[[0,1,3]])
    assert np.all(np.diff(tr.samples[:,2])>0)


def test_clock_reversal_is_rejected_not_sorted_or_replaced(tmp_path):
    rows=numeric_rows(PD); rows[10],rows[11]=rows[11],rows[10]
    path=tmp_path/'P_26060001.txt'; np.savetxt(path,rows,delimiter=';')
    audit=Counter(); assert parse_file(path,'uci::P_26060001',1,audit=audit)==[]
    assert audit['nonmonotonic_trajectories_skipped']==1


def test_nonfinite_source_rows_are_removed(tmp_path):
    rows=numeric_rows(PD); rows[10][0]=float('nan'); rows[20][3]=float('inf')
    path=tmp_path/'P_26060001.txt'; np.savetxt(path,rows,delimiter=';')
    audit=Counter(); tr=parse_file(path,'uci::P_26060001',1,audit=audit)[0]
    assert len(tr.samples)==698 and audit['nonfinite_rows_removed']==2


def test_task_aliases_keep_parent_and_full_numeric_codes(tmp_path):
    # File alias regression on ONE genuine recording, not extra subjects/data.
    root=tmp_path/'parkinson'; root.mkdir()
    for task in ('spiral','meander'):
        (root/f'P_26060001_{task}.txt').write_bytes(PD.read_bytes())
    records=load(tmp_path)
    assert {r.subject_id for r in records}=={'uci::P_26060001'}
    assert {r.task_id for r in records}=={'spiral','meander'}
    assert len({r.session_id for r in records})==1
    assert subject_code('P_26060001')!=subject_code('P_26060002')
    assert subject_code('H_P000-0001')!=subject_code('H_P000-0002')
    assert subject_code('H_P000-0001_spiral')=='H_P000-0001'


def test_named_six_column_header_adapter_is_separate_from_source_evidence(tmp_path):
    # Mechanical header projection of recorded numbers, NOT a claim that the
    # original GripAngle was an azimuth. Unknown azimuth units remain missing.
    raw=np.array(numeric_rows(PD)); columns=raw[:,[0,1,3,4,2,5]]
    path=tmp_path/'P_26060001.txt'
    with path.open('w') as f:
        f.write(';X ; Y ; Pressure ; Azimuth ; Tilt ; Time\n')
        np.savetxt(f,columns,delimiter=';')
    tr=parse_file(path,'uci::P_26060001',1,time_unit='milliseconds')[0]
    np.testing.assert_allclose(tr.samples[:,2],(raw[:,5]-raw[0,5])*.001)
    np.testing.assert_allclose(tr.samples[:,3],raw[:,3]); assert np.isnan(tr.samples[:,4]).all()
    angular=parse_file(path,'uci::P_26060001',1,time_unit='milliseconds',azimuth_unit='degrees')[0]
    np.testing.assert_allclose(angular.samples[:,4],columns[:,3]*np.pi/180)


def test_partition_roundtrip_uses_four_distinct_recorded_parents(tmp_path):
    records,partition,summary=build(ROOT,tmp_path/'records.json',tmp_path/'partition.json',feature_contract='v6-dwt19-v1')
    sets=[set(partition[k]) for k in ['development_subjects','holdout_subjects','unlabeled_subjects']]
    assert not sets[0]&sets[1] and len(sets[0])==len(sets[1])==2
    assert {t.subject_id for t in records}==set.union(*sets)
    labels={r.subject_id:r.label for r in records}
    for part in sets[:2]: assert Counter(labels[s] for s in part)=={0:1,1:1}
    assert partition==partition_subjects(records,seed=42)
    for raw in json.loads((tmp_path/'records.json').read_text()):
        t=Trajectory.from_dict(raw)
        assert sum(t.subject_id in part for part in sets)==1
        assert raw['schema_version']=='neurotrace-trajectory-1' and raw['time_units']=='seconds'
        assert raw['updrs'] is None and raw['age_bucket']==-1


def test_prepare_only_real_local_files_builds_missing_inputs(tmp_path):
    records=tmp_path/'records.json'; partition=tmp_path/'partition.json'; prepared=tmp_path/'prepared.json'
    command=[sys.executable,'-m','scripts.train_v6_gpu_standalone','--prepare-only',
             '--local-uci-dir',str(ROOT),'--records',str(records),'--partition',str(partition),
             '--prepared-records',str(prepared),'--ingestion-report',str(tmp_path/'summary.md')]
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert 'Real-only baseline' in result.stdout and 'CONFIGURED ASSUMPTION' in result.stdout
    assert records.is_file() and partition.is_file() and prepared.is_file()
    for value in json.loads(prepared.read_text()): Trajectory.from_dict(value)
    with np.load(prepared.with_suffix('.features.npz'),allow_pickle=False) as z:
        assert z['X'].shape[1:]==(19,512) and np.isfinite(z['X']).all()
        assert set(z['subject'])=={r['subject_id'] for r in json.loads(prepared.read_text())}
    before=records.read_bytes()
    again=subprocess.run(command,text=True,capture_output=True)
    assert again.returncode==0,again.stdout+'\n'+again.stderr
    assert records.read_bytes()==before # No reconversion or silent input repair.


def test_existing_invalid_records_report_errors_instead_of_replacing(tmp_path):
    records=tmp_path/'bad.json'
    for payload in ('[{}]','[0]'):
        records.write_text(payload)
        result=subprocess.run([sys.executable,'-m','scripts.train_v6_gpu_standalone','--prepare-only',
                 '--local-uci-dir',str(ROOT),'--records',str(records),'--partition',str(tmp_path/'partition.json'),
                 '--prepared-records',str(tmp_path/'out.json')],text=True,capture_output=True)
        assert result.returncode!=0 and 'record_index' in result.stdout and '1/1' in result.stderr
        assert records.read_text()==payload and not (tmp_path/'out.json').exists()
