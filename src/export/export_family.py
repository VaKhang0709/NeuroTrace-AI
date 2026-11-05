"""Immutable PRO copy, dynamic INT8 NANO, FP16 BASE; unchanged float32 IO.

Precision gates are prespecified,NOT tuned on holdout: FP32 <1e-5,FP16 <.01,
dynamic INT8 <.10 absolute raw-logit error. Lossy quantization is not 1e-5 parity.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from src.models.v6_tremorformer import TremorFormer as Pro
from src.models.tremorformer_family import load_student,CONTRACT
from src.training.family_data import sha256,load_context,windows

LIMITS=dict(nano=2_000_000,base=12_000_000)
TOLERANCE=dict(fp32=1e-5,fp16=.01,int8=.10)


def session(path):
    options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])


def check_signature(runtime):
    inputs=runtime.get_inputs(); outputs=runtime.get_outputs()
    expected=[('trajectory','tensor(float)',['batch',19,'sequence_length']),('device_id','tensor(int64)',['batch'])]
    actual=[(v.name,v.type,v.shape) for v in inputs]
    if actual!=expected or [(v.name,v.type,v.shape) for v in outputs]!=[('logits','tensor(float)',['batch'])]:
        raise ValueError(f'Family signatures must remain identical float32/int64 dynamic IO: {actual}')
    return dict(inputs=expected,outputs=[('logits','tensor(float)',['batch'])])


class Clinical(torch.nn.Module):
    def __init__(self,model): super().__init__(); self.model=model
    def forward(self,x,device_id): return self.model(x,device_id)['pd_logits']


def export_fp32(model,path):
    model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(Clinical(model),(torch.zeros(1,19,512),torch.zeros(1,dtype=torch.int64)),str(path),
            input_names=['trajectory','device_id'],output_names=['logits'],opset_version=17,dynamo=False,
            dynamic_axes=dict(trajectory={0:'batch',2:'sequence_length'},device_id={0:'batch'},logits={0:'batch'}))
    onnx.checker.check_model(onnx.load(path)); check_signature(session(path))


def compress(source,destination,precision):
    if precision=='int8':
        from onnxruntime.quantization import quantize_dynamic,QuantType
        # ConvInteger(QInt8) / grouped depthwise kernels are not portable on CPU.
        # Quantize constant linear weights only; CNN,normalization/attention math
        # remain FP32. This is standard MIXED dynamic INT8,not all-operator INT8.
        # Export deduplicates zero FiLM weights through Identity aliases. ORT's
        # Gemm transposition can otherwise mutate their shared initializer twice.
        import copy
        model=onnx.load(source); constants={v.name:v for v in model.graph.initializer}; nodes=[]
        for node in model.graph.node:
            if node.op_type=='Identity' and node.input[0] in constants:
                value=copy.deepcopy(constants[node.input[0]]); value.name=node.output[0]
                model.graph.initializer.append(value); constants[value.name]=value
            else: nodes.append(node)
        del model.graph.node[:]; model.graph.node.extend(nodes)
        quantize_dynamic(model,str(destination),weight_type=QuantType.QInt8,per_channel=True,
                         op_types_to_quantize=['MatMul','Gemm'],extra_options={'MatMulConstBOnly':True})
    elif precision=='fp16':
        from onnxconverter_common.float16 import convert_float_to_float16
        model=convert_float_to_float16(onnx.load(source),keep_io_types=True)
        # Converter can retag existing explicit float Cast outputs as FP16
        # without updating their `to` attribute (PyTorch's position/onehot casts).
        types={v.name:v.type.tensor_type.elem_type for v in list(model.graph.value_info)+list(model.graph.input)+list(model.graph.output)}
        for node in model.graph.node:
            if node.op_type=='Cast' and types.get(node.output[0])==onnx.TensorProto.FLOAT16:
                for attr in node.attribute:
                    if attr.name=='to' and attr.i==onnx.TensorProto.FLOAT: attr.i=onnx.TensorProto.FLOAT16
        onnx.save(model,str(destination))
    else: raise ValueError('Unknown compressed precision')
    onnx.checker.check_model(onnx.load(destination)); check_signature(session(destination))


def parity(model,path,precision,real_windows=None,seed=42):
    runtime=session(path); check_signature(runtime); model.cpu().eval(); rng=np.random.default_rng(seed)
    errors=[]; real_count=0
    with torch.no_grad():
        for i in range(100):
            if i>=50 and real_windows is not None and i-50<len(real_windows['X']):
                x=real_windows['X'][i-50:i-49]; d=real_windows['device_id'][i-50:i-49]; real_count+=1
            else:
                x=rng.normal(size=([1,2,4][i%3],19,[256,384,512,768][i%4])).astype(np.float32)
                d=rng.integers(0,5,len(x),dtype=np.int64)
            reference=model(torch.from_numpy(x),torch.from_numpy(d))['pd_logits'].numpy()
            predicted=runtime.run(['logits'],dict(trajectory=x,device_id=d))[0]
            if predicted.shape!=reference.shape or not np.isfinite(predicted).all(): raise ValueError('Nonfinite/shape-invalid ONNX output')
            errors.append(float(np.max(np.abs(predicted.astype(float)-reference.astype(float)))))
    return dict(tensors=100,recorded_validation_tensors=real_count,max_logit_error=max(errors),
                tolerance=TOLERANCE[precision],passed=max(errors)<TOLERANCE[precision],reference='PyTorch raw logits')


def export_all(args):
    records,partition,metadata,context=load_context(args.records,args.partition,args.scaler,args.teacher_metadata or args.teacher_checkpoint,args.teacher_onnx)
    context['records_sha256']=sha256(args.records)
    # REQUIRE actual teacher weights; no random auxiliary-head reconstruction
    # from a pruned ONNX graph masquerading as the original full checkpoint.
    checkpoint=torch.load(args.teacher_checkpoint,map_location='cpu',weights_only=True)
    if checkpoint.get('feature_contract')!=CONTRACT or any(set(checkpoint.get(k,[]))!=set(metadata.get(k,[])) for k in ('training_subjects','validation_subjects','ssl_subjects')):
        raise ValueError('Actual teacher checkpoint differs from bound provenance')
    pro=Pro(**checkpoint['model_config']).eval()
    if pro.config!=Pro().config: raise ValueError('PRO architecture must be unchanged')
    pro.load_state_dict(checkpoint['model_state_dict'],strict=True)
    valid=windows(records,metadata['validation_subjects'])
    models={}; checkpoints={}
    for tier in ('nano','base'):
        model,ck=load_student(getattr(args,tier))
        if ck['context']!=context or set(ck['training_subjects'])!=set(metadata['training_subjects']) or set(ck['validation_subjects'])!=set(metadata['validation_subjects']):
            raise ValueError('Mismatched student/teacher/cohort/scaler provenance')
        if ck['epochs_requested']<40 or ck['settings'].get('USE_SYNTHETIC') is not False: raise ValueError('Require completed >=40epoch real-only students')
        models[tier]=model; checkpoints[tier]=ck
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'family_manifest.json'
    # A failed rerun must never leave a stale accepted manifest beside changed files.
    manifest=dict(version='v6-family-1',accepted=False,feature_contract=CONTRACT,context=context,
                  scaler='scaler_config_v6.json',scoring='RAW subject-mean sigmoid,threshold 0.50; shared teacher calibration NOT applied to students',tiers={})
    manifest_path.write_text(json.dumps(manifest,indent=2))
    for tier,precision in [('nano','int8'),('base','fp16')]:
        raw=out/f'{tier}_fp32.onnx'; target=out/f'neurotrace_v6_{tier}.onnx'
        export_fp32(models[tier],raw); fp32=parity(models[tier],raw,'fp32',valid)
        compress(raw,target,precision); error=parity(models[tier],target,precision,valid)
        result=dict(file=target.name,precision=precision,bytes=target.stat().st_size,sha256=sha256(target),
                    parameters=sum(p.numel() for p in models[tier].parameters()),fp32_parity=fp32,parity=error,
                    max_bytes=LIMITS[tier],size_passed=target.stat().st_size<=LIMITS[tier],
                    checkpoint_sha256=sha256(getattr(args,tier)))
        manifest['tiers'][tier]=result; manifest_path.write_text(json.dumps(manifest,indent=2))
        if not fp32['passed'] or not error['passed'] or not result['size_passed']: raise ValueError(f'{tier} size/parity gate failed; manifest NOT accepted')
    target=out/'neurotrace_v6_pro.onnx'
    if Path(args.teacher_onnx).resolve()!=target.resolve(): shutil.copyfile(args.teacher_onnx,target)
    if sha256(target)!=context['teacher_onnx_sha256']: raise ValueError('Immutable PRO copy changed')
    pro_error=parity(pro,target,'fp32',valid)
    manifest['tiers']['pro']=dict(file=target.name,precision='fp32',bytes=target.stat().st_size,
        parameters=sum(p.numel() for p in pro.parameters()),sha256=sha256(target),parity=pro_error,
        checkpoint_sha256=sha256(args.teacher_checkpoint))
    manifest_path.write_text(json.dumps(manifest,indent=2))
    if not pro_error['passed']: raise ValueError('Original PRO checkpoint/ONNX parity failed')
    scaler_target=out/'scaler_config_v6.json'
    if Path(args.scaler).resolve()!=scaler_target.resolve(): shutil.copyfile(args.scaler,scaler_target)
    if sha256(scaler_target)!=context['scaler_sha256']: raise ValueError('Shared production scaler was modified')
    signatures=[check_signature(session(out/v['file'])) for v in manifest['tiers'].values()]
    if any(s!=signatures[0] for s in signatures): raise ValueError('Tier signatures differ')
    manifest.update(signature=signatures[0],accepted=True)
    manifest_path.write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2)); return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('nano','base','teacher-checkpoint','teacher-onnx','records','partition','scaler'): p.add_argument('--'+field,required=True)
    p.add_argument('--teacher-metadata',help='Optional original hash-bound provenance JSON; default actual checkpoint metadata')
    p.add_argument('--output',default='models/v6/family'); export_all(p.parse_args())

if __name__=='__main__': main()
