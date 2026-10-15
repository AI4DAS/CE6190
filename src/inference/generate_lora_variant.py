#!/usr/bin/env python3
"""Generate fixed two-seed LoRA ablation outputs from a supplied adapter."""

from __future__ import annotations

import argparse, json, os, subprocess, time
from pathlib import Path
import torch, yaml
from diffusers import StableDiffusionPipeline


def main():
    p=argparse.ArgumentParser(); p.add_argument('--project-root',type=Path,required=True); p.add_argument('--subject',required=True)
    p.add_argument('--weights',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--experiment',required=True); p.add_argument('--variant',required=True); p.add_argument('--rank',type=int,required=True)
    p.add_argument('--n-refs',type=int,required=True); p.add_argument('--prior',choices=('on','off'),required=True); p.add_argument('--resume',action='store_true')
    a=p.parse_args(); root=a.project_root.resolve(); weights=a.weights if a.weights.is_absolute() else root/a.weights; out=a.output_root if a.output_root.is_absolute() else root/a.output_root
    if torch.cuda.device_count()!=1: raise RuntimeError('Expected one MIG')
    subjects=yaml.safe_load((root/'configs/main_subjects.yaml').read_text())['subjects']; s=subjects[a.subject]
    cfg=yaml.safe_load((root/'configs/main_prompts.yaml').read_text()); g=cfg['generation']
    pipe=StableDiffusionPipeline.from_pretrained(root/'models/base/stable-diffusion-v1-5',torch_dtype=torch.float16,safety_checker=None,local_files_only=True)
    pipe.load_lora_weights(weights.parent,weight_name=weights.name); pipe.to('cuda:0'); torch.cuda.synchronize()
    vals={'instance_token':s['instance_token'],'class_name':s['class_name'],'placeholder_token':s['placeholder_token']}; revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    refs=sorted(path.name for path in (root/s['image_dir']).iterdir() if path.is_file())[:a.n_refs]
    for pid,pc in cfg['prompt_templates'].items():
      gp=pc['personalized'].format(**vals); ep=pc['clean'].format(**vals)
      for seed in (42,123):
       d=out/s['dataset']/a.subject/pid; image=d/f'seed_{seed}.png'; meta=image.with_suffix('.json')
       if image.exists()!=meta.exists(): raise RuntimeError(f'Partial pair {image}')
       if image.exists():
        if not a.resume: raise FileExistsError(image)
        continue
       d.mkdir(parents=True,exist_ok=True); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); tick=time.perf_counter()
       result=pipe(prompt=gp,num_inference_steps=g['inference_steps'],guidance_scale=g['guidance_scale'],height=g['resolution'],width=g['resolution'],generator=torch.Generator('cuda:0').manual_seed(seed)).images[0]
       torch.cuda.synchronize(); latency=time.perf_counter()-tick; result.save(image)
       row={'method':'lora','dataset':s['dataset'],'subject':a.subject,'prompt_id':pid,'prompt_category':pc['category'],'generation_prompt':gp,'evaluation_prompt':ep,'seed':seed,'base_model':'stable-diffusion-v1-5/stable-diffusion-v1-5','checkpoint':str(weights.relative_to(root)),'training_steps':800,'resolution':g['resolution'],'inference_steps':g['inference_steps'],'guidance_scale':g['guidance_scale'],'negative_prompt':None,'scheduler':pipe.scheduler.__class__.__name__,'generation_latency_sec':latency,'peak_inference_vram_gb':torch.cuda.max_memory_allocated()/1024**3,'precision':'fp16','gpu':torch.cuda.get_device_name(0),'mig_uuid':os.environ.get('CUDA_VISIBLE_DEVICES',''),'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'git_commit':revision,'experiment':a.experiment,'variant':a.variant,'scale':None,'rank':a.rank,'step':800,'n_refs':a.n_refs,'ref_files_used':refs,'dtype':'float16','prior_preservation':a.prior}
       meta.write_text(json.dumps(row,indent=2)+'\n'); print('SAVED',image,flush=True)


if __name__=='__main__': main()
