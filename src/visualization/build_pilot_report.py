#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

METHODS = ["dreambooth", "lora", "textual_inversion", "ip_adapter_plus"]
LABELS = {"dreambooth":"DreamBooth", "lora":"DreamBooth-LoRA", "textual_inversion":"Textual Inversion", "ip_adapter_plus":"IP-Adapter Plus"}
PROMPTS = ["P0","P1","P2","P3","P4","P5"]
PTITLES = ["Identity","Beach","Snow","Kitchen","Red Hat","Watercolor"]
SEEDS = [42, 123, 456, 789]

def tensor_count(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        return sum(math.prod(f.get_slice(k).get_shape()) for k in f.keys())

def mib(path): return path.stat().st_size / 1024**2
def dir_mib(path): return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not any(x in p.parts for x in ("checkpoint-200","checkpoint-400","checkpoint-600","checkpoint-800","runs")) and p.name not in {"training.log","training_command.txt","runtime.txt","peak_vram.txt","git_commit.txt","mig_uuid.txt","package_versions.txt","python_executable.txt","SUCCESS","FAILED","config.yaml"} and not p.name.startswith(("training.failed_","failure_summary_"))) / 1024**2

def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=300, bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)

def image_grid(root, out, rows, cols, getter, row_labels, col_labels, size=260, title=None):
    margin, header = 150, 55; canvas = Image.new("RGB", (margin+cols*size, header+rows*size), "white"); draw=ImageDraw.Draw(canvas)
    if title:
        title_box = draw.textbbox((0, 0), title)
        title_width = title_box[2] - title_box[0]
        draw.text((margin+(cols*size-title_width)//2,2),title,fill="black")
    label_y = 30 if title else 18
    for j,t in enumerate(col_labels): draw.text((margin+j*size+8,label_y),t,fill="black")
    for i,t in enumerate(row_labels): draw.text((8,header+i*size+size//2),t,fill="black")
    for i in range(rows):
        for j in range(cols):
            im=Image.open(getter(i,j)).convert("RGB"); im.thumbnail((size-8,size-8)); x=margin+j*size+(size-im.width)//2; y=header+i*size+(size-im.height)//2; canvas.paste(im,(x,y))
    out.parent.mkdir(parents=True,exist_ok=True); canvas.save(out, dpi=(300,300)); canvas.save(out.with_suffix(".pdf"), resolution=300)

def build_seed_comparison_grids(root):
    comparison_dir = root / "results/figures/comparison"
    refs = sorted((root / "data/processed/dreambooth/dog2/images").glob("*.jpg"))
    if len(refs) != len(PROMPTS):
        raise RuntimeError(
            f"Expected {len(PROMPTS)} dog2 references, found {len(refs)}"
        )

    summary_rows = []
    row_methods = [None] + METHODS
    for seed in SEEDS:
        source_paths = {
            (method, prompt_id): root
            / f"outputs/evaluation/{method}/dreambooth/dog2/{prompt_id}/seed_{seed}.png"
            for method in METHODS
            for prompt_id in PROMPTS
        }
        missing = [path for path in source_paths.values() if not path.is_file()]
        if missing:
            for path in missing:
                print(f"Missing required image: {path}")
            raise FileNotFoundError(
                f"Seed {seed} is incomplete; refusing to create its comparison grid"
            )

        for (method, prompt_id), path in source_paths.items():
            summary_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "prompt_id": prompt_id,
                    "image_path": str(path.relative_to(root)),
                    "status": "FOUND",
                }
            )

        output = comparison_dir / f"dog2_four_method_grid_seed{seed}.png"
        image_grid(
            root,
            output,
            5,
            6,
            lambda i, j, selected_seed=seed: refs[j]
            if i == 0
            else source_paths[(row_methods[i], PROMPTS[j])],
            ["Reference"] + [LABELS[method] for method in METHODS],
            PTITLES,
            title=f"Dog2 Four-Method Comparison (Seed={seed})",
        )
        print(f"Created {output}")

    summary_path = comparison_dir / "grid_generation_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Created {summary_path}")

def bar(values, ylabel, title, path, log=False):
    fig,ax=plt.subplots(figsize=(7,4.5)); names=[LABELS[m] for m in METHODS]; x=np.arange(4); vals=np.asarray(values,float)
    ax.bar(x,vals,color=["#4C78A8","#F58518","#54A24B","#E45756"]); ax.set_xticks(x,names,rotation=15,ha="right"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y",alpha=.25)
    if log and np.all(vals>0): ax.set_yscale("log")
    savefig(fig,path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",type=Path,required=True); ap.add_argument("--comparison-grids-only",action="store_true"); a=ap.parse_args(); root=a.project_root.resolve()
    if a.comparison_grids_only:
        build_seed_comparison_grids(root)
        return
    dino=pd.read_csv(root/"results/metrics/pilot_subject_fidelity.csv"); clip=pd.read_csv(root/"results/metrics/pilot_prompt_alignment.csv")
    merged=dino.merge(clip[["method","prompt_id","seed","clip_similarity"]],on=["method","prompt_id","seed"])
    meta=[]
    for m in METHODS:
        for p in sorted((root/f"outputs/evaluation/{m}/dreambooth/dog2").glob("P*/seed_*.json")): meta.append(json.loads(p.read_text()))
    md=pd.DataFrame(meta)
    base=root/"models/base/stable-diffusion-v1-5"
    total_base=sum(tensor_count(base/x) for x in ["unet/diffusion_pytorch_model.safetensors","vae/diffusion_pytorch_model.safetensors","text_encoder/model.safetensors"])
    db_train=tensor_count(base/"unet/diffusion_pytorch_model.safetensors")
    lora_train=tensor_count(root/"outputs/training/lora/dog2_lora_r16_s800_seed42/pytorch_lora_weights.safetensors")
    ti_train=768
    trainables=[db_train,lora_train,ti_train,0]
    runtimes=[649,308,3251,0]; peaks=[18.1641,7.8242,4.7422,np.nan]
    personalized=[dir_mib(root/"outputs/training/dreambooth/dog2_db_s800_seed42"),mib(root/"outputs/training/lora/dog2_lora_r16_s800_seed42/pytorch_lora_weights.safetensors"),mib(root/"outputs/training/textual_inversion/dog2_ti_s3000_seed42/learned_embeds.safetensors"),0]
    shared=[0,0,0,mib(root/"models/ip_adapter/IP-Adapter/models/ip-adapter-plus_sd15.bin")]
    rows=[]
    for i,m in enumerate(METHODS):
        lat=md[md.method==m].generation_latency_sec; infpeak=md[md.method==m].peak_inference_vram_gb.max()
        rows.append({"method":m,"subject_specific_training_required":m!="ip_adapter_plus","training_steps":{"dreambooth":800,"lora":800,"textual_inversion":3000}.get(m,0),"training_time_seconds":runtimes[i],"training_time_minutes":runtimes[i]/60,"peak_training_vram_gb":peaks[i],"debug_peak_vram_gb":np.nan,"trainable_parameters":trainables[i],"trainable_parameter_percent":trainables[i]/total_base*100,"personalized_checkpoint_size_mb":personalized[i],"shared_adapter_or_model_size_mb":shared[i],"reference_images_required":1 if m=="ip_adapter_plus" else 6,"class_images_required":200 if m in ("dreambooth","lora") else 0,"model_loading_time_sec":md[md.method==m].model_loading_time_sec.iloc[0],"inference_latency_mean_sec":lat.mean(),"inference_latency_std_sec":lat.std(ddof=1),"peak_inference_vram_gb":infpeak,"precision":"fp16","base_model":"stable-diffusion-v1-5/stable-diffusion-v1-5","notes":"Shared adapter excludes the separately shared IP-Adapter image encoder (%.2f MiB)."%mib(root/"models/ip_adapter/IP-Adapter/models/image_encoder/model.safetensors") if m=="ip_adapter_plus" else "Full-run measurement."})
    cost=pd.DataFrame(rows); cost.to_csv(root/"results/metrics/pilot_compute_cost.csv",index=False)
    quality=merged.groupby("method").agg(dino_mean=("mean_reference_similarity","mean"),dino_std=("mean_reference_similarity","std"),clip_mean=("clip_similarity","mean"),clip_std=("clip_similarity","std")).reset_index()
    summary=quality.merge(cost,on="method").rename(columns={"subject_specific_training_required":"training_required","training_time_minutes":"training_time_min","shared_adapter_or_model_size_mb":"shared_adapter_size_mb"})
    keep=["method","dino_mean","dino_std","clip_mean","clip_std","training_required","training_time_min","peak_training_vram_gb","trainable_parameters","trainable_parameter_percent","personalized_checkpoint_size_mb","shared_adapter_size_mb","reference_images_required","class_images_required","inference_latency_mean_sec","inference_latency_std_sec","peak_inference_vram_gb"]
    summary[keep].to_csv(root/"results/pilot_summary.csv",index=False)
    table=cost.copy(); table["Method"]=table.method.map(LABELS); table["Per-subject training?"]=table.subject_specific_training_required.map({True:"Yes",False:"No"}); table["Training time"]=table.training_time_seconds.map(lambda x:f"{int(x)//60}m {int(x)%60}s"); table["Peak training VRAM"]=table.peak_training_vram_gb.map(lambda x:"N/A" if pd.isna(x) else f"{x:.2f} GiB"); table["Trainable params"]=table.trainable_parameters.map(lambda x:f"{int(x):,}"); table["Per-subject storage"]=table.personalized_checkpoint_size_mb.map(lambda x:f"{x:.3f} MiB"); table["Shared adapter storage"]=table.shared_adapter_or_model_size_mb.map(lambda x:f"{x:.2f} MiB"); table["Reference images"]=table.reference_images_required; table["Class images"]=table.class_images_required; table["Inference latency"]=table.apply(lambda r:f"{r.inference_latency_mean_sec:.3f} ± {r.inference_latency_std_sec:.3f} s",axis=1); table["Peak inference VRAM"]=table.peak_inference_vram_gb.map(lambda x:f"{x:.2f} GiB")
    cols=["Method","Per-subject training?","Training time","Peak training VRAM","Trainable params","Per-subject storage","Shared adapter storage","Reference images","Class images","Inference latency","Peak inference VRAM"]
    out=root/"results/report_tables/pilot_compute_table.csv"; table[cols].to_csv(out,index=False)
    md_rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    md_rows += ["| " + " | ".join(str(v) for v in row) + " |" for row in table[cols].itertuples(index=False, name=None)]
    out.with_suffix(".md").write_text("\n".join(md_rows) + "\n")
    annot=pd.DataFrame([{"method":r.method,"prompt_id":r.prompt_id,"prompt_category":r.prompt_category,"seed":r.seed,"background_leakage":"","pose_copying":"","composition_copying":"","notes":""} for r in md.itertuples()]); annot.to_csv(root/"results/metrics/manual_overfitting_annotation.csv",index=False)

    comp=root/"results/figures/comparison/dog2_four_method_grid.png"; refs=sorted((root/"data/processed/dreambooth/dog2/images").glob("*.jpg")); row_methods=[None]+METHODS
    image_grid(root,comp,5,6,lambda i,j: refs[j] if i==0 else root/f"outputs/evaluation/{row_methods[i]}/dreambooth/dog2/{PROMPTS[j]}/seed_42.png",["Reference"]+[LABELS[m] for m in METHODS],PTITLES)
    quality_dir=root/"results/figures/comparison"
    q=quality.set_index("method").loc[METHODS]
    bar(q.dino_mean,"Mean cosine similarity","DINOv2 subject fidelity",quality_dir/"pilot_dino_fidelity.png")
    bar(q.clip_mean,"Cosine similarity","CLIP prompt alignment",quality_dir/"pilot_clip_alignment.png")
    fig,ax=plt.subplots(figsize=(6,5));
    for m in METHODS: z=merged[merged.method==m]; ax.scatter(z.clip_similarity,z.mean_reference_similarity,label=LABELS[m],alpha=.75)
    ax.set(xlabel="CLIP prompt alignment",ylabel="DINOv2 subject fidelity",title="Fidelity–alignment trade-off"); ax.legend(fontsize=8); ax.grid(alpha=.25); savefig(fig,quality_dir/"pilot_dino_vs_clip_scatter.png")
    cat=merged.groupby(["method","prompt_category"])[["mean_reference_similarity","clip_similarity"]].mean().reset_index(); fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for m in METHODS:
        z=cat[cat.method==m]; axs[0].plot(z.prompt_category,z.mean_reference_similarity,"o-",label=LABELS[m]); axs[1].plot(z.prompt_category,z.clip_similarity,"o-",label=LABELS[m])
    for ax,title,y in zip(axs,["Fidelity by prompt category","Alignment by prompt category"],["DINOv2","CLIP"]): ax.set_title(title); ax.set_ylabel(y); ax.tick_params(axis="x",rotation=30); ax.grid(alpha=.25)
    axs[1].legend(fontsize=8); savefig(fig,quality_dir/"pilot_performance_by_prompt_category.png")
    image_grid(root,quality_dir/"background_leakage_inspection.png",4,6,lambda i,j: root/f"outputs/evaluation/{METHODS[i]}/dreambooth/dog2/{PROMPTS[j]}/seed_42.png",[LABELS[m] for m in METHODS],PTITLES)
    worst=merged.assign(score=merged.mean_reference_similarity+merged.clip_similarity).sort_values("score").groupby("method").head(2).reset_index(drop=True)
    image_grid(root,quality_dir/"failure_case_grid.png",4,2,lambda i,j: root/worst.iloc[i*2+j].image_path,[LABELS[m] for m in METHODS],["Lowest combined score 1","Lowest combined score 2"])
    costdir=root/"results/figures/compute"; bar(np.array(runtimes)/60,"Minutes","Subject-specific training time",costdir/"pilot_training_time.png"); bar(np.nan_to_num(peaks),"GiB","Peak training VRAM (IP-Adapter: N/A)",costdir/"pilot_peak_training_vram.png"); bar(trainables,"Parameters","Per-subject trainable parameters",costdir/"pilot_trainable_parameters.png",log=False); bar(personalized,"MiB","Personalized checkpoint size",costdir/"pilot_checkpoint_size.png"); bar(shared,"MiB","Shared adapter checkpoint size",costdir/"pilot_shared_adapter_size.png"); bar([r["inference_latency_mean_sec"] for r in rows],"Seconds/image","Inference latency",costdir/"pilot_inference_latency.png"); bar([r["peak_inference_vram_gb"] for r in rows],"GiB","Peak inference VRAM",costdir/"pilot_peak_inference_vram.png"); bar([6,6,6,1],"Images","Reference-image requirement",costdir/"pilot_reference_images.png"); bar([200,200,0,0],"Images","Prior class-image requirement",costdir/"pilot_class_images.png")
    for xcol,xlabel,name in [("training_time_min","Training time (min)","quality_vs_training_time"),("personalized_checkpoint_size_mb","Per-subject storage (MiB)","quality_vs_checkpoint_size"),("inference_latency_mean_sec","Inference latency (s)","alignment_vs_inference_latency")]:
        fig,ax=plt.subplots(figsize=(6,5)); y="clip_mean" if "alignment" in name else "dino_mean"; data=summary.set_index("method").loc[METHODS]
        for m in METHODS: ax.scatter(data.loc[m,xcol],data.loc[m,y],s=70,label=LABELS[m]); ax.annotate(LABELS[m],(data.loc[m,xcol],data.loc[m,y]),xytext=(4,4),textcoords="offset points",fontsize=8)
        ax.set_xlabel(xlabel); ax.set_ylabel("CLIP alignment" if y=="clip_mean" else "DINOv2 fidelity"); ax.grid(alpha=.25); savefig(fig,costdir/f"pilot_{name}.png")

if __name__=="__main__": main()
