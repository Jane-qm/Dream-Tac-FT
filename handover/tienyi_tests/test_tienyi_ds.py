import pickle, time, json, os, numpy as np, torch
D = os.environ.get("D", "/media/jushen/zhixuan/tienyi_preprocessed/smoke3")
instr = json.load(open(f"{D}/preprocessing_metadata.json"))["dataset_info"]["task_name"]
pickle.dump({instr: torch.randn(1, 512, 1024)}, open("/media/jushen/zhixuan/tienyi_tests/fake_t5.pkl", "wb"))
if os.path.exists(f"{D}/dataset_statistics_tienyi.json"): os.remove(f"{D}/dataset_statistics_tienyi.json")
from cosmos_policy.datasets.tienyi_dataset import TienyiForceDataset, _EPISODE_CACHE
from torch.utils.data import DataLoader
for use_force in (True, False):
    for mem in (True, False):
        t0 = time.time()
        ds = TienyiForceDataset(D, "/media/jushen/zhixuan/tienyi_tests/fake_t5.pkl", is_train=True, use_force=use_force, load_videos_to_memory=mem)
        print(f"init {time.time()-t0:.1f}s len={len(ds)} pixel_frames={ds.num_pixel_frames}")
        s = ds[0]; s = ds[len(ds)-1]
        for k, v in s.items():
            print(" ", k, tuple(v.shape) if torch.is_tensor(v) else v, (v.dtype if torch.is_tensor(v) else ""))
        t0 = time.time()
        for i in np.random.randint(0, len(ds), 20): ds[int(i)]
        print(f"  getitem avg {(time.time()-t0)/20*1000:.1f} ms")
        b = next(iter(DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)))
        print("  batch video", b["video"].shape, "actions", b["actions"].shape, "range", float(b["actions"].min()), float(b["actions"].max()))
        if use_force:
            print("  force range", float(b["force"].min()), float(b["force"].max()), "future", float(b["future_force"].min()), float(b["future_force"].max()), "gate", b["tactile_self_attn_gate"])
print("cache entries", len(_EPISODE_CACHE))
# 二次实例化走缓存
t0=time.time(); TienyiForceDataset(D, "/media/jushen/zhixuan/tienyi_tests/fake_t5.pkl", is_train=True); print(f"cached re-init {time.time()-t0:.2f}s")
# val
dv = TienyiForceDataset(D, "/media/jushen/zhixuan/tienyi_tests/fake_t5.pkl", is_train=False); print("val len", len(dv))
print(json.load(open(f"{D}/dataset_statistics_tienyi.json")).keys())
