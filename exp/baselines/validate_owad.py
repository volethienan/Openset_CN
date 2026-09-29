"""Fidelity check of the OWAD code path used by sota.run_owad: re-runs the authors' NIDS demo
(sota/OWAD/demo/NIDS_KitNET_Kyoto.ipynb, Kyoto-2006 2007 -> 2011) and compares with the outputs stored in the notebook.
Stored (authors): AUC@2007 0.9472 | AUC@2011 before adaptation 0.8074 | after OWAD (20k labels) 0.8131 | retraining 0.7897
"""
import sys, os, numpy as np
OWAD = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sota", "OWAD").replace("\\", "/") + "/"
import yaml
_load = yaml.load; yaml.load = lambda f, Loader=yaml.SafeLoader: _load(f, Loader=Loader)   # PyYAML>=6 compat (as in sota.py)
for p in ("", "admodels", "moudles"): sys.path.insert(0, OWAD + p)
os.chdir(OWAD + "demo")
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import AE, myutils as utils
from calibrator import Calibrator
from shifthunter import ShiftHunter

utils.set_random_seed()
feat = np.load("./data/2007.npz"); train_num = 50000
X, y = feat["X"], feat["y"]; X_train = X[y == 0][:train_num]
scaler = MinMaxScaler().fit(X_train); X_train = scaler.transform(X_train)
model, thres = AE.train(X_train, X_train.shape[-1])
F0, F1 = np.load("./data/2007.npz"), np.load("./data/2011.npz")
X_0, y_0 = scaler.transform(F0["X"]), F0["y"]; X_1, y_1 = scaler.transform(F1["X"]), F1["y"]
_, p0 = AE.test(model, thres, X_0); _, p1 = AE.test(model, thres, X_1)
res = {"AUC@2007": roc_auc_score(y_0, p0), "AUC@2011 before": roc_auc_score(y_1, p1)}
vali_num = 100000; rmse_c, rmse_t = p0[:vali_num], p1[:vali_num]; X_con, X_tre = X_0[:vali_num], X_1[:vali_num]
utils.set_random_seed()
cb = Calibrator(X_con, method="Isotonic"); cb.set_calibrator(rmse_c, is_P_mal=True)
sh = ShiftHunter(rmse_c, rmse_t, cb, ad_type="Tab"); print("shift p-value", sh.permu_test())
utils.set_random_seed()
sh.explainer(X_tre, y_1[:vali_num], label_num=20000); sh.adapter(model)
_, pa = AE.test(sh.model, thres, X_1); res["AUC@2011 after OWAD"] = roc_auc_score(y_1, pa)
utils.set_random_seed()
rm, rt = AE.train(np.concatenate((X_0[:train_num], X_1[:20000])), X_1.shape[-1])
_, pr = AE.test(rm, rt, X_1); res["AUC@2011 retraining"] = roc_auc_score(y_1, pr)
ref = {"AUC@2007": 0.9472, "AUC@2011 before": 0.8074, "AUC@2011 after OWAD": 0.8131, "AUC@2011 retraining": 0.7897}
for k in ref: print(f"{k:22s} ours {res[k]:.4f} | authors' notebook {ref[k]:.4f}")
