import earthaccess
import os
import h5py
earthaccess.login()   # your Earthdata username/password

days = ["20180719", "20180720", "20180806", "20180815", "20180816",
        "20180824", "20180828", "20180829", "20180906"]

res = earthaccess.search_data(
    concept_id="C3880706941-LARC_CLOUD",
    temporal=("2018-07-19", "2018-09-07"),
    count=500,
)

def wanted(g):
    return any("GCAS-NO2" in l and "250MX250M" in l and "_PM_" in l
               and any(d in l for d in days)
               for l in g.data_links())

sel = [g for g in res if wanted(g)]
print(len(sel))
for g in sel:
    print(g.data_links())

out = "scratch4/willwu/datas/NY/airbone/"
os.makedirs(out, exist_ok = True)
# earthaccess.download(sel, out)

f = "/scratch4/willwu/datas/NY/airbone/LISTOS-GCAS-NO2_NASA-AIRCRAFT_20180719_R0_PM_250MX250M.h5"
with h5py.File(f, "r") as h:
    h.visititems(lambda name, obj: print(name, getattr(obj, "shape", ""), getattr(obj, "dtype", "")))
    # also print the attributes (units, fill values) of the columns once you see their names