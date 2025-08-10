import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def smooth_iqr(multidata):
    if not multidata:
        return None
    
    multidata = pd.concat((pd.read_csv(f) for f in multidata), axis=1, ignore_index=True).rename_axis(['epoch']).reset_index()
    aucs = multidata.loc[:,1:].apply(lambda run: np.trapz(run, range(49)) / 48)
    print(aucs)

    iqr = multidata.quantile((0.25,0.75), axis=1).unstack().to_frame('iqr').rename_axis(['epoch', 'quartile']).reset_index()
    iqr = iqr.pivot(index=['epoch'], columns='quartile' ,values = 'iqr').reset_index()
    iqr.columns = ['epoch', 'q25', 'q75']
    multidata = multidata.median(axis=1).to_frame('success_rate_median').rename_axis('epoch').merge(iqr, on='epoch', how='inner')

    multidata['success_rate_median'] = multidata['success_rate_median'].rolling(window=10, min_periods=1).mean()
    multidata['q25'] = multidata['q25'].rolling(window=10, min_periods=1).mean()
    multidata['q75'] = multidata['q75'].rolling(window=10, min_periods=1).mean()

    multidata.attrs['aucs_mean'] = aucs.mean()
    multidata.attrs['aucs_std'] = aucs.std()
    return multidata


noObs_sparse_original = smooth_iqr([
])
noObs_dense_original = smooth_iqr([
])

c1_sparse_baseline = smooth_iqr([
])
c1_sparse_shaped = smooth_iqr([
])
c1_dense_baseline = smooth_iqr([
])
c1_dense_shaped = smooth_iqr([
])

c1a_noObs_sparse_shaped = smooth_iqr([
])
c1a_noObs_dense_shaped = smooth_iqr([
])

c1b_sparse_shaped_undiscounted = smooth_iqr([
])
c1b_dense_shaped_undiscounted = smooth_iqr([
])

c2_hardbool_dist_dgs = smooth_iqr([
])
c2_softbool_dist_dgs = smooth_iqr([
])

c3_hardbool_full = smooth_iqr([
])

c4_hardbool_obsRed = smooth_iqr([
])

all = { # c1-obs
    # 'dense*-shaped (C1)': c1_dense_shaped,
    # 'dense*': c1_dense_baseline_goalAug,
    # 'dense': c1_dense_baseline,
    # 'dense* (noobs)': noObs_dense_original_goalAug,
    'hard-bool': c2_hardbool_dist_dgs,
    'hard-bool (full)': c3_hardbool_full,
}

# all = { # c1-obs
#     'sparse-shaped (C1)': c1_sparse_shaped,
#     'sparse-shaped (C1, noobs.)': c1a_noObs_sparse_shaped,
#     # 'sparse*':  TODO 0/1 rewards?
#     'sparse': c1_sparse_baseline,
# }

# all = { # c1a-no-obs
#     'dense*-shaped (C1)': c1a_noObs_dense_shaped,
#     'dense*': noObs_dense_original_goalAug,
#     'dense': noObs_dense_original,
    # 'sparse-shaped (C1)': c1a_noObs_sparse_shaped,
    # 'sparse': noObs_sparse_original,
# }

# all = { # c2, c3, c4
#     'reduced obs. (C4)': c4_hardbool_obsRed,
#     'augmented obs. (C3)': c3_hardbool_obsAug,
#     'goaldynamic (C2)': c2_hardbool_dist_dgs,
#     'dense*': c1_dense_baseline,
# }

success_rates = [experiment['success_rate_median'] for experiment in all.values()]
success_rates = pd.concat(success_rates, axis=1, ignore_index=True)
success_rates.columns = all.keys()
ax = success_rates.plot()
ax.set_title('FetchPush-v4 (SAC)')
ax.set_axisbelow(True)


# iqr
for experiment in all.values():
    ax.fill_between(x=experiment['epoch'], y1=experiment['q25'], y2=experiment['q75'], alpha=0.1)

plt.grid()
plt.show()