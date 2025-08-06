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
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-02/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-08/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-12/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-16/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-20/success_rate.dat',
])
noObs_dense_original = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/17-56-48/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/19-10-58/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/20-10-50/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/20-10-27/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/20-11-36/success_rate.dat',
])
noObs_dense_original_goalAug = smooth_iqr([
    # WIP top left
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-11-12/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-11-15/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-11-17/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-11-19/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-11-21/success_rate.dat',
])


c1_sparse_baseline = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-45/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-48/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-51/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-54/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-56/success_rate.dat',
])
c1_sparse_shaped = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/shaped-fetchpush-v4/16-25-32/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/shaped-fetchpush-v4/16-25-34/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/shaped-fetchpush-v4/16-25-37/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/shaped-fetchpush-v4/16-25-40/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/shaped-fetchpush-v4/16-25-43/success_rate.dat',
])
c1_dense_baseline = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-12/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-15/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-18/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-20/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-23/success_rate.dat',
])
c1_dense_baseline_goalAug = smooth_iqr([
    # WIP bot left
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-13-08/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-13-06/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-13-03/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-12-55/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-12-52/success_rate.dat',
])
c1_dense_shaped = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-14/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-17/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-20/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-22/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-25/success_rate.dat',
])
c1_dense_shaped_goalAug = smooth_iqr([
    # TODO
])

c1a_noObs_sparse_shaped = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-09/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-22/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-25/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-29/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-32/success_rate.dat',
])
c1a_noObs_dense_shaped = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-12/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-17/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-20/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-24/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-27/success_rate.dat',
])
c1a_noObs_dense_shaped_goalAug = smooth_iqr([
    # TODO
])

c1b_sparse_shaped_undiscounted = smooth_iqr([
])
c1b_dense_shaped_undiscounted = smooth_iqr([
])

c2_hardbool_dist_dgs = smooth_iqr([
    # WIP top right
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/mom-fetchpush-v4/21-50-29/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/mom-fetchpush-v4/21-50-27/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/mom-fetchpush-v4/21-50-26/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/mom-fetchpush-v4/21-50-21/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/mom-fetchpush-v4/21-50-18/success_rate.dat',
])
c2_softbool_dist_dgs = smooth_iqr([
])

c3_hardbool_full = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/mom-fetchpush-v4/17-00-32/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/mom-fetchpush-v4/17-00-31/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/mom-fetchpush-v4/17-00-29/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/mom-fetchpush-v4/17-00-25/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/mom-fetchpush-v4/17-00-22/success_rate.dat',
])

c4_hardbool_obsRed = smooth_iqr([
])

c1_sparse_shaped_origGoal = smooth_iqr([ # not improving, worse than without shaped
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/14-07-23/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/14-07-48/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/14-07-46/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/14-07-44/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/14-07-41/success_rate.dat'
])

c1_dense_shaped_origGoal = smooth_iqr([ # corrupt, not improving anyways, worse even
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/16-54-49/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/16-54-54/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/16-54-56/success_rate.dat'
])

# all = { # c1-obs
#     'dense*-shaped (C1)': c1_dense_shaped,
#     'dense*': c1_dense_baseline_goalAug,
#     'dense': c1_dense_baseline,
#     # 'dense* (noobs)': noObs_dense_original_goalAug,
#     # 'hard-bool': c2_hardbool_dist_dgs,
#     # 'hard-bool (full)': c3_hardbool_full,
# }

all = { # c1-obs
    'sparse-shaped* (C1)': c1_sparse_shaped,
    'sparse-shaped* (C1, noobs.)': c1a_noObs_sparse_shaped,
#     # 'sparse*':  TODO 0/1 rewards?
    'sparse': c1_sparse_baseline,
    'sparse-shaped, orig.goal': c1_sparse_shaped_origGoal,
}

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