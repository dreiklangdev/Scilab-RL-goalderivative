import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def smooth_iqr(multidata):
    if not multidata:
        return None

    multidata = pd.concat((pd.read_csv(f) for f in multidata), axis=1, ignore_index=True).rename_axis(['epoch']).reset_index()
    aucs = multidata.loc[:,1:].apply(lambda run: np.trapz(run, range(49)) / 48)

    iqr = multidata.quantile((0.25,0.75), axis=1).unstack().to_frame('iqr').rename_axis(['epoch', 'quartile']).reset_index()
    iqr = iqr.pivot(index=['epoch'], columns='quartile' ,values = 'iqr').reset_index()
    iqr.columns = ['epoch', 'q25', 'q75']
    multidata = multidata.median(axis=1).to_frame('metric').rename_axis('epoch').merge(iqr, on='epoch', how='inner')

    rolling_window = 20
    multidata['metric'] = multidata['metric'].rolling(window=rolling_window, min_periods=1).mean()
    multidata['q25'] = multidata['q25'].rolling(window=rolling_window, min_periods=1).mean()
    multidata['q75'] = multidata['q75'].rolling(window=rolling_window, min_periods=1).mean()

    multidata.attrs['aucs_mean'] = np.round(aucs.mean(), 2)
    multidata.attrs['aucs_std'] = np.round(aucs.std(), 2)
    return multidata


# FETCH PUSH =======================

# noObs_sparse_original = smooth_iqr([
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-02/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-08/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-12/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-16/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-20/success_rate.dat',
# ])
# noObs_dense_original = smooth_iqr([
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/17-56-48/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/19-10-58/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/20-10-50/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/20-10-27/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/20-11-36/success_rate.dat',
# ])
# noObs_dense_original_goalAug = smooth_iqr([
#     # WIP top left
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-11-12/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-11-15/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-11-17/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-11-19/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-11-21/success_rate.dat',
# ])


# c1_sparse_baseline = smooth_iqr([
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-45/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-48/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-51/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-54/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-56/success_rate.dat',
# ])
# c1_sparse_shaped = smooth_iqr([
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/shaped-fetchpush-v4/16-25-32/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/shaped-fetchpush-v4/16-25-34/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/shaped-fetchpush-v4/16-25-37/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/shaped-fetchpush-v4/16-25-40/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/shaped-fetchpush-v4/16-25-43/success_rate.dat',
# ])
# c1_dense_baseline = smooth_iqr([
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-12/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-15/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-18/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-20/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-23/success_rate.dat',
# ])
# c1_dense_baseline_goalAug = smooth_iqr([
#     # WIP bot left
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-13-08/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-13-06/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-13-03/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-12-55/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/01-12-52/success_rate.dat',
# ])
# c1_dense_shaped = smooth_iqr([
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-14/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-17/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-20/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-22/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-25/success_rate.dat',
# ])
# c1_dense_shaped_goalAug = smooth_iqr([
# ])

# c1a_noObs_sparse_shaped = smooth_iqr([
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-09/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-22/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-25/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-29/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-32/success_rate.dat',
# ])
# c1a_noObs_dense_shaped = smooth_iqr([
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-12/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-17/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-20/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-24/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-27/success_rate.dat',
# ])
# c1a_noObs_dense_shaped_goalAug = smooth_iqr([
# ])

# c1b_sparse_shaped_undiscounted = smooth_iqr([
# ])
# c1b_dense_shaped_undiscounted = smooth_iqr([
# ])

# c2_hardbool_dist_dgs = smooth_iqr([
#     # WIP top right
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/mom-fetchpush-v4/21-50-29/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/mom-fetchpush-v4/21-50-27/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/mom-fetchpush-v4/21-50-26/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/mom-fetchpush-v4/21-50-21/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/mom-fetchpush-v4/21-50-18/success_rate.dat',
# ])
# c2_softbool_dist_dgs = smooth_iqr([
# ])

# c3_hardbool_full = smooth_iqr([
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/mom-fetchpush-v4/17-00-32/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/mom-fetchpush-v4/17-00-31/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/mom-fetchpush-v4/17-00-29/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/mom-fetchpush-v4/17-00-25/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/ea53eec/mom-fetchpush-v4/17-00-22/success_rate.dat',
# ])

# c4_hardbool_obsRed = smooth_iqr([
# ])

# c1_sparse_shaped_origGoal = smooth_iqr([ # not improving, worse than without shaped
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/14-07-23/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/14-07-48/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/14-07-46/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/14-07-44/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/14-07-41/success_rate.dat'
# ])

# c1_dense_shaped_origGoal = smooth_iqr([ # corrupt, not improving anyways, worse even
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/16-54-49/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/16-54-54/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/abf72da/shaped-fetchpush-v4/16-54-56/success_rate.dat'
# ])

# c1_sparse_obsed_goalAug = smooth_iqr([
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aaf96d/shaped-fetchpush-v4/14-36-19/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aaf96d/shaped-fetchpush-v4/14-36-22/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aaf96d/shaped-fetchpush-v4/14-36-26/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aaf96d/shaped-fetchpush-v4/14-36-58/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aaf96d/shaped-fetchpush-v4/14-37-01/success_rate.dat',
# ])

# c1_sparse_allboolshaped_obsed_goalAug = smooth_iqr([
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aaf96d/shaped-fetchpush-v4/14-35-39/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aaf96d/shaped-fetchpush-v4/13-59-02/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aaf96d/shaped-fetchpush-v4/13-59-05/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aaf96d/shaped-fetchpush-v4/14-35-45/success_rate.dat',
#     '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/0aaf96d/shaped-fetchpush-v4/14-35-48/success_rate.dat',
# ])


# HAND REACH ======================= (sorted by performance)

GP_sparse_HER_unobs_baseline = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/09c63aa/shaped-handreach-v3/16-29-31/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/09c63aa/shaped-handreach-v3/16-33-58/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/09c63aa/shaped-handreach-v3/16-34-00/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/09c63aa/shaped-handreach-v3/16-34-02/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/09c63aa/shaped-handreach-v3/16-34-04/goalprogress.dat',
])
SR_sparse_HER_unobs_baseline = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/09c63aa/shaped-handreach-v3/16-29-31/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/09c63aa/shaped-handreach-v3/16-33-58/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/09c63aa/shaped-handreach-v3/16-34-00/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/09c63aa/shaped-handreach-v3/16-34-02/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/09c63aa/shaped-handreach-v3/16-34-04/success_rate.dat',
])

GP_sparse_HER_obs_baseline = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-42-45/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-42-54/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-43-14/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-43-37/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-44-02/goalprogress.dat',
])
SR_sparse_HER_obs_baseline = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-42-45/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-42-54/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-43-14/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-43-37/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-44-02/success_rate.dat',
])

GP_shaped_HER_obs = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-09-57/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-10-06/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-10-21/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-10-33/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-10-45/goalprogress.dat',
])
SR_shaped_HER_obs = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-09-57/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-10-06/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-10-21/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-10-33/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-10-45/success_rate.dat',
])

GP_shaped_HER_unobs = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-40-39/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-40-46/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-41-00/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-41-12/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-41-27/goalprogress.dat',
])
SR_shaped_HER_unobs = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-40-39/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-40-46/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-41-00/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-41-12/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/9cb58d8/shaped-handreach-v3/22-41-27/success_rate.dat',
])

GP_redesigned_HER_unobs = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-32-08/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-32-18/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-32-26/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-32-33/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-32-43/goalprogress.dat',
])
SR_redesigned_HER_unobs = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-32-08/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-32-18/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-32-26/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-32-33/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-32-43/success_rate.dat',
])

GP_redesigned_HER_augmented = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-34-21/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-34-29/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-34-46/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-34-55/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-35-06/goalprogress.dat',
])
SR_redesigned_HER_augmented = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-34-21/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-34-29/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-34-46/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-34-55/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/22-35-06/success_rate.dat',
])

GP_redesigned_HER_reduced = smooth_iqr([
])
SR_redesigned_HER_reduced = smooth_iqr([
])

GP_redesigned_reduced = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/11-10-55/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/11-37-26/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/11-37-33/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/11-37-44/goalprogress.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/12-26-35/goalprogress.dat',
])
SR_redesigned_reduced = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/11-10-55/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/11-37-26/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/11-37-33/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/11-37-44/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/dc09a61/shaped-handreach-v3/12-26-35/success_rate.dat',
])


# all = { # c1-obs
#     'dense*-shaped (C1)': c1_dense_shaped,
#     'dense*': c1_dense_baseline_goalAug,
#     'dense': c1_dense_baseline,
#     # 'dense* (noobs)': noObs_dense_original_goalAug,
#     # 'hard-bool': c2_hardbool_dist_dgs,
#     # 'hard-bool (full)': c3_hardbool_full,
# }

# all = { # c1-obs
#     'sparse-shaped* (C1)': c1_sparse_shaped,
#     'sparse-shaped* (C1, noobs.)': c1a_noObs_sparse_shaped,
# #     # 'sparse*':  TODO 0/1 rewards?
#     'sparse': c1_sparse_baseline,
#     'sparse-shaped, orig.goal': c1_sparse_shaped_origGoal,
# }

# all = { # c1a-no-obs
#     'dense*-shaped (C1)': c1a_noObs_dense_shaped,
#     'dense*': noObs_dense_original_goalAug,
#     'dense': noObs_dense_original,
    # 'sparse-shaped (C1)': c1a_noObs_sparse_shaped,
    # 'sparse': noObs_sparse_original,
# }

# all = { # c2, c3, c4
#     # 'reduced obs. (C4)': c4_hardbool_obsRed,
#     # 'augmented obs. (C3)': c3_hardbool_full,
#     'c1_sparse_allbool_obsed_goalAug': c1_sparse_allboolshaped_obsed_goalAug,
#     'c1_sparse_obsed_goalAug': c1_sparse_obsed_goalAug,
#     # 'dense*': c1_dense_baseline,
# }



all = { # C1
    # 'sparse (obs.)': GP_sparse_HER_obs_baseline,
    'sparse': GP_sparse_HER_unobs_baseline,
    'goalderiv. shape (obs.)': GP_shaped_HER_obs,
    'goalderiv. shape': GP_shaped_HER_unobs,
}

all = { # C2
    'sparse': GP_sparse_HER_unobs_baseline,
    'goalderiv. design': GP_redesigned_HER_unobs,
}

all = { # C3
    'sparse': GP_sparse_HER_unobs_baseline,
    'goalderiv. design (augm.)': GP_redesigned_HER_augmented,
}

all = { # C4
    'sparse': GP_sparse_HER_unobs_baseline,
    # 'goalderiv. design (red.)': GP_redesigned_HER_reduced,
    'goalderiv. design (red., no HER)': GP_redesigned_reduced,
}

# all = { # C2, C3, C4
#     'sparse': GP_sparse_HER_unobs_baseline,
#     'goalderiv. shape': GP_shaped_HER_unobs,
#     'goalderiv. design': GP_redesigned_HER_unobs,
#     'goalderiv. design (augm.)': GP_redesigned_HER_augmented,
#     'goalderiv. design (red., no HER)': GP_redesigned_reduced,
# }

# all = { # C2, C3, C4 (successrate)
#     'sparse': SR_sparse_HER_unobs_baseline,
#     'goalderiv. shape': SR_shaped_HER_unobs,
#     'goalderiv. design': SR_redesigned_HER_unobs,
#     'goalderiv. design (augm.)': SR_redesigned_HER_augmented,
#     'goalderiv. design (red., no HER)': SR_redesigned_reduced,
# }


metric = [ex['metric'] for ex in all.values()]
metric = pd.concat(metric, axis=1, ignore_index=True)
metric.columns = all.keys()

colormap = {
    'sparse (obs.)': '#17becf',
    'sparse': '#1f77b4',
    'goalderiv. shape (obs.)': '#bcbd22',
    'goalderiv. shape': '#ff7f0e',
    'goalderiv. design': '#2ca02c',
    'goalderiv. design (augm.)': '#d62728',
    'goalderiv. design (red., no HER)': '#9467bd',
    # '#8c564b',
    # '#e377c2',
    # '#7f7f7f',
    # '#bcbd22',
}

colors = [colormap[key] for key in all.keys()]

ax = metric.plot(color=colors)
ax.set_title('Observation Reduction (HandReach-v3, SAC+HER)')
ax.set_ylabel('Median Test Goalprogress')
ax.set_xlabel('Epoch (à 200 Episodes)')
ax.set_prop_cycle(plt.cycler(color=colors))
ax.set_axisbelow(True)



# iqr
for ex in all.values():
    ax.fill_between(x=ex['epoch'], y1=ex['q25'], y2=ex['q75'], alpha=0.1)
    ax.text(x=49, y=ex['metric'].tail(1), s=f"{ex.attrs['aucs_mean']} (±{ex.attrs['aucs_std']})")

# plt.gca().set_color_cycle(['red', 'green', 'blue', 'yellow'])
plt.tight_layout()
plt.box(False)
plt.grid()
plt.show()