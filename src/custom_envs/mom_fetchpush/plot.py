import pandas as pd
import matplotlib.pyplot as plt


def smooth_iqr(multidata):
    multidata = pd.concat((pd.read_csv(f) for f in multidata), axis=1, ignore_index=True).rename_axis(['epoch']).reset_index()


    print(multidata)
    exit()

    iqr = multidata.quantile((0.25,0.75), axis=1).unstack().to_frame('iqr').rename_axis(['epoch', 'quartile']).reset_index()
    iqr = iqr.pivot(index=['epoch'], columns='quartile' ,values = 'iqr').reset_index()
    iqr.columns = ['epoch', 'q25', 'q75']
    multidata = multidata.median(axis=1).to_frame('success_rate_median').rename_axis('epoch').merge(iqr, on='epoch', how='inner')

    multidata['success_rate_median'] = multidata['success_rate_median'].rolling(window=10, min_periods=1).mean()
    multidata['q25'] = multidata['q25'].rolling(window=10, min_periods=1).mean()
    multidata['q75'] = multidata['q75'].rolling(window=10, min_periods=1).mean()


    return multidata


dense_baseline = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-12/rl_model_finished',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-15/rl_model_finished',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-18/rl_model_finished',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-20/rl_model_finished',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-37-23/rl_model_finished',
])
dense_shaped = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-14/rl_model_finished',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-17/rl_model_finished',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-20/rl_model_finished',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-22/rl_model_finished',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-38-25/rl_model_finished',
])
sparse_baseline = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-45/rl_model_finished',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-48/rl_model_finished',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-51/rl_model_finished',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-54/rl_model_finished',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/07-39-56/rl_model_finished',
])
sparse_shaped = smooth_iqr([
])


noObs_dense_baseline = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/17-56-48/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/19-10-58/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/20-10-50/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/20-10-27/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/20-11-36/success_rate.dat',
])
noObs_dense_shaped = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-12/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-17/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-20/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-24/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/22-49-27/success_rate.dat',
])
noObs_sparse_baseline = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-02/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-08/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-12/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-16/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-01-20/success_rate.dat',
])
noObs_sparse_shaped = smooth_iqr([
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-09/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-22/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-25/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-29/success_rate.dat',
    '/home/t14/Documents/tuhh/dsf/Scilab-RL/data/d04860e/shaped-fetchpush-v4/23-05-32/success_rate.dat',
])



all = {
    'dense env., orig.': noObs_dense_baseline,
    'dense env., shaped': noObs_dense_shaped,
    'sparse env., orig.': noObs_sparse_baseline,
    'sparse env., shaped': noObs_sparse_shaped,
}

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