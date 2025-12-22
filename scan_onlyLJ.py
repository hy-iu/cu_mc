import os
# for LJ_EPSILON in [0.01, 0.05, 0.1]:
# for LJ_EPSILON in [0.001, 0.003, 0.005, 0.007, 0.011, 0.013, 0.015, 0.017, 0.02, 0.022, 0.025, 0.027]:
for LJ_EPSILON in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    t = open('warp_mc_simpleGrid_onlyLJ.py').read()
    to_replace = 'LJ_EPSILON = 0.1'
    assert to_replace in t
    open(f'warp_mc_simpleGrid_onlyLJ{LJ_EPSILON:.2f}.tmp.py', 'w').write(t.replace(to_replace, f'LJ_EPSILON = {LJ_EPSILON}'))
    print(f'Running LJ_EPSILON={LJ_EPSILON}')
    os.system(f'python warp_mc_simpleGrid_onlyLJ{LJ_EPSILON:.2f}.tmp.py')
