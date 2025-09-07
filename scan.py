import os

# for RADIUS in [0.01, 0.02, 0.03, 0.04, 0.05]:
#     open(f'warp_mc_simpleGrid_R{RADIUS:.2f}.py.tmp', 'w').write(open('warp_mc_simpleGrid.py').read().replace('RADIUS = 0.07', f'RADIUS = {RADIUS}'))
#     print(f'Running RADIUS={RADIUS}')
#     os.system(f'python warp_mc_simpleGrid_R{RADIUS:.2f}.py.tmp')
for LJ_EPSILON in [0.0, 0.12]:
# for LJ_EPSILON in [0.001, 0.003, 0.005, 0.007, 0.011, 0.013, 0.015, 0.017, 0.02, 0.022, 0.025, 0.027]:
    t = open('warp_mc_simpleGrid_lj.py').read()
    assert 'LJ_EPSILON = 1.0' in t
    open(f'warp_mc_simpleGrid_LJ{LJ_EPSILON:.2f}.py.tmp', 'w').write(t.replace('LJ_EPSILON = 1.0', f'LJ_EPSILON = {LJ_EPSILON}'))
    print(f'Running LJ_EPSILON={LJ_EPSILON}')
    os.system(f'python warp_mc_simpleGrid_LJ{LJ_EPSILON:.2f}.py.tmp')
