import os
for RADIUS in [0.01, 0.02, 0.03, 0.04, 0.05]:
    open(f'warp_mc_simpleGrid_R{RADIUS:.2f}.py.tmp', 'w').write(open('warp_mc_simpleGrid.py').read().replace('RADIUS = 0.07', f'RADIUS = {RADIUS}'))
    print(f'Running RADIUS={RADIUS}')
    os.system(f'python warp_mc_simpleGrid_R{RADIUS:.2f}.py.tmp')
