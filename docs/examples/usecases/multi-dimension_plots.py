import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_contour, plot_surface
import matplotlib.pyplot as plt
from pyswarms.utils.plotters.formatters import Mesher, Designer
import numpy as np

options = {'c1':0.5, 'c2':0.3, 'w':0.9}
iter = 50

def f_3(x):
        n_particles = x.shape[0]
        
        j = []
        for i in range(n_particles):
            #z = x[i][0] * x[i][0] + x[i][1] * x[i][1] + x[i][2]
            z =  (2 * x[i][0])**2 + (0.5 * x[i][1])**2 +  (x[i][2] - 0.5)**2
            j.append(z)

        return np.array(j)

def test_contour_2():
    func = fx.sphere
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=2, options=options)
    optimizer.optimize(func, iters=iter)
    
    m = Mesher(func=func, limits=[(-1,1), (-1,1)], levels=10, delta=0.01)
    d = Designer(limits=[(-1,1), (-1,1)], label=['1st dimension (x)', '2nd dimension (y)'])

    anim = plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d, mark=(0,0))
    plt.show()

def test_surface_2():
    func = fx.sphere
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=2, options=options)
    optimizer.optimize(func, iters=iter)

    m = Mesher(func=func, limits=[(-1,1), (-1,1)], levels=10, delta=0.1)
    d = Designer(limits=[(-1,1), (-1,1), (-0.1,1)], label=['1st dimension (x)', '2nd dimension (y)', 'fitness'])

    pos_history_3d = m.compute_history_3d(optimizer.pos_history)
    anim = plot_surface(pos_history=pos_history_3d, mesher=m, designer=d, mark=(0,0,0))
    plt.show()

def test_contour_3():
    func = f_3
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=3, options=options)
    cost , pos = optimizer.optimize(func, iters=iter)

    m = Mesher(func=func, limits=[(-1,1), (-1,1), (-1,1)], levels=10, delta=0.1)
    d = Designer(limits=[(-1,1), (-1,1), (-1,1)], label=['1st dimension (x)', '2nd dimension (y)', '3rd dimension (z)'])

    fig, axs = plt.subplots(1, 3, figsize=(17,5))

    anim_1 = plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d, mark=(0,0), x=0, y=1, best_pos=pos, canvas=(fig, axs[0]))
    anim_2 = plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d, mark=(0,0.5), x=0, y=2, best_pos=pos, canvas=(fig, axs[1]))
    anim_3 = plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d, mark=(0,0.5), x=1, y=2, best_pos=pos, canvas=(fig, axs[2]))
    plt.show()

def test_surface_3():
    func = f_3
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=3, options=options)
    cost , pos = optimizer.optimize(func, iters=iter)
    
    m = Mesher(func=func, limits=[(-1,1), (-1,1), (-1,1)], levels=10, delta=0.1)
    d = Designer(limits=[(-1,1), (-1,1), (-1,1), (-0.1,1)], label=['1st dimension (x)', '2nd dimension (y)', '3rd dimension (z)', 'fitness'])

    pos_history_3d = m.compute_history_3d(optimizer.pos_history)

    fig = plt.figure(figsize=(17,5))
    ax_1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax_2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax_3 = fig.add_subplot(1, 3, 3, projection='3d')

    anim_1 = plot_surface(pos_history=pos_history_3d, mesher=m, designer=d, mark=(0,0, 0), x=0, y=1, best_pos=pos, canvas=(fig, ax_1))
    anim_2 = plot_surface(pos_history=pos_history_3d, mesher=m, designer=d, mark=(0,0.5, 0), x=0, y=2, best_pos=pos, canvas=(fig, ax_2))
    anim_3 = plot_surface(pos_history=pos_history_3d, mesher=m, designer=d, mark=(0,0.5, 0), x=1, y=2, best_pos=pos, canvas=(fig, ax_3))
    plt.show()

def basic_test():

    # Set-up optimizer
    options = {'c1':0.5, 'c2':0.3, 'w':0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options)
    optimizer.optimize(fx.sphere, iters=100)

    # Plot the sphere function's mesh for better plots
    m = Mesher(func=fx.sphere,
            limits=[(-1,1), (-1,1)])
    # Adjust figure limits
    d = Designer(limits=[(-1,1), (-1,1)],
                label=['x-axis', 'y-axis'])

    anim = plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d, mark=(0,0))
    plt.show()

if __name__ == "__main__":
    # execute only if run as a script

    basic_test()

    test_contour_2()
    test_surface_2()
    
    test_contour_3()
    test_surface_3()
