"""
~~~~ Created by Ben Kroul, 2024 ~~~
Defines useful utility functions and constants for quantum physics related things. Of note:
- plot_op and plot_theory_exp for plotting 2D operators and comparing theoretical vs experimental operators
- pauli matrices defined like pauli_e, sigma_g, etc.
- number, phase, charge + x basis operators
- gates, rotation operators, common pulse sequences
- expectations of operators over states or density matrices
- full bloch sphere simulation/animation of quantum state under unitary evolution...
    see example at the end of the file
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# color mapping and bars, for PyLance type checking
import matplotlib.colors as mcolors
import matplotlib.cm as mplcm
import matplotlib.colorbar as mplcb
# bloch sphere / 3d plotting
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
# Range-Kutta approximation for ODEs
from scipy.integrate import solve_ivp
from tqdm import tqdm
import sys
from util import uFormat, timeIt, SAVEDIR, SAVEEXT

# --- PLOTTING --- #

# --- ANIMATE AND PLOT A BLOCH SPHERE --- #

# plot all edges of wireframe cube
def plot_wf_cube(ax, corner1, corner2, color='orange', linestyle='dashed', linewidth=1):
    x1,y1,z1 = corner1; x2,y2,z2 = corner2
    ax.plot([x1,x2,x2,x2,x2,x1,x1,x1,x1,x1,x1,x2,x2,x2,x2,x1],
            [y1,y1,y1,y2,y2,y2,y2,y1,y1,y2,y2,y2,y2,y1,y1,y1],
            [z1,z1,z2,z2,z1,z1,z2,z2,z1,z1,z2,z2,z1,z1,z2,z2],
            color=color,linestyle=linestyle,linewidth=1)

def plot_bloch_sphere(ax, frame_number=False, init_angle=False, angle_step=0):
    '''Plot bloch sphere and axes. adds frame_number in top left corner if specified'''
    default_angle = 24
    if isinstance(init_angle, bool): init_angle = default_angle
    # format axis
    if frame_number and angle_step:
        init_angle += frame_number*angle_step
    ax.view_init(elev=20., azim=init_angle) #  camera position
    ax.set_axis_off()  # turn off axis lines
    ax.set(xticks=[],yticks=[],zticks=[],xlabel='x',ylabel='y',zlabel='z',
           zlim=(-1, 1),proj_type='ortho',box_aspect=[1,1,1])
    ax.grid(False)
    # plot wireframe sphere
    sphere_res = 40j
    u, v = np.mgrid[0:2*np.pi:sphere_res, 0:np.pi:sphere_res]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color="blue", alpha=0.1, edgecolor="k", linewidth=0.1)
    # plot axis lines
    ax.plot([1,-1],[0,0],[0,0],color="black",linestyle='dashed',linewidth=0.5)
    ax.plot([0,0],[1,-1],[0,0],color="black",linestyle='dashed',linewidth=0.5)
    ax.plot([0,0],[0,0],[1,-1],color="black",linestyle='dashed',linewidth=0.5)
    # plot axis circles
    th = np.arange(0,2*np.pi+np.pi/20,np.pi/20)
    x, y, z = np.cos(th), np.sin(th), np.zeros_like(th)
    ax.plot(x, y, z, color="black",linestyle='dashed',linewidth=0.5)
    ax.plot(z, x, y, color="black",linestyle='dashed',linewidth=0.5)
    ax.plot(y, z, x, color="black",linestyle='dashed',linewidth=0.5)
    # add axis labels
    ax.text(0,0,-1,r"$|1\rangle$")
    ax.text(0,0,1,r"$|0\rangle$")
    ax.text(1,0,0,r"$|+\rangle$")
    ax.text(-1,0,0,r"$|-\rangle$")
    ax.text(0,1,0,r"$|y\rangle$")
    ax.text(0,-1,0,r"$|-y\rangle$")
    # label frame number in the top left of the plot
    if isinstance(frame_number,int):
        angle = np.pi*init_angle/180 - 3*np.pi/15 - np.pi/2
        ax.text(np.cos(angle),np.sin(angle),0.8,f"frame: {frame_number}")

def plot_bloch_vector(ax, vec, color="red", add_purity=True, scale_vector=False):
    ''' plot the 3-vector vec on 3d axis ax with color color
     if add_purity, shows purity of state at top-left corner of plot
     if scale_vector, will scale vector magnitude with purity of state '''
    x, y, z = vec
    purity = np.sqrt(x**2+y**2+z**2)
    if purity:  # normalize vector
        xn = x/purity; yn = y/purity; zn = z/purity
    else:
        xn = 0; yn = 0; zn = 0
    
    if scale_vector:
        ax.plot([x,xn],[y,yn],[z,zn],color="black", linewidth=0.5)
    # normalized vector
    ax.scatter(xn,yn,zn,color=color,marker='o')#,markersize=1)
    if scale_vector:   # plot "actual" vector
        ax.quiver(0,0,0,x,y,z,color=color, linewidth=2, arrow_length_ratio=0.1)
    else:    # plot the unit-normalized vector, which is easier to see
        ax.quiver(0,0,0,xn,yn,zn,color=color, linewidth=2, arrow_length_ratio=0.1)
    # plot cube as lighter color by dividing alpha by 2
    lighter_color = mcolors.to_rgba(color)[:3] + (mcolors.to_rgba(color)[3]/2,)
    plot_wf_cube(ax,(0,0,0),(xn,yn,zn),color=lighter_color)
    if add_purity: ax.text(0,0.8,1,f"purity: {uFormat(purity,0)}")

def animate_bloch(states, name: str, pbar=False, rot_vecs=[],
                  fps=60, dpi=200, add_purity=True, angle_step=0):
    '''Animates bloch sphere and saves to {SAVEDIR}{name}.mp4
     states[time] = (x,y,z).  state vector at each time step
     rot_vecs[time] = (x,y,z) Hamiltonian vector at each time step
     fps: frames per second to save animation at
     dpi: dots per inch for resolution of animation
     add_purity: if True, labels the purity of each frame '''
    nframes = states.shape[0]
    fig = plt.figure(figsize=(12,12))
    fig.tight_layout()
    ax = fig.add_subplot(projection='3d')
    def vector_from_data(frame: int) -> None:
        ax.clear()  # remove previous arrows
        plot_bloch_sphere(ax, frame, angle_step=angle_step)
        plot_bloch_vector(ax, states[frame], add_purity=add_purity)
        if len(rot_vecs):  # add arrow symbolizing rotation axis
            if len(rot_vecs.shape) > 1:
                plot_bloch_vector(ax, rot_vecs[frame], color="green", add_purity=False)
            else:  # single rotation specified
                plot_bloch_vector(ax, rot_vecs, color="green", add_purity=False)
        if not isinstance(pbar, bool): pbar.update(1)
    ani = animation.FuncAnimation(fig=fig, func=vector_from_data, frames=nframes)
    plot_name = SAVEDIR+name+".mp4"
    ani.save(plot_name, writer="ffmpeg", fps=fps, dpi=dpi)
    print('saved animation to',plot_name)
    return ani

def label_str_states(N):
    """ return ordered string of |00>, |01>, |10>, |11> for N qubits """
    ret = []
    for i in range(2**N):
        string = "$|"
        for n in range(N-1,-1,-1):
            if i >= 2**n:
                string += "1"
                i -= 2**n
            else:
                string += "0"
        string += r"\rangle$"
        ret.append(string)
    return ret

def plot_op(operators: list|tuple|np.ndarray, titles=[], saveplot=False, cmap_name="turbo", box_spec=False):
    """ Plot 2-dimensional operators
    Inputs: 
    - operators: takes in ndarray or list/tuple of ndarrays 
    - titles: title or list of titles for multiple ops 
    - saveplot: True or string to name file of plot
    - cmap_name: name of mpl.colormap to use
    - box_spec: if True, will show numbers of matrix elements
    """
    if not isinstance(operators, list) and not isinstance(operators, tuple):
        operators = [operators]
    if not isinstance(titles, list) and not isinstance(operators, tuple):
        titles = [titles]
    nops = len(operators)
    fig = plt.figure(figsize=(8,4*nops))
    gs = fig.add_gridspec(nops,3,width_ratios=[20,20,1])
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    # get abs max of all operators
    absmax = 0
    for op in operators:
        themin = min(op.real.min(), op.imag.min())
        themax = max(op.real.max(), op.imag.max())
        amax = max(abs(themin),abs(themax))
        absmax = max(absmax, amax)
    # make colorbar
    cax = fig.add_subplot(gs[:,2])
    norm = mcolors.Normalize(vmin=-absmax, vmax=absmax)
    cm_map = mplcm.ScalarMappable(norm=norm, cmap=cmap_name)
    cb = mplcb.ColorbarBase(cax, cmap=cmap_name, norm=norm, orientation='vertical')
    # plot operators
    for i in range(nops):
        operator = operators[i]
        ax1 = fig.add_subplot(gs[i,0])
        ax2 = fig.add_subplot(gs[i,1])
        ax1.matshow(operator.real, cmap=cmap_name, vmin = -absmax, vmax=absmax)
        ax2.matshow(operator.imag, cmap=cmap_name, vmin = -absmax, vmax=absmax)
        title = titles[i] if len(titles) > i else "_DEF_"
        ax1.set(xticks=[],yticks=[])
        ax1.set_title("Re{ "+title+" }")
        ax2.set(xticks=[],yticks=[])
        ax2.set_title("Im{ "+title+" }")
        if box_spec:
            for (i, j), val in np.ndenumerate(operator.real):
                ax1.text(j, i, uFormat(val,0), ha='center', va='center', color=('black' if abs(val) < absmax/2 else 'white'))
            for (i, j), val in np.ndenumerate(operator.imag):
                ax2.text(j, i, uFormat(val,0), ha='center', va='center', color=('black' if abs(val) < absmax/2 else 'white'))
    if saveplot:
        if isinstance(saveplot, str):
            plt_name = SAVEDIR + "/"+saveplot.replace(" ","_").replace("$","")+"_2d"+SAVEEXT
        else:
            plt_name = SAVEDIR + "/"+title.replace(" ","_").replace("$","")+"_2d"+SAVEEXT
        plt.savefig(plt_name,bbox_inches="tight")
        print(f'saved figure {plt_name}')
    plt.show()

def plot_theory_exp(op_th, op_exp, title, saveplot=False, cmap_name="turbo", labels=None):
    """ 3-d comparison of theoretical vs. experimental operators """
    zmax = max(op_th.real.max(), op_th.imag.max())
    zmin = min(op_th.real.min(), op_th.imag.min())
    cmax = max(abs(max(op_exp.real.max(), op_exp.imag.max())),abs(min(op_exp.real.min(), op_exp.imag.min())))
    cmax = max(cmax, max(abs(zmax),abs(zmin)))
    cmap = mpl.colormaps[cmap_name]
    _labels = labels if labels else label_str_states(int(np.log2(op_th.shape[0])))
    xrange = range(op_th.shape[0])
    X, Y = np.meshgrid(xrange,xrange)
    X = X.flatten(); Y = Y.flatten()
    Z = np.zeros_like(X)
    dx = 0.5 * np.ones_like(X)
    op_th = op_th.flatten()
    op_exp = op_exp.flatten()
    dy = dx.copy()
    fig, axs = plt.subplots(1,2,figsize=(12,6),subplot_kw={'projection':'3d','proj_type':'ortho'})
    ax1, ax2 = axs
    ax1.set_title("Re{ "+title+" }")
    nreal = op_exp.real/(2*cmax)+0.5
    colors = [cmap(x) for x in nreal]
    ax1.bar3d(X, Y, Z, dx, dy, op_exp.real, alpha=0.9, color=colors)
    ax1.bar3d(X, Y, Z, dx, dy, op_th.real, edgecolor='black', alpha=0)
    ax2.set_title("Im{ "+title+" }")
    nimag = op_exp.imag/(2*cmax)+0.5
    colors = [cmap(x) for x in nimag]
    ax2.bar3d(X, Y, Z, dx, dy, op_exp.imag, alpha=0.9, color=colors)
    ax2.bar3d(X, Y, Z, dx, dy, op_th.imag, edgecolor='black', alpha=0)
    fig.tight_layout()
    for ax in axs:
        ax.view_init(elev=50., azim=30) #  camera position
        #ax.set_axis_off()  # turn off axis lines
        ax.set_xticks(xrange, labels=_labels)
        ax.set_yticks(xrange, labels=_labels)
        if labels:
            ax.set(zlabel=r'$\hat{\rho}$', zlim=(zmin,zmax),proj_type='ortho')
        else:
            ax.set(xlabel='Input',ylabel='Output',zlabel=r'$\hat{\rho}$', zlim=(zmin,zmax),proj_type='ortho')
        #ax.grid(False)
    if saveplot:
        if isinstance(saveplot, str):
            plt_name = SAVEDIR + "/" + saveplot.replace(" ","_").replace("$","")+"_3d"+SAVEEXT
        else:
            plt_name = SAVEDIR + "/" + title.replace(" ","_").replace("$","")+"_3d"+SAVEEXT
        plt.savefig(plt_name,bbox_inches="tight")
        print(f'saved figure {plt_name}')
    plt.show()

# --- OPERATOR FUNCTIONS --- #

sigma_x = np.array([[0,1],[1,0]])    # |0><1| + |1><0|
sigma_y = np.array([[0,-1j],[1j,0]]) # i|0><1| - i|1><0|
sigma_z = np.array([[1,0],[0,-1]])   # |0><0| - |1><1|
sigma_i = np.array([[1,0],[0,1]])    # |0><0| + |1><1|
sigma_g = np.array([[1,0],[0,0]])    # |0><0|
sigma_e = np.array([[0,0],[0,1]])    # |1><1|
pauli_x = sigma_x
pauli_y = sigma_y
pauli_z = sigma_z
pauli_i = sigma_i
pauli_g = sigma_g
pauli_e = sigma_e

# ---- NUMBER (FOCK) BASIS OPERATORS ---- #
# 0-1 number states
def number_op(N):  # n|n><n|
    return np.diag(range(N))

def create_op(N):   #sqrt(n)|n+1><n|
    return np.diag(np.sqrt(np.arange(1,N)),k=-1)

def destroy_op(N):  #sqrt(n)|n-1><n|
    return np.diag(np.sqrt(np.arange(1,N)),k=1)

def phase_op(N):  # a + a†
    return create_op(N) + destroy_op(N)

def charge_op(N): # i(a - a†)
    return 1j*(destroy_op(N) - create_op(N))

def cosphi_op(N): # 2 cos(i(a+a†)) = |n+1><n| + |n-1><n|
    return (np.diag(np.ones(N-1),k=-1) + np.diag(np.ones(N-1),k=1))/2

def sinphi_op(N): # 2i sin(i(a+a†)) = |n+1><n| - |n+1><n|
    return (np.diag(np.ones(N-1),k=-1) - np.diag(np.ones(N-1),k=1))/(2j)

# ---- X BASIS OPERATORS ---- #
def xop(xpts):
    return np.diag(xpts)

def dop_forward(xpts):
    # We will assume a uniform spacing for now
    dx=xpts[1]-xpts[0]
    return (np.diag(np.ones(len(xpts)-1),k=1) - np.diag(np.ones(len(xpts)),k=0))/dx

def dop_central(xpts):
    # We will assume a uniform spacing for now
    dx=xpts[1]-xpts[0]
    return (np.diag(np.ones(len(xpts)-1),k=1) - np.diag(np.ones(len(xpts)-1),k=-1))/(2*dx)

def d2op(xpts):
    # We will assume a uniform spacing for now
    dx=xpts[1]-xpts[0]
    return (np.diag(np.ones(len(xpts)-1),k=1) - 2*np.diag(np.ones(len(xpts)),k=0) + np.diag(np.ones(len(xpts)-1),k=-1))/(dx**2)

# ------- GATE FUNCTIONS ------- #

# idx of qubit is 0 to N-1
def gate_on_nth(N, idx, gate):
    arr = np.array([1],dtype=complex)
    for i in range(N):
        arr = np.kron(arr, gate) if idx == i else np.kron(arr, np.identity(2))
    return arr

def gate_on_all(N, gate):
    arr = np.array([1],dtype=complex)
    for i in range(N):
        arr = np.kron(arr, gate)
    return arr

def mixed_on_all(N, gate):
    arr = np.zeros((2**N,2**N),dtype=complex)
    for i in range(N-1):
        arr += gate_on_nth(N, i, gate) @ gate_on_nth(N, i+1, gate)
    return arr

def rotation_op(theta, vec, N=2):
    vec = vec/np.linalg.norm(vec)  # normalize vector (it should be tho)
    x, y, z = vec
    vec_matrix = x*sigma_x + y*sigma_y + z*sigma_z
    single_rot = np.identity(2)*np.cos(theta/2)-1j*np.sin(theta/2)*vec_matrix
    return gate_on_all(N, single_rot)

# unitary operator for time-independent, diagonal hamiltonian
def unitary_diag(t, H):
    Ut = np.diag(np.exp(-1j*np.diag(H)*t))
    return Ut

# --- SEQUENCES --- #

def ramsey_sequence(T, H): #pi/2 +y -> T -> pi/2 -y
    N = int(np.log2(H.shape[0]))
    op1 = rotation_op(np.pi/2,(0,1,0),N)   # pi/2 +y
    op2 = unitary_diag(T,H)                # wait T
    op3 = rotation_op(np.pi/2,(0,-1,0),N)  # pi/2 -y
    return op3 @ op2 @ op1

def spin_echo_sequence(T, H): #pi/2 +y -> T/2 -> pi +x -> T/2 -> pi/2 -y
    N = int(np.log2(H.shape[0]))
    op1 = rotation_op(np.pi/2,(0,1,0),N)   # pi/2 +y
    op2 = unitary_diag(T/2,H)              # wait T/2
    op3 = rotation_op(np.pi,(1,0,0),N)     # pi +x
    op4 = rotation_op(np.pi/2,(0,-1,0),N)  # pi/2 -y
    return op4 @ op2 @ op3 @ op2 @ op1

def CPMG_N_sequence(T, N, H):
    nqubits = int(np.log2(H.shape[0]))
    half_pi_y = rotation_op(np.pi/2,(0,1,0),nqubits)   # pi/2 +y
    min_half_pi_y = rotation_op(np.pi/2,(0,-1,0),nqubits)   # pi/2 +y
    pi_x = rotation_op(np.pi, (1,0,0),nqubits)
    if N == 0:
        return min_half_pi_y @ unitary_diag(T,H) @ half_pi_y
    wait_half = unitary_diag(T/(2*N),H)    # wait T/2N
    wait_full = unitary_diag(T/N,H)
    op = pi_x @ wait_half @ half_pi_y
    for i in range(N-1):
        op = pi_x @ wait_full @ op
    return min_half_pi_y @ wait_half @ op

# --- OPERATOR FUNCTIONALS --- #
def complex_phase(complex_num):
    return np.arctan(complex_num.imag / complex_num.real)

def expect_op(operator, psi):
    """ Expectation value of operator over state psi
    - psi can be either a single state or a matrix of multiple states 
    - returns a vector of expectation values for each state in psi_t """
    if len(psi.shape) == 1:  # vectorized to psi_t
        psi = psi[np.newaxis]
    if psi.shape[0] != operator.shape[0]:  # num rows must be the same
        psi = psi.T
    # now psi.shape is  (state_dimension, number_of_states)
    # basically compute <psi|operator|psi> for all psis in psi_t
    ret = np.sum(np.conj(psi) * (operator @ psi), axis=0)
    return np.real(ret)

def expect_op_2(operator, rho):
    """ Expectation value of operator over VECTORIZED rho
    - can also be used to reverse-engineer hamiltonians 
    - returns a vector of expectation values for each state in rho_t"""
    if len(rho.shape) == 2:
        rho = rho[np.newaxis]
    assert(rho[0].shape == operator.shape)
    # now rho.shape is (number_of_states, state_dimension, state_dimension)
    ret = np.trace(operator @ rho, axis1=1, axis2=2)
    return np.real(ret)

#@ timeIt
def bloch_from_psi(psi_t):
    """ returns 2D array (len(times), 3) """
    return np.array([expect_op(sigma_x, psi_t), expect_op(sigma_y, psi_t), expect_op(sigma_z, psi_t)]).T

#@ timeIt
def bloch_from_op(rho_t):
    """ returns 2D array (len(times), 3) """
    return np.array([expect_op_2(sigma_x, rho_t), expect_op_2(sigma_y, rho_t), expect_op_2(sigma_z, rho_t)]).T

# use eigenvectors and eigenvalues of diagonalized, time-independent hamiltonian
def time_independent_H(times, es, evs, psi_0_ev):
    ''' returns psi_t from diagonalized H '''
    return np.conj(evs).T @ np.diag(np.exp(-1j * es * times)) @ psi_0_ev

def rabi_hamiltonian(z_omega, x_omega, t, nu):
    if isinstance(t, np.ndarray):  # vectorize the mf
        return 2*x_omega * np.cos(nu * t)[:,np.newaxis,np.newaxis] * sigma_x + z_omega * sigma_z
    return 2*x_omega * np.cos(nu * t) * sigma_x + z_omega * sigma_z


if __name__ == "__main__":
    # example bloch sphere
    z_omega = 0.1
    x_omega = 1
    nu = 1
    num_timesteps = 300
    num_periods = 1
    times = num_periods * 2*np.pi / nu * np.linspace(0, 1, num_timesteps)

    def ddt_psi_t(t, psi):
        return -1j * rabi_hamiltonian(z_omega, x_omega, t, nu) @ psi

    # COMPUTE PSI_T
    psi_0 = np.array([1,0], dtype=complex)
    psi_t = solve_ivp(ddt_psi_t, [times.min(), times.max()], psi_0, t_eval=times).y
    # psi_t.shape = (2, num_timesteps)
    bloch_states = bloch_from_psi(psi_t)
    Hs = rabi_hamiltonian(z_omega, x_omega, times, nu)
    rot_vecs = bloch_from_op(Hs)

    # animate 3D psi_t on bloch_sphere
    with tqdm(total=bloch_states.shape[0]) as pbar:
        animate_bloch(bloch_states, "hamiltonian_many", rot_vecs=rot_vecs, pbar=pbar, 
                    fps=15, dpi=100, add_purity=False, angle_step=1)
    