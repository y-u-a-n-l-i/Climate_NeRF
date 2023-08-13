import numpy as np
import math
from numpy import tanh, sqrt
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from matplotlib.colors import LightSource
from matplotlib.animation import FuncAnimation, PillowWriter


PI = 3.14159265359
G = 9.81;
KM = 370.0;
CM = 0.23
omega = lambda k: sqrt(G * k * (1.0 + (k / KM) ** 2))


def init_spectrum(u_wind=np.array([10, 10]), u_resolution= 512, u_size= 15.0):


    Omega = 0.84
    gamma = 1.7
    sigma = 0.08 * (1.0 + 4.0 * math.pow(Omega, -3.0))
    alphap = 0.006 * np.sqrt(Omega)

    print(Omega, gamma, sigma, alphap)

    n, m = np.meshgrid(np.linspace(0.5, u_resolution+0.5, u_resolution), np.linspace(0.5, u_resolution+0.5, u_resolution))

    mask_n = (n < u_resolution * 0.5)
    n = mask_n * n  + (1-mask_n) * (n - u_resolution)
    mask_m = (m < u_resolution * 0.5)
    m = mask_m * m  + (1-mask_m) * (m - u_resolution)
    # plt.imshow(n)
    K = np.stack((n, m), axis = 2) * 2.0 * PI / u_size
    k = np.linalg.norm(K, axis = 2) 
    # plt.figure(); plt.imshow(k)
    l_wind = np.linalg.norm(u_wind)
    kp = G * np.square(Omega / l_wind)
    c = omega(k) / k
    cp = omega(kp) / kp
    # plt.figure(); plt.imshow(c, 'jet')

    Lpm = np.exp(-1.25 * np.square(kp / k))
    Gamma = np.exp(-np.square(sqrt(k / kp) - 1.0) / 2.0 * np.square(sigma))
    Jp = gamma ** Gamma
    Fp = Lpm * Jp * np.exp(-Omega / np.sqrt(10.0) * (np.sqrt(k / kp) - 1.0))
    Bl = 0.5 * alphap * cp / c * Fp
    # plt.figure(); plt.imshow(Bl)


    z0 = 0.000037 * np.square(l_wind) / G * np.power(l_wind / cp, 0.9)
    uStar = 0.41 * l_wind / np.log(10.0 / z0)
    alpham = 0.01 * ((uStar < CM)*(1.0 + np.log(uStar / CM)) + (1 - (uStar < CM)) * (1.0 + 3.0 * np.log(uStar / CM)))
    Fm = np.exp(-0.25 * np.square(k / KM - 1.0))
    Bh = 0.5 * alpham * CM / c * Fm * Lpm
    # plt.figure(); plt.imshow(Bh)

    a0 = np.log(2.0) / 4.0
    am = 0.13 * uStar / CM
    Delta = np.tanh(a0 + 4.0 * np.power(c / cp, 2.5) + am * np.power(CM / c, 2.5))

    u_wind_norm = (u_wind) / np.linalg.norm(u_wind)
    K_norm = K / np.expand_dims(np.linalg.norm(K, axis = 2), axis = 2)
    cosPhi = K_norm.dot(u_wind_norm)

    S = (1.0 / (2.0 * PI)) * np.power(k, -4.0) * (Bl + Bh) * (1.0 + Delta * (2.0 * cosPhi * cosPhi - 1.0))
    dk = 2.0 * PI / u_size
    h = np.sqrt(S / 2.0) * dk

    # plt.figure(); plt.imshow(h)
    return h

def compute_phase(phase_0, t, u_resolution= 512, u_size= 250.0):
    deltaTime = 1.0 / 60.0
    omega = lambda k: sqrt(G * k * (1.0 + (k / KM) ** 2))
    n, m = np.meshgrid(np.linspace(0.5, u_resolution+0.5, u_resolution), np.linspace(0.5, u_resolution+0.5, u_resolution))
    mask_n = (n < u_resolution * 0.5)
    n = mask_n * n  + (1-mask_n) * (n - u_resolution)
    mask_m = (m < u_resolution * 0.5)
    m = mask_m * m  + (1-mask_m) * (m - u_resolution)
    K = np.stack((n, m), axis = 2) * 2.0 * PI / u_size
    k = np.linalg.norm(K, axis = 2)
    
    deltaPhase = omega(k) * t
    phase = np.mod(phase_0 + deltaPhase, 2.0*PI)
    return phase 

#     'varying vec2 v_coordinates;',

#     'uniform sampler2D u_phases;',

#     'uniform float u_deltaTime;',
#     'uniform float u_resolution;',
#     'uniform float u_size;',

#     'float omega (float k) {',
#         'return sqrt(G * k * (1.0 + k * k / KM * KM));',
#     '}',

#     'void main (void) {',
#         'float deltaTime = 1.0 / 60.0;',
#         'vec2 coordinates = gl_FragCoord.xy - 0.5;',
#         'float n = (coordinates.x < u_resolution * 0.5) ? coordinates.x : coordinates.x - u_resolution;',
#         'float m = (coordinates.y < u_resolution * 0.5) ? coordinates.y : coordinates.y - u_resolution;',
#         'vec2 waveVector = (2.0 * PI * vec2(n, m)) / u_size;',

#         'float phase = texture2D(u_phases, v_coordinates).r;',
#         'float deltaPhase = omega(length(waveVector)) * u_deltaTime;',
#         'phase = mod(phase + deltaPhase, 2.0 * PI);',

#         'gl_FragColor = vec4(phase, 0.0, 0.0, 0.0);',
#     '}'
# ].join('\n');


u_wind=np.array([-5, 5])
u_size= 500.0
freq = 15.0 # how many timestamps per cycle
ampl_const = 1e4 # amplitude for the wave

spec0 = init_spectrum(u_wind, u_size=u_size)
h0 = np.random.randn(512, 512) + 1.j * np.random.randn(512, 512)
h0 = h0*spec0
h0_star = np.fliplr(np.flipud(h0))
phase_0 = np.random.randn(512, 512)

# plt.imshow(np.real(h0))
eta = np.fft.ifft2(h0)
normalize = lambda x : (x - np.min(x)) / (np.max(x) - np.min(x))
# plt.figure(figsize=(10, 10));plt.imshow(np.uint8(normalize(np.real(eta))*255))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20, 20))
# Make data.
X = np.linspace(-1, 1, 512)
Y = np.linspace(-1, 1, 512)
X, Y = np.meshgrid(X, Y)


def animate(t, u_resolution= 512):
    phase = compute_phase(phase_0, t=t/freq, u_size=u_size)
    # plt.figure(figsize=(10, 10));plt.imshow(phase)
    phase_complex = np.cos(phase) + 1.j * np.sin(phase)
    phase_complex_star = np.cos(phase) - 1.j * np.sin(phase)
    ht = h0 * phase_complex + h0_star * phase_complex_star # this seems correct as ifft is not real number
    eta = np.fft.ifft2(ht)
    
    n, m = np.meshgrid(np.linspace(0.5, u_resolution+0.5, u_resolution), np.linspace(0.5, u_resolution+0.5, u_resolution))
    mask_n = (n < u_resolution * 0.5)
    n = mask_n * n  + (1-mask_n) * (n - u_resolution)
    mask_m = (m < u_resolution * 0.5)
    m = mask_m * m  + (1-mask_m) * (m - u_resolution)
    K = np.stack((n, m), axis = 2) * 2.0 * PI / u_size
    k = np.linalg.norm(K, axis = 2)
    waveVector = K / np.expand_dims(k, axis = 2)
    # plt.figure(figsize=(10, 10));plt.imshow(np.uint8(normalize(np.real(eta))*255))
    
    ax.clear()
    
    dZ = np.real(eta) * ampl_const 
    dX = np.real(eta*(1.j) * ampl_const * waveVector[:, :, 0])
    dY = np.real(eta*(1.j) * ampl_const * waveVector[:, :, 1])
    # Plot the surface.

    # Create light source object.
    # ls = LightSource(azdeg=0, altdeg=65)
    # Shade data, creating an rgb array.
    # rgb = ls.shade(Z, plt.cm.RdYlBu)

    surf = ax.plot_surface(X+dX, Y+dY, dZ, rstride=1, cstride=1, linewidth=0, cmap=cm.coolwarm,
                            antialiased=False) # , facecolors=rgb)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # fig.show()
    print(t)
    return surf

animate(0)

ani = FuncAnimation(fig, animate, interval=1, blit=False, repeat=True, frames=120)    
ani.save("3D_wave.gif", dpi=300, writer=PillowWriter(fps=10))