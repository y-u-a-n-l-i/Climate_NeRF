from distutils.log import WARN
import numpy as np 
import torch 
import torch.nn.functional as F
import math 

class WaveTMA():
    def __init__(self, 
        center,
        normal,
        vec_x,
        plane_len:float,
        ampl_const:float,
        device='cpu'
    ):
        self.device = device
        self.center = center 
        self.normal = normal.to(device)
        self.init_vec_xy(normal, vec_x)
        self.plane_len = plane_len

        self.u_resolution = 4096
        self.u_size = 1000.0
        self.u_wind = torch.tensor([-5.0, 5.0])

        self.G = 9.81
        self.KM = 370.0
        self.CM = 0.23
        self.Omega = 0.84
        self.gamma = 1.7
        self.sigma = 0.08 * (1.0 + 4.0 * math.pow(self.Omega, -3.0))
        self.alphap = 0.006 * np.sqrt(self.Omega)
        self.freq = 5.0 # how many timestamps per cycle
        self.ampl_const = ampl_const # amplitude for the wave
        self.choppiness = 1.0
        
        self.init_K_k()
        self.init_spectrum()

    def init_vec_xy(self, normal, vec_x):
        vec_y = torch.cross(normal, vec_x)
        vec_y = vec_y / torch.norm(vec_y)
        vec_x = torch.cross(vec_y, normal)
        vec_x = vec_x / torch.norm(vec_x)
        self.vec_x = vec_x.to(self.device)
        self.vec_y = vec_y.to(self.device)
    
    def omega_func(self, k):
        return torch.sqrt(self.G*k*(1.0 + (k/self.KM)**2))

    def init_K_k(self):
        n, m = np.meshgrid(np.linspace(0.5, self.u_resolution+0.5, self.u_resolution), np.linspace(0.5, self.u_resolution+0.5, self.u_resolution))
        n = torch.FloatTensor(n).to(self.device)
        m = torch.FloatTensor(m).to(self.device)

        mask_n = (n < self.u_resolution * 0.5)
        n = mask_n * n  + (~mask_n) * (self.u_resolution - n)
        mask_m = (m < self.u_resolution * 0.5)
        m = mask_m * m  + (~mask_m) * (self.u_resolution - m)
        
        K = torch.stack((n, m), dim=2) * 2.0 * np.pi / self.u_size
        k = torch.norm(K, dim=2) 
        self.K = K 
        self.k = k 

    def init_spectrum(self):
        K = self.K.cpu()
        k = self.k.cpu()

        l_wind = torch.norm(self.u_wind).item()
        kp = self.G * np.square(self.Omega / l_wind)
        c = self.omega_func(k) / k
        cp = self.omega_func(torch.FloatTensor([kp])) / kp

        Lpm = torch.exp(-1.25 * torch.square(kp / k))
        Gamma = torch.exp(-torch.square(torch.sqrt(k / kp) - 1.0) / 2.0 * np.square(self.sigma))
        Jp = self.gamma ** Gamma
        Fp = Lpm * Jp * torch.exp(-self.Omega / np.sqrt(10.0) * (torch.sqrt(k / kp) - 1.0))

        Bl = 0.5 * self.alphap * cp / c * Fp

        z0 = 0.000037 * np.square(l_wind) / self.G * np.power(l_wind / cp, 0.9)
        uStar = 0.41 * l_wind / np.log(10.0 / z0)
        alpham = 0.01 * ((uStar < self.CM)*(1.0 + np.log(uStar / self.CM)) + (~(uStar < self.CM)) * (1.0 + 3.0 * np.log(uStar / self.CM)))
        Fm = torch.exp(-0.25 * torch.square(k / self.KM - 1.0))
        Bh = 0.5 * alpham * self.CM / c * Fm * Lpm

        a0 = np.log(2.0) / 4.0
        am = 0.13 * uStar / self.CM
        Delta = np.tanh(a0 + 4.0 * np.power(c / cp, 2.5) + am * np.power(self.CM / c, 2.5))

        u_wind_norm = self.u_wind / l_wind
        K_norm = K / torch.unsqueeze(torch.norm(K, dim=2), dim=2)
        cosPhi = K_norm@u_wind_norm

        S = (1.0 / (2.0 * np.pi)) * (k**(-4.0)) * (Bl + Bh) * (1.0 + Delta * (2.0 * cosPhi * cosPhi - 1.0))
        dk = 2.0 * np.pi / self.u_size
        spec0 = torch.sqrt(S / 2.0) * dk
        self.spec0 = spec0 
        h0 = torch.randn(self.u_resolution, self.u_resolution) + \
             1.j * torch.randn(self.u_resolution, self.u_resolution)
        self.h0 = (h0*spec0).to(self.device)
        self.h0_star = torch.flip(self.h0, dims=[0, 1])
        self.phase0  = torch.randn(self.u_resolution, self.u_resolution, device=self.device)

    def compute_phase(self, t):
        deltaPhase = self.omega_func(self.k) * t/self.freq
        phase = torch.remainder(self.phase0 + deltaPhase, 2.0*np.pi)
        return phase 

    def get_wave_points(self, t):
        X, Y = np.meshgrid(np.linspace(-1, 1, self.u_resolution), np.linspace(-1, 1, self.u_resolution))
        X = torch.FloatTensor(X).to(self.device)
        Y = torch.FloatTensor(Y).to(self.device)
        
        phase = self.compute_phase(t)
        phase_complex = torch.cos(phase) + 1.j * torch.sin(phase)
        phase_complex_star = torch.cos(phase) - 1.j * torch.sin(phase)
        ht = self.h0 * phase_complex + self.h0_star * phase_complex_star # this seems correct as ifft is not real number

        waveVector_normalized = self.K / self.k.unsqueeze(2)
        wave_x = waveVector_normalized[:, :, 0]
        wave_y = waveVector_normalized[:, :, 1]
        dZ = torch.real(torch.fft.ifft2(ht)) * self.ampl_const 
        dX = torch.real(torch.fft.ifft2(ht*(1.j) * wave_x)) * self.ampl_const * self.choppiness
        dY = torch.real(torch.fft.ifft2(ht*(1.j) * wave_y)) * self.ampl_const * self.choppiness

        dZ_x = torch.real(torch.fft.ifft2(ht*(1.j) * wave_x)) * self.ampl_const * self.choppiness
        dX_x = torch.real(torch.fft.ifft2(ht*(-1) * wave_x * wave_x)) * self.ampl_const * self.choppiness
        # dY_x = torch.real(torch.fft.ifft2(ht*(-1) * wave_y * wave_x)) * self.ampl_const * self.choppiness

        dZ_y = torch.real(torch.fft.ifft2(ht*(1.j) * wave_y)) * self.ampl_const * self.choppiness
        # dX_y = torch.real(torch.fft.ifft2(ht*(-1) * wave_x * wave_y)) * self.ampl_const * self.choppiness
        dY_y = torch.real(torch.fft.ifft2(ht*(-1) * wave_y * wave_y)) * self.ampl_const * self.choppiness

        # grad_x = torch.stack([dX_x, dY_x, dZ_x], dim=-1)
        # grad_y = torch.stack([dX_y, dY_y, dZ_y], dim=-1)
        # normal = torch.cross(grad_x, grad_y, dim=-1)
        # basis = torch.stack([self.vec_x, self.vec_y, self.normal], dim=0) # (3, 3)
        # normal = torch.matmul(normal, basis)
        # normal = F.normalize(normal, dim=-1)
        
        sx = dZ_x / (1+dX_x)
        sy = dZ_y / (1+dY_y)
        ones = torch.ones_like(sx)
        normal = torch.stack([-sx, -sy, ones], dim=-1)
        basis = torch.stack([self.vec_x, self.vec_y, self.normal], dim=0) # (3, 3)
        normal = torch.matmul(normal, basis)
        normal = F.normalize(normal, dim=-1)

        # grad_x = torch.ones_like(dZ_x).unsqueeze(-1)*self.vec_x[None] + dZ_x.unsqueeze(-1)*self.normal[None]
        # grad_y = torch.ones_like(dZ_y).unsqueeze(-1)*self.vec_y[None] + dZ_y.unsqueeze(-1)*self.normal[None]
        # normal = torch.cross(grad_x, grad_y, dim=-1)
        # normal = F.normalize(normal, dim=-1)
        
        return X+dX, Y+dY, dZ, normal

    def sample_normals(self, t, points):
        '''
        Inputs
            points: (n, 3)
        '''
        device = points.device
        vec_xy = torch.stack([self.vec_x, self.vec_y], dim=-1).to(device) #(3, 2)
        center = self.center.to(device)
        plane_coord = torch.matmul(points - center.unsqueeze(0), vec_xy) #(n, 2)
        grid = plane_coord / (self.plane_len/2) 

        x, y, z, normal = self.get_wave_points(t)
        normal = normal.permute(2, 0, 1)[None].to(device)
        grid = grid[None, None]
        normal_samples = F.grid_sample(
            normal, 
            grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=True
        ) #(1, 3, 1, n)
        normal_samples = normal_samples.permute(0, 2, 3, 1)[0, 0] #(n, 3)
        normal_samples = F.normalize(normal_samples, dim=-1)
        return normal_samples

class WaveSimple():
    def __init__(self,
        center,
        normal,
        vec_x,
    ):
        self.center = center
        self.normal = normal 
        self.init_vec_xy(normal, vec_x)

        # f(x, y, t) = k*sin(omega_x*t + phi_x*x)*sin(omega_y*t + phi_y*y)
        self.k = 1e-3
        self.omega_x = 1e-1
        self.omega_y = 0 #5e-2
        self.phi_x   = 1e1
        self.phi_y   = 1e1


    def init_vec_xy(self, normal, vec_x):
        vec_y = torch.cross(normal, vec_x)
        vec_y = vec_y / torch.norm(vec_y)
        vec_x = torch.cross(vec_y, normal)
        vec_x = vec_x / torch.norm(vec_x)
        self.vec_x = vec_x 
        self.vec_y = vec_y

    def get_wave_points(self, t):
        res = 512
        x, y = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
        z = self.k * np.sin(self.omega_x*t + self.phi_x*x) * np.sin(self.omega_y*t + self.phi_y*y)
        return x, y, z

def test():
    wave = WaveTMA(
        center=torch.FloatTensor([0, 0, 0]),
        normal=torch.FloatTensor([0, 0, 1]),
        vec_x=torch.FloatTensor([1, 0, 0]),
        plane_len=5,
        device='cuda'
    )

    phase = wave.compute_phase(5)
    x, y, z, n = wave.get_wave_points(5)

    print('normal shape')
    print(n.size())
    print(n[:,:,-1])
    print(torch.norm(n, dim=-1))

    points = torch.randn(100, 3).to('cuda')
    normal_samples = wave.sample_normals(5, points)
    print('Normal samples')
    print('size:', normal_samples.size())
    print('device:', normal_samples.device)

def test_vis():
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    from matplotlib.animation import FuncAnimation, PillowWriter

    wave = WaveTMA(
        center=torch.FloatTensor([0, 0, 0]),
        normal=torch.FloatTensor([0, 0, 1]),
        vec_x=torch.FloatTensor([1, 0, 0]),
        plane_len=5
    )
    # wave = WaveSimple(
    #     center=torch.FloatTensor([0, 0, 0]),
    #     normal=torch.FloatTensor([0, 0, 1]),
    #     vec_x=torch.FloatTensor([1, 0, 0]),
    # )
    
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(20, 20))
    # def animate_surface(t):
    #     x, y, z, normal = wave.get_wave_points(t)
    #     ax.clear()

    #     surf = ax.plot_surface(x, y, z, 
    #         rstride=1, cstride=1, linewidth=0, cmap=cm.coolwarm, antialiased=False)
        
    #     ax.set_zlim(-1.01, 1.01)
    #     ax.zaxis.set_major_locator(LinearLocator(10))
    #     ax.zaxis.set_major_formatter('{x:.02f}')
    #     print(t)
    #     return surf

    # ani = FuncAnimation(fig, animate_surface, interval=1, blit=False, repeat=True, frames=100)
    # ani.save("output/3D_wave_surface.gif", dpi=300, writer=PillowWriter(fps=10))

    fig, ax = plt.subplots(figsize=(20, 20))
    x, y, z, normal = wave.get_wave_points(0)
    im = plt.imshow((normal + 1)/2, animated=True)
    def animate_normal(t):
        x, y, z, normal = wave.get_wave_points(t)
        im.set_array((normal + 1)/2)
        print(t)
        return im

    ani = FuncAnimation(fig, animate_normal, interval=1, blit=False, repeat=True, frames=100)
    ani.save("output/1028_waves/wave_gif/3D_wave_normal.gif", dpi=300, writer=PillowWriter(fps=10))

if __name__ == '__main__':
    test()