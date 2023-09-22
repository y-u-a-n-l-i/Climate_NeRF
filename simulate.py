import torch 
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import vren
from utility.fit_plane import * 
import imageio
from models.rendering import volume_render, NEAR_DISTANCE
from models.custom_functions import RayAABBIntersector
from utils import guided_filter
from render_panorama import sample_panorama
from simulate_wave import WaveTMA

def get_simulator(
    effect,
    device,
    **kwargs
):
    '''
    Return the simulator
    '''
    effect = effect.lower()
    assert effect in ['smog', 'water', 'snow']
    simulator = None
    if effect == 'smog':
        simulator = SmogSimulator(
            depth_bound=kwargs.get('depth_bound', 0.9),
            sigma=kwargs.get('sigma', 1.0), 
            rgb=kwargs.get('rgb_smog', [1.0, 1.0, 1.0]),
            device=device
        )
    elif effect == 'water':
        simulator = WaterSimulator(
            water_height=kwargs.get('water_height', 0.0),
            plane_path=kwargs.get('plane_path', None),
            color=kwargs.get('rgb_water', [1.0, 1.0, 1.0]),
            refraction_idx=kwargs.get('refraction_idx', 1.333),
            pano_path=kwargs.get('pano_path', None),
            v_forward=kwargs.get('v_forward', [1, 0, 0]),
            v_down=kwargs.get('v_down', [1, 0, 0]),
            v_right=kwargs.get('v_right', [1, 0, 0]),
            theta=kwargs.get('theta', 0.01),
            sharpness=kwargs.get('sharpness', 1000),
            wave_len=kwargs.get('wave_len', 1.0),
            wave_ampl=kwargs.get('wave_ampl', 1e6),
            refract_decay=kwargs.get('refract_decay', -1.0),
            device=device
        )
    elif effect == 'snow':
        pass 
    return simulator 
class SmogSimulator():
    def __init__(self, 
        depth_bound:float,
        sigma:float,
        rgb:float=[1.0, 1.0, 1.0],
        device='cuda'
    ):
        '''
        Input
            depth_path: path to depth_raw.npy, depth of all testing views
            depth_bound: range in (0, 1), simulate smog in depth_bound * depth

        '''
        self.depth_bound = depth_bound
        self.sigma = sigma
        self.rgb = torch.Tensor(rgb).to(device)
        self.device = device
    
    def simulate_before_marching(self, 
        **sim_kwargs
    ):
        '''
        Input
            view_idx: index of testing view of video
            rays_o: (h*w, 3)
            rays_d: (h*w, 3)
            hits_t: (h*w, 2) near & far bound of ray
            opacity:(h*w)
            depth: (h*w)
            rgb: (h*w, 3)
        NO Return 
            update values of:
            hits_t,
            opacity,
            depth,
            rgb
        '''
        img_idx = sim_kwargs.get('img_idx', 0)
        model = sim_kwargs.get('model', None)
        rays_o = sim_kwargs.get('rays_o', None)
        rays_d = sim_kwargs.get('rays_d', None)
        hits_t = sim_kwargs.get('hits_t', None)
        opacity = sim_kwargs.get('opacity', None)
        depth = sim_kwargs.get('depth', None)
        rgb = sim_kwargs.get('rgb', None)
        kwargs = sim_kwargs.get('kwargs', {})

        n = opacity.size(0)
        classes = kwargs.get('classes', 7)
        device = opacity.device
        opacity_clear = torch.zeros(n, device=device)
        depth_clear   = torch.zeros(n, device=device)
        volume_render(
            model, rays_o, rays_d, hits_t.clone(),
            opacity_clear, depth_clear, 
            torch.zeros(n, 3, device=device), torch.zeros(n, 3, device=device), torch.zeros(n, 3, device=device), torch.zeros(n, classes, device=device),
            **kwargs
        )
        depth_clear += (1 - opacity_clear) * depth_clear.max()
        n_pixels = len(depth_clear)

        # march rays, update htis
        hit_near, hit_far = hits_t[:, 0], hits_t[:, 1]
        
        ts = depth_clear * self.depth_bound 
        ts[ts < hit_near] = hit_near[ts < hit_near]
        ts[ts > hit_far] = hit_far[ts > hit_far]
        # hits_t[:, 0] = ts
        ts = ts.unsqueeze(-1) #(hw, 1)
        deltas = ts.clone()
        N_eff_samples = torch.ones_like(depth_clear).int()

        # get sigma~(0, 500), color~(0, 1) 
        sigmas  = torch.zeros(n_pixels, device=self.device)
        rgbs    = torch.zeros(n_pixels, 3, device=self.device)
        normals = torch.zeros(n_pixels, 3, device=self.device)
        normals_raw = torch.zeros(n_pixels, 3, device=self.device)
        normal = torch.zeros(n_pixels, 3, device=self.device)
        normal_raw = torch.zeros(n_pixels, 3, device=self.device)
        sems    = torch.zeros(n_pixels, classes, device=self.device)
        sem    = torch.zeros(n_pixels, classes, device=self.device)

        alive_indices = torch.arange(n_pixels, device=self.device)
        sigmas[:] = self.sigma
        rgbs[:] = self.rgb
        N_samples = 1
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals = rearrange(normals, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_raw = rearrange(normals_raw, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        sems = rearrange(sems, '(n1 n2) c -> n1 n2 c', n2=N_samples)

        # alpha composite
        vren.composite_test_fw(
            sigmas, rgbs, normals, normals_raw, sems, deltas, ts,
            hits_t, alive_indices, 1e-2, classes,
            N_eff_samples, opacity, depth, rgb, normal, normal_raw, sem)

    def simulate_after_marching(self, 
        **sim_kwargs
    ):
        pass
class WaterSimulator():
    def __init__(self, 
        water_height:float,
        plane_path:str,
        color:float,
        refraction_idx:float,
        # panorama
        pano_path:str,
        v_forward,
        v_down,
        v_right,
        # glossy effect
        theta:float,
        sharpness:float,
        wave_len:float,
        wave_ampl:int,
        refract_decay:float,
        device='cuda'
    ):
        self.device = device
        self.water_height = water_height
        self.initialize_plane(plane_path)
        self.color = torch.FloatTensor(color).to(device)
        self.refraction_idx = refraction_idx
        
        self.panorama = None 
        if pano_path is not None:
            panorama = imageio.imread(pano_path).astype(np.float32)/255.0
            self.panorama = torch.FloatTensor(panorama).to(device)
        self.v_forward = torch.FloatTensor(v_forward).to(device)
        self.v_down = torch.FloatTensor(v_down).to(device)
        self.v_right = torch.FloatTensor(v_right).to(device)

        self.theta = theta
        self.sharpness = sharpness 

        self.wave = WaveTMA(
            center=torch.FloatTensor(self.plane.center),
            normal=torch.FloatTensor(self.plane.normal),
            vec_x=torch.FloatTensor(v_forward),
            plane_len=wave_len,
            ampl_const=wave_ampl,
            device='cuda'
        )
        self.refract_decay = refract_decay

    def initialize_plane(self, plane_path):
        plane_param = np.load(plane_path)
        normal = plane_param[0]
        center = plane_param[1]
        plane = Plane(normal, center)
        plane.move_by_distance(self.water_height)
        self.plane = plane 

    def initialize_plane_ransac(self, points_path):
        points = np.load(points_path) #(n, h, w, 3)
        _, h, _, c = points.shape
        points = points[:,-(h//3):]
        points = points.reshape(-1, c)

        n_sample = 5000
        points = random_sample(points, n_sample)
        plane = plane_ransac(points, n_iter=20, n_sample=500, threshold=0.01)
        plane.move_by_distance(self.water_height)
        self.plane = plane 

    def simulate_before_marching(self, 
        **sim_kwargs
    ):
        pass

    def simulate_after_marching(self, 
        **sim_kwargs
    ):
        '''
        Input
            view_idx: index of testing view of video
            rays_o: (h*w, 3)
            rays_d: (h*w, 3)
            depth: (h*w)
            rgb: (h*w, 3)
        No return    
            Update values of 
            opacity, depth, rgb 
            according to water height 
        '''
        model  = sim_kwargs.get('model', None)
        rays_o = sim_kwargs.get('rays_o', None)
        rays_d = sim_kwargs.get('rays_d', None)
        depth = sim_kwargs.get('depth', None)
        rgb = sim_kwargs.get('rgb', None)
        kwargs = sim_kwargs.get('kwargs', {})
        device = rays_o.device

        # img_idx = kwargs.get('img_idx', 0)
        # rate = (np.sin(img_idx/30) + 1)/2
        # b_top, b_bot = 0.0, -0.1
        # dist = rate * b_top + (1 - rate)*b_bot
        # self.plane.move_by_distance(dist)
        depth_to_water = depth2plane(self.plane, rays_o, rays_d)
        depth_to_water[depth_to_water < 0] = depth.max()
        is_water = depth_to_water < depth
        if torch.sum(is_water) == 0:
            return is_water 
        
        depth[is_water] = depth_to_water[is_water]

        # Render reflection on water surface
        rays_o = rays_o[is_water]
        rays_d = rays_d[is_water]
        depth_to_water = depth_to_water[is_water]

        rays_o_ref = rays_o + depth_to_water.unsqueeze(-1) * rays_d

        img_idx = kwargs.get('img_idx', 0)
        normals = self.wave.sample_normals(img_idx, rays_o_ref)
        
        rgb_reflect, reflection_rate = self.render_reflection(
            rays_o_ref=rays_o_ref, 
            rays_d=rays_d,
            normals=normals,
            model=model,
            device=device,
            kwargs=kwargs
        )

        if self.refract_decay >= 0.0:
            rgb_refract = self.render_refraction(
                rays_o_ref=rays_o_ref, 
                rays_d=rays_d,
                normals=normals,
                model=model,
                device=device,
                kwargs=kwargs
            )
            rgb_water = reflection_rate * rgb_reflect + (1 - reflection_rate)  * rgb_refract
        else:
            rgb_water = reflection_rate * rgb_reflect + (1 - reflection_rate) * self.color
        rgb[is_water] = rgb_water
        # self.plane.move_by_distance(-dist)
        return is_water

    def render_reflection(self, rays_o_ref, rays_d, normals, model, device, kwargs):
        rays_d_ref = reflect_by_normals(normals, rays_d)
        rays_d_ref = F.normalize(rays_d_ref, dim=1)
        
        normal = torch.tensor(self.plane.normal, device=self.device)
        rays_d_scatter, w_scatter = sample_from_SG_sigma(rays_d_ref, normal, self.theta, self.sharpness)
        rays_o_scatter = rays_o_ref.unsqueeze(1).repeat(1, 5, 1)
        rays_o_scatter = rearrange(rays_o_scatter, 'n f c -> (n f) c')
        rays_d_scatter = rearrange(rays_d_scatter, 'n f c -> (n f) c')

        _, hits_t, _ = RayAABBIntersector.apply(rays_o_scatter, rays_d_scatter, model.center, model.half_size, 1)
        hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE
        hits_t = hits_t[:,0,:] 
        
        N_rays = len(rays_o_scatter)
        opacity_ref = torch.zeros(N_rays, device=device)
        depth_ref   = torch.zeros(N_rays, device=device)
        rgb_ref = torch.zeros(N_rays, 3, device=device)
        normal_ref = torch.zeros(N_rays, 3, device=device)
        normal_raw_ref = torch.zeros(N_rays, 3, device=device)
        up_sem_ref = torch.zeros(N_rays, device=device)
        sem_ref = torch.zeros(N_rays, kwargs.get('classes', 7), device=device)
        volume_render(
            model, rays_o_scatter, rays_d_scatter, hits_t,
            opacity_ref, depth_ref, rgb_ref, normal_ref, normal_raw_ref, sem_ref,
            **kwargs
        )

        # Fill incomplete regions with inpainted panorama
        if self.panorama is not None:
            pano_ref = sample_panorama(
                rays_d_scatter, self.panorama,
                self.v_forward, self.v_down, self.v_right
            )
            rgb_ref += pano_ref * (1 - opacity_ref).unsqueeze(-1)
        
        rgb_ref = rearrange(rgb_ref, '(n f) c -> n f c', f=5)
        rgb_ref = torch.sum(rgb_ref * w_scatter.unsqueeze(-1), dim=1)

        # Fresnel effect for reflection
        reflection_rate = calculate_reflection_rate(rays_d_ref, normals, self.refraction_idx)
        reflection_rate = reflection_rate.unsqueeze(-1)
        return rgb_ref, reflection_rate

    def render_refraction(self, rays_o_ref, rays_d, normals, model, device, kwargs):
        rays_d_refract = refract_by_normals(normals, rays_d, self.refraction_idx)

        _, hits_t, _ = RayAABBIntersector.apply(rays_o_ref, rays_d_refract, model.center, model.half_size, 1)
        hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE
        hits_t = hits_t[:,0,:] 
        
        N_rays = rays_o_ref.size(0)
        opacity_ref = torch.zeros(N_rays, device=device)
        depth_ref   = torch.zeros(N_rays, device=device)
        rgb_ref = torch.zeros(N_rays, 3, device=device)
        normal_ref = torch.zeros(N_rays, 3, device=device)
        normal_raw_ref = torch.zeros(N_rays, 3, device=device)
        up_sem_ref = torch.zeros(N_rays, device=device)
        sem_ref = torch.zeros(N_rays, kwargs.get('classes', 7), device=device)
        volume_render(
            model, rays_o_ref, rays_d_refract, hits_t,
            opacity_ref, depth_ref, rgb_ref, normal_ref, normal_raw_ref, sem_ref,
            **kwargs
        )
        alpha = torch.exp(-self.refract_decay * depth_ref).unsqueeze(-1)
        rgb_refract = alpha * rgb_ref + (1 - alpha) * self.color.unsqueeze(0)
        return rgb_refract

def depth2plane(plane, rays_o, rays_d):
    '''
    Input
        rays_o: (n, 3)
        rays_d: (n, 3)
    Return 
        depth: (n)
    '''
    device = rays_o.device 
    center = torch.FloatTensor(plane.center).to(device)
    normal = torch.FloatTensor(plane.normal).to(device)
    numerator = (center - rays_o) @ normal 
    denominator = rays_d @ normal
    depth = numerator / denominator
    return depth

def reflect_by_plane(plane, points):
    '''
    Reflect points according to plane function
    Input & return:
        points: (n, 3)
    '''
    device = points.device
    normal = torch.tensor(plane.normal, device=device) #normalized 
    center = torch.tensor(plane.center, device=device)

    dist = (points - center) @ normal #(n, )
    dist = dist.unsqueeze(1)
    normal = normal.unsqueeze(0)
    points_reflect = points - 2 * (dist @ normal) 
    return points_reflect

def reflect_by_normals(normals, vectors):
    dot = torch.sum(vectors*normals, dim=1) #(n, )
    dot = dot.unsqueeze(1)
    vectors_reflect = vectors - 2*(dot*normals)
    return vectors_reflect

def refract_by_normals(normals, vectors, refraction_index):
    '''
    calculate refraction light direction from empty space into medium (water)
    equation: n_1*sin(theta_1) = n_2*sin(theta_2)
    Args
        normals: (n, 3) unit length
        vectors: (n, 3) rays shooting at surface
    return 
        rafracted_rays: (n, 3)
    '''
    vectors = F.normalize(vectors, dim=-1)
    dot = torch.sum(vectors*normals, dim=-1)
    dot = dot.unsqueeze(1)
    v_surf = vectors - (dot*normals)
    v_surf_unit = F.normalize(v_surf, dim=-1)

    cross = torch.cross(vectors, normals)
    sin_1 = torch.norm(cross, dim=-1)
    sin_2 = sin_1 / refraction_index
    tan_2 = sin_2 / torch.sqrt(1 - sin_2**2)
    v_refract = -normals + tan_2.unsqueeze(-1) * v_surf_unit
    v_refract = F.normalize(v_refract, dim=-1)
    return v_refract

def calculate_reflection_rate(rays_d, normal, refraction_idx:float):
    '''
    Follow Fresnel equation
    assume that light shoot from medium whose refraction index = 1
    Input:
        rays_d: (n, 3)
        normal: (n, 3) or (1, 3)
        both should be normalized
    Return:
        rate: (n, )
    '''
    rays_d /= torch.norm(rays_d, dim=1).unsqueeze(-1)
    normal /= torch.norm(normal, dim=1).unsqueeze(-1)

    cos_i = torch.sum(rays_d * normal, dim=1) #(n, )
    sin_i = torch.sqrt(1 - cos_i**2)
    sin_t = sin_i * 1 / refraction_idx

    theta_i = torch.arcsin(sin_i)
    theta_t = torch.arcsin(sin_t)

    eps = 1e-8
    R_s = (torch.sin(theta_t - theta_i)/(torch.sin(theta_t + theta_i)+eps))**2
    R_p = (torch.tan(theta_t - theta_i)/(torch.tan(theta_t + theta_i)+eps))**2
    rate = (R_s + R_p) / 2
    return rate

def sample_from_SG_sigma(
    rays_lobe,
    normal,
    theta,
    sharpness,
):
    '''
    Sample 4 points deviated with angle theta from lobe direction
    also calculate the SG weight
    Inputs
        ray_lobe: (n, 3) direction of lobes
        normal: (3, )
    Return:
        ray_scatter: (n, 5, 3)
        w_scatter: (n, 5)
    '''
    n = rays_lobe.size(0)
    normal = normal[None].repeat(n, 1) #(n, 3)
    y = torch.cross(normal, rays_lobe) #(n, 3)
    y = F.normalize(y, dim=1)
    x = torch.cross(y, normal) #(n, 3)
    x = F.normalize(x, dim=1)
    rays_y0 = np.cos(theta)*rays_lobe + np.sin(theta)*y
    rays_y1 = np.cos(theta)*rays_lobe - np.sin(theta)*y

    prod_norm = torch.sum(rays_lobe*normal, dim=1)
    phi = torch.arccos(prod_norm) # (n) angle between lobe and normal
    rays_x0 = torch.cos(phi-theta).unsqueeze(-1)*normal + torch.sin(phi-theta).unsqueeze(-1)*x
    rays_x1 = torch.cos(phi+theta).unsqueeze(-1)*normal + torch.sin(phi+theta).unsqueeze(-1)*x

    rays_scatter = torch.stack([rays_lobe, rays_x0, rays_x1, rays_y0, rays_y1], dim=1)
    rays_scatter = F.normalize(rays_scatter, dim=-1)
    w_scatter = torch.ones(n, 5, device=rays_lobe.device)
    w_side = np.exp(-sharpness*(1 - np.cos(theta)))
    w_scatter[:,1:] = w_side
    w_scatter /= (1+w_side*4)
    return rays_scatter, w_scatter

def sample_sphere(num, device='cuda'):
    '''
    Uniformly sample points on unit sphere
    Return
        points: (num, 3)
    '''
    theta = 2*np.pi*np.random.rand(num)
    phi   = np.arccos(1 - 2*np.random.rand(num))
    
    theta = torch.tensor(theta, device=device).float()
    phi = torch.tensor(phi, device=device).float()
    x = torch.sin(phi)*torch.cos(theta)
    y = torch.sin(phi)*torch.sin(theta)
    z = torch.cos(phi)
    points = torch.stack([x, y, z], dim=-1).to(device)
    return points

def sample_hemisphere(num, normal):
    device = normal.device
    points = sample_sphere(num, device)
    prod = torch.matmul(points, normal)
    below = prod < 0
    diff = (-2*prod[below]).unsqueeze(-1)
    points[below] += diff*normal[None]
    return points

def sample_from_SG(
    ray_lobe,
    normal, 
    sharpness, 
    n_coarse,
    n_fine,
    eps=1e-8
):
    '''
    sample rays and weights according to BRDF, 
    which is approximated with a Spherical Gaussian (SG) over the upper hemisphere
    Inputs
        ray_lobe: (n, 3) direction of lobes
        normal: (3, )
    Return 
        ray_scatter: (n, n_fine, 3)
        w_scatter: (n, n_fine)
    '''
    device = ray_lobe.device
    n = ray_lobe.size(0)
    ray_coarse = sample_hemisphere(n_coarse, normal) #(n_coarse, 3)
    prod = torch.matmul(ray_lobe, ray_coarse.T)
    # ray_coarse = sample_hemisphere(n*n_coarse, normal) # (n*n_coarse, 3)
    # ray_coarse = rearrange(ray_coarse, '(n c) d -> n c d', n=n) # (n, n_coarse, 3)
    # prod = torch.matmul(ray_coarse, ray_lobe.unsqueeze(-1)).squeeze(-1) #(n, n_coarse)

    w_SG = torch.exp(-sharpness * (1-prod)) + eps #(n, n_coarse)
    sample_idx = torch.multinomial(w_SG, num_samples=n_fine, replacement=True) #(n, n_fine)

    idx_0 = torch.arange(n).unsqueeze(-1).to(device)
    ray_coarse_all = ray_coarse.unsqueeze(0).repeat(n, 1, 1)
    ray_scatter = ray_coarse_all[idx_0, sample_idx] # (n, n_fine, 3)
    # w_fine = w_SG[idx_0, sample_idx] # (n, n_fine)
    # w_coarse_mean = torch.mean(w_SG, dim=1).unsqueeze(-1) #(n, 1)
    # w_scatter = w_coarse_mean / w_fine
    w_scatter = torch.ones(n, n_fine, device=device)/n_fine
    
    # import vedo 
    # v_lobe = vedo.Points(ray_lobe[0][None], c='red', r=10)
    # v_coarse = vedo.Points(ray_coarse, c='gray', r=3)
    # v_fine  = vedo.Points(ray_scatter[0], c='blue', r=5)
    # vedo.show([v_lobe, v_coarse, v_fine], axes=1)
    return ray_scatter, w_scatter

def test():
    import vedo 
    normal = torch.tensor([0.0, 0.0, 1.0])
    rays = sample_hemisphere(2, normal)
    rays_scatter, w_scatter = sample_from_SG_sigma(rays, normal, theta=0.1, sharpness=10)

    points_0 = vedo.Points(rays, c='red', r=7)
    points_1 = vedo.Points(rays_scatter.view(-1, 3), c='blue', r=5)
    vedo.show([points_0, points_1], axes=1)

def test_refract():
    n = 10
    refraction_index = 1.33
    normals = torch.zeros(n, 3)
    normals[:, -1] = 1
    vectors = torch.randn(n, 3)
    v_refract = refract_by_normals(normals, vectors, refraction_index)
    print(v_refract.size())
    print(v_refract)

if __name__ == '__main__':
    test_refract()
