from logging.config import valid_ident
import numpy as np
import torch
import torch.nn.functional as F
from .custom_functions import \
    RayAABBIntersector, RayMarcher, RefLoss, VolumeRenderer
from einops import rearrange
import vren
import trimesh

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01
prev = {}

def srgb_to_linear(c):
    limit = 0.04045
    return torch.where(c>limit, ((c+0.055)/1.055)**2.4, c/12.92)

def linear_to_srgb(c):
    limit = 0.0031308
    c = torch.where(c>limit, 1.055*c**(1/2.4)-0.055, 12.92*c)
    c[c>1] = 1 # "clamp" tonemapper
    return c

def render(model, rays_o, rays_d, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final rgb and depth
    """
    rays_o = rays_o.contiguous(); rays_d = rays_d.contiguous()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE
    if kwargs.get('cal_snow_occ', False) and not kwargs['test_time']:
        _, sky_hits_t, _ = RayAABBIntersector.apply(kwargs['sky_rays_o'].contiguous(), kwargs['sky_rays_d'].contiguous(), \
                                                     model.center, model.half_size, 1)
        kwargs['sky_hits_t'] = sky_hits_t

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results

def volume_render(
    model, 
    rays_o,
    rays_d,
    hits_t,
    # Image properties to be updated
    opacity,
    depth,
    rgb,
    normal_pred,
    normal_raw,
    sem,
    # Other parameters
    **kwargs
):
    N_rays = len(rays_o)
    device = rays_o.device
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
        embedding_a = kwargs['embedding_a']

    classes = kwargs.get('classes', 7)
    samples = 0
    total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4
    
    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t, alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        ## Shapes
        # xyzs: (N_alive*N_samples, 3)
        # dirs: (N_alive*N_samples, 3)
        # deltas: (N_alive, N_samples) intervals between samples (with previous ones)
        # ts: (N_alive, N_samples) ray length for each samples
        # N_eff_samples: (N_alive) #samples along each ray <= N_smaples

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        normals_pred = torch.zeros(len(xyzs), 3, device=device)
        normals_raw = torch.zeros(len(xyzs), 3, device=device)
        sems = torch.zeros(len(xyzs), classes, device=device)
        if isinstance(kwargs.get('embedding_a', None), torch.Tensor):
            kwargs['embedding_a'] = torch.repeat_interleave(embedding_a, len(xyzs), 0)[valid_mask]
        
        _sigmas, _rgbs, _normals_pred, _sems = model.forward_test(xyzs[valid_mask], dirs[valid_mask], **kwargs)

        if kwargs.get("snow", False):
            _sigmas_mb, _rgbs_mb, _normals_mb = kwargs['mb_model'].forward_test(xyzs[valid_mask], geometry_model=model, **kwargs)
            _rgbs_mb = linear_to_srgb(_rgbs_mb.float())
            # thres_ratio = kwargs.get('mb_thres', 1/8)
            # thres = weighted_sigmoid(_sigmas_mb, 50, kwargs.get('center_density', 2e3) * thres_ratio).detach()
            # _sigmas_mb = _sigmas_mb * thres
            _rgbs = torch.clamp((_rgbs * _sigmas[:, None] + _rgbs_mb * _sigmas_mb[:, None]) / (_sigmas[:, None] + _sigmas_mb[:, None] + 1e-9), max=1.)
            _sigmas = _sigmas + _sigmas_mb
       
        if kwargs.get('render_rgb', False) or kwargs.get('render_depth', False):
            sigmas[valid_mask], rgbs[valid_mask] = _sigmas.detach().float(), _rgbs.detach().float()
        if kwargs.get('render_semantic', False):
            sems[valid_mask] = _sems.float()
        if kwargs.get('render_normal', False):
            normals_pred[valid_mask] = _normals_pred.float()
            
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_pred = rearrange(normals_pred, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        normals_raw = rearrange(normals_raw, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        sems = rearrange(sems, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        vren.composite_test_fw(
            sigmas, rgbs, normals_pred, normals_raw, sems, deltas, ts,
            hits_t, alive_indices, kwargs.get('T_threshold', 1e-4), classes,
            N_eff_samples, opacity, depth, rgb, normal_pred, normal_raw, sem)
        alive_indices = alive_indices[alive_indices>=0]

    if kwargs.get('use_skybox', False):
        rgb_bg = model.forward_skybox(rays_d)
        rgb += rgb_bg*rearrange(1 - opacity, 'n -> n 1')
    else: # real
        rgb_bg = torch.zeros(3, device=device)
        rgb += rgb_bg*rearrange(1 - opacity, 'n -> n 1')

    return total_samples

@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Input:
        rays_o: [h*w, 3] rays origin
        rays_d: [h*w, 3] rays direction

    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    hits_t = hits_t[:,0,:]
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    classes = kwargs.get('classes', 7)
    # output tensors to be filled in
    N_rays = len(rays_o) # h*w
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)
    normal_pred = torch.zeros(N_rays, 3, device=device)
    normal_raw = torch.zeros(N_rays, 3, device=device)
    sem = torch.zeros(N_rays, classes, device=device)
    mask = torch.zeros(N_rays, device=device)

    simulator = kwargs.get('simulator', False)
    if simulator:
        img_idx = kwargs.get('img_idx', 0)
        sim_kwargs = {
            # Smog params
            'img_idx': img_idx,
            'model': model,
            'rays_o': rays_o,
            'rays_d': rays_d,
            'hits_t': hits_t, 
            'opacity': opacity,
            'depth': depth,
            'rgb': rgb,
            'kwargs': kwargs
            # Water params
        }
        simulator.simulate_before_marching(**sim_kwargs)
    # Perform volume rendering
    total_samples = \
        volume_render(
            model, rays_o, rays_d, hits_t,
            opacity, depth, rgb, normal_pred, normal_raw, sem,
            **kwargs
        )

    if simulator:
        depth_smooth = kwargs.get('depth_smooth', None)
        if depth_smooth is not None:
            depth_sim = depth_smooth
        else:
            depth_sim = depth
        sim_kwargs = {
            # Smog params
            # Water params
            'model': model,
            'rays_o': rays_o,
            'rays_d': rays_d,
            'depth': depth_sim,
            'rgb': rgb,
            'kwargs': kwargs 
        }
        mask = simulator.simulate_after_marching(**sim_kwargs)
    
    results = {}
    results['opacity'] = opacity # (h*w)
    results['depth'] = depth # (h*w)
    results['depth_xyz'] = rays_o + rays_d*depth[:, None]
    results['rgb'] = rgb # (h*w, 3)
    results['normal_pred'] = normal_pred
    results['normal_raw'] = normal_raw # zeros
    results['semantic'] = torch.argmax(sem, dim=-1, keepdim=True)
    results['total_samples'] = total_samples # total samples for all rays
    results['points'] = rays_o + rays_d * depth.unsqueeze(-1)
    results['mask'] = mask

    return results


def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        rays_a, xyzs, dirs, results['deltas'], results['ts'], total_samples = \
            RayMarcher.apply(
                rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
                model.cascades, model.scale,
                exp_step_factor, model.grid_size, MAX_SAMPLES)
    results['total_samples'] = total_samples
    
    for k, v in kwargs.items(): # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor) and k not in ["up_vector", "ground_height", "R", "R_inv", "sky_rays_o", "sky_rays_d"]:
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)

    if not kwargs.get("stylize", False) and not kwargs.get('make_snow', False):
        sigmas, rgbs, normals_raw, normals_pred, sems, _ = model(xyzs, dirs, **kwargs)
    else:
        
        sigmas, rgbs, normals_pred, sems = model.forward_test(xyzs, dirs, **kwargs)

    results['sigma'] = sigmas
    results['xyzs'] = xyzs
    results['rays_a'] = rays_a

    results['vr_samples'], results['opacity'], results['depth'], results['rgb'], results['normal_pred'], results['semantic'], results['ws'] = \
        VolumeRenderer.apply(sigmas.contiguous(), rgbs.contiguous(), normals_pred.contiguous(), 
                                sems.contiguous(), results['deltas'].contiguous(), results['ts'].contiguous(),
                                rays_a, kwargs.get('T_threshold', 1e-4), kwargs.get('classes', 7))

    if not kwargs.get("stylize", False) and not kwargs.get('make_snow', False):
        normals_raw = normals_raw.detach()
        normals_diff = (normals_raw-normals_pred)**2
        dirs = F.normalize(dirs, p=2, dim=-1, eps=1e-6)
        normals_ori = torch.clamp(torch.sum(normals_pred*dirs, dim=-1), min=0.)**2 # don't keep dim!
        results['Ro'], results['Rp'] = \
            RefLoss.apply(sigmas.detach().contiguous(), normals_diff.contiguous(), normals_ori.contiguous(), results['deltas'].contiguous(), results['ts'].contiguous(),
                                rays_a, kwargs.get('T_threshold', 1e-4))

    if kwargs.get('pred_shadow', False):
        sigmas_shadow = model.density(xyzs, grad=False)
        pred_t = kwargs['sun_vis_net'](xyzs).unsqueeze(-1)
        place_holder = torch.zeros((sigmas_shadow.shape[0], 3)).cuda()
        _, _, _, _, _, results['pred_shadow'], _ = \
            VolumeRenderer.apply(sigmas_shadow.contiguous(), place_holder.contiguous(), place_holder.contiguous(),
                                pred_t.contiguous(), results['deltas'].contiguous(), results['ts'].contiguous(),
                                rays_a, kwargs.get('T_threshold', 1e-4), 1)
        
    if kwargs.get('cal_snow_occ', False):
        with torch.no_grad():
            sky_rays_a, sky_xyzs, sky_dirs, sky_deltas, sky_ts, total_samples = \
                RayMarcher.apply(
                    kwargs['sky_rays_o'], kwargs['sky_rays_d'], 
                    kwargs['sky_hits_t'][:, 0], model.density_bitfield,
                    model.cascades, model.scale,
                    exp_step_factor, model.grid_size, MAX_SAMPLES)
            sigmas_sky = model.density(sky_xyzs, grad=False)
            results['model_t'], mask_ = vren.composite_trans(sigmas_sky.contiguous(), sky_deltas.contiguous(), sky_ts.contiguous(), sky_rays_a, 1e-3) # calculate transparency
            mask_ = mask_.bool()
            results['model_t'] = results['model_t'][mask_]
        results['pred_t'] = kwargs['snow_occ_net'](sky_xyzs[mask_])
    
    if kwargs.get('make_snow', False):
        with torch.no_grad():
            _, _, normals_pred_origin, semantics_origin = kwargs['origin_model'].forward_test(xyzs, dirs, **kwargs)
            weighted_sigmoid = lambda x, weight, bias : 1./(1+torch.exp(-weight*(x-bias)))
            cos_theta = torch.sum(kwargs['up_vector'][None, :]*normals_pred_origin, dim=-1) # for every sample
            semantic_mask = torch.argmax(semantics_origin, dim=-1) != kwargs.get("sky_label", 4)
            scaled_weights = (results['ws'] * 10 * (results['ts'].clamp(min=1.0, max=1.e3))).clamp(0, 1)
            results['alphas'] = (scaled_weights * weighted_sigmoid(cos_theta, 50, 0.8) * semantic_mask.float()) # work as g.t.

            rgbs_l = srgb_to_linear(rgbs.float())
            results['rgbs'] = rgbs_l[:, 0] * 0.2126 + rgbs_l[:, 1] * 0.7152 + rgbs_l[:, 2] * 0.0722
        results['alphas_mb'], results['rgbs_mb'] = kwargs['mb_model'](xyzs) # 3D hash table

    if kwargs.get('use_skybox', False):
        rgb_bg = model.forward_skybox(rays_d)
    elif exp_step_factor==0: # synthetic
        rgb_bg = torch.zeros(3, device=rays_o.device)
    else: # real
        if kwargs.get('random_bg', False):
            rgb_bg = torch.rand(3, device=rays_o.device)
        else:
            rgb_bg = torch.zeros(3, device=rays_o.device)
    
    results['rgb'] = results['rgb'] + \
                rgb_bg*rearrange(1-results['opacity'], 'n -> n 1')

    for (i, n) in results.items():
        if torch.any(torch.isnan(n)):
            print(f'nan in results[{i}]')
        if torch.any(torch.isinf(n)):
            print(f'inf in results[{i}]')

    return results
