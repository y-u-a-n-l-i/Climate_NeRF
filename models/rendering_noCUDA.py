from logging.config import valid_ident
import torch
import torch.nn.functional as F
from .custom_functions import \
    RayAABBIntersector, sample_pdf, raw2outputs
from einops import rearrange
import vren
# from torchviz import make_dot

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01
prev = {}

def render(models, rays_o, rays_d, **kwargs):
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
    model = models[0]
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(models, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results


@torch.no_grad()
def __render_rays_test(models, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    results = {}
    ret = __render_rays_train(models, rays_o, rays_d, hits_t, **kwargs)
    # import ipdb; ipdb.set_trace()
    results['total_samples'] = ret['total_samples']
    samples = kwargs['samples']
    len_ = len(samples)
    for (key, item) in ret.items():
        if f'{len_-1}' in key:
            results[key.strip(f'{len_-1}')] = item
    model = models[1]
    for k in results:
        if k in ['total_samples']:
            continue
            
        if kwargs.get('use_skybox', False):
            rgb_bg = model.forward_skybox(rays_d)
        else: # real
            if kwargs.get('random_bg', False):
                rgb_bg = torch.rand(3, device=rays_o.device)
            else:
                rgb_bg = torch.zeros(3, device=rays_o.device)
        
        results['rgb'] = results['rgb'] + \
                    rgb_bg*rearrange(1-results['opacity'], 'n -> n 1')
    
    if kwargs.get('render_sem', False):
        mask = results['opacity']<1e-2
        results['semantic'] = torch.argmax(results['semantic'], dim=-1, keepdim=True)
        results['semantic'][mask] = 4

    if kwargs.get('use_skybox', False):
        # print('rendering skybox')
        rgb_bg = model.forward_skybox(rays_d)
    else: # real
        rgb_bg = torch.zeros(3, device=rays_o.device)
    results['rgb'] += rgb_bg*rearrange(1-results['opacity'], 'n -> n 1')
        
    return results


# @torch.cuda.amp.autocast()
def __render_rays_train(models, rays_o, rays_d, hits_t, **kwargs):
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
    results = {}
    # import ipdb; ipdb.set_trace()
    samples = kwargs['samples']
    results['total_samples'] = 0
    for i, sample in enumerate(samples):
        sample = int(sample)
        if i <len(samples)-1:
            model = models[0]
        else:
            model = models[1]
        ############################### sampling along each ray for coarse samples
        
        if i>0:
            mids = .5 * (results[f'z_vals{i-1}'][:, 1:]+results[f'z_vals{i-1}'][:, :-1])
            z_vals = sample_pdf(mids.detach(), results[f'ws{i-1}'][...,1:-1].detach(), sample, det=True)
            # z_vals = torch.cat([results[f'z_vals{i-1}'].detach(), z_vals], dim=-1)
            # sample += results[f'z_vals{i-1}'].shape[-1]
            z_vals, _ = torch.sort(z_vals, -1)
            z_vals = z_vals.detach()
        else:
            t_vals = torch.linspace(0., 1.-1e-3, steps=sample).cuda()
            near = hits_t[:, 0, 0].unsqueeze(-1).repeat(1, sample)
            far = hits_t[:, 0, 1].unsqueeze(-1).repeat(1, sample)
            num_rays = near.shape[0]
            t_vals = t_vals.unsqueeze(0).repeat(near.shape[0],1)
            t_rand = 5e-5*torch.rand(t_vals.shape[0]).cuda()
            t_vals = t_vals + t_rand[:, None]
            z_vals = near * (1.-t_vals) + far * t_vals
            exp_factor = 1.+1/16
            t_vals_pow = (exp_factor**(z_vals-near)-1.)/(exp_factor**(far-near)-1.)
            z_vals = near * (1.-t_vals_pow) + far * t_vals_pow
            # mids = .5 * (z_vals[:, 1:]+z_vals[:, :-1])
            # upper = torch.cat([mids, z_vals[...,-1:]], -1)
            # lower = torch.cat([z_vals[...,:1], mids], -1)
            # z_vals = lower + (upper - lower) * t_rand
            z_vals = z_vals + t_rand[:, None]
        
        xyzs = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
        xyzs = xyzs.reshape(-1, 3)
        dirs = rays_d.unsqueeze(1).repeat(1, sample, 1)
        dirs = dirs.reshape(-1, 3)
        xyzs = xyzs.detach()
        dirs = dirs.detach()
        
        ###############################
        results['total_samples'] += sample
        
        if kwargs.get('test_time', False):
            embed_a = torch.repeat_interleave(kwargs[f'embedding_a{i}'], num_rays*sample, 0)
        else:
            embed_a = torch.repeat_interleave(kwargs[f'embedding_a{i}'], sample, 0)
        
        if i==len(samples)-1:
            if kwargs.get('test_time', False):
                with torch.no_grad():
                    sigmas, rgbs, normals_raw, normals_pred, sems, _ = model.forward_test(xyzs, dirs, embed_a, **kwargs)
            else:
                sigmas, rgbs, normals_raw, normals_pred, sems, _ = model(xyzs, dirs, embed_a, **kwargs)
        else:
            sigmas, rgbs, sems = model(xyzs, dirs, embed_a, **kwargs)
            # sigmas, rgbs, sems = model(xyzs, dirs, **kwargs)
            normals_raw = torch.zeros_like(rgbs).cuda()
            normals_pred = torch.zeros_like(rgbs).cuda()
            # sems = torch.ones((rgbs.shape[0], kwargs['num_classes'])).cuda()/kwargs['num_classes']
        results[f'sigma{i}'] = sigmas
        results[f'xyzs{i}'] = xyzs
        normals_raw = normals_raw.detach()
        
        sigmas = sigmas.reshape(num_rays, sample, 1)
        rgbs = rgbs.reshape(num_rays, sample, 3)
        normals_raw = normals_raw.reshape(num_rays, sample, 3)
        normals_pred = normals_pred.reshape(num_rays, sample, 3)
        sems = sems.reshape(num_rays, sample, kwargs['num_classes'])
        raw = torch.cat([sigmas, rgbs, normals_raw, normals_pred, sems], dim=-1)
        results[f'opacity{i}'], results[f'rgb{i}'], results[f'normal_raw{i}'], results[f'normal_pred{i}'], results[f'semantic{i}'], results[f'ws{i}'], results[f'depth{i}'] = \
            raw2outputs(raw, z_vals, rays_d, classes=kwargs['num_classes'])
        results[f'z_vals{i}'] = z_vals
        
        normals_diff = torch.sum((normals_raw-normals_pred)**2, dim=-1)
        dirs = F.normalize(dirs, p=2, dim=-1, eps=1e-6)
        results[f'Rp{i}'] = (normals_diff.reshape(num_rays, sample) * results[f'ws{i}']).reshape(-1)
        
        if kwargs.get('use_skybox', False):
            rgb_bg = model.forward_skybox(rays_d)
        else: # real
            if kwargs.get('random_bg', False) and not kwargs['test_time']:
                rgb_bg = torch.rand(3, device=rays_o.device)
            else:
                rgb_bg = torch.zeros(3, device=rays_o.device)
        
        results[f'rgb{i}'] = results[f'rgb{i}'] + \
                    rgb_bg*rearrange(1-results[f'opacity{i}'], 'n -> n 1')

    for (i, n) in results.items():
        if isinstance(n, torch.Tensor):
            if torch.any(torch.isnan(n)):
                print(f'nan in results[{i}]')
            if torch.any(torch.isinf(n)):
                print(f'inf in results[{i}]')

    return results