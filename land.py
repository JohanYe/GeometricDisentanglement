import torch

def land_auto(loc, A, z_points, grid, dv, model, constant=None, batch_size=1024, metric_grid_sum=None, grid_sampled=None):
    """
    Checks if scale is tensor or scale and calls corresponding LAND function
    :param loc: mu
    :param A: std in either full cov or scalar
    :param z_points: latent points to optimize for
    :param grid: grid for constant estimation
    :param dv: grid square size
    :param constant: constant input to save recalculation, by default None
    :param model: model to get metric from
    :param batch_size: batch_size
    :return: lpz, init_curve, D2, constant
    """

    if metric_grid_sum is not None:
        lpz, init_curve, D2, constant = LAND_fullcov_sampled(loc=loc,
                                                             A=A,
                                                             z_points=z_points,
                                                             sampled_grid_points=grid_sampled,
                                                             metric_sum=metric_grid_sum,
                                                             dv=dv,
                                                             model=model,
                                                             logspace=True,
                                                             init_curve=None,
                                                             batch_size=batch_size)

    elif A.dim() == 1:
        lpz, init_curve, D2, constant = LAND_scalar_variance(loc=loc,
                                                             scale=A,
                                                             z_points=z_points,
                                                             grid_points=grid,
                                                             dv=dv,
                                                             constant=constant,
                                                             model=model,
                                                             logspace=True,
                                                             init_curve=None,
                                                             batch_size=batch_size)
    else:
        lpz, init_curve, D2, constant = LAND_fullcov(loc=loc,
                                                     A=A,
                                                     z_points=z_points,
                                                     dv=dv,
                                                     grid_points=grid,
                                                     constant=constant,
                                                     model=model,
                                                     logspace=True,
                                                     init_curve=None,
                                                     batch_size=batch_size)

    return lpz, init_curve, D2, constant


def LAND_fullcov(loc, A, z_points, dv, grid_points, constant=None, model=None, logspace=True,
                 init_curve=None, batch_size=1024):
    """
    full covariance matrix
    """
    if constant == None:
        constant = estimate_constant_full(mu=loc, A=A, grid=grid_points, dv=dv, model=model, batch_size=batch_size)
    loc = loc.repeat([z_points.shape[0], 1])
    if init_curve is not None:
        D2, init_curve, _ = model.dist2_explicit(loc, z_points, C=init_curve, A=A)
    else:
        D2, init_curve, _ = model.dist2_explicit(loc, z_points, A=A)

    inside = (-1 * D2 / 2).squeeze(-1)
    pz = (1 / constant) * inside.exp()

    if logspace:
        lpz = -1 * pz.log()
        return lpz, init_curve, D2, constant
    else:
        return pz, init_curve, D2, constant


def estimate_constant_full(mu, A, grid, dv, model, batch_size=512, sum=True):
    """ Estimate constant using a full covariance matrix. """
    iters = (grid.shape[0] // batch_size) + 1
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    for i in range(iters):
        # data
        grid_points_batch = grid[i * batch_size:(i + 1) * batch_size, :] if i < (iters - 1) else grid[
                                                                                                 i * batch_size:, :]
        mu_repeated = mu.repeat([grid_points_batch.shape[0], 1])

        # calcs
        D2, _, _ = model.dist2_explicit(mu_repeated, grid_points_batch.to(device), A=A)
        exponential_term = (-D2 / 2).squeeze(-1).exp()
        metric_term = model.metric(grid_points_batch.to(device)).det().sqrt()
        constant = metric_term * exponential_term * dv

        if i == 0:
            approx_constant = constant
            if not sum:
                metric_vector = metric_term.cpu()
        else:
            approx_constant = torch.cat((approx_constant, constant), dim=0)
            if not sum:
                metric_vector = torch.cat((metric_vector, constant.cpu()), dim=0)
    if sum:
        return approx_constant.sum()
    else:
        return approx_constant, metric_vector




def LAND_fullcov_sampled(loc, A, z_points, dv, sampled_grid_points, metric_sum, model=None, logspace=True,
                         init_curve=None, batch_size=256):
    """
    full covariance matrix, expecting sampled grid data points.
    """

    iters = (sampled_grid_points.shape[0] // batch_size)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    for i in range(iters):
        # data
        grid_points_batch = sampled_grid_points[i * batch_size:(i + 1) * batch_size, :] if i < (
                    iters - 1) else sampled_grid_points[
                                    i * batch_size:, :]
        mu_repeated = loc.repeat([grid_points_batch.shape[0], 1])

        # calcs
        D2, _, _ = model.dist2_explicit(mu_repeated, grid_points_batch.to(device), A=A)
        exponential_term = (-D2 / 2).squeeze(-1).exp()

        if i == 0:
            approx_constant = exponential_term
        else:
            approx_constant = torch.cat((approx_constant, exponential_term), dim=0)
    constant = approx_constant.mean() * metric_sum

    loc = loc.repeat([z_points.shape[0], 1])
    if init_curve is not None:
        D2, init_curve, success = model.dist2_explicit(loc, z_points, A=A)
    else:
        D2, init_curve, success = model.dist2_explicit(loc, z_points, A=A)

    inside = (-(D2) / 2).squeeze(-1)
    pz = (1 / constant) * inside.exp()
    if logspace:
        lpz = -1 * (pz).log()
        return lpz, init_curve, D2, constant
    else:
        return pz, init_curve, D2, constant


def LAND_grid_prob(grid, model, batch_size=1024, device="cuda"):
    tmp = (grid.shape[0] // batch_size)
    iters = tmp + 1 if grid.shape[0] / batch_size > tmp else tmp
    model.eval()
    with torch.no_grad():
        for i in range(iters):
            z = grid[i * batch_size:(i + 1) * batch_size, :] if i < (iters - 1) else grid[i * batch_size:, :]
            metric_determinant = model.metric(z.to(device)).det()

            if i == 0:  # for metric
                grid_save = metric_determinant
            else:
                grid_save = torch.cat((grid_save, metric_determinant), dim=0)

    print("negative grid metric:", grid_save[grid_save < 0].shape[0])
    grid_save[grid_save < 0] = model.metric(grid[grid_save < 0].to(device)).det()
    grid_metric = grid_save.sqrt()
    grid_prob = grid_metric / grid_metric.sum()
    return grid_prob.cpu(), grid_metric.cpu(), grid_metric.sum().cpu(), grid_save


def LAND_scalar_variance(loc, scale, z_points, grid_points, dv, constant=None, model=None, logspace=True,
                         init_curve=None,
                         batch_size=1024):
    """
    Uses a scalar covariance instead of a covariance matrix
    OBSOLETE
    """
    if constant is None:
        constant = estimate_constant_simple(mu=loc,
                                            std=scale,
                                            grid=grid_points,
                                            dv=dv,
                                            model=model,
                                            batch_size=batch_size)

    loc = loc.repeat([z_points.shape[0], 1])
    if init_curve is not None:
        D2, init_curve, success = model.dist2_explicit(loc, z_points, C=init_curve)
    else:
        D2, init_curve, success = model.dist2_explicit(loc, z_points)

    inside = -D2 / (2 * scale ** 2)

    pz = (1 / constant) * inside.exp()
    if logspace:
        lpz = -1 * pz.log()
        return lpz, init_curve, D2, constant
    else:
        return pz, init_curve, D2, constant


def estimate_constant_simple(mu, std, grid, dv, model, batch_size=512):
    """ OBSOLETE """
    iters = (grid.shape[0] // batch_size) + 1
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    for i in range(iters):
        # data
        grid_points_batch = grid[i * batch_size:(i + 1) * batch_size, :] if i < (iters - 1) else grid[
                                                                                                 i * batch_size:, :]
        mu_repeated = mu.repeat([grid_points_batch.shape[0], 1])

        # calcs
        D2, _, _ = model.dist2_explicit(mu_repeated, grid_points_batch.to(device), A=None)
        exponential_term = (-D2 / (2 * std ** 2)).squeeze(-1).exp()
        metric_term = model.metric(grid_points_batch.to(device)).det().sqrt()
        constant = metric_term * exponential_term * dv

        if i == 0:
            approx_constant = constant
        else:
            approx_constant = torch.cat((approx_constant, constant), dim=0)
    return approx_constant.sum()
