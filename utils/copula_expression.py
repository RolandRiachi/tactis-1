import torch
from torch.distributions import MultivariateNormal, Normal

def gaussian_copula(covariance_matrix):
    n = covariance_matrix.shape[0]

    prod_var_matrix = torch.outer(torch.diag(covariance_matrix), torch.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / torch.sqrt(prod_var_matrix)

    mv_normal = MultivariateNormal(loc=torch.zeros(n), covariance_matrix=correlation_matrix)
    std_mv_normal = MultivariateNormal(loc=torch.zeros(n), covariance_matrix=torch.eye(n))
    std_uni_normal = Normal(loc=0, scale=1)

    def log_copula(u):
        quantiles = std_uni_normal.icdf(u)
        mv_log_probs = mv_normal.log_prob(quantiles)
        std_log_prob = torch.nan_to_num(std_mv_normal.log_prob(quantiles), nan=0.)

        return mv_log_probs - std_log_prob

    return log_copula

def sklar_log_gaussian_pdf(mean, covariance_matrix):
    marginals = [Normal(mean[i], covariance_matrix[i, i]**0.5) for i in range(mean.shape[0])]
    log_copula = gaussian_copula(covariance_matrix)
    
    def sklar_log_pdf(x):
        marginal_cdfs = torch.stack([marg.cdf(x[..., i]) for i, marg in enumerate(marginals)], dim=-1)
        marginal_log_pdfs = torch.stack([marg.log_prob(x[..., i]) for i, marg in enumerate(marginals)], dim=-1)

        return log_copula(marginal_cdfs) + marginal_log_pdfs.sum(dim=-1)

    return sklar_log_pdf

def sklar_log_gaussian_timeseries_copula(series_mean, series_covariance_matrix):
    log_sklar_copula = sklar_log_gaussian_pdf(series_mean, series_covariance_matrix)

    def _log_copula_wrapper(hist_time, hist_value, pred_time, pred_value):
        pred_cumsums = pred_value - hist_value[..., -1]
        return log_sklar_copula(pred_cumsums)
    
        pred_increments = pred_value - torch.cat([hist_value[..., -1].unsqueeze(dim=-1), pred_value[..., :-1]], dim=-1)
        pred_cumsums = pred_value - hist_value[..., -1]

        incr_log_probs = incr_joint_dist.log_prob(pred_increments)

        cumsum_log_probs = [dist_fcn.log_prob(pred_value[..., i]) for i, dist_fcn in enumerate(series_marginal_dists)]
        cumsum_log_probs = torch.stack(cumsum_log_probs, dim=-1)
        sum_cumsum_log_probs = torch.nan_to_num(cumsum_log_probs.sum(dim=-1), nan=0.)

        return incr_log_probs - sum_cumsum_log_probs
    
    return _log_copula_wrapper