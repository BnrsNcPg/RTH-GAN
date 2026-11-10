"""Lazy PINN/fusion exports; full RTH-GAN does not import legacy experiments."""
from importlib import import_module
from importlib.util import find_spec

_EXPORTS = {
    'BoundaryDerivativeTerms': ('pinn_retinex', 'BoundaryDerivativeTerms'),
    'PINNLosses': ('pinn_retinex', 'PINNLosses'),
    'PDEResidualTerms': ('pinn_retinex', 'PDEResidualTerms'),
    'PINNRetinex': ('pinn_retinex', 'PINNRetinex'),
    'PINNRetinexConfig': ('pinn_retinex', 'PINNRetinexConfig'),
    'RetinexPriors': ('pinn_retinex', 'RetinexPriors'),
    'build_intensity_map': ('pinn_retinex', 'build_intensity_map'),
    'compute_boundary_normal_derivative': ('pinn_retinex', 'compute_boundary_normal_derivative'),
    'compute_retinex_pde_residual': ('pinn_retinex', 'compute_retinex_pde_residual'),
    'differentiable_bilinear_sample': ('pinn_retinex', 'differentiable_bilinear_sample'),
    'sample_boundary_coordinates': ('pinn_retinex', 'sample_boundary_coordinates'),
    'sample_interior_coordinates': ('pinn_retinex', 'sample_interior_coordinates'),
    'EuclideanControlFusion': ('euclidean_hyperbolic_fusion', 'EuclideanControlFusion'),
    'EuclideanHyperbolicFusion': ('euclidean_hyperbolic_fusion', 'EuclideanHyperbolicFusion'),
    'FusionConfig': ('euclidean_hyperbolic_fusion', 'FusionConfig'),
    'build_stage3a_fusion': ('euclidean_hyperbolic_fusion', 'build_stage3a_fusion'),
    'expmap0': ('euclidean_hyperbolic_fusion', 'expmap0'),
    'logmap0': ('euclidean_hyperbolic_fusion', 'logmap0'),
    'project_to_ball': ('euclidean_hyperbolic_fusion', 'project_to_ball'),
    'EuclideanControlFusionV2': ('euclidean_hyperbolic_fusion_v2', 'EuclideanControlFusionV2'),
    'EuclideanHyperbolicFusionV2': ('euclidean_hyperbolic_fusion_v2', 'EuclideanHyperbolicFusionV2'),
    'FusionV2Config': ('euclidean_hyperbolic_fusion_v2', 'FusionV2Config'),
    'build_stage3b_fusion': ('euclidean_hyperbolic_fusion_v2', 'build_stage3b_fusion'),
    'euclidean_residual': ('euclidean_hyperbolic_fusion_v2', 'euclidean_residual'),
    'hyperbolic_residual': ('euclidean_hyperbolic_fusion_v2', 'hyperbolic_residual'),
    'mobius_add': ('euclidean_hyperbolic_fusion_v2', 'mobius_add'),
    'mobius_matvec': ('euclidean_hyperbolic_fusion_v2', 'mobius_matvec'),
    'radial_tangent_squash': ('euclidean_hyperbolic_fusion_v2', 'radial_tangent_squash'),
    'AdaptiveEuclideanControlFusion': ('task_adaptive_dual_space_fusion', 'AdaptiveEuclideanControlFusion'),
    'AdaptiveEuclideanHyperbolicFusion': ('task_adaptive_dual_space_fusion', 'AdaptiveEuclideanHyperbolicFusion'),
    'FusionV3Config': ('task_adaptive_dual_space_fusion', 'FusionV3Config'),
    'build_stage3c_fusion': ('task_adaptive_dual_space_fusion', 'build_stage3c_fusion'),
    'channel_rms_normalize': ('task_adaptive_dual_space_fusion', 'channel_rms_normalize'),
    'FusionV4Config': ('residual_stabilized_soft_dual_space_fusion', 'FusionV4Config'),
    'ResidualStabilizedAdaptiveEuclideanControlFusion': ('residual_stabilized_soft_dual_space_fusion', 'ResidualStabilizedAdaptiveEuclideanControlFusion'),
    'ResidualStabilizedAdaptiveEuclideanHyperbolicFusion': ('residual_stabilized_soft_dual_space_fusion', 'ResidualStabilizedAdaptiveEuclideanHyperbolicFusion'),
    'build_stage3d_fusion': ('residual_stabilized_soft_dual_space_fusion', 'build_stage3d_fusion'),
    'detached_rms_residual_scale': ('residual_stabilized_soft_dual_space_fusion', 'detached_rms_residual_scale'),
}

__all__ = [name for name, (module, _) in _EXPORTS.items()
           if find_spec(__name__ + "." + module) is not None]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _EXPORTS[name]
    value = getattr(import_module("." + module, __name__), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
