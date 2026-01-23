"""
Architecture optimization utilities for hardware-aware model optimization in medical imaging.

This module provides comprehensive implementations of modern neural network optimization
techniques specifically designed for clinical deployment scenarios. Focuses on reducing
computational overhead, memory usage, and inference latency while maintaining diagnostic
accuracy for the PneumoniaMNIST binary classification task.

Key optimization strategies:
    - Interpolation Removal: Eliminates computational overhead from resolution upscaling
    - Depthwise Separable Convolutions: Reduces parameters and FLOPs significantly
    - Grouped Convolutions: Parallel channel processing for improved efficiency
    - Inverted Residual Blocks: Mobile-optimized residual architectures
    - Low-Rank Factorization: Matrix decomposition for parameter reduction
    - Channel Optimization: Memory layout and activation optimizations
    - Parameter Sharing: Weight reuse across similar layer configurations
"""

import copy
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def create_optimized_model(base_model: nn.Module, optimizations: Dict[str, Any]) -> nn.Module:
    """
    Apply selected optimization strategies in order to create a clinically-optimized model.

    Args:
        base_model: Original ResNet model to optimize for clinical deployment
        optimizations: Dictionary specifying which optimizations to apply with parameters:
            - 'interpolation_removal': bool - Remove upscaling overhead (recommended: True)
            - 'depthwise_separable': bool - Apply depthwise separable convolutions
            - 'grouped_conv': bool - Use grouped convolutions for parallel processing
            - 'channel_optimization': bool - Optimize memory layout and activations
            - 'inverted_residuals': bool - Replace blocks with inverted residuals
            - 'lowrank_factorization': bool - Apply matrix factorization to linear layers
            - 'parameter_sharing': bool - Share weights between similar layers
            
    Returns:
        Optimized model with selected techniques applied, ready for clinical deployment
        
    Example:
        >>> base_model = create_baseline_model()
        >>> optimization_config = {
        ...     'interpolation_removal': True,
        ...     'depthwise_separable': True,
        ...     'channel_optimization': True
        ... }
        >>> optimized_model = create_optimized_model(base_model, optimization_config)
        >>> print("Clinical deployment model ready")
    """
    model = copy.deepcopy(base_model)
  
    print("Starting clinical model optimization pipeline...")
    
    # Define the optimization order
    # 1. Interpolation Removal (Change input flow)
    # 2. Structural changes (Depthwise, Inverted, Grouped, LowRank)
    # 3. Parameter Sharing (Weights)
    # 4. Channel Optimization (Hardware layout)
    optimization_order = [
        'interpolation_removal',
        'depthwise_separable',
        'grouped_conv',
        'inverted_residuals',
        'lowrank_factorization',
        'parameter_sharing',
        'channel_optimization'
    ]
    
    # Optimization function mapping - connects optimization names to their implementation
    # IMPORTANT: Make sure to experiment with different input parameters for each optimization function, if performance is suboptimal
    optimization_functions = {
        'interpolation_removal': lambda m: apply_interpolation_removal_optimization(
            m,
            **optimizations.get('interpolation_removal_params', {})
        ),
        'depthwise_separable': lambda m: apply_depthwise_separable_optimization(
            m,
            **optimizations.get('depthwise_separable_params', {})
        ),
        'grouped_conv': lambda m: apply_grouped_convolution_optimization(
            m,
            **optimizations.get('grouped_conv_params', {})
        ),
        'channel_optimization': lambda m: apply_channel_optimization(
            m,
            **optimizations.get('channel_optimization_params', {})
        ),
        'inverted_residuals': lambda m: apply_inverted_residual_optimization(
            m,
            **optimizations.get('inverted_residuals_params', {})
        ),
        'lowrank_factorization': lambda m: apply_lowrank_factorization(
            m,
            **optimizations.get('lowrank_factorization_params', {})
        ),
        'parameter_sharing': lambda m: apply_parameter_sharing(
            m,
            **optimizations.get('parameter_sharing_params', {})
        )
    }
    
    # Smart iteration through the defined optimization order
    applied_optimizations = []
    for opt_name in optimization_order:
        # Check if this optimization is requested and available
        if optimizations.get(opt_name, False) and opt_name in optimization_functions:
            print(f"   Applying {opt_name.replace('_', ' ')} optimization...")
            try:
                # Apply the optimization using the mapped function
                model = optimization_functions[opt_name](model)
                applied_optimizations.append(opt_name)
            except Exception as e:
                print(f"   ERROR: {opt_name} optimization failed: {e}")
        elif opt_name not in optimization_functions and optimizations.get(opt_name, False):
            print(f"   WARNING: Unknown optimization: {opt_name}")
    
    # Report results
    if applied_optimizations:
        print(f"Applied optimizations in order: {' → '.join(applied_optimizations)}")
    else:
        print("No optimizations were applied")
        
    return model

# --------------------------------------
# INTERPOLATION REMOVAL (NATIVE RESOLUTION)
# --------------------------------------

def apply_interpolation_removal_optimization(model: nn.Module, native_size: int = 64) -> nn.Module:
    """
    Remove interpolation overhead by processing images at native resolution.
    
    Args:
        model: Model with interpolation capability (e.g., ResNetBaseline)
        native_size: Native input resolution to process (64 for clinical deployment)
        
    Returns:
        Optimized model that processes at native resolution without interpolation

    Note: 
        In `data_loader.py`, we would also want to replace ImageNet stats with chest 
        X-ray specific to check if accuracy improves, but you can skip this for simplicity 
        as normalization affects accuracy/sensitivity and not operational efficiency.
        
    Example:
        >>> baseline_model = create_baseline_model()
        >>> optimized_model = apply_interpolation_removal_optimization(baseline_model, 64)
        >>> # Model now processes 64x64 images directly without upscaling
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)

    print(f"Applying native resolution optimization ({native_size}x{native_size})...")
    
    # The ResNetBaseline model automatically interpolates input images from 64x64 to 224x224.
    # We can modify the target_size attribute to match the native size.
    if hasattr(optimized_model, 'target_size'):
        optimized_model.target_size = native_size
        optimized_model.input_size = native_size
        print(f"   Updated target_size to {native_size}")
    else:
        print("   WARNING: Model does not have 'target_size' attribute. Skipping interpolation removal configuration.")

    # Report optimization status and provide deployment guidance
    print("INTERPOLATION REMOVAL completed.")
    
    return optimized_model

# --------------------------------------
# DEPTHWISE SEPARABLE CONVOLUTION MODULES
# --------------------------------------

def apply_depthwise_separable_optimization(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    min_channels: int = 16,
    preserve_residuals: bool = True
) -> nn.Module:
    """
    Convert suitable Conv2d layers to DepthwiseSeparableConv2d for clinical efficiency.
    
    Systematically replaces standard convolutions with depthwise separable alternatives
    to reduce computational cost and memory usage while preserving diagnostic accuracy.
    Essential for deploying medical imaging models on resource-constrained devices.
    
    Args:
        model: Input model to optimize for clinical deployment
        layer_names: Specific layer names to convert (None = convert all suitable layers)
        min_channels: Minimum input/output channels required for conversion
        preserve_residuals: Use residual-compatible configurations for ResNet models
        
    Returns:
        Optimized model with depthwise separable convolutions applied
        
    Note:
        Only converts layers that benefit from depthwise separation (kernel_size > 1,
        sufficient channels, not already grouped). Preserves ResNet compatibility by
        maintaining residual connection requirements.
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_depthwise_separable_optimization(
        ...     model, min_channels=32
        ... )
        >>> # Suitable Conv2d layers now use depthwise separable convolutions
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print("Applying depthwise separable convolution optimization...")

    def replace_conv_with_depthwise(module):
        nonlocal replacements
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                # Criteria: Kernel > 1, Groups == 1 (Standard Conv), Channels >= min
                if (child.kernel_size[0] > 1 and 
                    child.groups == 1 and 
                    child.in_channels >= min_channels and
                    child.out_channels >= min_channels):
                    
                    # Create Depthwise Separable Block
                    # Depthwise: Groups = In_Channels
                    depthwise = nn.Conv2d(
                        in_channels=child.in_channels,
                        out_channels=child.in_channels,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        groups=child.in_channels,
                        bias=False # Usually bias in pointwise
                    )
                    
                    # Pointwise: Kernel = 1, Groups = 1
                    pointwise = nn.Conv2d(
                        in_channels=child.in_channels,
                        out_channels=child.out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=child.bias is not None
                    )
                    
                    # Sequential Block
                    separable_conv = nn.Sequential(depthwise, pointwise)
                    
                    # Replace
                    setattr(module, name, separable_conv)
                    replacements += 1
            else:
                replace_conv_with_depthwise(child)
    
    replace_conv_with_depthwise(optimized_model)

    # Report optimization status
    if replacements > 0:
        print(f"DEPTHWISE SEPARABLE completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: DEPTHWISE SEPARABLE not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# GROUPED CONVOLUTION MODULES
# --------------------------------------

def apply_grouped_convolution_optimization(
    model: nn.Module,
    groups: int = 2,
    min_channels: int = 32,
    layer_names: Optional[List[str]] = None,
    do_depthwise: Optional[bool] = False,
) -> nn.Module:
    """
    Convert suitable Conv2d layers to grouped convolutions for parallel efficiency.
    
    Args:
        model: Input model to optimize
        groups: Number of groups for grouped convolution (typically 2-8)
        min_channels: Minimum channels required for conversion
        layer_names: Specific layers to convert (None = all suitable layers)
        do_depthwise: Whether to apply depthwise grouping (groups=in_channels)
        
    Returns:
        Model with grouped convolutions applied for enhanced efficiency
        
    Note:
        Grouped convolutions can be highly efficient on certain hardware backends, 
        especially when used with memory formats like channels_last and mixed precision (AMP)
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_grouped_convolution_optimization(
        ...     model, groups=4, min_channels=64
        ... )
        >>> # Suitable layers now use 4-group parallel processing
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    # Track number of successful and skipped replacements
    replacements = 0
    skipped = 0

    print(f"Applying grouped convolution optimization (groups={groups})...")

    def replace_with_grouped(module):
        nonlocal replacements, skipped
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                if (child.kernel_size[0] > 1 and 
                    child.groups == 1 and
                    child.in_channels >= min_channels and
                    child.in_channels % groups == 0 and
                    child.out_channels % groups == 0):
                    
                    # Create Grouped Conv
                    grouped_conv = nn.Conv2d(
                        in_channels=child.in_channels,
                        out_channels=child.out_channels,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        groups=groups,
                        bias=child.bias is not None
                    )
                    
                    setattr(module, name, grouped_conv)
                    replacements += 1
                elif child.kernel_size[0] > 1:
                    skipped += 1
            else:
                replace_with_grouped(child)

    replace_with_grouped(optimized_model)

    # Report optimization status and provide deployment tipes
    if replacements > 0:
        print(f"GROUPED CONV completed: Successfully applied to layers with {replacements} replacements. Skipped {skipped} layers.")
        print("\nDEPLOYMENT TIP: For some hardware (like NVIDIA GPUs), grouped convolutions may require specific memory formats (channels_last) and mixed precision to achieve maximum throughput.")
    else:
        print("WARNING: GROUPED CONV not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# INVERTED RESIDUAL BLOCKS
# --------------------------------------

def apply_inverted_residual_optimization(
    model: nn.Module,
    target_layers: Optional[List[str]] = None,
    expand_ratio: int = 6
) -> nn.Module:
    """
    Replace suitable blocks with mobile-optimized InvertedResidual blocks.

    Args:
        model: Original model for mobile optimization
        target_layers: Specific layer names to convert (None = auto-detect suitable blocks)
        expand_ratio: Channel expansion factor for inverted residuals (6 is optimal)
        
    Returns:
        Model with mobile-optimized inverted residual blocks
        
    Note:
        This optimization targets BasicBlock structures and converts them to mobile-friendly
        inverted residuals. Most effective for deployment on edge devices and mobile platforms
        common in point-of-care medical applications.
        
    Example:
        >>> model = create_baseline_model()
        >>> mobile_model = apply_inverted_residual_optimization(
        ...     model, expand_ratio=6
        ... )
        >>> # Suitable blocks now use mobile-optimized inverted residuals
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print(f"Applying mobile inverted residual optimization...")
    
    # Implementing Inverted Residuals involves replacing ResNet BasicBlocks.
    # This requires defining the InvertedResidual block class.
    
    class InvertedResidual(nn.Module):
        def __init__(self, inp, oup, stride, expand_ratio):
            super(InvertedResidual, self).__init__()
            self.stride = stride
            hidden_dim = int(round(inp * expand_ratio))
            self.use_res_connect = self.stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # pw
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=True))
            
            # dw
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
            
            # pw-linear
            layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(oup))
            
            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

    # Iterate and replace BasicBlocks
    def replace_blocks(module):
        nonlocal replacements
        for name, child in module.named_children():
            # Check if it looks like a BasicBlock (has conv1, bn1, relu, conv2, bn2)
            # Or assume it is if it's in the layer containers of ResNet
            if "BasicBlock" in str(type(child)):
                # We need input and output channels and stride
                # ResNet BasicBlock: conv1 (3x3), conv2 (3x3).
                # We replace the whole block.
                # In ResNet, blocks have in_planes and planes.
                
                # Try to extract params
                if hasattr(child, 'conv1'):
                     stride = child.conv1.stride[0]
                     in_channels = child.conv1.in_channels
                     out_channels = child.conv2.out_channels
                     
                     inv_res = InvertedResidual(in_channels, out_channels, stride, expand_ratio)
                     setattr(module, name, inv_res)
                     replacements += 1
            else:
                replace_blocks(child)

    replace_blocks(optimized_model)

    # Report optimization status
    if replacements > 0:
        print(f"INVERTED RESIDUALS completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: INVERTED RESIDUALS not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# LOW-RANK FACTORIZATION MODULES
# --------------------------------------

def apply_lowrank_factorization(
    model: nn.Module,
    min_params: int = 10_000,
    rank_ratio: float = 0.25
) -> nn.Module:
    """
    Apply low-rank factorization to large linear layers for parameter reduction.
    
    Args:
        model: Input model to optimize for clinical deployment
        min_params: Minimum parameter count to consider for factorization
        rank_ratio: Fraction of minimum dimension to use as factorization rank
    
    Returns:
        Model with low-rank factorized linear layers for reduced memory usage
        
    Note:
        Only factorizes layers with sufficient parameters to benefit from compression.
        Rank selection balances compression ratio with accuracy preservation - lower
        ranks provide more compression but may impact diagnostic performance.
        
    Example:
        >>> model = create_baseline_model()
        >>> compressed_model = apply_lowrank_factorization(
        ...     model, min_params=5000, rank_ratio=0.5
        ... )
        >>> # Large linear layers now use low-rank factorization
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    replacements = 0  # Track number of successful replacements

    print("Applying low-rank factorization optimization...")

    def replace_linear(module):
        nonlocal replacements
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                params = child.in_features * child.out_features
                if params >= min_params:
                    rank = int(min(child.in_features, child.out_features) * rank_ratio)
                    if rank < 1: rank = 1
                    
                    # W ~ U @ V
                    # U: (out, rank), V: (rank, in)
                    # Linear(in, out) -> Sequential(Linear(in, rank), Linear(rank, out))
                    
                    # We need to approximate weights using SVD
                    # child.weight is (out, in)
                    if child.weight is not None:
                        U, S, Vh = torch.linalg.svd(child.weight.data, full_matrices=False)
                        # Truncate
                        U = U[:, :rank]
                        S = S[:rank]
                        Vh = Vh[:rank, :]
                        
                        # Absorb S into U or V
                        sqrt_S = torch.diag(torch.sqrt(S))
                        U_prime = U @ sqrt_S
                        V_prime = sqrt_S @ Vh
                        
                        # Create layers
                        # Layer 1: in -> rank. Weight shape (rank, in) = V_prime
                        l1 = nn.Linear(child.in_features, rank, bias=False)
                        l1.weight.data = V_prime
                        
                        # Layer 2: rank -> out. Weight shape (out, rank) = U_prime
                        l2 = nn.Linear(rank, child.out_features, bias=child.bias is not None)
                        l2.weight.data = U_prime
                        if child.bias is not None:
                            l2.bias.data = child.bias.data
                            
                        seq = nn.Sequential(l1, l2)
                        setattr(module, name, seq)
                        replacements += 1
            else:
                replace_linear(child)

    replace_linear(optimized_model)

    # Report optimization status
    if replacements > 0:
        print(f"LOW RANK FACTORIZATION completed: Successfully applied to layers with {replacements} replacements")
    else:
        print("WARNING: LOW RANK FACTORIZATION not applied: No suitable layers found for replacement")

    return optimized_model

# --------------------------------------
# CHANNEL OPTIMIZATION FUNCTIONS
# --------------------------------------

def apply_channel_optimization(
    model: nn.Module,
    enable_channels_last: bool = True,
    enable_inplace_relu: bool = True
) -> nn.Module:
    """
    Apply channel-level optimizations for enhanced hardware efficiency.

    Implements memory layout and activation optimizations to improve hardware utilization
    and reduce memory bandwidth requirements.

    Args:
        model: Model to optimize for hardware efficiency
        enable_channels_last: E.g., you'd use NHWC memory layout for faster GPU convolutions
        enable_inplace_relu: Convert ReLU layers to in-place for memory savings
    
    Returns:
        Hardware-optimized model with improved memory efficiency
        
    Note:
        The 'channels last' memory format can significantly improve convolution performance on certain hardware 
        (e.g., modern GPUs with specialized cores) but requires input tensors to be converted...
        
    Example:
        >>> model = create_baseline_model()
        >>> optimized_model = apply_channel_optimization(model)
        >>> # Remember to convert inputs: input.to(memory_format=torch.channels_last)
    """
    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    
    print("Applying channel-level hardware optimizations...")
    
    if enable_channels_last:
        # Note: This changes the memory format of the parameters.
        # Input tensors also need to be in channels_last to benefit.
        optimized_model = optimized_model.to(memory_format=torch.channels_last)
        print("   Converted model to channels_last memory format")
        
    if enable_inplace_relu:
        count = 0
        for module in optimized_model.modules():
            if isinstance(module, (nn.ReLU, nn.ReLU6)):
                module.inplace = True
                count += 1
        print(f"   Enabled in-place for {count} ReLU/ReLU6 layers")

    # Report optimization status
    print("CHANNEL OPTIMIZATION completed")

    return optimized_model

# --------------------------------------
# PARAMETER SHARING FUNCTIONS
# --------------------------------------

def apply_parameter_sharing(
    model: nn.Module,
    sharing_groups: Optional[List[List[str]]] = None,
    layer_types: Optional[List[Type[nn.Module]]] = None
) -> nn.Module:
    """
    Apply parameter sharing between layers to reduce memory and improve efficiency.

    Shares weight parameters between layers with identical shapes to reduce memory
    footprint and potentially improve generalization. 

    Args:
        model: Model to optimize through parameter sharing
        sharing_groups: Manual specification of layer groups to share parameters.
                       If None, automatically groups layers with identical weight shapes.
        layer_types: Types of layers to consider for parameter sharing 
                    (defaults to Conv2d for maximum impact)
    
    Returns:
        Memory-optimized model with parameter sharing applied
        
    Note:
        Parameter sharing can improve model generalization by enforcing weight
        consistency across similar layers. Most effective when applied to layers
        with identical computational roles and sufficient parameter count.
        
    Example:
        >>> model = create_baseline_model()
        >>> shared_model = apply_parameter_sharing(model)
        >>> # Layers with identical shapes now share parameters
    """    
    # Default to Conv2d layers (largest parameter count and memory footprint)
    if layer_types is None:
        layer_types = [nn.Conv2d]

    # Deep copy model to avoid modifying original
    optimized_model = copy.deepcopy(model)
    # Track number of sharing layers and shared parameters
    total_shared = 0
    total_parameters_shared = 0
    
    print("Applying parameter sharing optimization...")

    # We need to find layers by name
    named_modules = dict(optimized_model.named_modules())
    
    if sharing_groups is None:
        # Auto-detect groups
        from collections import defaultdict
        groups = defaultdict(list)
        
        for name, module in optimized_model.named_modules():
             if any(isinstance(module, t) for t in layer_types):
                 # Key by shape of weight
                 if hasattr(module, 'weight') and module.weight is not None:
                     shape = tuple(module.weight.shape)
                     groups[shape].append(name)
        
        sharing_groups = [g for g in groups.values() if len(g) > 1]

    # Apply sharing
    for group in sharing_groups:
        if len(group) < 2: continue
        
        # Master layer is the first one
        master_name = group[0]
        master_layer = named_modules[master_name]
        
        for slave_name in group[1:]:
            slave_layer = named_modules[slave_name]
            
            # Share weight
            slave_layer.weight = master_layer.weight
            total_parameters_shared += master_layer.weight.numel()
            total_shared += 1
            
            # Share bias if present
            if master_layer.bias is not None and slave_layer.bias is not None:
                 slave_layer.bias = master_layer.bias
   
    # Report optimization status
    if total_shared > 0:
        print(f"PARAMETER SHARING completed - Successfully shared parameters for {total_shared} layers")
        print(f"   Total parameters shared: {total_parameters_shared:,}")
    else:
        print("WARNING: PARAMETER SHARING failed - No suitable layer groups found for optimization")
    
    return optimized_model
