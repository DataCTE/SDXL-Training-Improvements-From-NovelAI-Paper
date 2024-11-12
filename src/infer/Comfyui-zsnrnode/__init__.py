from .zsnrnode import ZsnrVpredConditioningNode, CFGRescaleNode, CustomKSamplerAdvanced

NODE_CLASS_MAPPINGS = {
    "ZsnrVpredConditioningNode": ZsnrVpredConditioningNode,
    "CFGRescaleNode": CFGRescaleNode,
    "CustomKSampler-sigma": CustomKSamplerAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZsnrVpredConditioningNode": "ZTSNR + V-Prediction",
    "CFGRescaleNode": "CFG Rescale",
    "CustomKSampler-sigma": "KSampler sigma"
}