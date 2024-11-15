from .zsnrnode import ZsnrVpredConditioningNode, CFGRescaleNode

NODE_CLASS_MAPPINGS = {
    "ZsnrVpredConditioningNode": ZsnrVpredConditioningNode,
    "CFGRescaleNode": CFGRescaleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZsnrVpredConditioningNode": "ZTSNR + V-Prediction",
    "CFGRescaleNode": "CFG Rescale"
}
