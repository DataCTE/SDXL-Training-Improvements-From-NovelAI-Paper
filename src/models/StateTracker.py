from typing import Any, Dict, Optional, Union
import torch
from diffusers.callbacks import PipelineCallback


class StateTracker(PipelineCallback):
    """
    StateTracker for monitoring and managing SDXL pipeline state during training.
    Inherits from PipelineCallback to integrate with the diffusers pipeline.
    """
    
    def __init__(self):
        super().__init__()
        self.current_timestep: Optional[int] = None
        self.latents: Optional[torch.Tensor] = None
        self.text_encoder_hidden_states: Optional[torch.Tensor] = None
        self.text_encoder_2_hidden_states: Optional[torch.Tensor] = None
        self.prompt_embeds: Optional[torch.Tensor] = None
        self.pooled_prompt_embeds: Optional[torch.Tensor] = None
        self.state: Dict[str, Any] = {}

    def on_step_begin(self, step: int, timestep: int, callback_kwargs: Dict[str, Any]) -> None:
        """Called at the beginning of each denoising step."""
        self.current_timestep = timestep
        # Store latents if available
        if "latents" in callback_kwargs:
            self.latents = callback_kwargs["latents"]

    def on_step_end(self, step: int, timestep: int, callback_kwargs: Dict[str, Any]) -> None:
        """Called at the end of each denoising step."""
        pass

    def store_text_encoder_outputs(self, 
                                 text_encoder_hidden_states: Optional[torch.Tensor] = None,
                                 text_encoder_2_hidden_states: Optional[torch.Tensor] = None,
                                 prompt_embeds: Optional[torch.Tensor] = None,
                                 pooled_prompt_embeds: Optional[torch.Tensor] = None) -> None:
        """Store text encoder outputs for later use."""
        if text_encoder_hidden_states is not None:
            self.text_encoder_hidden_states = text_encoder_hidden_states
        if text_encoder_2_hidden_states is not None:
            self.text_encoder_2_hidden_states = text_encoder_2_hidden_states
        if prompt_embeds is not None:
            self.prompt_embeds = prompt_embeds
        if pooled_prompt_embeds is not None:
            self.pooled_prompt_embeds = pooled_prompt_embeds

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the pipeline."""
        return {
            "current_timestep": self.current_timestep,
            "latents": self.latents,
            "text_encoder_hidden_states": self.text_encoder_hidden_states,
            "text_encoder_2_hidden_states": self.text_encoder_2_hidden_states,
            "prompt_embeds": self.prompt_embeds,
            "pooled_prompt_embeds": self.pooled_prompt_embeds,
            **self.state
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set custom state values."""
        self.state.update(state)

    def reset(self) -> None:
        """Reset the state tracker to initial state."""
        self.current_timestep = None
        self.latents = None
        self.text_encoder_hidden_states = None
        self.text_encoder_2_hidden_states = None
        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.state = {}