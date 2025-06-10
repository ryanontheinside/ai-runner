import logging
from typing import Dict, List, Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field
from StreamDiffusionWrapper import StreamDiffusionWrapper

from .interface import Pipeline


class ControlNetConfig(BaseModel):
    """Configuration for a single ControlNet"""
    model_id: str = "thibaud/controlnet-sd21-depth-diffusers"
    preprocessor: str = "depth_tensorrt"
    conditioning_scale: float = 0.5
    enabled: bool = True
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    preprocessor_params: Optional[Dict] = None


class ControlNetStreamDiffusionParams(BaseModel):
    class Config:
        extra = "forbid"

    # Base StreamDiffusion parameters
    prompt: str = "an anime render of a girl with purple hair, masterpiece"
    model_id: str = "stabilityai/sd-turbo"
    lora_dict: Optional[Dict[str, float]] = None
    use_lcm_lora: bool = True
    lcm_lora_id: str = "latent-consistency/lcm-lora-sdv1-5"
    num_inference_steps: int = 50
    t_index_list: Optional[List[int]] = [0, 16]  # SD Turbo optimized
    scale: float = 1.0
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt"
    use_denoising_batch: bool = True
    enable_similar_image_filter: bool = False
    seed: int = 789
    guidance_scale: float = 1.1
    do_add_noise: bool = True
    similar_image_filter_threshold: float = 0.98
    negative_prompt: str = "blurry, low quality, flat, 2d"
    
    # Multiple ControlNets configuration
    controlnets: List[ControlNetConfig] = [
        ControlNetConfig(
            model_id="thibaud/controlnet-sd21-depth-diffusers",
            preprocessor="depth_tensorrt",
            conditioning_scale=0.5,
            enabled=True
        )
    ]
    
    # Pipeline type for ControlNet patching
    pipeline_type: str = "sdturbo"
    

class ControlNetStreamDiffusion(Pipeline):
    """
    Multi-ControlNet StreamDiffusion pipeline for ai-runner.
    
    Supports multiple ControlNets with independent strength controls.
    Uses list of dicts configuration pattern to match updated StreamDiffusion examples.
    """
    
    def __init__(self, **params):
        super().__init__(**params)
        self.pipe: Optional[StreamDiffusionWrapper] = None
        self.first_frame = True
        self.update_params(**params)

    def process_frame(self, image: Image.Image) -> Image.Image:
        """Process a single frame through the ControlNet StreamDiffusion pipeline"""
        img_tensor = self.pipe.preprocess_image(image)
        img_tensor = self.pipe.stream.image_processor.denormalize(img_tensor)

        if self.first_frame:
            self.first_frame = False
            # Warmup the pipeline with multiple iterations
            for _ in range(self.pipe.batch_size):
                self.pipe(image=img_tensor)

        return self.pipe(image=img_tensor)

    def update_params(self, **params):
        """Update pipeline parameters, recreating pipeline if necessary"""
        new_params = ControlNetStreamDiffusionParams(**params)
        
        if self.pipe is not None:
            # Optimize: avoid resetting the pipe if only the prompt changed
            only_prompt = self.params.model_copy(update={"prompt": new_params.prompt})
            if new_params == only_prompt:
                logging.info(f"ControlNetStreamDiffusion: Updating prompt: {new_params.prompt}")
                self.pipe.stream.update_prompt(new_params.prompt)
                self.params = new_params
                return

        logging.info(f"ControlNetStreamDiffusion: Resetting pipeline for params change")
        logging.info(f"ControlNetStreamDiffusion: Configured with {len(new_params.controlnets)} ControlNets")
        
        # Create ControlNet configurations list
        controlnet_configs = self._create_controlnet_configs(new_params)
        
        # Initialize StreamDiffusionWrapper with ControlNet support
        pipe = StreamDiffusionWrapper(
            model_id_or_path=new_params.model_id,
            lora_dict=new_params.lora_dict,
            use_lcm_lora=new_params.use_lcm_lora,
            lcm_lora_id=new_params.lcm_lora_id,
            t_index_list=new_params.t_index_list,
            frame_buffer_size=1,  # Optimized for real-time
            width=512,
            height=512,
            warmup=10,
            acceleration=new_params.acceleration,
            do_add_noise=new_params.do_add_noise,
            mode="img2img",
            enable_similar_image_filter=new_params.enable_similar_image_filter,
            similar_image_filter_threshold=new_params.similar_image_filter_threshold,
            use_denoising_batch=new_params.use_denoising_batch,
            seed=new_params.seed,
            # Multi-ControlNet parameters
            use_controlnet=True,
            controlnet_config=controlnet_configs,  # Now expects a list
        )
        
        # Prepare the pipeline
        pipe.prepare(
            prompt=new_params.prompt,
            negative_prompt=new_params.negative_prompt,
            num_inference_steps=new_params.num_inference_steps,
            guidance_scale=new_params.guidance_scale,
        )

        self.params = new_params
        self.pipe = pipe
        self.first_frame = True
        
        logging.info("ControlNetStreamDiffusion: Multi-ControlNet pipeline ready for inference")

    def _create_controlnet_configs(self, params: ControlNetStreamDiffusionParams) -> List[dict]:
        """Create list of ControlNet configuration dictionaries"""
        configs = []
        
        for cn_config in params.controlnets:
            config = {
                'model_id': cn_config.model_id,
                'preprocessor': cn_config.preprocessor,
                'conditioning_scale': cn_config.conditioning_scale,
                'enabled': cn_config.enabled,
                'pipeline_type': params.pipeline_type,
                'control_guidance_start': cn_config.control_guidance_start,
                'control_guidance_end': cn_config.control_guidance_end,
            }
            
            # Add preprocessor params if provided
            if cn_config.preprocessor_params:
                config['preprocessor_params'] = cn_config.preprocessor_params
            
            configs.append(config)
            
            logging.info(f"ControlNetStreamDiffusion: Added ControlNet - {cn_config.model_id} ({cn_config.preprocessor}) strength={cn_config.conditioning_scale}")
        
        return configs
    
    def update_controlnet_strength(self, index: int, strength: float):
        """Update strength of a specific ControlNet by index"""
        if self.pipe and hasattr(self.pipe, 'update_controlnet_scale'):
            try:
                self.pipe.update_controlnet_scale(index, strength)
                logging.info(f"ControlNetStreamDiffusion: Updated ControlNet {index} strength to {strength}")
                
                # Update params to keep in sync
                if hasattr(self, 'params') and index < len(self.params.controlnets):
                    self.params.controlnets[index].conditioning_scale = strength
                    
            except Exception as e:
                logging.error(f"ControlNetStreamDiffusion: Failed to update ControlNet {index} strength: {e}")
    
    def toggle_controlnet(self, index: int, enabled: bool):
        """Enable/disable a specific ControlNet by index"""
        if self.pipe and hasattr(self.pipe, 'update_controlnet_scale'):
            try:
                # Set strength to configured value if enabling, 0 if disabling
                if hasattr(self, 'params') and index < len(self.params.controlnets):
                    strength = self.params.controlnets[index].conditioning_scale if enabled else 0.0
                    self.pipe.update_controlnet_scale(index, strength)
                    self.params.controlnets[index].enabled = enabled
                    
                    logging.info(f"ControlNetStreamDiffusion: {'Enabled' if enabled else 'Disabled'} ControlNet {index}")
                    
            except Exception as e:
                logging.error(f"ControlNetStreamDiffusion: Failed to toggle ControlNet {index}: {e}")
    
    def get_controlnet_info(self) -> List[Dict]:
        """Get information about all configured ControlNets"""
        if not hasattr(self, 'params'):
            return []
            
        info = []
        for i, cn_config in enumerate(self.params.controlnets):
            info.append({
                'index': i,
                'model_id': cn_config.model_id,
                'preprocessor': cn_config.preprocessor,
                'strength': cn_config.conditioning_scale,
                'enabled': cn_config.enabled,
                'guidance_start': cn_config.control_guidance_start,
                'guidance_end': cn_config.control_guidance_end,
            })
        return info
    
    def get_health(self) -> dict:
        """Health check method following ai-runner patterns"""
        health = {
            "status": "OK" if self.pipe else "LOADING",
            "pipeline": "controlnet_streamdiffusion",
            "model_id": self.params.model_id if hasattr(self, 'params') else "unknown"
        }
        
        if hasattr(self, 'params'):
            health["controlnets"] = len(self.params.controlnets)
            health["controlnet_info"] = self.get_controlnet_info()
            
        return health
    
    def __str__(self) -> str:
        if hasattr(self, 'params'):
            num_controlnets = len(self.params.controlnets)
            return f"ControlNetStreamDiffusion model_id={self.params.model_id} controlnets={num_controlnets}"
        return "ControlNetStreamDiffusion model_id=unknown" 