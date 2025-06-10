import logging
from typing import Dict, List, Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field
from StreamDiffusionWrapper import StreamDiffusionWrapper

from .interface import Pipeline


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
    
    # ControlNet specific parameters
    controlnet_model: str = "thibaud/controlnet-sd21-depth-diffusers"
    controlnet_preprocessor: str = "depth_tensorrt"
    controlnet_strength: float = 0.5
    negative_prompt: str = "blurry, low quality, flat, 2d"
    
    # TensorRT engine path for depth preprocessing
    tensorrt_engine_path: Optional[str] = None
    

class ControlNetStreamDiffusion(Pipeline):
    """
    ControlNet-enabled StreamDiffusion pipeline for ai-runner.
    
    Extends the base StreamDiffusion implementation with ControlNet conditioning.
    Follows ai-runner patterns exactly for seamless integration.
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
        
        # Create ControlNet configuration
        controlnet_config = self._create_controlnet_config(new_params)
        
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
            # ControlNet parameters
            use_controlnet=True,
            controlnet_config=controlnet_config,
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
        
        logging.info("ControlNetStreamDiffusion: Pipeline ready for inference")

    def _create_controlnet_config(self, params: ControlNetStreamDiffusionParams) -> dict:
        """Create ControlNet configuration dictionary"""
        config = {
            'model_id': params.controlnet_model,
            'preprocessor': params.controlnet_preprocessor,
            'conditioning_scale': params.controlnet_strength,
            'enabled': True,
            'pipeline_type': 'sdturbo',  # Hardcoded for SD Turbo optimization
            'control_guidance_start': 0.0,
            'control_guidance_end': 1.0,
        }
        
        # Add TensorRT engine path for depth preprocessing if provided
        if params.tensorrt_engine_path and params.controlnet_preprocessor == "depth_tensorrt":
            config['preprocessor_params'] = {
                'engine_path': params.tensorrt_engine_path,
                'detect_resolution': 518,
                'image_resolution': 512
            }
        
        return config
    
    def get_health(self) -> dict:
        """Health check method following ai-runner patterns"""
        return {
            "status": "OK" if self.pipe else "LOADING",
            "pipeline": "controlnet_streamdiffusion",
            "model_id": self.params.model_id if hasattr(self, 'params') else "unknown"
        }
    
    def __str__(self) -> str:
        return f"ControlNetStreamDiffusion model_id={getattr(self.params, 'model_id', 'unknown')}" 