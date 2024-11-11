import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import traceback
from torchvision import models, transforms
from transformers import (
    CLIPModel, 
    CLIPProcessor,
    DetrImageProcessor, 
    DetrForObjectDetection
)
from torch.nn import MLP
from typing import Dict, Any, List, Union
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIPFeatureExtractor:
    """
    Handles CLIP model operations for extracting semantic features from images and text.
    Used primarily for attribute binding and semantic relationship evaluation.
    """
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = None):
        logger.info(f"Initializing CLIP Feature Extractor with model: {model_name}")
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Initialize CLIP model and processor
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
            # Setup model
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully initialized CLIP model on {self.device}")
            
        except Exception as e:
            logger.error("Failed to initialize CLIP model")
            logger.error(traceback.format_exc())
            raise

    def extract_features(self, 
                        images: Union[torch.Tensor, List[Image.Image]], 
                        text: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Extract features from images and optionally text using CLIP.
        
        Args:
            images: Either a batch tensor [B,C,H,W] or list of PIL images
            text: Optional list of text prompts to encode
            
        Returns:
            Dictionary containing 'image_features' and optionally 'text_features'
        """
        logger.debug(f"Extracting features for batch of {len(images) if isinstance(images, list) else images.shape[0]} images")
        
        try:
            # Process inputs based on type
            if isinstance(images, (list, tuple)) and isinstance(images[0], Image.Image):
                inputs = self.processor(images=images, return_tensors="pt")
            else:
                inputs = {"pixel_values": images}
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                if text:
                    # Process both images and text
                    text_inputs = self.processor(
                        text=text, 
                        return_tensors="pt", 
                        padding=True
                    )
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    
                    outputs = self.model(**inputs, **text_inputs)
                    features = {
                        "image_features": outputs.image_embeds,
                        "text_features": outputs.text_embeds
                    }
                else:
                    # Process only images
                    outputs = self.model.get_image_features(**inputs)
                    features = {"image_features": outputs}
                
                logger.debug("Successfully extracted features")
                return features
                    
        except Exception as e:
            logger.error("Error during feature extraction")
            logger.error(traceback.format_exc())
            raise

class DETRObjectDetector:
    """
    Handles object detection using DETR model.
    Used for spatial relationship evaluation and object localization.
    """
    def __init__(self, model_name: str = "facebook/detr-resnet-50", device: str = None):
        logger.info(f"Initializing DETR Object Detector with model: {model_name}")
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Initialize DETR model and processor
            self.model = DetrForObjectDetection.from_pretrained(model_name)
            self.processor = DetrImageProcessor.from_pretrained(model_name)
            
            # Setup model
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully initialized DETR model on {self.device}")
            
        except Exception as e:
            logger.error("Failed to initialize DETR model")
            logger.error(traceback.format_exc())
            raise

    def detect_objects(self, 
                      images: List[Image.Image], 
                      confidence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Detect objects in a batch of images.
        
        Args:
            images: List of PIL images
            confidence_threshold: Minimum confidence score for detections
            
        Returns:
            List of dictionaries containing detection results for each image
        """
        logger.debug(f"Detecting objects in {len(images)} images")
        
        try:
            # Prepare inputs
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run detection
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Post-process outputs
            target_sizes = torch.tensor([image.size[::-1] for image in images])
            results = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes.to(self.device),
                threshold=confidence_threshold
            )
            
            logger.debug(f"Successfully detected objects with threshold {confidence_threshold}")
            return results
            
        except Exception as e:
            logger.error("Error during object detection")
            logger.error(traceback.format_exc())
            raise

class ModelManager:
    """
    Central manager for all models and preprocessing operations.
    Provides unified access to CLIP and DETR functionality.
    """
    def __init__(self, device: str = None):
        logger.info("Initializing Model Manager")
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Initialize models
            self.clip = CLIPFeatureExtractor(device=self.device)
            self.detr = DETRObjectDetector(device=self.device)
            
            # Define common preprocessing pipeline
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info("Successfully initialized Model Manager")
            
        except Exception as e:
            logger.error("Failed to initialize Model Manager")
            logger.error(traceback.format_exc())
            raise
    
    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply standard preprocessing to image tensors
        
        Args:
            images: Input image tensor [B,C,H,W]
            
        Returns:
            Preprocessed image tensor
        """
        try:
            processed = self.preprocess(images)
            logger.debug(f"Preprocessed images of shape {processed.shape}")
            return processed
        except Exception as e:
            logger.error("Error during image preprocessing")
            logger.error(traceback.format_exc())
            raise

# Initialize global model manager
try:
    logger.info("Initializing global model manager")
    model_manager = ModelManager()
    logger.info("Global model manager initialized successfully")
except Exception as e:
    logger.error("Failed to initialize global model manager")
    logger.error(traceback.format_exc())
    raise

def get_model_manager() -> ModelManager:
    """Helper function to access the global model manager"""
    return model_manager

class BaseRewardModel(nn.Module):
    """
    Base class for all reward models implementing common MLP architecture
    
    Architecture:
    - 3-layer MLP with ReLU activations
    - Progressive dimension reduction (feature_dim -> hidden_dim -> hidden_dim/2 -> 1)
    - Final squeeze to remove singleton dimension
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 768):
        super().__init__()
        logger.debug(f"Initializing BaseRewardModel with dims: feature={feature_dim}, hidden={hidden_dim}")
        
        try:
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            logger.debug("Successfully created MLP architecture")
            
        except Exception as e:
            logger.error("Failed to initialize BaseRewardModel")
            logger.error(f"Input dimensions: feature_dim={feature_dim}, hidden_dim={hidden_dim}")
            logger.error(traceback.format_exc())
            raise
    
    def forward(self, features):
        """Forward pass through MLP network"""
        try:
            # Log input statistics for debugging
            logger.debug(f"Forward pass input shape: {features.shape}")
            logger.debug(f"Input stats - Mean: {features.mean():.4f}, Std: {features.std():.4f}")
            
            output = self.mlp(features).squeeze(-1)
            
            # Log output statistics
            logger.debug(f"Output shape: {output.shape}")
            logger.debug(f"Output stats - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
            
            return output
            
        except Exception as e:
            logger.error("Error in BaseRewardModel forward pass")
            logger.error(f"Input tensor shape: {features.shape}")
            logger.error(traceback.format_exc())
            raise

class AttributeBindingRewardModel(BaseRewardModel):
    """
    Reward model for attribute binding and semantic relationships
    Uses CLIP features to evaluate semantic consistency between image and text
    
    Input features:
    - CLIP image embeddings (dim=768)
    - CLIP text embeddings (dim=768)
    Total feature dim = 768 * 2 = 1536
    """
    def __init__(self, clip_feature_dim: int = 768):
        logger.info(f"Initializing AttributeBindingRewardModel with CLIP dim: {clip_feature_dim}")
        super().__init__(feature_dim=clip_feature_dim * 2)
        
    def forward(self, clip_features, text_embeddings):
        try:
            # Validate input dimensions
            if clip_features.shape[-1] != text_embeddings.shape[-1]:
                raise ValueError(
                    f"Mismatched feature dimensions: CLIP={clip_features.shape}, text={text_embeddings.shape}"
                )
            
            # Log feature statistics
            logger.debug(f"CLIP features - Mean: {clip_features.mean():.4f}, Std: {clip_features.std():.4f}")
            logger.debug(f"Text features - Mean: {text_embeddings.mean():.4f}, Std: {text_embeddings.std():.4f}")
            
            # Combine features
            combined = torch.cat([clip_features, text_embeddings], dim=-1)
            logger.debug(f"Combined feature shape: {combined.shape}")
            
            return super().forward(combined)
            
        except Exception as e:
            logger.error("Error in AttributeBindingRewardModel forward pass")
            logger.error(f"CLIP features shape: {clip_features.shape}")
            logger.error(f"Text embeddings shape: {text_embeddings.shape}")
            logger.error(traceback.format_exc())
            raise

class SpatialRewardModel(BaseRewardModel):
    """
    Reward model for spatial relationships between objects
    Uses DETR object detection features to evaluate spatial composition
    
    Input features:
    - DETR spatial features (dim=256)
    - Text embeddings (dim=768)
    Total feature dim = 256 + 768 = 1024
    """
    def __init__(self, detr_feature_dim: int = 256):
        logger.info(f"Initializing SpatialRewardModel with DETR dim: {detr_feature_dim}")
        super().__init__(feature_dim=detr_feature_dim + 768)
        
    def forward(self, object_detections, text_embeddings):
        try:
            # Extract spatial features from detections
            spatial_features = self._extract_spatial_features(object_detections)
            logger.debug(f"Extracted spatial features shape: {spatial_features.shape}")
            
            # Combine with text embeddings
            combined = torch.cat([spatial_features, text_embeddings], dim=-1)
            logger.debug(f"Combined feature shape: {combined.shape}")
            
            return super().forward(combined)
            
        except Exception as e:
            logger.error("Error in SpatialRewardModel forward pass")
            logger.error(f"Object detections: {[d.keys() for d in object_detections]}")
            logger.error(f"Text embeddings shape: {text_embeddings.shape}")
            logger.error(traceback.format_exc())
            raise
    
    def _extract_spatial_features(self, detections):
        """Convert DETR detections into spatial relationship features"""
        try:
            features = []
            for detection in detections:
                # Validate detection dictionary
                required_keys = ["boxes", "labels", "scores"]
                if not all(k in detection for k in required_keys):
                    raise KeyError(f"Missing required keys in detection. Found: {detection.keys()}")
                
                boxes = detection["boxes"]
                labels = detection["labels"]
                scores = detection["scores"]
                
                logger.debug(f"Processing detection with {len(boxes)} objects")
                logger.debug(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                
                # Calculate spatial relationships
                spatial_encoding = self._encode_spatial_relationships(boxes, labels, scores)
                features.append(spatial_encoding)
                
            return torch.stack(features)
            
        except Exception as e:
            logger.error("Error extracting spatial features")
            logger.error(f"Detection format: {type(detections)}")
            logger.error(traceback.format_exc())
            raise
    
    def _encode_spatial_relationships(self, boxes, labels, scores):
        """Encode relative positions and relationships between objects"""
        try:
            num_objects = len(boxes)
            spatial_features = torch.zeros(256, device=boxes.device)
            
            if num_objects > 0:
                logger.debug(f"Encoding relationships between {num_objects} objects")
                
                # Calculate pairwise relationships
                for i in range(num_objects):
                    for j in range(i + 1, num_objects):
                        # Get spatial metrics
                        rel_pos = self._get_relative_position(boxes[i], boxes[j])
                        rel_size = self._get_relative_size(boxes[i], boxes[j])
                        
                        # Encode into feature vector
                        idx = (i * num_objects + j) % 256
                        confidence = scores[i] * scores[j]
                        spatial_features[idx] = rel_pos * confidence
                        
                        logger.debug(
                            f"Object pair ({i},{j}) - "
                            f"RelPos: {rel_pos:.4f}, "
                            f"RelSize: {rel_size:.4f}, "
                            f"Confidence: {confidence:.4f}"
                        )
                        
            return spatial_features
            
        except Exception as e:
            logger.error("Error encoding spatial relationships")
            logger.error(f"Boxes shape: {boxes.shape}")
            logger.error(f"Labels shape: {labels.shape}")
            logger.error(f"Scores shape: {scores.shape}")
            logger.error(traceback.format_exc())
            raise

class NonSpatialRewardModel(BaseRewardModel):
    """Reward model for non-spatial compositional aspects"""
    def __init__(self, clip_dim: int = 768, detr_dim: int = 256):
        super().__init__(feature_dim=clip_dim + detr_dim + 768)  # CLIP + DETR + text
        
    def forward(self, clip_features, object_detections, text_embeddings):
        # Combine all features
        detr_features = self._process_detections(object_detections)
        combined = torch.cat([clip_features, detr_features, text_embeddings], dim=-1)
        return super().forward(combined)
    
    def _process_detections(self, detections):
        # Convert DETR detections into fixed-size feature vector
        features = []
        for detection in detections:
            # Aggregate detection features
            scores = detection["scores"]
            labels = detection["labels"]
            
            # Create fixed-size encoding (simplified)
            feature_vec = torch.zeros(256, device=scores.device)
            for score, label in zip(scores, labels):
                idx = label.item() % 256
                feature_vec[idx] = max(feature_vec[idx], score)
                
            features.append(feature_vec)
            
        return torch.stack(features)

def collect_preferences(reward_type: str, 
                       train_dataloader: torch.utils.data.DataLoader,
                       model_manager: ModelManager,
                       args: Any) -> List[Dict[str, torch.Tensor]]:
    """
    Collect preference pairs for reward model training
    
    Process:
    1. Extract features from winning/losing image pairs
    2. Use appropriate feature extractor based on reward type
    3. Combine with text embeddings
    4. Return list of preference pairs for training
    
    Args:
        reward_type: Type of reward model ("attribute", "spatial", "non_spatial")
        train_dataloader: Training data loader
        model_manager: Model manager instance
        args: Training arguments
        
    Returns:
        List of preference pairs with features
    """
    preferences = []
    logger.info(f"Collecting preferences for {reward_type} reward model")
    
    try:
        for batch_idx, batch in enumerate(train_dataloader):
            logger.debug(f"Processing batch {batch_idx}")
            
            # Validate batch contents
            required_keys = ["winning_images", "losing_images", "prompts"]
            if not all(k in batch for k in required_keys):
                raise KeyError(f"Missing required keys in batch. Found: {batch.keys()}")
            
            # Get winning and losing images
            winning_images = batch["winning_images"].to(args.device)
            losing_images = batch["losing_images"].to(args.device)
            prompts = batch["prompts"]
            
            logger.debug(
                f"Batch stats - "
                f"Winning shape: {winning_images.shape}, "
                f"Losing shape: {losing_images.shape}, "
                f"Prompts: {len(prompts)}"
            )
            
            # Extract features based on reward type
            if reward_type == "attribute":
                logger.debug("Extracting CLIP features")
                winning_features = model_manager.clip.extract_features(
                    winning_images, text=prompts
                )["image_features"]
                losing_features = model_manager.clip.extract_features(
                    losing_images, text=prompts
                )["image_features"]
                
            elif reward_type == "spatial":
                logger.debug("Extracting DETR features")
                winning_features = model_manager.detr.detect_objects(winning_images)
                losing_features = model_manager.detr.detect_objects(losing_images)
                
            else:  # non_spatial
                logger.debug("Extracting both CLIP and DETR features")
                winning_clip = model_manager.clip.extract_features(winning_images)["image_features"]
                winning_detr = model_manager.detr.detect_objects(winning_images)
                losing_clip = model_manager.clip.extract_features(losing_images)["image_features"]
                losing_detr = model_manager.detr.detect_objects(losing_images)
                
                winning_features = (winning_clip, winning_detr)
                losing_features = (losing_clip, losing_detr)
            
            # Add to preferences list
            preferences.append({
                "winning_features": winning_features,
                "losing_features": losing_features,
                "text_embeddings": batch["text_embeddings"].to(args.device)
            })
            
            logger.debug(f"Successfully processed batch {batch_idx}")
            
        logger.info(f"Collected {len(preferences)} preference pairs")
        return preferences
        
    except Exception as e:
        logger.error("Error collecting preferences")
        logger.error(f"Current batch index: {batch_idx if 'batch_idx' in locals() else 'N/A'}")
        logger.error(f"Reward type: {reward_type}")
        logger.error(traceback.format_exc())
        raise

def train_reward_model(reward_model: nn.Module,
                      preference_pairs: List[Dict[str, torch.Tensor]],
                      args: Any) -> float:
    """
    Train reward model on collected preferences using preference learning
    
    Training process:
    1. For each preference pair:
        - Calculate rewards for winning and losing examples
        - Compute preference loss using sigmoid
        - Update model weights
    2. Return average loss across all pairs
    
    Args:
        reward_model: Reward model to train
        preference_pairs: List of preference pairs with features
        args: Training arguments
        
    Returns:
        Average training loss
    """
    logger.info(f"Training reward model of type {type(reward_model).__name__}")
    
    try:
        reward_model.train()
        optimizer = torch.optim.AdamW(
            reward_model.parameters(), 
            lr=args.reward_lr,
            weight_decay=args.reward_weight_decay
        )
        
        total_loss = 0
        num_batches = 0
        
        for pair_idx, pair in enumerate(preference_pairs):
            logger.debug(f"Processing preference pair {pair_idx}")
            
            # Get features
            winning_features = pair["winning_features"]
            losing_features = pair["losing_features"]
            text_embeddings = pair["text_embeddings"]
            
            # Calculate rewards based on model type
            if isinstance(reward_model, AttributeBindingRewardModel):
                winning_reward = reward_model(winning_features, text_embeddings)
                losing_reward = reward_model(losing_features, text_embeddings)
                
            elif isinstance(reward_model, SpatialRewardModel):
                winning_reward = reward_model(winning_features, text_embeddings)
                losing_reward = reward_model(losing_features, text_embeddings)
                
            else:  # NonSpatialRewardModel
                winning_clip, winning_detr = winning_features
                losing_clip, losing_detr = losing_features
                winning_reward = reward_model(winning_clip, winning_detr, text_embeddings)
                losing_reward = reward_model(losing_clip, losing_detr, text_embeddings)
            
            # Log reward statistics
            logger.debug(
                f"Rewards - "
                f"Winning: {winning_reward.mean():.4f} ± {winning_reward.std():.4f}, "
                f"Losing: {losing_reward.mean():.4f} ± {losing_reward.std():.4f}"
            )
            
            # Compute preference loss
            loss = -torch.log(torch.sigmoid(winning_reward - losing_reward)).mean()
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Log gradients
            if args.log_gradients:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    reward_model.parameters(), 
                    args.max_grad_norm
                )
                logger.debug(f"Gradient norm: {grad_norm:.4f}")
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            logger.debug(f"Pair {pair_idx} - Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        logger.info(f"Training completed - Average loss: {avg_loss:.6f}")
        
        reward_model.eval()
        return avg_loss
        
    except Exception as e:
        logger.error("Error training reward model")
        logger.error(f"Model type: {type(reward_model).__name__}")
        logger.error(f"Number of preference pairs: {len(preference_pairs)}")
        logger.error(traceback.format_exc())
        raise