# src/data/processors/text_processor.py
import torch
from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from pathlib import Path

from src.data.processors.utils.caption.text_embedder import TextEmbedder
from src.data.processors.utils.caption.tag_weighter import parse_tags, TagWeighter
from src.data.processors.utils.system_utils import get_optimal_workers, create_thread_pool

logger = logging.getLogger(__name__)

class TextProcessor:
    """Process text data with tag weighting and text embeddings."""
    
    def __init__(
        self,
        text_embedder: TextEmbedder,
        tag_weighter: Optional[TagWeighter] = None,
        num_workers: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        """Initialize text processor.
        
        Args:
            text_embedder: Text embedding model
            tag_weighter: Optional tag weighting model
            num_workers: Number of worker threads
            device: Torch device for processing
        """
        self.text_embedder = text_embedder
        self.tag_weighter = tag_weighter
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize thread pool for parallel processing
        self.num_workers = num_workers or get_optimal_workers()
        self.executor = create_thread_pool(self.num_workers)
        
        logger.info(
            f"Initialized TextProcessor:\n"
            f"- Device: {self.device}\n"
            f"- Workers: {self.num_workers}\n"
            f"- Tag Weighter: {'Yes' if tag_weighter else 'No'}"
        )

    async def process_text(self, text: str) -> Tuple[Dict[str, Any], List[str]]:
        """Process text and extract tags with optional weighting.
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple of (text_data, tags)
        """
        try:
            # Parse tags
            tags = parse_tags(text)
            
            # Update tag frequencies if weighter is available
            if self.tag_weighter is not None:
                await asyncio.to_thread(
                    self.tag_weighter.update_frequencies,
                    text
                )
            
            # Process text embeddings
            text_data = await self.text_embedder.process_text(text, tags)
            
            # Add tag weights if available
            if self.tag_weighter is not None:
                text_data['tag_weights'] = await asyncio.to_thread(
                    self.tag_weighter.get_weight,
                    tags
                )
            
            return text_data, tags
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)[:200]}...")
            return None, []

    async def process_text_file(self, text_path: Path) -> Optional[Tuple[Dict[str, Any], List[str]]]:
        """Process text from a file.
        
        Args:
            text_path: Path to text file
            
        Returns:
            Tuple of (text_data, tags) if successful, None otherwise
        """
        try:
            if not text_path.exists():
                logger.error(f"Text file not found: {text_path}")
                return None
                
            # Read text file
            text = await asyncio.to_thread(
                lambda: text_path.read_text(encoding='utf-8')
            )
            
            # Process text and tags
            return await self.process_text(text)
            
        except Exception as e:
            logger.error(f"Error processing text file {text_path}: {str(e)[:200]}...")
            return None

    async def process_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Tuple[Dict[str, Any], List[str]]]:
        """Process a batch of texts in parallel.
        
        Args:
            texts: List of texts to process
            batch_size: Size of processing batches
            
        Returns:
            List of (text_data, tags) tuples
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process batch items in parallel
            tasks = [self.process_text(text) for text in batch]
            batch_results = await asyncio.gather(*tasks)
            
            # Filter out failed results
            results.extend([r for r in batch_results if r[0] is not None])
            
        return results

    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clean up thread pool
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            
            # Clean up text embedder
            if hasattr(self.text_embedder, 'cleanup'):
                await self.text_embedder.cleanup()
            
            # Clean up tag weighter
            if self.tag_weighter and hasattr(self.tag_weighter, 'cleanup'):
                await self.tag_weighter.cleanup()
            
            logger.info("Successfully cleaned up text processor resources")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")