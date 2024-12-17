from pathlib import Path
import os
import shutil
from typing import List, Set, Dict, Optional, Tuple, Generator
import logging
import gc
from contextlib import contextmanager
import tempfile
import time

logger = logging.getLogger(__name__)

@contextmanager
def temp_file_handler(suffix: str = '.tmp') -> Generator[Path, None, None]:
    """Context manager for handling temporary files with automatic cleanup."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            yield temp_path
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_path}: {e}")

def ensure_dir(path: str) -> Path:
    """Ensure directory exists and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_file_size(path: str) -> int:
    """Get file size in bytes safely."""
    try:
        return Path(path).stat().st_size
    except Exception:
        return 0

def find_matching_files(
    directory: str,
    extensions: Set[str],
    recursive: bool = True,
    require_text_pair: bool = False,
    batch_size: int = 1000
) -> Generator[str, None, None]:
    """Find files with given extensions, optionally requiring text pairs.
    Returns a generator to avoid loading all paths into memory at once."""
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return
        
    # Get all text files first if needed
    text_files = set()
    if require_text_pair:
        try:
            text_files = {
                os.path.splitext(f)[0]
                for f in os.listdir(directory)
                if f.endswith('.txt')
            }
        except Exception as e:
            logger.error(f"Error reading text files: {e}")
            return
    
    # Find matching files in batches
    pattern = "**/*" if recursive else "*"
    batch = []
    
    for ext in extensions:
        try:
            for file_path in directory.glob(f"{pattern}{ext}"):
                if not require_text_pair or os.path.splitext(file_path.name)[0] in text_files:
                    batch.append(str(file_path))
                    if len(batch) >= batch_size:
                        yield from batch
                        batch.clear()
                        gc.collect()  # Help clean up path strings
                        
            # Yield remaining files
            if batch:
                yield from batch
                batch.clear()
                
        except Exception as e:
            logger.error(f"Error processing extension {ext}: {e}")
            
    # Clear text files set
    text_files.clear()
    gc.collect()

def safe_file_write(path: str, data: bytes) -> bool:
    """Write file atomically using temporary file with proper cleanup."""
    path = Path(path)
    
    with temp_file_handler(suffix='.tmp') as tmp_path:
        try:
            # Write to temporary file
            with open(tmp_path, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
                
            # Atomic rename
            os.replace(tmp_path, path)
            return True
            
        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            return False

def get_cache_paths(
    base_path: str,
    cache_dir: str,
    suffixes: Optional[Dict[str, str]] = None
) -> Dict[str, Path]:
    """Get cache file paths for a base file."""
    if suffixes is None:
        suffixes = {
            'latent': '.pt',
            'text': '.pt'
        }
        
    base_name = Path(base_path).stem
    cache_dir = ensure_dir(cache_dir)
    
    return {
        name: cache_dir / f"{base_name}{suffix}"
        for name, suffix in suffixes.items()
    }

def cleanup_temp_files(
    directory: str,
    pattern: str = "*.tmp",
    min_age_seconds: int = 3600  # 1 hour
) -> int:
    """Clean up temporary files in directory that are older than min_age_seconds."""
    directory = Path(directory)
    count = 0
    current_time = time.time()
    
    try:
        for tmp_file in directory.glob(pattern):
            try:
                # Check file age
                if current_time - tmp_file.stat().st_mtime > min_age_seconds:
                    tmp_file.unlink()
                    count += 1
                    
                    # Periodic GC to prevent memory buildup during large cleanups
                    if count % 1000 == 0:
                        gc.collect()
                        
            except Exception as e:
                logger.error(f"Error deleting {tmp_file}: {e}")
                
    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")
        
    finally:
        # Final cleanup
        gc.collect()
            
    return count

def validate_image_text_pair(
    img_path: str,
    txt_path: Optional[str] = None,
    min_text_size: int = 1,
    max_text_size: int = 100000,  # 100KB max text size
    required_text_encoding: str = 'utf-8',
    chunk_size: int = 8192  # Read text in chunks
) -> Tuple[bool, str]:
    """Validate that an image-text pair exists and is valid.
    
    Args:
        img_path: Path to image file
        txt_path: Optional path to text file (if None, uses img_path with .txt extension)
        min_text_size: Minimum text file size in bytes
        max_text_size: Maximum text file size in bytes
        required_text_encoding: Required text file encoding
        chunk_size: Size of chunks to read text file in
        
    Returns:
        Tuple of (is_valid, reason)
        - is_valid: Whether the pair is valid
        - reason: Reason for invalidity if not valid
    """
    img_path = Path(img_path)
    
    # Check image file exists
    if not img_path.exists():
        return False, "Image file not found"
        
    # Get text path
    if txt_path is None:
        txt_path = img_path.with_suffix('.txt')
    else:
        txt_path = Path(txt_path)
        
    # Check text file exists
    if not txt_path.exists():
        return False, "Text file not found"
        
    # Check text file size
    try:
        text_size = txt_path.stat().st_size
        if text_size < min_text_size:
            return False, "Text file empty"
        if text_size > max_text_size:
            return False, "Text file too large"
    except Exception as e:
        return False, f"Error checking text file size: {str(e)}"
        
    # Check text file can be read with correct encoding
    try:
        has_content = False
        with open(txt_path, 'r', encoding=required_text_encoding) as f:
            # Read in chunks to avoid loading entire file
            while chunk := f.read(chunk_size):
                if chunk.strip():
                    has_content = True
                    break
                    
        if not has_content:
            return False, "Text file contains only whitespace"
            
    except UnicodeDecodeError:
        return False, f"Text file not in {required_text_encoding} encoding"
    except Exception as e:
        return False, f"Error reading text file: {str(e)}"
        
    return True, "Valid pair"