from pathlib import Path
import os
import shutil
from typing import List, Set, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

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
    require_text_pair: bool = False
) -> List[str]:
    """Find files with given extensions, optionally requiring text pairs."""
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
        
    # Get all text files first if needed
    text_files = set()
    if require_text_pair:
        text_files = {
            os.path.splitext(f)[0]
            for f in os.listdir(directory)
            if f.endswith('.txt')
        }
    
    # Find matching files
    matched_files = []
    pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        for file_path in directory.glob(f"{pattern}{ext}"):
            if not require_text_pair or os.path.splitext(file_path.name)[0] in text_files:
                matched_files.append(str(file_path))
                
    return matched_files

def safe_file_write(path: str, data: bytes) -> bool:
    """Write file atomically using temporary file."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    
    try:
        # Write to temporary file
        with open(tmp_path, 'wb') as f:
            f.write(data)
            
        # Atomic rename
        os.replace(tmp_path, path)
        return True
        
    except Exception as e:
        logger.error(f"Error writing file {path}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
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

def cleanup_temp_files(directory: str, pattern: str = "*.tmp") -> int:
    """Clean up temporary files in directory."""
    directory = Path(directory)
    count = 0
    
    for tmp_file in directory.glob(pattern):
        try:
            tmp_file.unlink()
            count += 1
        except Exception as e:
            logger.error(f"Error deleting {tmp_file}: {e}")
            
    return count

def validate_image_text_pair(
    img_path: str,
    txt_path: Optional[str] = None,
    min_text_size: int = 1,
    max_text_size: int = 100000,  # 100KB max text size
    required_text_encoding: str = 'utf-8'
) -> Tuple[bool, str]:
    """Validate that an image-text pair exists and is valid.
    
    Args:
        img_path: Path to image file
        txt_path: Optional path to text file (if None, uses img_path with .txt extension)
        min_text_size: Minimum text file size in bytes
        max_text_size: Maximum text file size in bytes
        required_text_encoding: Required text file encoding
        
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
    text_size = txt_path.stat().st_size
    if text_size < min_text_size:
        return False, "Text file empty"
    if text_size > max_text_size:
        return False, "Text file too large"
        
    # Check text file can be read with correct encoding
    try:
        with open(txt_path, 'r', encoding=required_text_encoding) as f:
            text = f.read().strip()
            if not text:
                return False, "Text file contains only whitespace"
    except UnicodeDecodeError:
        return False, f"Text file not in {required_text_encoding} encoding"
    except Exception as e:
        return False, f"Error reading text file: {str(e)}"
        
    return True, "Valid pair"