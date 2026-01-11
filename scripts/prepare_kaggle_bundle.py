
import os
import zipfile
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def create_kaggle_bundle():
    """Create a zip bundle for Kaggle/Colab training."""
    
    project_root = Path(__file__).resolve().parents[1]
    output_zip = project_root / 'urban_drainage_kaggle_bundle.zip'
    
    logger.info(f"Creating Kaggle bundle at: {output_zip}")
    
    # Directories to include
    dirs_to_include = [
        'src',
        'scripts',
        'config',
        'outputs'  # CRITICAL: Contains the training data (GeoJSONs)
    ]
    
    # Files to include (root level)
    files_to_include = [
        'requirements.txt',
        'README.md',
    ]
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add directories
        for dir_name in dirs_to_include:
            dir_path = project_root / dir_name
            if not dir_path.exists():
                logger.warning(f"Warning: Directory not found: {dir_path}")
                continue
                
            logger.info(f"  Adding directory: {dir_name}/")
            for root, _, files in os.walk(dir_path):
                for file in files:
                    # Skip __pycache__ and other junk
                    if '__pycache__' in root or file.endswith('.pyc') or file.endswith('.zip'):
                        continue
                        
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(project_root)
                    zipf.write(file_path, arcname)
        
        # Add root files
        for file_name in files_to_include:
            file_path = project_root / file_name
            if file_path.exists():
                logger.info(f"  Adding file: {file_name}")
                zipf.write(file_path, file_name)
    
    logger.info("\n✅ Bundle created successfully!")
    logger.info(f"Size: {output_zip.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info("\nINSTRUCTIONS FOR KAGGLE/COLAB:")
    logger.info("1. Upload 'urban_drainage_kaggle_bundle.zip' to your notebook environment.")
    logger.info("2. Run the extraction and training script (see KAGGLE_INSTRUCTIONS.md).")

if __name__ == '__main__':
    create_kaggle_bundle()
