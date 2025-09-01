import requests
import json
from pathlib import Path
import time

def test_endpoint(url, image_path):
    """Test an endpoint with a single image"""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status code: {response.status_code}", "detail": response.text}
    except Exception as e:
        return {"error": str(e)}

def main():
    # API endpoints
    base_url = "http://localhost:8000"
    detect_url = f"{base_url}/api/detect"
    metadata_url = f"{base_url}/api/metadata"
    
    # Test images directory (now in the same directory as the script)
    test_dir = Path(__file__).parent
    
    # Test both real and fake images
    for category in ['real', 'fake']:
        print(f"\n{'='*50}")
        print(f"Testing {category.upper()} images:")
        print(f"{'='*50}")
        
        category_dir = test_dir / category
        for image_path in category_dir.glob('*'):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                print(f"\nTesting image: {image_path.name}")
                
                # Test detection endpoint
                print("\nDetection Results:")
                detect_result = test_endpoint(detect_url, image_path)
                print(json.dumps(detect_result, indent=2))
                
                # Test metadata endpoint
                print("\nMetadata Results:")
                metadata_result = test_endpoint(metadata_url, image_path)
                print(json.dumps(metadata_result, indent=2))
                
                # Add a small delay between requests
                time.sleep(1)

if __name__ == "__main__":
    main() 