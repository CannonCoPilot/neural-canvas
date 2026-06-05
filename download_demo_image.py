import requests
import os

def download_demo_image():
    """Download the demo image for the image-upscaling API demo."""
    url = "https://mdl.artvee.com/sftb/516847ld.jpg"
    output_path = "test_image.jpg"
    
    try:
        # Disable SSL verification due to self-signed certificate issue
        response = requests.get(url, verify=False)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded the demo image to {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the demo image: {e}")

if __name__ == "__main__":
    download_demo_image()