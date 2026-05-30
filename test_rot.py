import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import math

def test_rotation():
    # Create a 100x100 black image with a white dot at (80, 50)
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([78, 48, 82, 52], fill=(255, 255, 255))
    
    # Center is (50, 50). Point is at (80, 50) -> x=30, y=0 relative to center
    cx, cy = 50, 50
    pt = [80, 50]
    
    # Rotate 90 degrees CCW
    angle = 90
    new_img = F.affine(img, angle=angle, translate=[0, 0], scale=1.0, shear=0, center=[cx, cy])
    
    # Save the rotated image to see where the dot went
    new_img.save("rotated_img.jpg")
    
    # Now use the formula from the code
    x = pt[0] - cx
    y = pt[1] - cy
    
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    # Code's formula
    rx_code = x * cos_a - y * sin_a
    ry_code = x * sin_a + y * cos_a
    nx_code = rx_code + cx
    ny_code = ry_code + cy
    print(f"Code's formula point: ({nx_code:.1f}, {ny_code:.1f})")
    
    # Correct math for Y-down coordinate system counter-clockwise
    rx_correct = x * cos_a + y * sin_a
    ry_correct = -x * sin_a + y * cos_a
    nx_correct = rx_correct + cx
    ny_correct = ry_correct + cy
    print(f"Correct math point: ({nx_correct:.1f}, {ny_correct:.1f})")

if __name__ == "__main__":
    test_rotation()
