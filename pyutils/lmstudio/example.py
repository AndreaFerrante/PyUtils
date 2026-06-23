# 1. From file
response = call_lmstudio_with_image(
    "What's in this image?",
    image_input="/home/user/photo.jpg"
)
print(response)

# 2. From URL
response = call_lmstudio_with_image(
    "Describe the scene",
    image_input="https://example.com/image.jpg"
)
print(response)

# 3. From PIL Image
from PIL import Image
img = Image.open("photo.jpg")
response = call_lmstudio_with_image("Analyze this", image_input=img)
print(response)