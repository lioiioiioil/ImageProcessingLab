"""Generate deterministic synthetic input images (Pillow required)."""
from pathlib import Path
from PIL import Image, ImageDraw
import math

out = Path(__file__).resolve().parents[1] / "examples"
out.mkdir(exist_ok=True)
images = []
# Gradient with sharp transitions.
im = Image.new("L", (256, 256))
im.putdata([int(255*x/255) for y in range(256) for x in range(256)])
d = ImageDraw.Draw(im)
d.rectangle((48, 48, 96, 192), fill=25)
d.ellipse((128, 72, 224, 168), fill=235)
images.append(im)
# Separate light and dark shapes for connected-component analysis.
im = Image.new("L", (256, 256), 35)
d = ImageDraw.Draw(im)
d.ellipse((24, 24, 100, 100), fill=230)
d.rectangle((140, 24, 230, 78), fill=230)
d.ellipse((32, 140, 116, 224), fill=230)
d.ellipse((58, 166, 90, 198), fill=35)
d.polygon([(160, 130), (230, 224), (140, 224)], fill=230)
images.append(im)
# Checkerboard.
im = Image.new("L", (256, 256))
im.putdata([220 if ((x//24+y//24)%2) else 40 for y in range(256) for x in range(256)])
images.append(im)
# Sum of sinusoids.
im = Image.new("L", (256, 256))
im.putdata([int(127+60*math.sin(2*math.pi*x/32)+60*math.sin(2*math.pi*y/64)) for y in range(256) for x in range(256)])
images.append(im)
for i, im in enumerate(images, 1):
    im.save(out/f"{i}.png")
    im.save(out/f"{i}.jpg", quality=95)
images[-1].save(out/"input.png")
print(f"Created {len(images)} image pairs and input.png in {out}")
