from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageEnhance, ImageFont

from anim.geom import BasePoint
from anim.typing import *
from anim.transform import Transform


def bbox_max(bbox_a, bbox_b):
    compare = (min, min, max, max)
    return tuple(compare(a, b) for compare, a, b in zip(compare, bbox_a, bbox_b))


def test_bbm():
    a = (1, 2, 3, 4)
    b = (0, 0, 6, 8)
    print(bbox_max(a, b))
    assert bbox_max(a, b) == bbox_max(b, a)
    a = (1, 0, 6, 4)
    b = (0, 2, 3, 8)
    print(bbox_max(a, b))
    assert bbox_max(a, b) == bbox_max(b, a)


# test_bbm(); sys.exit()

def make_shadow(base):
    r, g, b, a = base.split()
    [i.paste(i.getextrema()[1], (0, 0) + i.size) for i in (r, g, b)]
    a0 = a
    sa = ImageChops.offset(a, 4, 3)
    for n in range(6):
        a = ImageChops.offset(a, 4, 3)
        a = a.filter(ImageFilter.BLUR)
        a = ImageEnhance.Brightness(a).enhance(0.95)
        sa = ImageChops.lighter(sa, a)
    sa = sa.filter(ImageFilter.BLUR)
    sa = ImageChops.lighter(sa, a0)
    return Image.merge('RGBA', (r, g, b, sa))


class DrawingContext(object):
    IDENTITY = (1, 0, 0, 0, 1, 0)

    def __init__(self, image: Optional[Image.Image], matrix: Optional[Transform], crop=None, prod=True):
        image = image or Image.new('RGBA', (1000, 1000), 'white')
        matrix = matrix or Transform()
        if not image.mode == 'RGBA':
            image = image.convert('RGBA')
        self.background = image
        self.layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        self.shadow: Optional[Image.Image] = None
        self.shadow_offset = (0, 0)
        self.draw = ImageDraw.ImageDraw(self.layer)
        self.matrix = matrix
        self.points_per_curve = 80
        self.set_matrix(None)
        self.prod = prod
        self.color = 'black'
        self.debug = True
        self.showall = False        # Show image at each step during drawing
        self.show_cps = False
        self.shadow_color = None
        self.raw_crop = crop
        self.crop = crop and self.crop_to_pixels(crop)

    def crop_to_pixels(self, crop):
        # Map into image space
        x1, y1, x2, y2 = map(int, self.scale_pt(crop[0], crop[1]) + self.scale_pt(crop[2], crop[3]))
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def image(self, crop=True) -> Image.Image:
        """Return the composed image (layer over background)"""

        # print(self.background.mode, self.layer.mode, self.shadow.mode if self.shadow else None)
        if self.shadow:
            background = Image.alpha_composite(self.background, self.shadow)
        else:
            background = self.background
        im = Image.alpha_composite(background, self.layer)
        if isinstance(crop, tuple):
            im = im.crop(self.crop_to_pixels(crop))
        elif crop:
            im = im.crop(self.crop)
        return im

    def stroke_width(self, width: float):
        """Scale width appropriately based on xform"""
        return self.matrix.transform_width(width)

    def alloc_shadow(self, offset=(4, -3)):
        if not self.shadow:
            self.shadow = Image.new('RGBA', self.layer.size, (0, 0, 0, 0))
            self.shadow_offset = offset

    def shadow_draw(self, model, color=(140, 140, 170, 255)):
        # TODO - just draw into an L image?
        draw = self.draw
        self.alloc_shadow()
        shadow_draw = ImageDraw.Draw(self.shadow)
        try:
            self.draw = shadow_draw
            self.shadow_color = color
            model.draw(self)
        finally:
            self.draw = draw
            self.shadow_color = None

        self.shadow = make_shadow(self.shadow)

    def show(self):
        self.image().show()

    def set_matrix(self, matrix: Optional[Transform]):
        if matrix:
            self.matrix.set_matrix(matrix)

    def scale_pt(self, x, y):
        return self.matrix.transform(x, y)

    def unscale_pt(self, px, py):
        return self.matrix.untransform(px, py)

    def line(self, x1, y1, x2, y2, color: Union[str, tuple] = 'black', width=1):
        x1, y1 = self.scale_pt(x1, y1)
        x2, y2 = self.scale_pt(x2, y2)
        self.draw.line((x1, y1, x2, y2), fill=color, width=width)

    def text(self, x, y, text: str, color=(20, 20, 20, 255)):
        # font = ImageFont.truetype('/Library/Fonts/Arial Unicode.ttf', 15)
        font = ImageFont.truetype('/System/Library/Fonts/SFNSMono.ttf', 40)
        self.draw.text((x, y), text, fill=color, font=font)

    def polygon(self, points: List[Union[Tuple[Number, Number], BasePoint]]):
        pts = self.matrix.transform_pts(points, True)
        self.draw.polygon(pts, fill='orange', outline='cyan')

    def grid(self, spacing=10, center=(0, 0)):
        unused(spacing, center)

        self.line(1000, 0, -1000, 0, 'red')
        self.line(0, 1000, 0, -1000, 'red')

        for r, c in [(100, 'black'), (50, 'green'), (10, 'blue')]:
            self.line(-r, r, r, r, c)
            self.line(r, -r, r, r, c)
            self.line(r, -r, -r, -r, c)
            self.line(-r, -r, -r, r, c)
        # self.draw.line([(10, 10), (100, 10), (100, 100), (50, 50)], fill='blue', width=10, joint='curve')


def test_drawing():
    img = Image.new('RGBA', (1000, 1000), 'grey')
    t = Transform.canvas_fit((1000, 1000), zoom=10.0, zero_zero=(500, 500))
    print(t)
    draw = DrawingContext(img, t)
    print('scale_pt(0,0)->', draw.scale_pt(0, 0))
    draw.line(0, -10, 0, 10, 'red')
    draw.line(-10, 0, 10, 0, 'red')
    draw.show()


# TODO - add demo/test of shadow


def test_all():
    test_bbm()
    test_drawing()


if __name__ == '__main__':
    test_all()
