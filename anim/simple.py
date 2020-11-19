"""
Simple animations
"""
import math
import sys
import os
from PIL import Image
from typing import List

from anim.drawing import DrawingContext
from anim.geom import BBox, Point
from anim.transform import Transform

if sys.platform == 'something for windows':
    TMP_ANIM = '/temp/sa.gif'
else:
    TMP_ANIM = '/tmp/sa.gif'


class StepContext(object):
    """Context used by SimpleAnim.next_step"""

    def __init__(self, sa: 'SimpleAnimation', dc: DrawingContext):
        self.sa = sa
        self.dc = dc

    def __enter__(self):
        return self.dc

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sa.finish_step(self.dc)


class SimpleAnimation(object):
    """Master animation object."""

    def __init__(self, draw_func, image_size=(500, 500), draw_bbox=(-250, -250, 250, 250),
                 steps=20, t_low=0.0, t_high=1.0,
                 frame_display_time=0.1,
                 show_steps=False):
        """
        :param draw_func: a function of (DrawingContext, step_time)
        :param image_size: size of images
        :param draw_bbox: bounding box of drawing space (used by draw_func)
        :param steps: number of steps (minus one) to go from t_low to t_high
        :param t_low: starting step time
        :param t_high: ending step time
        :param frame_display_time: number of seconds to display each animation frame
        :param show_steps: debugging, show each step as separate image
        """

        self.draw_func = draw_func
        self.image_size = Point(*image_size)
        self.model_bbox = BBox(draw_bbox)
        self.steps = steps
        self.background = 'grey'
        assert(t_low != t_high)
        self.t_low = t_low
        self.t_high = t_high
        self.frame_display_time = frame_display_time
        self.show_steps = show_steps

        self.images: List[Image.Image] = []

    def next_step(self) -> StepContext:
        """Create images, call draw_func for each step"""

        this_image = Image.new('RGBA', self.image_size.xy(), self.background)
        xf = Transform.canvas_fit(self.image_size.xy(), zoom=1.0, zero_zero=(0, self.image_size.y))
        xf.scale_bbox(self.model_bbox, BBox(0, 0, *self.image_size))
        dc = DrawingContext(this_image, xf)
        return StepContext(self, dc)

    def finish_step(self, dc: DrawingContext):
        if self.show_steps:
            dc.show()
        self.images.append(dc.image())

    def animate(self):
        """Create images, call draw_func for each step"""

        step_size = (self.t_high - self.t_low) / self.steps
        for step in range(self.steps+1):
            with self.next_step() as dc:
                step_t = self.t_low + step * step_size
                self.draw_func(dc, step_t)
            if self.show_steps:
                dc.show()
            self.images.append(dc.image())

    def save_animation(self, output):
        if not output:
            output = TMP_ANIM
            print('Output to %r' % output)
        images = self.images
        ms_per_frame = int(self.frame_display_time * 1000)
        # palette = self.make_palette(images)
        # images = [rgb_to_transparent(img, palette_image=palette) for img in images]
        if sys.platform == 'darwin':
            images[0].save('/tmp/images.png')
            images[0].save(output, save_all=True, append_images=images[1:], disposal=1, duration=ms_per_frame, loop=0)
            # images[0].save(output, save_all=True, append_images=images[1:], disposal=2, duration=ms_per_frame, loop=0)
            os.system('qlmanage -p %r 2&>/dev/null' % output)
        elif sys.platform == 'something for windows':
            # Not sure if this will work or not
            images[0].save('/temp/images.png')
            images[0].save(output, save_all=True, append_images=images[1:], disposal=1, duration=ms_per_frame, loop=0)
            os.system('start %r' % output)
        else:
            raise NotImplementedError('Unknown platform: %s' % sys.platform)


def test_simple():
    def draw_it(draw: DrawingContext, t: float):
        theta = t * math.tau * 2
        print('%8.5f '*4 % (t, theta, 100 * math.sin(theta), 100 * math.cos(theta)))

        draw.line(0, 0, 100 * math.sin(theta), 100 * math.cos(theta), color='red', width=10)
        draw.line(0, 0, 50 * math.sin(theta/2), 50 * math.cos(theta/2), color='green', width=10)
        x = -10 * math.sin(theta/2)
        y = -10 * math.cos(theta/2)
        draw.polygon([(x-10, y), (x, y-10), (x+10, y), (x, y+10)])
        draw.line(-10, -10, 10, 10, color='blue', width=2)
        draw.line(10, -10, -10, 10, color='blue', width=2)
        draw.line(-99, -99, -99, 99, color='yellow')
        draw.line(-99, 99, 99, 99, color='yellow')
        draw.line(99, 99, 99, -99, color='yellow')
        draw.line(99, -99, -99, -99, color='yellow')

    sa = SimpleAnimation(draw_it, image_size=(400, 400), draw_bbox=(-100, -100, 100, 100), steps=120, t_high=1, frame_display_time=1 / 60)
    sa.animate()
    sa.save_animation(None)


def main():
    test_simple()


if __name__ == '__main__':
    main()
