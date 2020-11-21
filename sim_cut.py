"""
Simulate cuts in a gear and display images
"""

from PIL import Image
from anim.simple import SimpleAnimation
from anim.drawing import DrawingContext
from anim.transform import Transform
from anim.geom import BBox

from rack import Rack


class SimCut(object):
    def __init__(self):
        self.rack = Rack()
        self.points = []
        self.gear_teeth = 32
        self.pitch_radius = self.rack.module * self.gear_teeth

    def one_step(self, dc: DrawingContext, t: float):
        rack_offset = -t * self.rack.circular_pitch
        (cut_x, cut_y), (low_x, low_y) = self.rack.cut_points()
        cut_x += rack_offset
        low_x += rack_offset
        print('%8.5f, %8.5f' % (cut_x, cut_y))
        self.points[:] = [(cut_x, cut_y), (low_x, low_y)]

        if len(self.points) > 1:
            dc.polygon(self.points, outline='green')
            dc.arc((rack_offset, self.pitch_radius), self.pitch_radius, 60, 120, 'red')
            if False:
                dc.arc((0, 0), 8, 0, 45, 'green')
                dc.arc((0, 0), 8, 50, 100, 'blue')
                dc.arc((0, 0), 7, 0, 300, 'white')
                dc.line(-9, -9, -9, 9, color='yellow')
                dc.line(-9, 9, 9, 9, color='yellow')
                dc.line(9, 9, 9, -9, color='yellow')
                dc.line(9, -9, -9, -9, color='yellow')


def sim():
    sc = SimCut()
    sa = SimpleAnimation(sc.one_step, image_size=(500, 500), draw_bbox=(-10, -10, 10, 10),
                         steps=20, t_high=1, frame_display_time=1 / 20)
    sa.animate()
    sa.save_animation(None)


def main():
    sim()


if __name__ == '__main__':
    main()
