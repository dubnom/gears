"""
Various plots to document gear calculations
"""

import matplotlib.pyplot as plt
from x7.geom.utils import plot

from gear_involute import GearInvolute


def doc_radii():
    g = GearInvolute(17, profile_shift=0.5, **GearInvolute.HIGH_QUALITY)
    g.plot(gear_space=False, mill_space=False, pressure_line=False)
    plt.title('Involute Radii with Profile Shift of 0.5')
    plt.legend(loc=6)
    g.plot_show((5, 1, 3))

    g = GearInvolute(17, profile_shift=0, **GearInvolute.HIGH_QUALITY)
    g.plot(gear_space=False, mill_space=False, pressure_line=False)
    plt.title('Involute Radii')
    plt.legend(loc=6)
    g.plot_show((4.5, 1.5, 3))


def doc_tooth_parts():
    for teeth in [7, 27]:
        g = GearInvolute(teeth, **GearInvolute.HIGH_QUALITY)
        g.curved_root = False
        plot(g.instance().poly, 'lightgrey')
        plt.title('Tooth path for %d teeth' % g.teeth)
        plot(g.gen_rack_tooth(teeth=5, rot=0.5), color='c:', label='Rack')
        colors = dict(face='blue', root_cut='red', dropcut='orange', root_arc='green', tip_arc='cyan')
        seen = set()
        parts = g.gen_gear_tooth_parts(True)
        for tag, points in parts:
            if tag == '_face_root_intersection':
                continue
            plot(points, color=colors[tag], label=tag if tag not in seen else None)
            seen.add(tag)
            if tag == 'root_arc':
                # Plot this again mirrored so that plot is symmetric
                plot([(p.x, -p.y) for p in points], color=colors[tag])
        plt.legend(loc=6)
        g.plot_show((3, 1, 2))


def main():
    # doc_radii()
    doc_tooth_parts()


if __name__ == '__main__':
    main()
