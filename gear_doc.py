"""
Various plots to document gear calculations
See also: https://www.tec-science.com/mechanical-power-transmission/involute-gear/calculation-of-involute-gears/
"""

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from x7.geom.utils import plot

from gear_involute import GearInvolute


def doc_radii():
    def legend2():
        line, = plot([(0, 0)], color='white')
        leg1 = plt.legend(loc=3)
        leg2: Legend = plt.legend([line], [extra], loc=2)
        leg2.set_title('Inputs')
        plt.gca().add_artist(leg1)

    rot = -0.35

    g = GearInvolute(17, profile_shift=0.5, rot=rot, **GearInvolute.HIGH_QUALITY)
    plot(g.gen_rack_tooth(teeth=5, rot=g.rot+0.5), color='lightgrey')
    extra = g.plot(gear_space=False, mill_space=False, pressure_line=True, linewidth=1)
    plt.title('Involute Radii with Profile Shift of 0.5')
    legend2()
    g.plot_show((6, 1, 3))

    g = GearInvolute(17, profile_shift=0, rot=rot, **GearInvolute.HIGH_QUALITY)
    plot(g.gen_rack_tooth(teeth=5, rot=g.rot+0.5), color='lightgrey')
    extra = g.plot(gear_space=False, mill_space=False, pressure_line=False, linewidth=1)
    plt.title('Involute Radii')
    legend2()
    g.plot_show((5.5, 1.5, 3))


def doc_tooth_parts():
    for teeth in [7, 27]:
        g = GearInvolute(teeth, **GearInvolute.HIGH_QUALITY)
        g.curved_root = False
        plot(g.instance().poly, 'lightgrey')
        plt.title('Tooth path for %d teeth' % g.teeth)
        plot(g.gen_rack_tooth(teeth=5, rot=0.5), color='c', linestyle=':')
        colors = dict(face='blue', root_cut='red', dropcut='orange', root_arc='green', tip_arc='cyan')
        labels = dict(face='Gear Face', root_cut='Root Cut', dropcut='Drop Cut', root_arc='Root Arc', tip_arc='Tip Arc')
        seen = set()
        parts = g.gen_gear_tooth_parts(True)
        for tag, points in parts:
            if tag == '_face_root_intersection':
                continue
            plot(points, color=colors[tag], linewidth=3, label=labels[tag] if tag not in seen else None)
            seen.add(tag)
            if tag == 'root_arc':
                # Plot this again mirrored so that plot is symmetric
                plot([(p.x, -p.y) for p in points], color=colors[tag], linewidth=3)
        plt.legend(loc=6)
        g.plot_show((3, 1, 2))


def main():
    doc_radii()
    doc_tooth_parts()


if __name__ == '__main__':
    main()
