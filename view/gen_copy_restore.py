def gen_args():
    data = """teeth=30, center=Point(0, 0), rot=0.0,
                 module=1.0, relief_factor=1.25,
                 steps=4, tip_arc=0.0, root_arc=0.0, curved_root=False, debug=False,
                 pressure_angle=20.0, pressure_line=True"""
    data = data.replace('Point(0, 0)', 'PointZeroZero')
    for e in data.split(','):
        e = e.partition('=')[0].strip()
        print('%s=self.%s,' % (e, e))
    print()
    for e in data.split(','):
        e = e.partition('=')[0].strip()
        print('self.%s = other.%s' % (e, e))

    print()
    for e in data.split(','):
        e = e.partition('=')[0].strip()
        print('and self.%s == other.%s \\' % (e, e))


gen_args()
