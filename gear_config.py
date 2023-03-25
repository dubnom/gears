from typing import Tuple

GEARS = {
    # Name        X   Y
    'mercury':  (33, 137, 0.89647),
    'venus':    (80, 130, 0.72571),
    'earth':    (108, 108, 0.70556),
    'mars':     (79, 42, 1.25950),
    'jupiter':  (197, 20, 0.70230),
    'saturn':   (197, 20, 0.70230),
}


def lookup(planet: str) -> Tuple[int, int, float]:
    """Lookup planet and return x, y, module.  Also things like 'example:23:17:.8'"""

    if planet in GEARS:
        return GEARS[planet]

    parts = planet.split(':')
    if len(parts) == 4:
        return int(parts[1]), int(parts[2]), float(parts[3])

    raise ValueError('Unknown planet: %r' % planet)


if __name__ == '__main__':
    for p in 'mercury,venus,test:1:2:3,error,other:error:3:4'.split(','):
        try:
            print(p, lookup(p))
        except ValueError as e:
            print(p, e)
