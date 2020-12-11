from math import *

planets = [
    ("Mercury",     87.97 ),
    ("Venus",       224.7 ),
    ("Earth",       365.24 ),
    ("Mars",        686.98 ),
    ("Jupiter",     4332.59 ),
    ("Saturn",      10755.70 ),
    ("Uranus",      30687.17 ),
    ("Neptune",     60190.03 ),
]

year = 365.24

# Module = diameter / teeth
totalDiameter = 6 * 25.4
minModule = .7
maxModule = 1.3
minTeeth = 20
maxTeeth = 200

print("{:10} {:>3} {:>3} {:>8}  {:>6} {:>6} {:>6} {:>6} {:>6}".format('Planet', 'x', 'y','mod', 'err','x Diam', 'y Diam', 'x Rad', 'y rad'), '\n')
for planet in planets:
    best = ()
    prevError = 1000
    planetYear = planet[1] / year
    for y in range(minTeeth, maxTeeth):
        for x in range(minTeeth, maxTeeth):
            ratio = x/y
            error = abs((planetYear - ratio) / planetYear)
            if error < prevError:
                module = totalDiameter / (x + y)
                if module < minModule or module > maxModule:
                    continue
                prevError = error
                best = (x,y,module,round(100*error,2), (x+2)*module, (y+2)*module, (x+2)*module/2, (y+2)*module/2)

    print("{:10} {:3d} {:3d} {:8.5f} {:6.2f}% {:6.2f} {:6.2f} {:6.2f} {:6.2f}".format(planet[0], *best))
