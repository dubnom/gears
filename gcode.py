class Gcode(object):
    def __init__(self):
        self.gcode = []

    def move(self,a=None,x=None,y=None,z=None):
        """Record a high-speed move operation"""
        self.gcode.append( "G0" + (' A%g'%a if a!=None else '') + (' X%g'%x if x!=None else '') + (' Y%g'%y if y!=None else '') + (' Z%g'%z if z!=None else ''))
    
    def cut(self,a=None,x=None,y=None,z=None):
        """Record a cutting speed linear operation"""
        self.gcode.append( "G1" + (' A%g'%a if a!=None else '') + (' X%g'%x if x!=None else '') + (' Y%g'%y if y!=None else '') + (' Z%g'%z if z!=None else ''))
    
    def comment(self,c=None):
        """Record comments"""
        self.gcode.append( "(%s)" % c if c else '' )

    def append(self,c):
        self.gcode.append( c )

    def output(self):
        return '\n'.join(self.gcode)
    

