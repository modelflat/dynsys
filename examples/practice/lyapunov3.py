# Program LORENZLE.BAS (c) 1997 by J. C. Sprott
# Compile with PowerBASIC 3.2 or later
import numpy

# defext a-z       # Use extended (80-bit) precision throughout
nb = 100           # Number of boxes per dimension for capacity dimension
ns = 6553          # Number of points saved for correlation dimension
nd = 32000         # Number of initial iterations to discard
h  = .001          # Iteration step size

# dim huge c(nb%\2,nb%\2,nb%\2) as shared byte
c = numpy.ndarray(shape=(nb / 2, nb / 2, nb / 2), dtype=numpy.int8)

# dim xs(ns%-1),ys(ns%-1),zs(ns%-1) as shared ext
xs = numpy.ndarray(shape=(ns - 1,), dtype=numpy.float)
ys = numpy.ndarray(shape=(ns - 1,), dtype=numpy.float)
zs = numpy.ndarray(shape=(ns - 1,), dtype=numpy.float)

# screen 12
# print"Lorenz"
# randomize timer                    #Reseed the random number generator

x = -3.16
y = -5.31
z = 13.31          # Initial conditions (close to attractor)

while inkey$<>chr$(27)             #Loop until user presses the <Esc> key
    if (n > nd // 10) and (n < nd):
        minmax(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax)
    # if n == nd: # Set up the screen
    #     xmm = xmax - xmin
    #     ymm = ymax - ymin
    #     zmm = zmax - zmin
    #     zav=(zmax+zmin)/2
    #     line(213,320)-(639,320),,,&H4444: locate 22,80: print"X";
    #     line(213,320)-(213,0),,,&H4444: locate 2,25: print"Y";
    #     line(213,320)-(0,479),,,&H4444: locate 29,1: print"Z";
    # end if
    rk4(x, y, z, h)  # Advance (x,y,z) by time step h
    if n > nd:    # Assume transient has settled
        # !here screen pixel shit is computed
        # dx = 32*(z-zav)/zmm
        # xp = 213+427*(x-xmin)/xmm-213*(z-zmin)/zmm
        # yp = 320-320*(y-ymin)/ymm+160*(z-zmin)/zmm
        # pset(xp-dx,yp),3: pset(xp+dx,yp),4  #Make stereo cyan/red plot

        # capdim  (x-xmin,y-ymin,z-zmin,d0,dd0,xmm,ymm,zmm,nb%)
        # cordim  (x,y,z,d2,dd2,xmm,ymm,zmm,ns%,n&)

        lyapunov(x, y, z, h, l1)
        jacobian(x, y, z, l1, l2, l3)
    # end if
    n = n + 1
# wend
# end

def rk4(x, y, z, h):        #Advance (x,y,z) with fourth order Runge-Kutta
    k1x = h * fnx(x, y, z)
    k1y = h * fny(x, y, z)
    k1z = h * fnz(x, y, z)
    k2x = h * fnx(x + k1x / 2, y + k1y / 2, z + k1z / 2)
    k2y = h * fny(x + k1x / 2, y + k1y / 2, z + k1z / 2)
    k2z = h * fnz(x + k1x / 2, y + k1y / 2, z + k1z / 2)
    k3x = h * fnx(x + k2x / 2, y + k2y / 2, z + k2z / 2)
    k3y = h * fny(x + k2x / 2, y + k2y / 2, z + k2z / 2)
    k3z = h * fnz(x + k2x / 2, y + k2y / 2, z + k2z / 2)
    k4x = h * fnx(x + k3x, y + k3y, z + k3z)
    k4y = h * fny(x + k3x, y + k3y, z + k3z)
    k4z = h * fnz(x + k3x, y + k3y, z + k3z)
    x = x + (k1x + 2*(k2x + k3x) + k4x) / 6
    y = y + (k1y + 2*(k2y + k3y) + k4y) / 6
    z = z + (k1z + 2*(k2z + k3z) + k4z) / 6
# end sub

def fnx(x, y, z): return 10*(y-x)
def fny(x, y, z): return -x*z+28*x-y
def fnz(x, y, z): return x*y-(8/3)*z

def minmax(x,y,z, xmin, xmax, ymin, ymax, zmin, zmax): #Find min and max of (x,y,z)
    xmin = min(x, xmin)
    xmax = max(x, xmax)
    ymin = min(y, ymin)
    ymax = max(y, ymax)
    zmin = min(z, zmin)
    zmax = max(z, zmax)
# end sub

# def capdim(x, y, z, d0, dd0, xmm, ymm, zmm, nb%):    #Get capacity dimension
#     shared c?()
#     static n1&,n2&,j%
#     j%=(j%+1) mod 1000
#     xi%=int(nb%*x/xmm): if xi%<0 or xi%>nb% then exit sub  #Out of range
#     yi%=int(nb%*y/ymm): if yi%<0 or yi%>nb% then exit sub  #Out of range
#     zi%=int(nb%*z/zmm): if zi%<0 or zi%>nb% then exit sub  #Out of range
#     ci?=0: bit set ci?,(xi% mod 2)+2*(yi% mod 2)+4*(zi% mod 2)
#     c?=c?(xi%\2,yi%\2,zi%\2)
#     if c?=0 then                            #A new large box was visited
#         if j% mod 8 then exit sub else n1&=n1&+1
#     # end if
#     if (c? or ci?)<>c? then n2&=n2&+1       #A new small box was visited
#     if j%=0 then                            #Don#t print results too often
#         if n1&=0 then exit if
#         d0=log2(n2&/n1&)
#         dd0=1.4427*sqr(1/n1&-1/n2&)
#         locate 1,10: print using"D0 =##.###ñ#.###";d0;dd0;
#     # end if
#     c?(xi%\2,yi%\2,zi%\2)=c? or ci?         #Update record of boxes visited
# # end sub

# def cordim(x, y, z, d2, dd2, xmm, ymm, zmm, ns%, n&): #Get correlation dimension
#     shared xs(),ys(),zs()
#     static n1&,n2&,j%,i%
#     j%=(j%+1) mod 1000
#     k%=(i%+int(ns%*rnd/2)) mod ns%          #Choose a random reference point
#     dx=(x-xs(k%))/xmm                       #Rescale to fit into a cube
#     dy=(y-ys(k%))/ymm
#     dz=(z-zs(k%))/zmm
#     rsq=dx*dx+dy*dy+dz*dz                   #Calculate square of separation
#     if rsq<1e-5 then n1&=n1&+1              #Point was inside small sphere
#     if rsq<5e-5 then n2&=n2&+1              #Point was inside larger sphere
#     if j%=0 then                            #Don#t print results too often
#         if n1&=0 then exit if
#         d2=1.242669869*log(n2&/n1&)
#         dd2=1.242669869*sqr(1/n1&-1/n2&)
#         locate 1,26: print using"   D2 =##.###ñ#.###";d2;dd2;
#     # end if
#     if (j% mod (1+n&\ns%\2))=0 then         #Occasionally update the reference point
#         xs(i%)=x: ys(i%)=y: zs(i%)=z: i%=(i%+1) mod ns%
#     # end if
# # end sub

def lyapunov(x, y, z, h, l1):                    #Get largest Lyapunov exponent
    static xe, ye, ze, lsum, nl&, j%
    if xe is None:
        xe = x + 1e-8; ye=y; ze=z; lsum = 0; nl = 0
        return

    j = (j + 1) % 1000

    rk4(xe, ye, ze, h)

    dlx = xe - x
    dly = ye - y
    dlz = ze - z
    dl2 = dlx**2 + dly**2 + dlz**2

    if cdbl(dl2) > 0:
        df = 1e16 * dl2
        rs = 1 / sqr(df)
        xe = x + rs*(xe - x)
        ye = y + rs*(ye - y)
        ze = z + rs*(ze - z)
        lsum += log(df)
        nl += 1
    # end if

    if j == 0: # then                            #Don#t print results too often
        l1 = .5*lsum / nl / abs(h)
        if abs(l1) < 10:
            print("L = {}".format(l1))
    # end if
# end sub

def jacobian(x, y, z, l1, l2, l3):    #Get smallest Lyapunov exponent from Jacobian
    static lnj, nl&, j%
    j = (j + 1) % 1000
    lnj = lnj - 10 - 1 - 8/3                        #Trace of Jacobian
    nl += 1
    if j == 0: # then                            #Don#t print results too often
        l2 = 0
        l3 = lnj / nl - l1
        if l3 != 0:
            dl = 2 + l1 / abs(l3)
        locate 1,57: print", 0,";
        if abs(l3) < 100:
            print(l3)
        print(dl)
    # end if
# end sub