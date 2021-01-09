def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df / mx) * 100
    v = mx * 100
    return h, s, v


def rgb_to_ycbcr(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    y = float(0.2989 * r + 0.5866 * g + 0.1145 * b)
    cb = float(-0.1687 * r - 0.3313 * g + 0.5000 * b)
    cr = float(0.5000 * r - 0.4184 * g - 0.0816 * b)

    return y, cb, cr
