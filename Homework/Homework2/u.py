def coef_to_exp(coef, names):
    out = ""
    for c, n in zip(coef, names):
        if c < 0:
            out += " - {0:.3f}*{1}".format(-1 * c, n)
        else:
            out += " + {0:.3f}*{1}".format(1 * c, n)

    return "\log(odd(y=1)) = " + out[1:]