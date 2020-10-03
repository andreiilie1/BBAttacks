def cap(value, inf, sup):
    if value <= inf:
        return inf
    if value >= sup:
        return sup
    return value