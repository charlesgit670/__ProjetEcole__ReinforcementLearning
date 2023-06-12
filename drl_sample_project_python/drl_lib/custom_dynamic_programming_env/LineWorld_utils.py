def env_line_world():
    S_Line_World = [i for i in range(7)]
    A_Line_World = [0, 1]  # Gauche, Droite
    R_Line_World = [-1.0, 0.0, 1.0]
    return S_Line_World, A_Line_World, R_Line_World


def p_line_world(s, a, s_p, r):
    assert (s >= 0 and s <= 6)
    assert (s_p >= 0 and s_p <= 6)
    assert (a >= 0 and a <= 1)
    assert (r >= 0 and r <= 2)
    if s == 0 or s == 6:
        return 0.0
    if s + 1 == s_p and a == 1 and r == 1 and s != 5:
        return 1.0
    if s + 1 == s_p and a == 1 and r == 2 and s == 5:
        return 1.0
    if s - 1 == s_p and a == 0 and r == 1 and s != 1:
        return 1.0
    if s - 1 == s_p and a == 0 and r == 0 and s == 1:
        return 1.0
    return 0.0


def pi_random_line_world(s, a):
    if s == 0 or s == 6:
        return 0.0
    return 0.5