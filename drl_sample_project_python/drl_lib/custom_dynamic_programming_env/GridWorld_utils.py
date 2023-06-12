def env_grid_world():
    S_Line_World = [(i,j) for j in range(5) for i in range(5)]
    A_Line_World = [0, 1, 2, 3]  # Gauche, Droite, Bas, Haut
    R_Line_World = [-1.0, 0.0, 1.0]
    return S_Line_World, A_Line_World, R_Line_World


def p_grid_world(s, a, s_p, r):
    assert (s[0] >= 0 and s[0] <= 4)
    assert (s[1] >= 0 and s[1] <= 4)
    assert (s_p[0] >= 0 and s_p[0] <= 4)
    assert (s_p[1] >= 0 and s_p[1] <= 4)
    assert (a >= 0 and a <= 3)
    assert (r >= 0 and r <= 2)
    if (s[0] == 0 and s[1] == 4) or (s[0] == 4 and s[1] == 4):
        return 0.0

    # déplacement horizontal
    if s[0] == s_p[0]:
        if s[1] + 1 == s_p[1] and a == 1 and r == 1 and (s[0] != 0 or s[1] != 3) and (s[0] != 4 or s[1] != 3):
            return 1.0
        if s[1] == s_p[1] and a == 1 and r == 1 and s[1] == 4:
            return 1.0
        if s[1] + 1 == s_p[1] and a == 1 and r == 2 and (s[0] == 4 and s[1] == 3):
            return 1.0
        if s[1] + 1 == s_p[1] and a == 1 and r == 0 and (s[0] == 0 and s[1] == 3):
            return 1.0
        if s[1] - 1 == s_p[1] and a == 0 and r == 1:
            return 1.0
        if s[1] == s_p[1] and a == 0 and r == 1 and s[1] == 0:
            return 1.0
    # déplacement vertical
    if s[1] == s_p[1]:
        if s[0] - 1 == s_p[0] and a == 3 and r == 1 and (s[0] != 1 or s[1] != 4):
            return 1.0
        if s[0] == s_p[0] and a == 3 and r == 1 and s[0] == 0:
            return 1.0
        if s[0] - 1 == s_p[0] and a == 3 and r == 0 and (s[0] == 1 and s[1] == 4):
            return 1.0
        if s[0] + 1 == s_p[0] and a == 2 and r == 1 and (s[0] != 3 or s[1] != 4):
            return 1.0
        if s[0] == s_p[0] and a == 2 and r == 1 and s[0] == 4:
            return 1.0
        if s[0] + 1 == s_p[0] and a == 2 and r == 2 and (s[0] == 3 and s[1] == 4):
            return 1.0

    return 0.0


def pi_random_grid_world(s, a):
    if (s[0] == 0 and s[1] == 4) or (s[0] == 4 and s[1] == 4):
        return 0.0
    return 0.25


