def build_bracket_svg(region_name, teams, results=None, color="#f30b45"):
    BOX_W = 150
    BOX_H = 22
    R1_X = 8
    COL_GAP = 158
    R1_GAP = 4
    r1_y_start = 18
    r1_positions = []
    for i in range(16):
        y = r1_y_start + i * (BOX_H + R1_GAP)
        r1_positions.append(y)
    total_h = r1_positions[-1] + BOX_H + 14
    total_w = R1_X + 4 * COL_GAP + BOX_W + 8
    svg = []
    svg.append(
        f'<svg viewBox="0 0 {total_w} {total_h}"'
        f' xmlns="http://www.w3.org/2000/svg"'
        f' style="width:100%; max-height:550px;'
        f' font-family: Source Sans 3, sans-serif;">'
    )

    def team_box(x, y, seed, name, is_winner=False, pct=None):
        fill = "#f0faf0" if is_winner else "#ffffff"
        border = color if is_winner else "#c4cec0"
        bw = 1.2 if is_winner else 0.8
        txt_color = "#000000" if is_winner else "#444444"
        fw = "700" if is_winner else "400"
        x3 = x + 3
        x11 = x + 11
        x26 = x + 26
        x_end = x + BOX_W - 4
        y_mid = y + BOX_H / 2 + 3.5
        seed_w = 20
        p = []
        p.append(
            f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}"'
            f' rx="3" fill="{fill}" stroke="{border}" stroke-width="{bw}"/>'
        )
        p.append(
            f'<rect x="{x}" y="{y}" width="{seed_w}" height="{BOX_H}"'
            f' rx="3" fill="{color}"/>'
        )
        p.append(
            f'<rect x="{x3}" y="{y}" width="{seed_w - 3}" height="{BOX_H}"'
            f' fill="{color}"/>'
        )
        p.append(
            f'<text x="{x11}" y="{y_mid}" text-anchor="middle" fill="#fff"'
            f' font-size="9" font-weight="600"'
            f' font-family="Oswald,sans-serif">{seed}</text>'
        )
        display_name = name if len(name) <= 15 else name[:14] + "."
        p.append(
            f'<text x="{x26}" y="{y_mid}" fill="{txt_color}"'
            f' font-size="9" font-weight="{fw}">{display_name}</text>'
        )
        if pct is not None:
            p.append(
                f'<text x="{x_end}" y="{y_mid}" text-anchor="end"'
                f' fill="{color}" font-size="8"'
                f' font-weight="600">{pct:.0f}%</text>'
            )
        return "\n".join(p)

    def connector(x1, y1_top, y1_bot, x2, y2):
        mid = x1 + (x2 - x1) * 0.5
        p = []
        p.append(f'<path d="M{x1},{y1_top} H{mid}" fill="none" stroke="#c4cec0" stroke-width="1"/>')
        p.append(f'<path d="M{x1},{y1_bot} H{mid}" fill="none" stroke="#c4cec0" stroke-width="1"/>')
        p.append(f'<path d="M{mid},{y1_top} V{y1_bot}" fill="none" stroke="#c4cec0" stroke-width="1"/>')
        p.append(f'<path d="M{mid},{y2} H{x2}" fill="none" stroke="#c4cec0" stroke-width="1"/>')
        return "\n".join(p)

    round_labels = ["Round of 64", "Round of 32", "Sweet 16", "Elite Eight", "Region Champ"]
    for rnd_idx, label in enumerate(round_labels):
        lx = R1_X + rnd_idx * COL_GAP + BOX_W / 2
        svg.append(
            f'<text x="{lx}" y="{r1_y_start - 6}" text-anchor="middle"'
            f' fill="#888" font-size="7.5"'
            f' font-family="Oswald,sans-serif" letter-spacing="1">{label}</text>'
        )

    round_ys = [[] for _ in range(5)]
    for i in range(16):
        seed, name = teams[i]
        y = r1_positions[i]
        is_w = False
        pct_val = None
        if results and len(results) > 0:
            game_idx = i // 2
            games = results[0]["games"]
            if game_idx < len(games):
                g = games[game_idx]
                is_w = (g["winner"] == name)
                if is_w:
                    pct_val = g["w_pct"] * 100
        svg.append(team_box(R1_X, y, seed, name, is_winner=is_w, pct=pct_val))
        round_ys[0].append(y)

    for rnd in range(1, 5):
        prev = round_ys[rnd - 1]
        col_x = R1_X + rnd * COL_GAP
        prev_right = R1_X + (rnd - 1) * COL_GAP + BOX_W
        for i in range(0, len(prev), 2):
            if i + 1 >= len(prev):
                break
            y_top_center = prev[i] + BOX_H / 2
            y_bot_center = prev[i + 1] + BOX_H / 2
            new_y = (prev[i] + prev[i + 1]) / 2
            new_center = new_y + BOX_H / 2
            svg.append(connector(prev_right, y_top_center, y_bot_center, col_x, new_center))
            game_idx = i // 2
            if results and rnd - 1 < len(results):
                games = results[rnd - 1]["games"]
                if game_idx < len(games):
                    g = games[game_idx]
                    svg.append(team_box(col_x, new_y, g["w_seed"], g["winner"], is_winner=True, pct=g["w_pct"] * 100))
                else:
                    svg.append(team_box(col_x, new_y, "?", "TBD"))
            else:
                svg.append(team_box(col_x, new_y, "?", "TBD"))
            round_ys[rnd].append(new_y)

    svg.append('</svg>')
    return "\n".join(svg)


def build_final_four_svg(ff_results, ncg_result, champion):
    W = 580
    H = 180
    BOX_W = 130
    BOX_H = 22
    color = "#f30b45"
    svg = []
    svg.append(
        f'<svg viewBox="0 0 {W} {H}"'
        f' xmlns="http://www.w3.org/2000/svg"'
        ' style="width:100%; max-height:400px;'
        ' font-family: Source Sans 3, sans-serif;">'
    )

    def team_box(x, y, seed, name, pct=None, is_champ=False):
        fill = "#fff9f0" if is_champ else "#ffffff"
        border = "#f30b45" if is_champ else "#c4cec0"
        bw = 1.5 if is_champ else 0.8
        x3 = x + 3
        x10 = x + 10
        x22 = x + 22
        x_end = x + BOX_W - 4
        y_mid = y + BOX_H / 2 + 3.5
        seed_w = 18
        p = []
        p.append(
            f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}"'
            f' rx="3" fill="{fill}" stroke="{border}" stroke-width="{bw}"/>'
        )
        p.append(
            f'<rect x="{x}" y="{y}" width="{seed_w}" height="{BOX_H}"'
            f' rx="3" fill="{color}"/>'
        )
        p.append(
            f'<rect x="{x3}" y="{y}" width="{seed_w - 3}" height="{BOX_H}"'
            f' fill="{color}"/>'
        )
        p.append(
            f'<text x="{x10}" y="{y_mid}" text-anchor="middle" fill="#fff"'
            f' font-size="9" font-weight="600"'
            f' font-family="Oswald,sans-serif">{seed}</text>'
        )
        p.append(
            f'<text x="{x22}" y="{y_mid}" fill="#000"'
            f' font-size="9" font-weight="700">{name}</text>'
        )
        if pct is not None:
            p.append(
                f'<text x="{x_end}" y="{y_mid}" text-anchor="end"'
                f' fill="{color}" font-size="8"'
                f' font-weight="600">{pct:.0f}%</text>'
            )
        return "\n".join(p)

    w_half = W / 2
    svg.append(
        f'<text x="{w_half}" y="14" text-anchor="middle" fill="#3b6e3f"'
        ' font-size="11" font-weight="700"'
        ' font-family="Oswald,sans-serif" letter-spacing="2">FINAL FOUR</text>'
    )
    s1 = ff_results[0]
    svg.append(team_box(8, 30, s1["w_seed"], s1["winner"], pct=s1["w_pct"] * 100))
    svg.append(team_box(8, 56, s1["l_seed"], s1["loser"], pct=s1["l_pct"] * 100))
    s1_label_x = 8 + BOX_W / 2
    svg.append(
        f'<text x="{s1_label_x}" y="26" text-anchor="middle" fill="#888"'
        ' font-size="7" font-family="Oswald,sans-serif"'
        ' letter-spacing="1">SEMIFINAL 1</text>'
    )
    s2 = ff_results[1]
    s2_x = W - BOX_W - 8
    svg.append(team_box(s2_x, 30, s2["w_seed"], s2["winner"], pct=s2["w_pct"] * 100))
    svg.append(team_box(s2_x, 56, s2["l_seed"], s2["loser"], pct=s2["l_pct"] * 100))
    s2_label_x = s2_x + BOX_W / 2
    svg.append(
        f'<text x="{s2_label_x}" y="26" text-anchor="middle" fill="#888"'
        ' font-size="7" font-family="Oswald,sans-serif"'
        ' letter-spacing="1">SEMIFINAL 2</text>'
    )
    ncg_x = (W - BOX_W) / 2
    line_left = 8 + BOX_W
    line_y = 30 + BOX_H / 2
    line_right = W - BOX_W - 8
    ncg_right = ncg_x + BOX_W
    svg.append(f'<path d="M{line_left},{line_y} H{ncg_x}" fill="none" stroke="#c4cec0" stroke-width="0.7"/>')
    svg.append(f'<path d="M{line_right},{line_y} H{ncg_right}" fill="none" stroke="#c4cec0" stroke-width="0.7"/>')
    svg.append(
        f'<text x="{w_half}" y="100" text-anchor="middle" fill="#888"'
        ' font-size="7" font-family="Oswald,sans-serif"'
        ' letter-spacing="1">CHAMPIONSHIP</text>'
    )
    n = ncg_result
    svg.append(team_box(ncg_x, 106, n["w_seed"], n["winner"], pct=n["w_pct"] * 100))
    svg.append(team_box(ncg_x, 132, n["l_seed"], n["loser"], pct=n["l_pct"] * 100))
    champ_y = H - 4
    champ_label = f"🏆 CHAMPION: ({champion[0]}) {champion[1]}"
    svg.append(
        f'<text x="{w_half}" y="{champ_y}" text-anchor="middle"'
        f' fill="{color}" font-size="11" font-weight="700"'
        f' font-family="Oswald,sans-serif" letter-spacing="2">{champ_label}</text>'
    )
    svg.append('</svg>')
    return "\n".join(svg)
