def build_bracket_svg(region_name, teams, results=None, color="#f30b45"):
    BOX_W = 150
    BOX_H = 28
    R1_X = 10
    COL_GAP = 180
    R1_GAP = 6
    r1_y_start = 22
    r1_positions = []
    for i in range(16):
        y = r1_y_start + i * (BOX_H + R1_GAP)
        r1_positions.append(y)
    total_h = r1_positions[-1] + BOX_H + 20
    total_w = R1_X + 4 * COL_GAP + BOX_W + 10
    svg = []
    svg.append(f'<svg viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg" style="width:100%; font-family: Source Sans 3, sans-serif;">')

    def team_box(x, y, seed, name, is_winner=False, pct=None):
        fill = "#f0faf0" if is_winner else "#ffffff"
        border = color if is_winner else "#c4cec0"
        bw = 1.5 if is_winner else 1
        txt_color = "#000000" if is_winner else "#444444"
        fw = "700" if is_winner else "400"
        p = []
        p.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="4" fill="{fill}" stroke="{border}" stroke-width="{bw}"/>')
        p.append(f'<rect x="{x}" y="{y}" width="24" height="{BOX_H}" rx="4" fill="{color}"/>')
        p.append(f'<rect x="{x+4}" y="{y}" width="20" height="{BOX_H}" fill="{color}"/>')
        p.append(f'<text x="{x+12}" y="{y+BOX_H/2+4.5}" text-anchor="middle" fill="#fff" font-size="11" font-weight="600" font-family="Oswald,sans-serif">{seed}</text>')
        display_name = name if len(name) <= 16 else name[:15] + "."
        p.append(f'<text x="{x+30}" y="{y+BOX_H/2+4.5}" fill="{txt_color}" font-size="11" font-weight="{fw}">{display_name}</text>')
        if pct is not None:
            p.append(f'<text x="{x+BOX_W-5}" y="{y+BOX_H/2+4.5}" text-anchor="end" fill="{color}" font-size="9" font-weight="600">{pct:.0f}%</text>')
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
        svg.append(f'<text x="{lx}" y="{r1_y_start - 6}" text-anchor="middle" fill="#888" font-size="8.5" font-family="Oswald,sans-serif" letter-spacing="1">{label}</text>')

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
            if results and rnd < len(results):
                games = results[rnd]["games"]
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
    W = 700
    H = 220
    BOX_W = 150
    BOX_H = 30
    color = "#f30b45"
    svg = []
    svg.append(f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%; font-family: Source Sans 3, sans-serif;">')

    def team_box(x, y, seed, name, pct=None, is_champ=False):
        fill = "#fff9f0" if is_champ else "#ffffff"
        border = "#f30b45" if is_champ else "#c4cec0"
        bw = 2 if is_champ else 1
        p = []
        p.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="5" fill="{fill}" stroke="{border}" stroke-width="{bw}"/>')
        p.append(f'<rect x="{x}" y="{y}" width="26" height="{BOX_H}" rx="5" fill="{color}"/>')
        p.append(f'<rect x="{x+4}" y="{y}" width="22" height="{BOX_H}" fill="{color}"/>')
        p.append(f'<text x="{x+13}" y="{y+BOX_H/2+5}" text-anchor="middle" fill="#fff" font-size="12" font-weight="600" font-family="Oswald,sans-serif">{seed}</text>')
        p.append(f'<text x="{x+32}" y="{y+BOX_H/2+5}" fill="#000" font-size="12" font-weight="700">{name}</text>')
        if pct is not None:
            p.append(f'<text x="{x+BOX_W-5}" y="{y+BOX_H/2+5}" text-anchor="end" fill="{color}" font-size="10" font-weight="600">{pct:.0f}%</text>')
        return "\n".join(p)

    svg.append(f'<text x="{W/2}" y="18" text-anchor="middle" fill="#3b6e3f" font-size="14" font-weight="700" font-family="Oswald,sans-serif" letter-spacing="2">FINAL FOUR</text>')
    s1 = ff_results[0]
    svg.append(team_box(10, 40, s1["w_seed"], s1["winner"], pct=s1["w_pct"] * 100))
    svg.append(team_box(10, 76, s1["l_seed"], s1["loser"], pct=s1["l_pct"] * 100))
    svg.append(f'<text x="85" y="36" text-anchor="middle" fill="#888" font-size="8.5" font-family="Oswald,sans-serif" letter-spacing="1">SEMIFINAL 1</text>')
    s2 = ff_results[1]
    svg.append(team_box(W - BOX_W - 10, 40, s2["w_seed"], s2["winner"], pct=s2["w_pct"] * 100))
    svg.append(team_box(W - BOX_W - 10, 76, s2["l_seed"], s2["loser"], pct=s2["l_pct"] * 100))
    svg.append(f'<text x="{W - 85}" y="36" text-anchor="middle" fill="#888" font-size="8.5" font-family="Oswald,sans-serif" letter-spacing="1">SEMIFINAL 2</text>')
    ncg_x = (W - BOX_W) / 2
    svg.append(f'<path d="M{10+BOX_W},{40+BOX_H/2} H{ncg_x}" fill="none" stroke="#c4cec0" stroke-width="1"/>')
    svg.append(f'<path d="M{W-BOX_W-10},{40+BOX_H/2} H{ncg_x+BOX_W}" fill="none" stroke="#c4cec0" stroke-width="1"/>')
    svg.append(f'<text x="{W/2}" y="130" text-anchor="middle" fill="#888" font-size="8.5" font-family="Oswald,sans-serif" letter-spacing="1">CHAMPIONSHIP</text>')
    n = ncg_result
    svg.append(team_box(ncg_x, 136, n["w_seed"], n["winner"], pct=n["w_pct"] * 100))
    svg.append(team_box(ncg_x, 172, n["l_seed"], n["loser"], pct=n["l_pct"] * 100))
    svg.append(f'<text x="{W/2}" y="{H - 4}" text-anchor="middle" fill="{color}" font-size="13" font-weight="700" font-family="Oswald,sans-serif" letter-spacing="2">🏆 CHAMPION: ({champion[0]}) {champion[1]}</text>')
    svg.append('</svg>')
    return "\n".join(svg)
