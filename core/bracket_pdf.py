import io
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfgen import canvas


def _draw_team_box(c, x, y, seed, name, w_pct=None, is_winner=False, color="#3b82f6"):
    box_w = 140
    box_h = 22
    seed_w = 22
    if is_winner:
        c.setFillColor(HexColor("#eef7ee"))
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(1.2)
    else:
        c.setFillColor(white)
        c.setStrokeColor(HexColor("#c4cec0"))
        c.setLineWidth(0.6)
    c.roundRect(x, y, box_w, box_h, 3, fill=1, stroke=1)
    c.setFillColor(HexColor(color))
    c.rect(x, y, seed_w, box_h, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(x + seed_w / 2, y + 7, str(seed))
    if is_winner:
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 8)
    else:
        c.setFillColor(HexColor("#555555"))
        c.setFont("Helvetica", 8)
    display_name = name if len(name) <= 16 else name[:15] + "."
    c.drawString(x + seed_w + 4, y + 7, display_name)
    if w_pct is not None:
        c.setFillColor(HexColor(color))
        c.setFont("Helvetica-Bold", 7)
        c.drawRightString(x + box_w - 4, y + 7, f"{w_pct:.0f}%")
    return box_w, box_h


def _draw_connector(c, x1, y1_top, y1_bot, x2, y2):
    mid = x1 + (x2 - x1) * 0.5
    c.setStrokeColor(HexColor("#c4cec0"))
    c.setLineWidth(0.5)
    c.line(x1, y1_top, mid, y1_top)
    c.line(x1, y1_bot, mid, y1_bot)
    c.line(mid, y1_top, mid, y1_bot)
    c.line(mid, y2, x2, y2)


def _draw_region_bracket(c, x_origin, y_origin, region_name, teams, results, color, going_right=True):
    box_w = 140
    box_h = 22
    col_gap = 160
    r1_gap = 4
    c.setFillColor(HexColor(color))
    c.setFont("Helvetica-Bold", 12)
    header_x = x_origin + 70 if going_right else x_origin + 70
    c.drawCentredString(header_x, y_origin + 16 * (box_h + r1_gap) + 20, region_name.upper())
    # round_labels = ["R64", "R32", "S16", "E8"]
    r1_positions = []
    for i in range(16):
        y = y_origin + i * (box_h + r1_gap)
        r1_positions.append(y)
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
        col_x = x_origin
        _draw_team_box(c, col_x, y, seed, name, w_pct=pct_val, is_winner=is_w, color=color)
        round_ys[0].append(y)
    for rnd in range(1, 5):
        prev = round_ys[rnd - 1]
        col_x = x_origin + rnd * col_gap
        prev_right = x_origin + (rnd - 1) * col_gap + box_w
        for i in range(0, len(prev), 2):
            if i + 1 >= len(prev):
                break
            y_top_center = prev[i] + box_h / 2
            y_bot_center = prev[i + 1] + box_h / 2
            new_y = (prev[i] + prev[i + 1]) / 2
            new_center = new_y + box_h / 2
            _draw_connector(c, prev_right, y_top_center, y_bot_center, col_x, new_center)
            game_idx = i // 2
            if results and rnd - 1 < len(results):
                games = results[rnd - 1]["games"]
                if game_idx < len(games):
                    g = games[game_idx]
                    _draw_team_box(c, col_x, new_y, g["w_seed"], g["winner"], w_pct=g["w_pct"] * 100, is_winner=True, color=color)
                else:
                    _draw_team_box(c, col_x, new_y, "?", "TBD", color=color)
            else:
                _draw_team_box(c, col_x, new_y, "?", "TBD", color=color)
            round_ys[rnd].append(new_y)


def build_bracket_pdf(bracket_2026, region_rounds, region_winners, ff_results, ncg_result, champion, gear):
    buf = io.BytesIO()
    page_w, page_h = landscape(letter)
    c = canvas.Canvas(buf, pagesize=landscape(letter))
    region_colors = {
        "East": "#3b82f6",
        "South": "#ef4444",
        "West": "#22c55e",
        "Midwest": "#d97706",
    }
    for region_name, teams in bracket_2026.items():
        color = region_colors.get(region_name, "#3b82f6")
        results = region_rounds.get(region_name, [])
        c.setFont("Helvetica-Bold", 16)
        c.setFillColor(HexColor("#f30b45"))
        c.drawCentredString(page_w / 2, page_h - 30, f"2026 NCAA Tournament - {region_name} Region")
        c.setFont("Helvetica", 9)
        c.setFillColor(HexColor("#888888"))
        c.drawCentredString(page_w / 2, page_h - 44, f"Confidence Gear: {gear:+d}")
        round_labels = ["Round of 64", "Round of 32", "Sweet 16", "Elite Eight", "Region Champ"]
        for ri, label in enumerate(round_labels):
            lx = 40 + ri * 160 + 70
            c.setFont("Helvetica", 7)
            c.setFillColor(HexColor("#999999"))
            c.drawCentredString(lx, page_h - 58, label)
        bracket_h = 16 * 26
        y_start = (page_h - 70 - bracket_h) / 2
        _draw_region_bracket(c, 40, y_start, region_name, teams, results, color)
        winner = region_winners.get(region_name)
        if winner:
            c.setFont("Helvetica-Bold", 11)
            c.setFillColor(HexColor(color))
            c.drawCentredString(page_w / 2, 20, f"Region Champion: ({winner[0]}) {winner[1]}")
        c.showPage()
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(HexColor("#f30b45"))
    c.drawCentredString(page_w / 2, page_h - 40, "2026 NCAA Tournament - Final Four & Championship")
    c.setFont("Helvetica", 10)
    c.setFillColor(HexColor("#888888"))
    c.drawCentredString(page_w / 2, page_h - 58, f"Confidence Gear: {gear:+d}")
    mid_y = page_h / 2
    box_w = 140
    # box_h = 22
    if ff_results and len(ff_results) >= 2:
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(HexColor("#3b6e3f"))
        c.drawCentredString(150, mid_y + 80, "SEMIFINAL 1")
        s1 = ff_results[0]
        _draw_team_box(c, 80, mid_y + 40, s1["w_seed"], s1["winner"], w_pct=s1["w_pct"] * 100, is_winner=True, color="#3b6e3f")
        _draw_team_box(c, 80, mid_y + 10, s1["l_seed"], s1["loser"], w_pct=s1["l_pct"] * 100, is_winner=False, color="#3b6e3f")
        c.drawCentredString(page_w - 150, mid_y + 80, "SEMIFINAL 2")
        s2 = ff_results[1]
        _draw_team_box(c, page_w - 220, mid_y + 40, s2["w_seed"], s2["winner"], w_pct=s2["w_pct"] * 100, is_winner=True, color="#3b6e3f")
        _draw_team_box(c, page_w - 220, mid_y + 10, s2["l_seed"], s2["loser"], w_pct=s2["l_pct"] * 100, is_winner=False, color="#3b6e3f")
    if ncg_result:
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(HexColor("#f30b45"))
        ncg_x = (page_w - box_w) / 2
        c.drawCentredString(page_w / 2, mid_y - 20, "CHAMPIONSHIP")
        _draw_team_box(c, ncg_x, mid_y - 50, ncg_result["w_seed"], ncg_result["winner"], w_pct=ncg_result["w_pct"] * 100, is_winner=True, color="#f30b45")
        _draw_team_box(c, ncg_x, mid_y - 80, ncg_result["l_seed"], ncg_result["loser"], w_pct=ncg_result["l_pct"] * 100, is_winner=False, color="#f30b45")
    if champion:
        c.setFont("Helvetica-Bold", 18)
        c.setFillColor(HexColor("#f30b45"))
        c.drawCentredString(page_w / 2, mid_y - 130, f"CHAMPION: ({champion[0]}) {champion[1]}")
    c.save()
    buf.seek(0)
    return buf.getvalue()
