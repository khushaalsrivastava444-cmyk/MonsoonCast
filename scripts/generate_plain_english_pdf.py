#!/usr/bin/env python3
"""
Generate a clear, human-language, intuitive PDF guide for MonsoonCast.
Explains the entire project from scratch in plain English, with clear analogies,
zero unnecessary jargon, and deep explanations of every architectural, meteorological,
and machine learning decision.
"""

import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header
        self.drawString(54, 11 * 72 - 36, "MonsoonCast — The Human-Language Guide to Extreme Rainfall AI")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)

        # Running Footer
        self.line(54, 46, 8.5 * 72 - 54, 46)
        self.drawString(54, 34, "MonsoonCast Plain-English Technical & Conceptual Manual")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 34, page_text)
        self.restoreState()

def create_callout(text, title=None, bg_color="#F8FAFC", border_color="#2563EB", text_color="#1E293B"):
    styles = getSampleStyleSheet()
    title_html = f"<b><font color='{border_color}'>{title}</font></b><br/><br/>" if title else ""
    p_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor(text_color)
    )
    p = Paragraph(f"{title_html}{text}", p_style)
    t = Table([[p]], colWidths=[7.0 * 72])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
        ('LINELEFT', (0,0), (-1,-1), 3.5, colors.HexColor(border_color)),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    return t

def build_plain_english_pdf(output_path: str):
    pdf_path = Path(output_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    base_styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CoverTitle',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#0F172A")
    )
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=17,
        textColor=colors.HexColor("#2563EB")
    )
    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#475569")
    )
    h1_style = ParagraphStyle(
        'Header1',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Header2',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1D4ED8"),
        spaceBefore=11,
        spaceAfter=5,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=4,
        spaceAfter=6
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=base_styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#1E293B")
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=base_styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.white
    )

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 15))
    story.append(Paragraph("MonsoonCast Explained", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("The Plain-English, Human-Language Guide to Predicting India's Extreme Monsoon Rain with Machine Learning", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2563EB"), spaceBefore=4, spaceAfter=14))

    intro_summary = (
        "<b>What is this document?</b> Most machine learning and meteorological papers are dense, full of Greek letters, "
        "and written in a way that obscures the simple, elegant ideas underneath. This guide is different. "
        "It explains MonsoonCast in straightforward, conversational human language — starting from the physical reality of rain in India, "
        "walking through why classical supercomputer forecasts fail at localized downpours, explaining why we picked every single tool in our tech stack, "
        "and detailing how our machine learning models, probability calibrations, and uncertainty estimates work under the hood."
    )
    story.append(Paragraph(intro_summary, body_style))
    story.append(Spacer(1, 12))

    meta_data = [
        [
            Paragraph("<b>Creator & Engineer:</b> Khushaal Srivastava<br/><b>Project Name:</b> MonsoonCast (v0.3.0)<br/><b>Focus Region:</b> Central India Core Monsoon Zone", meta_style),
            Paragraph("<b>Web App:</b> monsooncast-popwidywnbpu3rgwkn5skh.streamlit.app<br/><b>Cloud API:</b> monsooncast-api.onrender.com<br/><b>GitHub:</b> khushaalsrivastava444-cmyk/MonsoonCast", meta_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[3.5 * 72, 3.5 * 72])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Table of Contents: What You Will Learn</b>", h2_style))
    toc_data = [
        [Paragraph("<b>Chapter</b>", table_header), Paragraph("<b>What We Discuss in Plain English</b>", table_header)],
        [Paragraph("1. The Problem", table_cell), Paragraph("Why monsoon rain makes or breaks India, and why extreme downpours (>50 mm) cause disasters.", table_cell)],
        [Paragraph("2. Why Supercomputers Struggle", table_cell), Paragraph("The 'butterfly effect', why 15-kilometer grid boxes miss clouds, and the double-penalty problem.", table_cell)],
        [Paragraph("3. The Ground Rules", table_cell), Paragraph("Why we picked Central India, what '50 mm/day' means, and why extreme rain is a 3% rarity puzzle.", table_cell)],
        [Paragraph("4. The Ingredients (Data)", table_cell), Paragraph("IMD rain gauges, ERA5 winds and atmospheric moisture, and our physics-based synthetic data generator.", table_cell)],
        [Paragraph("5. The Rule Against Cheating", table_cell), Paragraph("Why random cross-validation is fatal in weather, chronological splitting, and the May 31 bug.", table_cell)],
        [Paragraph("6. Feature Engineering", table_cell), Paragraph("How we turn yesterday's weather, spatial storm clusters, and atmospheric moisture into AI clues.", table_cell)],
        [Paragraph("7. Why This Tech Stack?", table_cell), Paragraph("Why Python 3.12, NumPy, Xarray, XGBoost, PyTorch, SciPy, FastAPI, Streamlit, and Render.", table_cell)],
        [Paragraph("8. The 7 Models in Our Zoo", table_cell), Paragraph("From simple calendar lookups and lazy guesses to Random Forests, XGBoost, and PyTorch deep neural nets.", table_cell)],
        [Paragraph("9. The Council of Models", table_cell), Paragraph("Why simple averaging fails, and how we use math to give the best models the highest votes.", table_cell)],
        [Paragraph("10. Making AI Tell the Truth", table_cell), Paragraph("What probability calibration is, why raw AI scores lie, and how we fix it so '70%' actually means 70%.", table_cell)],
        [Paragraph("11. Knowing What You Don't Know", table_cell), Paragraph("The difference between nature's chaos (aleatoric) and model confusion (epistemic uncertainty).", table_cell)],
        [Paragraph("12. The 97% Accuracy Trap", table_cell), Paragraph("Why accuracy is useless for rare events, and why PR-AUC and Brier Skill Score are the real tests.", table_cell)],
        [Paragraph("13. How It Works in Real Life", table_cell), Paragraph("How a user clicks on the Streamlit map, how the FastAPI backend answers, and what's next.", table_cell)]
    ]
    toc_table = Table(toc_data, colWidths=[2.0 * 72, 5.0 * 72])
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(toc_table)
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 1: THE BIG PICTURE
    # =========================================================================
    story.append(Paragraph("Chapter 1: The Big Picture — What is MonsoonCast?", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "Every year between June and September, India receives more than 75% of its entire annual rainfall. "
        "This four-month period is known as the <b>Indian Summer Monsoon (JJAS)</b>. "
        "It fills the reservoirs that supply drinking water to over a billion people, waters millions of acres of farmland, and powers the rural economy. "
        "When the monsoon arrives on time and in moderation, India prospers.",
        body_style
    ))
    story.append(Paragraph(
        "<b>But the monsoon has a dark side.</b> Rainfall does not fall gently and evenly across four months. "
        "Instead, massive storm systems dump immense volumes of water in short, violent bursts. "
        "When 50, 80, or 120 millimeters of rain falls on a city or farming district in a single day, the ground cannot absorb it. "
        "Streets turn into rushing rivers, subways flood, earthen dam reservoirs breach, topsoil washes away, and lives are lost. "
        "These catastrophic downpours are called <b>Extreme Rainfall Events</b>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The goal of MonsoonCast is simple to state, but very hard to achieve:</b> "
        "Given yesterday's weather observations and the current state of the atmosphere, can we predict whether a specific 50 km by 50 km region will experience an extreme downpour tomorrow? "
        "And even more importantly: <b>can we tell disaster response officials exactly how confident we are in that prediction?</b>",
        body_style
    ))

    analogy_1 = (
        "<b>An Everyday Analogy:</b> Imagine planning your morning commute. If an app tells you 'there is a 10% chance of rain', you probably leave your umbrella at home. "
        "If it tells you 'there is an 80% chance of rain', you bring one. But if a disaster official is deciding whether to evacuate 10,000 families from a low-lying river basin, "
        "they cannot work with a simple 'Yes' or 'No'. A false alarm costs millions of dollars in panic and disrupts lives; a missed prediction kills people. "
        "They need honest, calibrated probabilities: 'There is a 65% likelihood of torrential rain, and we estimate between 45 and 90 millimeters will fall.'"
    )
    story.append(create_callout(analogy_1, "Why Probabilities Matter More Than Yes/No Guesses", "#EFF6FF", "#2563EB", "#1E3A8A"))
    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 2: WHY SUPERCOMPUTERS STRUGGLE
    # =========================================================================
    story.append(Paragraph("Chapter 2: Why Supercomputers Struggle with Extreme Rain", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "You might wonder: <i>'Doesn't the government already run giant supercomputers for weather forecasting?'</i> Yes, they do. "
        "Agencies like ECMWF in Europe, NOAA in the United States, and NCMRWF/IMD in India run what are called <b>Numerical Weather Prediction (NWP) models</b>. "
        "These are massive physics simulations that divide the entire atmosphere into millions of 3D grid boxes and solve fluid dynamics equations. "
        "NWP models are fantastic at predicting large-scale weather — like whether next Tuesday will be cooler than today, or where a massive 500-kilometer cyclone is heading across the ocean. "
        "<b>However, when it comes to localized extreme rainfall, supercomputers struggle badly. Here is why:</b>",
        body_style
    ))

    nwp_reasons = (
        "<b>1. The Resolution Problem (Looking Through a Blurry Lens):</b> Operational global supercomputers divide the world into boxes that are 9 to 25 kilometers wide. "
        "A violent thunderstorm cloud (a cumulonimbus) is often only 1 to 3 kilometers wide. The computer literally cannot 'see' the thunderstorm. "
        "Instead, the model has to make an educated guess using an approximation called a 'parameterization scheme'. These approximations tend to spread rain out smoothly over huge areas, predicting gentle drizzle everywhere and completely missing the intense cloudburst in the middle.<br/><br/>"
        "<b>2. The 'Double Penalty' Problem:</b> Imagine a storm cloud forms and dumps 90 mm of rain on City A. The supercomputer predicted that exact 90 mm downpour, but it placed it 50 kilometers away over City B. "
        "In the scoring book, the supercomputer gets penalized twice: it gave a false alarm for City B, and it completely missed the storm in City A. "
        "Even when the physics model knew an extreme storm was brewing, its spatial displacement ruins the forecast.<br/><br/>"
        "<b>3. The Overconfidence Problem:</b> Standard physics models output a single deterministic number (e.g. '14 mm'). They don't tell emergency managers whether the atmosphere is stable and predictable, or on a razor's edge where a tiny gust of wind could trigger a massive deluge."
    )
    story.append(create_callout(nwp_reasons, "The Three Traps of Classical Physics Weather Models", "#FEF2F2", "#DC2626", "#7F1D1D"))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 3: THE GROUND RULES
    # =========================================================================
    story.append(Paragraph("Chapter 3: The Ground Rules — Central India & The 3% Puzzle", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph("3.1 Why Central India? (18.0°–25.0°N, 73.0°–82.0°E)", h2_style))
    story.append(Paragraph(
        "India has many different climate regions. In Mumbai and the Western Ghats mountains, heavy rain is forced by topography: "
        "moisture-laden ocean winds hit a wall of mountains, rise rapidly, cool, and drop rain in the same spot day after day. "
        "Predicting rain there is relatively easy because the mountains never move. "
        "MonsoonCast focuses instead on the <b>Central India Core Monsoon Zone</b> (spanning parts of Maharashtra, Madhya Pradesh, Chhattisgarh, and Gujarat). "
        "Here, there are no giant mountain ranges to force rain. Rain here is driven entirely by dynamic atmospheric battles: "
        "swirling cyclonic storm vortices, called <b>Monsoon Depressions</b>, that spawn in the warm waters of the Bay of Bengal and march inland across the plains. "
        "If an AI model can predict rain here, it means the model is genuinely learning how the atmosphere moves, not just memorizing the location of mountains.",
        body_style
    ))

    story.append(Paragraph("3.2 What Counts as 'Extreme Rain'?", h2_style))
    story.append(Paragraph(
        "We set our threshold at <b>50.0 millimeters of rain in a 24-hour day</b> (measured from 08:30 in the morning to 08:30 the next morning). "
        "For perspective, 50 mm of rain is nearly 2 inches of continuous water standing across every square meter of land. "
        "In agricultural basins, this causes immediate waterlogging; in cities, it drowns streets.",
        body_style
    ))

    story.append(Paragraph("3.3 The 3% Imbalance Puzzle: The Needle in the Haystack", h2_style))
    story.append(Paragraph(
        "Across our historical dataset of Central India, an extreme rain day happens in a given grid cell only about <b>2% to 3.5% of the time</b>. "
        "That means on 97 days out of 100, the weather is normal, light, or dry. "
        "This sounds great for daily life, but it is a massive trap for machine learning! "
        "If you build a lazy AI model that simply outputs <i>'No extreme rain today'</i> every single day, that lazy model will achieve <b>97% accuracy</b>! "
        "It looks brilliant on paper, but it is 100% useless because it fails to predict every single disaster. "
        "MonsoonCast is built from the ground up to conquer this severe class imbalance.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 4: THE INGREDIENTS (DATA)
    # =========================================================================
    story.append(Paragraph("Chapter 4: The Ingredients — Where Does the Data Come From?", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "To teach an AI to predict tomorrow's storms, you need two kinds of data: "
        "<b>what happened on the ground</b>, and <b>what is happening up in the sky</b>.",
        body_style
    ))

    data_cards = (
        "<b>1. Ground Truth (Rain Gauges on Earth):</b> We use high-resolution daily gridded rainfall data from the <b>India Meteorological Department (IMD)</b>. "
        "The IMD collects rain gauge measurements from over 6,500 weather stations across India and mathematically interpolates them onto a regular grid. "
        "We aggregate this onto a standard 0.5-degree grid (boxes roughly 55 km by 55 km, giving 285 active cells across Central India).<br/><br/>"
        "<b>2. Atmospheric Variables (Winds, Pressure & Moisture in the Sky):</b> Rain does not come from nowhere; it is cooked in the atmosphere. "
        "We ingest five crucial variables from the European weather agency's <b>ERA5 atmospheric reanalysis</b>:<br/>"
        "&bull; <i>u850 and v850 Winds (at 1.5 km altitude):</i> The low-level 'Somali Jet' that acts as a giant conveyor belt, dragging millions of tons of water vapor from the Arabian Sea across India.<br/>"
        "&bull; <i>z500 Geopotential Height (at 5.5 km altitude):</i> The mid-level steering wheel of the storm. Dips in this layer show low-pressure troughs pulling air upward.<br/>"
        "&bull; <i>Total Precipitable Water (TPW / tcwv):</i> How much water vapor is hanging in the air column directly above that grid cell. If TPW is low (< 45 kg/m²), extreme rain is virtually impossible; if it exceeds 60 kg/m², the air is soaked like a wet sponge.<br/>"
        "&bull; <i>CAPE (Convective Energy):</i> The atmospheric gunpowder. It measures how much buoyant energy is available to blast warm air upward into towering thunderclouds.<br/><br/>"
        "<b>3. The Synthetic Data Generator (Why It Was Built):</b> Raw IMD binary files are protected by government registration agreements and cannot be freely redistributed on GitHub. "
        "To make sure MonsoonCast can be downloaded, run, and verified by any engineer or researcher instantly without broken paths, we built a <b>physics-consistent synthetic generator</b>. "
        "It doesn't just create random numbers; it simulates realistic moving weather fronts, spatial storm clusters, and Gamma-distributed rainfall tails."
    )
    story.append(create_callout(data_cards, "The Raw Datasets We Feed Into the AI", "#F0FDF4", "#16A34A", "#14532D"))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 5: THE RULE AGAINST CHEATING
    # =========================================================================
    story.append(Paragraph("Chapter 5: The Golden Rule — Don't Cheat (Anti-Leakage)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "In machine learning, there is a deadly trap known as <b>Data Leakage</b> — where the model accidentally peeks at the answers during training. "
        "In weather forecasting, data leakage is shockingly easy to commit by accident. Here is how MonsoonCast guarantees causal integrity:",
        body_style
    ))

    story.append(Paragraph("5.1 Why Random K-Fold Shuffling is Fatal in Weather", h2_style))
    story.append(Paragraph(
        "In standard tabular machine learning (like predicting whether someone will click an ad), data scientists routinely shuffle all the rows and randomly pick 80% for training and 20% for testing. "
        "<b>If you do that with weather data, you have completely broken reality.</b> "
        "Think about it: suppose a massive monsoon storm hits central India on July 15, 2021. "
        "If you randomly shuffle the data, the city of Bhopal on July 15 ends up in the training set, while the neighboring city of Indore (only 180 km away) on that exact same day ends up in the test set. "
        "The model doesn't have to learn how weather evolves; it just peeks at what Bhopal did 2 hours ago and copies the answer for Indore! "
        "The test score will look 99% perfect. But when you try to use that model next year, it fails completely because it never learned real forecasting.",
        body_style
    ))

    story.append(Paragraph("5.2 Strict Chronological Splitting (Time Travel is Forbidden)", h2_style))
    story.append(Paragraph(
        "MonsoonCast uses strict chronological time partitions. The model only ever trains on the past, tunes on the middle past, and is tested on the future:",
        body_style
    ))

    time_splits_data = [
        [Paragraph("<b>Time Partition</b>", table_header), Paragraph("<b>Seasons</b>", table_header), Paragraph("<b>Sample Count</b>", table_header), Paragraph("<b>How It Is Used</b>", table_header)],
        [
            Paragraph("<b>Training Partition</b>", table_cell),
            Paragraph("2010 – 2018 (9 years)", table_cell),
            Paragraph("<b>310,365 samples</b>", table_cell),
            Paragraph("The model trains only on these past years. All scaling statistics (mean, variance) are learned strictly here.", table_cell)
        ],
        [
            Paragraph("<b>Validation Partition</b>", table_cell),
            Paragraph("2019 – 2020 (2 years)", table_cell),
            Paragraph("<b>68,970 samples</b>", table_cell),
            Paragraph("Used as an independent exam to tune decision thresholds, calibrate probabilities, and find the best ensemble weights.", table_cell)
        ],
        [
            Paragraph("<b>Held-Out Test Partition</b>", table_cell),
            Paragraph("2021 – 2023 (3 years)", table_cell),
            Paragraph("<b>103,455 samples</b>", table_cell),
            Paragraph("The final exam. The model has never seen these three years. This proves whether the system works on unseen future monsoon seasons.", table_cell)
        ]
    ]
    t_table = Table(time_splits_data, colWidths=[1.5 * 72, 1.4 * 72, 1.5 * 72, 2.6 * 72])
    t_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("5.3 The 'May 31 for June 1' Invariant", h2_style))
    story.append(Paragraph(
        "Here is a real engineering bug that trips up many weather pipelines: "
        "The official monsoon runs from June 1 to September 30. If you filter your data to June 1 before calculating lag features (what happened yesterday, two days ago, etc.), "
        "what happens to June 1? Yesterday was May 31, but you deleted May! "
        "Naive code fills those missing lags with zeros. As a result, the model is crippled on June 1 because it falsely thinks not a single drop of rain fell anywhere in India yesterday. "
        "<b>How MonsoonCast solved it:</b> We keep the full unbroken 365-day rainfall stream intact, calculate all 1-day, 2-day, and 7-day lags across May and June smoothly, and only filter to the target dates *after* the features are built. "
        "June 1 gets its authentic May 31 history.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 6: FEATURE ENGINEERING
    # =========================================================================
    story.append(Paragraph("Chapter 6: Feature Engineering — Turning Weather into Clues", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "A machine learning algorithm doesn't know what a 'monsoon' is. It only sees rows of numbers. "
        "<b>Feature Engineering</b> is the art of translating our meteorological domain knowledge into numbers that give the AI the strongest possible clues. "
        "At forecast issue time $t$, MonsoonCast calculates 20 distinct clues for every grid cell:",
        body_style
    ))

    feat_explanations = (
        "<b>1. Autoregressive Lags (Yesterday, 2 Days Ago, 3 Days Ago):</b> If a grid cell received 60 mm of rain yesterday, an active storm system is already on top of it. Rain today is much more likely than if it had been bone-dry for two weeks.<br/><br/>"
        "<b>2. Rolling 3-Day & 7-Day Rainfall (Soil Moisture Proxy):</b> When heavy rain falls for a week, the soil becomes totally saturated with water (like a soaked towel). "
        "Saturated soil cannot absorb new rain, so 100% of new precipitation turns into surface runoff. Furthermore, wet ground evaporates moisture directly back into the lower air, feeding the storm.<br/><br/>"
        "<b>3. Neighborhood Spatial Pooling (Looking at the 8 Neighbors):</b> Extreme storms are not 1-pixel pinpricks; they are giant mesoscale clusters spanning 100 to 200 km. "
        "By looking at the mean and maximum rainfall in the $3 \\times 3$ grid cells around each point, the AI detects whether a storm system is roaring in from the east.<br/><br/>"
        "<b>4. Spatial Gradients (Storm Fronts):</b> When cold rain-cooled air rushes out of a thunderstorm, it slams into warm moist air. This boundary is called a gust front, and it creates sharp spatial differences in rainfall. High spatial gradients tell the AI where new thunderstorms are being triggered.<br/><br/>"
        "<b>5. Climatological Anomalies (Is This Normal?):</b> 20 mm of rain on June 5th (early monsoon) is a huge anomaly. 20 mm of rain on July 25th (peak monsoon) is completely ordinary. "
        "By subtracting the 30-year historical average for that specific day of the year, we tell the AI whether the atmosphere is acting unusually crazy.<br/><br/>"
        "<b>6. Atmospheric Kinematics (Wind Shear & Vorticity):</b> When the low-level jet (1.5 km) and the mid-level winds (5.5 km) blow in different directions, that wind shear tilts convective updrafts so they don't drown in their own rain, allowing storms to survive for hours.<br/><br/>"
        "<b>7. Cyclic Day-of-Year Encoding:</b> Instead of feeding the calendar day as a number from 1 to 365 (which makes day 365 and day 1 look infinitely far apart), we convert the day into $\\sin$ and $\\cos$ angles around a 360-degree circle. The AI understands seasonal cycles continuously."
    )
    story.append(create_callout(feat_explanations, "The 20 Meteorological Clues We Feed the AI", "#F8FAFC", "#2563EB", "#0F172A"))
    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 7: WHY THIS TECH STACK?
    # =========================================================================
    story.append(Paragraph("Chapter 7: Why This Tech Stack? Plain-English Decisions", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "Every tool in MonsoonCast was chosen for a specific engineering reason. We avoided vanity tools and heavy frameworks in favor of speed, reliability, and precision:",
        body_style
    ))

    stack_cards_data = [
        [Paragraph("<b>Tool / Library</b>", table_header), Paragraph("<b>What It Does</b>", table_header), Paragraph("<b>Why We Chose It (The Plain English Reason)</b>", table_header)],
        [
            Paragraph("<b>Python 3.12</b>", table_cell),
            Paragraph("Core programming language", table_cell),
            Paragraph("Runs about 15% faster than older Python versions, has clean typing, and is universally supported across every scientific ML library.", table_cell)
        ],
        [
            Paragraph("<b>Xarray & NetCDF4</b>", table_cell),
            Paragraph("Multi-dimensional climate data", table_cell),
            Paragraph("Pandas is great for 2D tables, but weather is a 4D cube (Time, Altitude, Latitude, Longitude). Xarray lets us slice through weather cubes like butter without running out of RAM.", table_cell)
        ],
        [
            Paragraph("<b>XGBoost</b>", table_cell),
            Paragraph("Gradient boosted decision trees", table_cell),
            Paragraph("The undefeated king of tabular data. It naturally handles missing sensors, handles non-linear weather thresholds, and lets us configure <code>scale_pos_weight=5.0</code> to punish missed rare storms.", table_cell)
        ],
        [
            Paragraph("<b>PyTorch</b>", table_cell),
            Paragraph("Deep learning neural networks", table_cell),
            Paragraph("Allows us to build a custom tabular neural network and keep Dropout turned ON during testing to sample epistemic model uncertainty.", table_cell)
        ],
        [
            Paragraph("<b>SciPy SLSQP</b>", table_cell),
            Paragraph("Constrained mathematical optimizer", table_cell),
            Paragraph("The referee that calculates the exact weights for combining our models. It guarantees that weights are never negative and always add up to 100%.", table_cell)
        ],
        [
            Paragraph("<b>FastAPI & Pydantic</b>", table_cell),
            Paragraph("RESTful microservice backend", table_cell),
            Paragraph("Blazing fast asynchronous web server. Pydantic validates incoming weather requests in microseconds, preventing corrupted inputs from ever reaching the models.", table_cell)
        ],
        [
            Paragraph("<b>Streamlit</b>", table_cell),
            Paragraph("Interactive scientific frontend", table_cell),
            Paragraph("Turns pure Python into an interactive visual dashboard with sliders, map coordinate pickers, and live charts without writing separate JavaScript.", table_cell)
        ],
        [
            Paragraph("<b>Pytest & Ruff</b>", table_cell),
            Paragraph("Testing & quality enforcement", table_cell),
            Paragraph("Ruff lints the whole repo in 30 milliseconds. Pytest runs 28 rigorous unit tests in 5 seconds to verify no data leakage ever enters the code.", table_cell)
        ]
    ]
    st_table = Table(stack_cards_data, colWidths=[1.4 * 72, 1.8 * 72, 3.8 * 72])
    st_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(st_table)
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 8: THE MODEL ZOO
    # =========================================================================
    story.append(Paragraph("Chapter 8: The Model Zoo — Meet the 7 Contestants", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "Instead of betting the entire project on a single model, MonsoonCast trains a diverse lineup of 7 different estimators. "
        "Think of them as a team of specialists with completely different backgrounds:",
        body_style
    ))

    model_zoo_text = (
        "<b>1. Seasonal Climatology (The Calendar Baseline):</b> "
        "<i>How it thinks:</i> 'What percentage of July 15ths in history had an extreme downpour at this spot?' "
        "It knows that late July is wetter than early June, but it is blind to today's weather. Any real AI model MUST beat this baseline to prove it has skill.<br/><br/>"
        "<b>2. Markov Persistence (The Lazy Forecaster):</b> "
        "<i>How it thinks:</i> 'If it rained heavily today, it will probably rain heavily tomorrow. If it was dry, tomorrow will be dry.' "
        "It has moderate skill during multi-day storm spells, but gives terrible false alarms when the storm moves away overnight.<br/><br/>"
        "<b>3. Penalized Logistic Regression (The Balanced Line):</b> "
        "A fast linear model. Because extreme events are so rare, we use class weights that multiply the penalty of missing an extreme event by 30x. Ridge ($L_2$) regularization stops correlated weather variables from blowing up the weights.<br/><br/>"
        "<b>4. Random Forest Classifier (The Committee of 100 Trees):</b> "
        "100 decision trees vote. Each tree looks at a random subset of weather clues. Trees excel at non-linear rules (e.g. 'If atmospheric moisture > 58 kg/m² AND wind is westerly, probability jumps to 70%').<br/><br/>"
        "<b>5. XGBoost Classifier (The Champion Gradient Booster):</b> "
        "Unlike Random Forest where trees work independently, XGBoost trains trees sequentially — each new tree specifically tries to fix the mistakes of the previous trees. With <code>scale_pos_weight=5.0</code>, it is intensely focused on catching rare extremes.<br/><br/>"
        "<b>6. PyTorch Tabular Neural Network (The Deep Thinker):</b> "
        "A deep neural network with Batch Normalization and Dropout layers. It maps complex non-linear interactions across all 20 features simultaneously.<br/><br/>"
        "<b>7. XGBoost Quantile Regressors (The Millimeter Estimator):</b> "
        "The first 6 models output probability of crossing 50 mm. This model predicts actual rainfall amounts: a lower bound ($q_{10}$), an expected median ($q_{50}$), and a worst-case storm scenario ($q_{90}$) in millimeters."
    )
    story.append(create_callout(model_zoo_text, "The 7 Models Inside MonsoonCast", "#F0F9FF", "#0284C7", "#0C4A6E"))
    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 9: THE COUNCIL OF MODELS
    # =========================================================================
    story.append(Paragraph("Chapter 9: The Council of Models — Probabilistic Ensembling", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "Once you have trained multiple models, how do you combine their answers? "
        "The easiest and most common way in amateur data science is simple averaging: $(M_1 + M_2 + M_3 + M_4) / 4$. "
        "<b>Simple averaging is a terrible idea in extreme weather forecasting.</b> "
        "If you have an Olympic sprinter (XGBoost) and a novice runner (Persistence baseline), averaging their race times just slows down the sprinter! "
        "You should give the strongest models high weight, and weak baselines very low or zero weight.",
        body_style
    ))
    story.append(Paragraph(
        "<b>How MonsoonCast builds its ensemble:</b> We set up a mathematical competition on the independent <b>Validation Partition (2019–2020)</b>. "
        "We ask a quadratic optimizer (SciPy's SLSQP algorithm) to find the combination of weights that minimizes the total prediction error (Brier Score):",
        body_style
    ))

    ens_analogy = (
        "<b>The Two Rules of the Optimizer:</b><br/>"
        "1. <b>No negative weights (w_i &ge; 0):</b> A model can have zero influence, but it can never have negative influence (which would invert probabilities and make no sense).<br/>"
        "2. <b>The weights must add up to exactly 100% (&Sigma; w_i = 1):</b> This guarantees that the final blended answer is always a clean, valid probability between 0% and 100%.<br/><br/>"
        "<b>The Magic of Error Decorrelation:</b> Why combine them at all? Because decision trees and neural networks make different mistakes. "
        "When an XGBoost tree makes an error on an unusual day, the PyTorch neural net often gets it right, and vice versa. "
        "By blending their opinions, the random errors cancel out, leaving a far more stable and reliable prediction."
    )
    story.append(create_callout(ens_analogy, "Why We Optimize Ensemble Weights Instead of Simple Averaging", "#FEF3C7", "#D97706", "#78350F"))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 10: MAKING AI TELL THE TRUTH (CALIBRATION)
    # =========================================================================
    story.append(Paragraph("Chapter 10: Making AI Tell the Truth — Probability Calibration", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "Here is a shocking secret in machine learning: <b>modern AI models are pathological liars about their own confidence.</b> "
        "When an uncalibrated XGBoost model or neural network outputs '0.85', that does NOT mean there is an 85% chance of rain. "
        "Because XGBoost is trained with <code>scale_pos_weight=5.0</code> to aggressively catch rare storms, it pushes its raw scores outward toward 0% and 100%. "
        "A raw score of 0.85 might only correspond to a real-world event 50% of the time. This is called an <b>uncalibrated model</b>.",
        body_style
    ))

    calib_story = (
        "<b>What is Calibration in Plain English?</b><br/>"
        "Imagine a weather forecaster who predicts 'a 70% chance of rain' on 100 different days throughout the year. "
        "If you check the record books at the end of the year, exactly 70 of those days should have had rain. "
        "If only 30 days had rain, the forecaster was wildly overconfident. If 95 days had rain, the forecaster was timidly underconfident. "
        "When the forecaster's stated percentage perfectly matches real-world historical frequency, that forecaster is <b>calibrated</b>.<br/><br/>"
        "<b>How MonsoonCast Calibrates:</b> We use <b>Isotonic Regression</b>. "
        "On our 68,970 validation examples, we map raw model outputs to true empirical frequencies using a non-decreasing monotonic curve. "
        "If the raw model says 0.82, the calibration engine looks at what actually happened historically when the model said 0.82, and corrects it to 0.61. "
        "This shrinks our <b>Expected Calibration Error (ECE)</b> from a dangerous 0.14 down to an exceptionally honest <b>0.02 - 0.04</b>."
    )
    story.append(create_callout(calib_story, "The Difference Between Raw Scores and Real Probabilities", "#F0FDF4", "#16A34A", "#14532D"))
    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 11: UNCERTAINTY QUANTIFICATION
    # =========================================================================
    story.append(Paragraph("Chapter 11: Knowing What You Don't Know (UQ)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "A great engineer or scientist does not just give an answer; they explain their uncertainty. "
        "MonsoonCast splits uncertainty into two fundamentally different human concepts:",
        body_style
    ))

    uq_human_text = (
        "<b>1. Nature's Chaos (Aleatoric Uncertainty):</b><br/>"
        "This is the irreducible randomness of the universe. Even if you had a supercomputer with infinite memory and a perfect model, "
        "the atmosphere is chaotic (the butterfly effect). Small turbulent air currents mean that under identical synoptic conditions, a storm might drop 40 mm one day and 60 mm the next. "
        "<i>How we express it:</i> We give a calibrated probability (e.g., '62% chance of exceeding 50 mm') and a quantile range (e.g., 'Expect between 35 mm and 85 mm').<br/><br/>"
        "<b>2. The Model's Confusion (Epistemic Uncertainty):</b><br/>"
        "This happens when the model encounters weather it has never seen before — like an unprecedented cyclonic storm track or record sea-surface temperatures. "
        "The model is operating in unfamiliar territory and might be guessing.<br/>"
        "<i>How we catch it:</i> We measure <b>Ensemble Spread</b> (do our 4 different models agree or are they arguing?) and <b>Monte Carlo Dropout</b> "
        "(we run the PyTorch neural network 30 times with random sub-circuits turned off; if the answers vary wildly, the neural net is uncertain). "
        "When spread is high, the system flags: <i>'Warning: High model uncertainty — verify with human forecaster.'</i>"
    )
    story.append(create_callout(uq_human_text, "Two Very Different Kinds of Doubt", "#EFF6FF", "#2563EB", "#1E3A8A"))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 12: THE 97% ACCURACY TRAP
    # =========================================================================
    story.append(Paragraph("Chapter 12: The 97% Accuracy Trap — How to Score Weather AI", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "If someone shows you a machine learning model for extreme weather and brags that it has <b>'98% Accuracy'</b>, "
        "you should be deeply skeptical. In fact, they probably built a completely broken model.",
        body_style
    ))

    metrics_guide = (
        "<b>Why Accuracy is a Lie for Rare Events:</b><br/>"
        "In our Central India dataset, extreme rain happens on only ~3% of grid-days. "
        "If a broken model predicts 'Zero rain' every single day, it gets 97 out of 100 days correct. Accuracy = 97%! "
        "It looks like an A+ student, but it failed to catch a single disaster. That is why accuracy is forbidden as our primary score.<br/><br/>"
        "<b>The Real Metrics MonsoonCast Uses:</b><br/>"
        "&bull; <b>Precision-Recall AUC (PR-AUC):</b> The gold standard. It measures how well the model catches rare storms (Recall) without spamming emergency officials with false alarms (Precision). An uninformative coin-flip gets only 0.03; our models score significantly higher.<br/>"
        "&bull; <b>Brier Skill Score (BSS):</b> Measures whether the AI beats the historical calendar. If BSS &le; 0, your AI is worse than just looking up historical averages. If BSS > 0, the AI has genuine predictive skill.<br/>"
        "&bull; <b>Critical Success Index (CSI / Threat Score):</b> The favorite metric of operational weather agencies. It completely throws away the easy 'dry days' and scores the model strictly on how well it predicts the hard storms.<br/>"
        "&bull; <b>Date-Aware Block Bootstrapping:</b> When we calculate confidence intervals (e.g. 'PR-AUC is 0.42 &plusmn; 0.04'), we resample entire calendar days together. This keeps the 285 spatial points of each day intact, preventing artificially narrow and dishonest error bars."
    )
    story.append(create_callout(metrics_guide, "The Real Scorecard for Extreme Weather", "#FEF2F2", "#DC2626", "#7F1D1D"))
    story.append(Spacer(1, 10))

    # =========================================================================
    # CHAPTER 13: HOW IT WORKS IN REAL LIFE
    # =========================================================================
    story.append(Paragraph("Chapter 13: How It Works in Real Life — Web App & Cloud API", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "MonsoonCast is not just a research script on a laptop; it is deployed as a live cloud system with two public interfaces:",
        body_style
    ))

    deploy_boxes = (
        "<b>1. The Cloud REST API (FastAPI on Render):</b><br/>"
        "&bull; <i>Live URL:</i> <code>https://monsooncast-api.onrender.com</code> (Docs: <code>/docs</code>)<br/>"
        "&bull; <i>How it works:</i> Any client (a mobile app, disaster agency portal, or Python script) sends an HTTP POST request to <code>/predict</code> with latitude, longitude, and 20 weather variables. "
        "The API runs the models, performs isotonic calibration, checks ensemble spread, computes quantile bounds, and returns a clean JSON response in under 5 milliseconds.<br/><br/>"
        "<b>2. The Interactive Visual Dashboard (Streamlit Cloud):</b><br/>"
        "&bull; <i>Live URL:</i> <code>https://monsooncast-popwidywnbpu3rgwkn5skh.streamlit.app</code><br/>"
        "&bull; <i>How it works:</i> Anyone with a web browser can click on a map of Central India, adjust risk sensitivity sliders (balancing false alarms vs missed alerts), and view real-time probability dials, uncertainty meters, and rainfall interval charts."
    )
    story.append(create_callout(deploy_boxes, "Public Cloud Endpoints", "#F8FAFC", "#2563EB", "#0F172A"))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Chapter 14: Honest Limitations & The Road Ahead", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    story.append(Paragraph(
        "Great engineering requires honest transparency about what your system can and cannot do: "
        "<br/>&bull; <b>Observation Delay:</b> Official IMD gridded rain data is produced retrospectively after a 1 to 3 day delay for quality checking. A true operational real-time system needs live satellite telemetry (like NASA GPM-IMERG) or automated weather station (AWS) feeds. "
        "<br/>&bull; <b>Grid Resolution:</b> Our 0.5-degree grid is roughly 55 km wide. A sudden cloudburst that floods a single urban neighborhood in 45 minutes is smoothed out over that 55 km box. "
        "<br/>&bull; <b>Future Vision:</b> The next evolution of MonsoonCast aims to downscale to 10 km resolution, extend forecast horizons to 3 and 5 days, and integrate real-time Doppler Weather Radar matrices for 0–6 hour nowcasting.",
        body_style
    ))
    story.append(Spacer(1, 10))

    final_conclusion = (
        "<b>The Bottom Line:</b> MonsoonCast shows that machine learning and atmospheric physics are not competitors — they are partners. "
        "By enforcing strict anti-leakage rules, engineering features grounded in real meteorology, ensembling diverse models, "
        "and calibrating probabilities, we can transform raw weather data into honest, reliable, life-saving forecasts."
    )
    story.append(create_callout(final_conclusion, "Final Takeaway", "#F0FDF4", "#16A34A", "#14532D"))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated Plain-English PDF at: {pdf_path}")
    return str(pdf_path)

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "docs/MonsoonCast_Project_Guide_Plain_English.pdf"
    build_plain_english_pdf(out_file)
