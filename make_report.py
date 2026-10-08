"""Builds reports/Freight_Rate_Report.pdf from experiment_results.json + scorer chart."""
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

R = json.load(open("reports/experiment_results.json"))
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=18, textColor=colors.HexColor("#064A56"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, textColor=colors.HexColor("#064A56"), spaceBefore=10)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
S = ParagraphStyle("S", parent=B, fontSize=8.5, textColor=colors.HexColor("#455A60"))

def tbl(rows, widths):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#064A56")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9DAFB3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF3F4")])]))
    return t

def fold_rows(key):
    return [[x["test"].replace("2025-", "").replace("..", " to "), x["MAE"], f'{x["MAPE_%"]}%', f'{x["MedAPE_%"]}%', x["R2"]] for x in R[key]]

fin = R["final_config_on_folds"]; fe = R["folds_final_features(no_market,no_qs)"]; mk = R["folds_plus_market_index"]; qs = R["folds_plus_market_and_quote_signal"]
cs = R["cold_start_8_unseen_cities"]; corr = R["quote_signal_corr_with_log_distance"]
E = []
E += [Paragraph("Freight Rate Prediction: Validation and Split Approach", H1),
      Paragraph("Model: LightGBM on log(posted_rate). Labeled data: train_test.csv (48,000 loads, 2025-01-01 to 2025-10-31). "
                "Scored data: validation.csv (12,000 loads, 2025-11-01 to 2025-12-31).", B)]

E += [Paragraph("1. How the labeled data was split", H2),
      Paragraph("The labeled loads end on Oct 31 and the scored loads start on Nov 1, so the model is deployed forward in time. "
                "I therefore validated forward in time and did not use random K-fold, which would let the model see neighbouring days "
                "and overstate accuracy.", B),
      Paragraph("<b>Forward-chaining folds</b> (used for every modelling decision): train Jan-Apr, test May-Jun; train Jan-Jun, test Jul-Aug.<br/>"
                "<b>Untouched holdout</b> (final check): train Jan-Aug, test Sep-Oct. Two months matches the Nov-Dec scoring horizon.<br/>"
                "<b>Cold-start check:</b> validation.csv contains 8 cities that never appear in training, so I also held out 8 whole cities "
                "(3 random draws) and scored every load touching them.<br/>"
                "<b>Final model:</b> refit on all 48,000 labeled loads, with a 60-day recency half-life.", B)]

E += [Paragraph("2. Data cleaning and features", H2),
      Paragraph("Cleaning never drops a row, because every validation load must be scored. 292 negative weights are sign errors "
                "(normal magnitude) and were made positive. 300 missing weights stay NaN with a flag. About 1.2% of training rates sit far "
                "from their usual rate per mile; a Huber loss on log(rate) limits their influence.", B),
      Paragraph("Features: distance (raw, log, inverse), haversine distance, road-to-air ratio, weight, equipment, origin and destination "
                "coordinates, direction. Distance dominates (about 97% of gain). Coordinates rather than city IDs let the model handle unseen cities.", B)]

E += [Paragraph("3. Two signals excluded on evidence", H2),
      Paragraph("<b>quote_signal is regime-dependent.</b> In Jan, Feb, Mar, Jun and Sep it matches rate per mile within about 0.7%, which behaves like a leak. "
                "In Apr, May, Jul and Oct the relationship is inverted. In Aug it is uncorrelated. A label-free test separates the regimes: the correlation of "
                f"quote_signal with log distance is about -0.8, +0.8 or 0. For Nov it is {corr['validation-11']} and for Dec {corr['validation-12']}, "
                f"matching Aug ({corr['train-08']}). On an Aug-matched test, adding it raised MAE from {R['regime_matched_Aug']['final']['MAE']} to "
                f"{R['regime_matched_Aug']['with_market_and_qs']['MAE']}.", B),
      Paragraph("<b>market_index is not stable out of time.</b> It helps on May-Jun and fails on Sep-Oct (MAE by test window, below).", B),
      tbl([["Test window", "Final features, no recency wt. MAE", "+ market_index MAE", "+ market_index + quote_signal MAE"]] +
          [[fe[i]["test"].replace("2025-", "").replace("..", " to "), fe[i]["MAE"], mk[i]["MAE"], qs[i]["MAE"]] for i in range(3)],
          [1.5*inch, 1.4*inch, 1.5*inch, 2.2*inch])]

E += [Paragraph("4. Results of the final configuration", H2),
      tbl([["Test window", "MAE ($)", "MAPE", "MedAPE", "R2"]] + fold_rows("final_config_on_folds"), [2.0*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch]),
      Spacer(1, 4),
      Paragraph(f"Sep-Oct is the untouched holdout. Cold-start (8 unseen cities, 3 draws): MedAPE {min(c['MedAPE_%'] for c in cs)}%-{max(c['MedAPE_%'] for c in cs)}%, "
                f"MAPE {min(c['MAPE_%'] for c in cs)}%-{max(c['MAPE_%'] for c in cs)}%. Hyper-parameter changes (leaves, loss, trees) moved mean MAE only between 133 and 135, "
                "so the gains came from data handling and validation design. RMSE (about $630) is dominated by long-haul loads.", B)]

E += [PageBreak(), Paragraph("5. Fixed December chart (produced by score.py)", H2),
      Image("scorer_results/candidate_december.png", width=7.0*inch, height=7.0*inch*0.4444),
      Spacer(1, 6),
      Paragraph("The model has no stable date-dependent signal: day-of-week effects were about 0.3% after detrending and not consistent across periods, "
                "and market_index failed out of time. The chart is therefore nearly flat, a repeating weekly ripple between $829 and $845 (about 1%). "
                "It does not model a holiday effect, because no December history exists in the training data.", B),
      Paragraph("6. Limitations", H2),
      Paragraph("Rate level can drift between Sep-Oct and Nov-Dec in ways the available features cannot see; the May-Jun fold shows the cost of such a shift "
                "(MAE 179). Excluding market_index is a judgment based on its failure on the most recent holdout. Reproduce with run_experiments.py, train.py, score.py.", B)]

SimpleDocTemplate("reports/Freight_Rate_Report.pdf", pagesize=letter, leftMargin=0.7*inch, rightMargin=0.7*inch,
                  topMargin=0.7*inch, bottomMargin=0.7*inch, title="Freight Rate Prediction Report").build(E)
print("ok")
