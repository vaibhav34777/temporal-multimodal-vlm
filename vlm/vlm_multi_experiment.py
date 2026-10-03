import os
import json
import base64
import random
import argparse
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CLASS_NAMES = [
    "Pushing [something] from left to right",
    "Pushing [something] from right to left",
    "Moving [something] up",
    "Moving [something] down",
    "Tearing [something] into two pieces"
]

VLM_PROMPT = (
    "You are shown {n} video frames in chronological order of a single hand action.\n\n"
    "Step 1 - Briefly describe the motion you observe across the frames (1-2 sentences). "
    "Pay close attention to the DIRECTION of movement (left, right, up, down).\n"
    "Step 2 - State which frame number (1 to {n}) provides the clearest evidence of direction and why.\n"
    "Step 3 - Select exactly one label from the numbered list below. "
    "Read each description carefully before deciding:\n"
    "   Label 0 = a hand moves an object HORIZONTALLY from the LEFT side toward the RIGHT side\n"
    "   Label 1 = a hand moves an object HORIZONTALLY from the RIGHT side toward the LEFT side\n"
    "   Label 2 = a hand moves an object VERTICALLY UPWARD (away from the surface, toward the top of frame)\n"
    "   Label 3 = a hand moves an object VERTICALLY DOWNWARD (toward the surface, toward the bottom of frame)\n"
    "   Label 4 = a hand TEARS an object, splitting it into two separate pieces\n\n"
    "End your entire response with this line (replace X with your chosen digit):\n"
    "FINAL_LABEL: X"
)


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Set it in your .env file or as an environment variable.")
    return OpenAI(api_key=api_key)


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_label(response_text):
    for line in reversed(response_text.strip().split('\n')):
        line = line.strip()
        if 'final_label' in line.lower():
            for ch in line:
                if ch in '01234':
                    return int(ch)
    for line in reversed(response_text.strip().split('\n')):
        line = line.strip()
        if line in ('0', '1', '2', '3', '4'):
            return int(line)
    import re
    matches = re.findall(r'(?<!\d)([0-4])(?!\d)', response_text)
    if matches:
        return int(matches[-1])
    return -1


def get_frame_paths(video_id, frames_base_dir, num_frames=4):
    vid_dir = os.path.join(frames_base_dir, str(video_id))
    if not os.path.exists(vid_dir):
        return []
    frame_files = sorted([f for f in os.listdir(vid_dir) if f.endswith('.jpg')])
    if len(frame_files) < 2:
        return []
    indices = [int(i * (len(frame_files) - 1) / (num_frames - 1)) for i in range(num_frames)]
    return [os.path.join(vid_dir, frame_files[idx]) for idx in indices]


def run_vlm_classification(client, frame_paths, model_name="gpt-4o", shuffled=False):
    images = [Image.open(p).convert("RGB").resize((336, 336)) for p in frame_paths]
    if shuffled:
        images = random.sample(images, len(images))

    n = len(images)
    prompt_text = VLM_PROMPT.format(n=n)

    content = []
    for i, img in enumerate(images):
        b64 = image_to_base64(img)
        content.append({
            "type": "text",
            "text": f"Frame {i+1}:"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low"
            }
        })
    content.append({"type": "text", "text": prompt_text})

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=400,
        temperature=0.0
    )

    response_text = response.choices[0].message.content.strip()
    return response_text, parse_label(response_text)


def run_accuracy_experiment(client, df, frames_base_dir, model_name="gpt-4o", num_videos=25, num_frames=4, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    sampled_rows = []
    per_class = num_videos // len(CLASS_NAMES)
    for cls_name in CLASS_NAMES:
        subset = df[df['class_name'] == cls_name]
        n_sample = min(per_class, len(subset))
        sampled_rows.append(subset.sample(n=n_sample, random_state=seed))
    sampled_df = pd.concat(sampled_rows).reset_index(drop=True)

    results = []
    for _, row in sampled_df.iterrows():
        video_id = str(row['video_id'])
        true_label = int(row['class_id'])
        frame_paths = get_frame_paths(video_id, frames_base_dir, num_frames=num_frames)
        if len(frame_paths) < 2:
            continue

        ordered_response, ordered_pred = run_vlm_classification(client, frame_paths, model_name=model_name, shuffled=False)
        shuffled_response, shuffled_pred = run_vlm_classification(client, frame_paths, model_name=model_name, shuffled=True)

        results.append({
            "video_id": video_id,
            "class_name": row['class_name'],
            "true_label": true_label,
            "ordered_pred": ordered_pred,
            "shuffled_pred": shuffled_pred,
            "ordered_correct": int(ordered_pred == true_label),
            "shuffled_correct": int(shuffled_pred == true_label),
            "ordered_response": ordered_response,
            "shuffled_response": shuffled_response,
            "frame_paths": frame_paths
        })
        print(f"  [{video_id}] GT={true_label} ({row['class_name'][:30]}) | Ordered={ordered_pred} ({'OK' if ordered_pred==true_label else 'X'}) | Shuffled={shuffled_pred} ({'OK' if shuffled_pred==true_label else 'X'})")
        if len(results) <= 3:
            print(f"  --- Raw response ---")
            print(ordered_response)
            print(f"  --------------------")

    ordered_acc = 100.0 * sum(r['ordered_correct'] for r in results) / len(results) if results else 0.0
    shuffled_acc = 100.0 * sum(r['shuffled_correct'] for r in results) / len(results) if results else 0.0

    print(f"\n========== VLM Classification Results ({model_name}) ==========")
    print(f"  Videos evaluated: {len(results)}")
    print(f"  Ordered Frames  Accuracy: {ordered_acc:.2f}%")
    print(f"  Shuffled Frames Accuracy: {shuffled_acc:.2f}%")
    print(f"  Accuracy drop (ordered - shuffled): {ordered_acc - shuffled_acc:.2f}pp")
    print(f"==============================================================\n")

    return results, ordered_acc, shuffled_acc


def pick_report_cases(results):
    seen_classes = {}
    for rec in results:
        cls = rec['class_name']
        if cls not in seen_classes:
            seen_classes[cls] = rec
        if len(seen_classes) == len(CLASS_NAMES):
            break
    return [seen_classes[cls] for cls in CLASS_NAMES if cls in seen_classes]


def create_vlm_pdf_report(records, ordered_acc, shuffled_acc, model_name, output_pdf_path):
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36, leftMargin=36,
        topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=14, leading=16,
        textColor=colors.HexColor("#1A365D"), spaceAfter=6)

    subtitle_style = ParagraphStyle('DocSub', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10, leading=13,
        textColor=colors.HexColor("#4A5568"), spaceAfter=12)

    heading_style = ParagraphStyle('SectionHead', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=11, leading=14,
        textColor=colors.HexColor("#2B6CB0"), spaceBefore=8, spaceAfter=4)

    body_style = ParagraphStyle('BodyText', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=12,
        textColor=colors.HexColor("#2D3748"))

    story = []

    story.append(Paragraph(f"VLM Experiment: {model_name} — Ordered vs. Shuffled Frame Classification", title_style))
    story.append(Paragraph(
        f"Model: {model_name} &nbsp;|&nbsp; Frames per video: 4 &nbsp;|&nbsp; Videos evaluated: {len(records)}",
        subtitle_style))

    summary_data = [
        ["Condition", "Accuracy"],
        ["Ordered Frames", f"{ordered_acc:.2f}%"],
        ["Shuffled Frames", f"{shuffled_acc:.2f}%"],
        ["Drop (Ordered − Shuffled)", f"{ordered_acc - shuffled_acc:.2f}pp"],
    ]
    summary_table = Table(summary_data, colWidths=[300, 200])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F7FAFC"), colors.HexColor("#EDF2F7")]),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 18))

    story.append(Paragraph("Case Studies — One Example per Action Class (Ordered Frames)", heading_style))
    story.append(Spacer(1, 6))

    report_cases = pick_report_cases(records)

    for idx, rec in enumerate(report_cases):
        story.append(Paragraph(f"Case {idx+1}: {rec['class_name']}", heading_style))
        story.append(Paragraph(
            f"<b>Video ID:</b> {rec['video_id']} &nbsp;|&nbsp; "
            f"<b>Ground Truth Label:</b> {rec['true_label']} &nbsp;|&nbsp; "
            f"<b>Ordered Prediction:</b> {rec['ordered_pred']} ({'Correct' if rec['ordered_correct'] else 'Wrong'}) &nbsp;|&nbsp; "
            f"<b>Shuffled Prediction:</b> {rec['shuffled_pred']} ({'Correct' if rec['shuffled_correct'] else 'Wrong'})",
            subtitle_style))

        img_cells = []
        label_cells = []
        for f_idx, img_path in enumerate(rec['frame_paths']):
            rl_img = RLImage(img_path, width=110, height=82)
            img_cells.append(rl_img)
            label_cells.append(Paragraph(f"<font size=8><b>Frame {f_idx+1}</b></font>", body_style))

        col_w = [120] * len(rec['frame_paths'])
        img_table = Table([img_cells, label_cells], colWidths=col_w)
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 6))

        truncated_response = rec['ordered_response'][:700] + ("..." if len(rec['ordered_response']) > 700 else "")
        story.append(Paragraph("<b>Model Chain-of-Thought (Ordered):</b>", body_style))
        story.append(Paragraph(truncated_response.replace('\n', '<br/>'), body_style))
        story.append(Spacer(1, 4))

        shuffled_truncated = rec['shuffled_response'][:400] + ("..." if len(rec['shuffled_response']) > 400 else "")
        story.append(Paragraph("<b>Model Response (Shuffled):</b>", body_style))
        story.append(Paragraph(shuffled_truncated.replace('\n', '<br/>'), body_style))

        if idx < len(report_cases) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"VLM Report PDF saved to: {output_pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_csv', type=str, default='/kaggle/working/val_subset.csv')
    parser.add_argument('--frames_dir', type=str, default='/kaggle/working/frames')
    parser.add_argument('--num_videos', type=int, default=25)
    parser.add_argument('--num_frames', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/results')
    parser.add_argument('--results_json', type=str, default='/kaggle/working/results/vlm_accuracy_results.json')
    parser.add_argument('--model_name', type=str, default='gpt-4o')
    args, _ = parser.parse_known_args()

    val_csv = args.val_csv
    if not os.path.exists(val_csv):
        val_csv = "data/val_subset.csv"
    frames_dir = args.frames_dir
    if not os.path.exists(frames_dir):
        frames_dir = "frames"

    df = pd.read_csv(val_csv)
    client = get_client()

    print(f"\nRunning classification on {args.num_videos} videos (ordered + shuffled) using {args.model_name}...")
    results, ordered_acc, shuffled_acc = run_accuracy_experiment(
        client, df, frames_dir,
        model_name=args.model_name,
        num_videos=args.num_videos,
        num_frames=args.num_frames
    )

    os.makedirs(args.output_dir, exist_ok=True)
    serializable = [{k: v for k, v in r.items() if k != 'frame_paths'} for r in results]
    with open(args.results_json, "w") as f:
        json.dump({
            "model": args.model_name,
            "ordered_acc": ordered_acc,
            "shuffled_acc": shuffled_acc,
            "records": serializable
        }, f, indent=2)
    print(f"Results JSON saved to: {args.results_json}")

    pdf_path = os.path.join(args.output_dir, "vlm_experiment_report.pdf")
    create_vlm_pdf_report(results, ordered_acc, shuffled_acc, args.model_name, pdf_path)
