import os
import json
import random
import torch
import pandas as pd
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

CLASS_NAMES = [
    "Pushing [something] from left to right",
    "Pushing [something] from right to left",
    "Moving [something] up",
    "Moving [something] down",
    "Tearing [something] into two pieces"
]

LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

VLM_PROMPT = (
    "You are shown {n} sequential video frames of a hand action.\n"
    "1. Briefly explain what is happening across the sequence.\n"
    "2. Identify which specific frame (e.g. Frame 1, Frame 4) is the most decisive or helpful in recognizing the true action, and explain why.\n"
    "3. Classify the action into exactly one of the following categories:\n"
    "0: Pushing [something] from left to right\n"
    "1: Pushing [something] from right to left\n"
    "2: Moving [something] up\n"
    "3: Moving [something] down\n"
    "4: Tearing [something] into two pieces\n\n"
    "Provide your response in the following exact format:\n"
    "Explanation: <your reasoning>\n"
    "Decisive Frame: <frame number or description>\n"
    "Label: <integer label>"
)


def parse_label(response_text):
    for line in response_text.split('\n'):
        if line.lower().startswith('label:'):
            for ch in line:
                if ch in "01234":
                    return int(ch)
    # fallback
    for ch in response_text[::-1]:
        if ch in "01234":
            return int(ch)
    return -1


def load_model():
    print("Loading LLaVA-OneVision model in 4-bit...")
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    return processor, model


def get_frame_paths(video_id, frames_base_dir, num_frames=4):
    vid_dir = os.path.join(frames_base_dir, str(video_id))
    if not os.path.exists(vid_dir):
        return []
    frame_files = sorted([f for f in os.listdir(vid_dir) if f.endswith('.jpg')])
    if len(frame_files) < 2:
        return []
    indices = [int(i * (len(frame_files) - 1) / (num_frames - 1)) for i in range(num_frames)]
    return [os.path.join(vid_dir, frame_files[idx]) for idx in indices]


def run_vlm_classification(processor, model, frame_paths, shuffled=False):
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    if shuffled:
        images = random.sample(images, len(images))

    n = len(images)
    prompt_text = VLM_PROMPT.format(n=n)
    image_tokens = "".join([f"<image>\n" for _ in range(n)])
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": image_tokens + prompt_text}]
        }
    ]
    text_input = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=text_input, images=images, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=150, do_sample=False)

    decoded = processor.decode(output[0], skip_special_tokens=True)
    response = decoded.split("assistant")[-1].strip() if "assistant" in decoded.lower() else decoded.strip()
    return response, parse_label(response)


def run_accuracy_experiment(processor, model, df, frames_base_dir, num_videos=25, num_frames=4, seed=42):
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

        ordered_response, ordered_pred = run_vlm_classification(processor, model, frame_paths, shuffled=False)
        shuffled_response, shuffled_pred = run_vlm_classification(processor, model, frame_paths, shuffled=True)

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
        print(f"  [{video_id}] GT={true_label} | Ordered={ordered_pred} ({'OK' if ordered_pred==true_label else 'X'}) | Shuffled={shuffled_pred} ({'OK' if shuffled_pred==true_label else 'X'})")

    ordered_acc = 100.0 * sum(r['ordered_correct'] for r in results) / len(results) if results else 0.0
    shuffled_acc = 100.0 * sum(r['shuffled_correct'] for r in results) / len(results) if results else 0.0

    print(f"\n========== VLM Classification Results ==========")
    print(f"  Videos evaluated: {len(results)}")
    print(f"  Ordered Frames  Accuracy: {ordered_acc:.2f}%")
    print(f"  Shuffled Frames Accuracy: {shuffled_acc:.2f}%")
    print(f"  Accuracy drop (ordered - shuffled): {ordered_acc - shuffled_acc:.2f}pp")
    print(f"=================================================\n")

    return results, ordered_acc, shuffled_acc


def create_vlm_pdf_report(records, ordered_acc, shuffled_acc, output_pdf_path):
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
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

    body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=12,
        textColor=colors.HexColor("#2D3748"))

    box_style = ParagraphStyle('BoxText', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=11.5,
        textColor=colors.HexColor("#1A202C"))

    story = []

    story.append(Paragraph("VLM Classification Experiment: LLaVA-OneVision (Ordered vs Shuffled)", title_style))
    story.append(Paragraph(
        f"Model: LLaVA-OneVision-Qwen2-0.5B &nbsp;|&nbsp; Frames per video: 4 &nbsp;|&nbsp; Videos evaluated: {len(records)}",
        subtitle_style))

    summary_data = [
        ["Condition", "Accuracy"],
        ["Ordered Frames", f"{ordered_acc:.2f}%"],
        ["Shuffled Frames", f"{shuffled_acc:.2f}%"],
        ["Drop (Ordered - Shuffled)", f"{ordered_acc - shuffled_acc:.2f}pp"],
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

    display_records = records[:3]
    for idx, rec in enumerate(display_records):
        story.append(Paragraph(f"Case Study {idx+1}: {rec['class_name']}", heading_style))
        story.append(Paragraph(
            f"<b>Video ID:</b> {rec['video_id']} &nbsp;|&nbsp; "
            f"<b>Ground Truth:</b> {rec['true_label']} ({rec['class_name']})",
            subtitle_style))

        img_cells = []
        label_cells = []
        for f_idx, img_path in enumerate(rec['frame_paths']):
            rl_img = RLImage(img_path, width=120, height=90)
            img_cells.append(rl_img)
            label_cells.append(Paragraph(f"<font size=8><b>Frame {f_idx+1}</b></font>", body_style))

        img_table = Table([img_cells, label_cells], colWidths=[130] * 4)
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 8))

        result_data = [
            ["Condition", "Predicted Label", "Correct?"],
            ["Ordered", str(rec['ordered_pred']), "Yes" if rec['ordered_correct'] else "No"],
            ["Shuffled", str(rec['shuffled_pred']), "Yes" if rec['shuffled_correct'] else "No"],
        ]
        result_table = Table(result_data, colWidths=[160, 160, 160])
        result_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F7FAFC"), colors.HexColor("#EDF2F7")]),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E0")),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(result_table)

        if idx < len(display_records) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"VLM Report PDF saved to: {output_pdf_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_csv', type=str, default='/kaggle/working/val_subset.csv')
    parser.add_argument('--frames_dir', type=str, default='/kaggle/working/frames')
    parser.add_argument('--num_videos', type=int, default=25)
    parser.add_argument('--num_frames', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/results')
    parser.add_argument('--results_json', type=str, default='/kaggle/working/results/vlm_accuracy_results.json')
    args, _ = parser.parse_known_args()

    val_csv = args.val_csv
    if not os.path.exists(val_csv):
        val_csv = "data/val_subset.csv"
    frames_dir = args.frames_dir
    if not os.path.exists(frames_dir):
        frames_dir = "frames"

    df = pd.read_csv(val_csv)

    processor, model = load_model()

    print(f"\nRunning classification on {args.num_videos} videos (ordered + shuffled)...")
    results, ordered_acc, shuffled_acc = run_accuracy_experiment(
        processor, model, df, frames_dir,
        num_videos=args.num_videos,
        num_frames=args.num_frames
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.results_json, "w") as f:
        serializable = [{k: v for k, v in r.items() if k != 'frame_paths'} for r in results]
        json.dump({"ordered_acc": ordered_acc, "shuffled_acc": shuffled_acc, "records": serializable}, f, indent=2)
    print(f"Results JSON saved to: {args.results_json}")

    pdf_path = os.path.join(args.output_dir, "vlm_experiment_report.pdf")
    create_vlm_pdf_report(results, ordered_acc, shuffled_acc, pdf_path)
