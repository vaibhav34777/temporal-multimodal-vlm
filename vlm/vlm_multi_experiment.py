import os
import torch
import pandas as pd
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_vlm_pdf_report(records, output_pdf_path="/kaggle/working/visualizations/vlm_experiment_report.pdf"):
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
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=16,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12
    )
    
    heading_style = ParagraphStyle(
        'SectionHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2D3748")
    )
    
    box_style = ParagraphStyle(
        'BoxText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1A202C")
    )
    
    story = []
    
    for idx, rec in enumerate(records):
        story.append(Paragraph(f"VLM Sequential Analysis: Case {idx+1} of {len(records)}", title_style))
        story.append(Paragraph(f"<b>Ground Truth Action:</b> {rec['class_name']} &nbsp;|&nbsp; <b>Video ID:</b> {rec['video_id']}", subtitle_style))
        story.append(Paragraph("<b>Input Frame Sequence (4 Evenly-Spaced Frames):</b>", heading_style))
        
        img_cells = []
        label_cells = []
        for f_idx, img_path in enumerate(rec['frame_paths']):
            rl_img = RLImage(img_path, width=125, height=95)
            img_cells.append(rl_img)
            label_cells.append(Paragraph(f"<font size=8><b>Frame {f_idx+1}</b></font>", body_style))
            
        img_table = Table([img_cells, label_cells], colWidths=[135]*4)
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("<b>Evaluation Prompt:</b>", heading_style))
        prompt_p = Paragraph(rec['prompt'].replace('\n', '<br/>'), box_style)
        prompt_table = Table([[prompt_p]], colWidths=[540])
        prompt_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(prompt_table)
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("<b>LLaVA 1.5 7B Output & Temporal Reasoning:</b>", heading_style))
        resp_p = Paragraph(rec['response'].replace('\n', '<br/>'), box_style)
        resp_table = Table([[resp_p]], colWidths=[540])
        resp_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(resp_table)
        
        if idx < len(records) - 1:
            story.append(PageBreak())
            
    doc.build(story)
    print(f"VLM 3-Page Report PDF generated at: {output_pdf_path}")

def run_multi_vlm_experiment(num_frames=4):
    print("Loading LLaVA Model in 4-bit...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    val_csv = "/kaggle/working/val_subset.csv"
    frames_base_dir = "/kaggle/working/frames"
    
    if not os.path.exists(val_csv):
        val_csv = "data/val_subset.csv"
    if not os.path.exists(frames_base_dir):
        frames_base_dir = "frames"
        
    df = pd.read_csv(val_csv)
    
    classes_to_test = [
        "Pushing [something] from left to right",
        "Moving [something] up",
        "Tearing [something] into two pieces"
    ]
    
    records = []
    
    for cls in classes_to_test:
        matching = df[df['class_name'] == cls]
        if len(matching) == 0:
            continue
        sample = matching.iloc[0]
        video_id = str(sample['video_id'])
        video_dir = os.path.join(frames_base_dir, video_id)
        
        print(f"\n==========================================")
        print(f"Target Class: {cls}")
        print(f"Video Directory: {video_dir}")
        print(f"==========================================\n")
        
        if not os.path.exists(video_dir):
            print(f"Directory {video_dir} not found. Skipping.")
            continue
            
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        selected_indices = [int(i * (len(frame_files)-1) / (num_frames-1)) for i in range(num_frames)]
        selected_frames = [frame_files[idx] for idx in selected_indices]
        full_frame_paths = [os.path.join(video_dir, f) for f in selected_frames]
        
        images = [Image.open(p).convert("RGB") for p in full_frame_paths]
        
        image_tags = "\n".join([f"Frame {i+1}: <image>" for i in range(num_frames)])
        prompt_text = (
            "You are analyzing 4 sequential frames from an action video.\n"
            "1. Describe what is changing across each sequential frame.\n"
            "2. What action and direction of movement is taking place?\n"
            "3. Is Frame 1 alone ambiguous to determine the full action? At which frame does the movement become clear and why?"
        )
        full_prompt = f"USER: {image_tags}\n{prompt_text}\nASSISTANT:"
        
        inputs = processor(text=full_prompt, images=images, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=300)
        decoded_output = processor.decode(output[0], skip_special_tokens=True)
        
        response_part = decoded_output
        if "ASSISTANT:" in decoded_output:
            response_part = decoded_output.split("ASSISTANT:")[-1].strip()
            
        print(response_part)
        print("\n")
        
        records.append({
            "class_name": cls,
            "video_id": video_id,
            "frame_paths": full_frame_paths,
            "prompt": prompt_text,
            "response": response_part
        })
        
    pdf_out = "/kaggle/working/visualizations/vlm_experiment_report.pdf"
    create_vlm_pdf_report(records, pdf_out)

if __name__ == "__main__":
    run_multi_vlm_experiment()
