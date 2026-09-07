from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any
import shutil

from .gemini_client import GeminiClient
from .utils import ensure_dir, read_json, write_json

VISUAL_JUDGE_PROMPT = """You are an expert AI visual judge for video generation models.
Your task is to analyze the provided storyboard image(s) or video keyframes of a generated video candidate against the original prompt and the specific prompt variant, and output a detailed, structured quality assessment.

Original Prompt: {original_prompt}
Candidate/Final Prompt: {prompt}

Please analyze the visual content very carefully and provide a structured JSON response with the following keys:
1. "human_reasonable_score": A float between 0.0 and 1.0 indicating how reasonable, realistic, and high-quality this generation is from a human perspective (where 1.0 is perfect and 0.0 is completely broken/meaningless).
2. "prompt_alignment": A detailed description of how well the visual content aligns with the prompt's instructions and intent.
3. "object_presence": A description of whether all requested objects/characters are present and recognizable.
4. "attribute_consistency": A description of whether the colors, shapes, styles, and other attributes of objects/characters are consistent with the prompt.
5. "motion_or_state_change": A description of whether the requested actions, motions, or physical state changes are visible and sensible.
6. "temporal_consistency": A description of whether the identity of characters/objects, backgrounds, and action continuity are stable across the frames.
7. "visual_quality": A description of the aesthetic and technical quality (sharpness, lighting, realistic textures, presence of artifacts/morphing).
8. "major_failures": A list of strings listing any major issues, bugs, physical glitches, severe morphing, or missing critical elements. If none, return an empty list [].
9. "evidence": The visual evidence from the frames (e.g. "in frame 2, the cat morphs into a dog") supporting your scores and judgments.
10. "recommendation": Concrete actionable feedback and suggestions on how to rewrite the prompt or adjust parameters to fix the issues.

Ensure your output is a single valid JSON object containing exactly these 10 keys. Do not include markdown code block syntax except standard json formatting.
"""

def extract_keyframes(video_path: str | Path, output_dir: Path, num_frames: int = 8) -> list[str]:
    """Extracts a uniform sample of frames from a video file using OpenCV."""
    try:
        import cv2
    except ImportError:
        print("Warning: opencv-python (cv2) is not installed. Frame extraction skipped.")
        return []

    video_path = Path(video_path)
    if not video_path.exists():
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    # ensure indices are unique and sorted
    indices = sorted(list(set(indices)))

    output_paths = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = output_dir / f"frame_{idx:02d}.jpg"
        cv2.imwrite(str(out_path), frame)
        output_paths.append(str(out_path))

    cap.release()
    return output_paths

def generate_markdown_report(data: dict[str, Any]) -> str:
    """Generates a beautifully formatted Markdown report from visual judge results."""
    md = []
    md.append("# SparseMAC Visual Judge Report\n")
    md.append(f"- **Status**: `{data.get('status')}`")
    md.append(f"- **Run ID**: `{data.get('run_id')}`")
    md.append(f"- **Original Prompt**: {data.get('original_prompt')}\n")

    if data.get("status") == "unavailable":
        md.append(f"> [!WARNING]\n> Visual judge is unavailable: {data.get('reason')}\n")
        return "\n".join(md)

    def format_judgment(name: str, j_data: dict[str, Any]):
        sub_md = []
        sub_md.append(f"### {name}")
        prompt = j_data.get("prompt", "N/A")
        seed = j_data.get("seed", "N/A")
        sub_md.append(f"- **Prompt**: {prompt}")
        sub_md.append(f"- **Seed**: `{seed}`")

        j = j_data.get("judgment")
        if not j:
            sub_md.append("- **Judgment**: N/A (Missing or failed response)\n")
            return "\n".join(sub_md)

        score = j.get("human_reasonable_score", 0.0)
        sub_md.append(f"- **Human Reasonable Score**: `{score:.2f}/1.00`")
        sub_md.append(f"- **Prompt Alignment**: {j.get('prompt_alignment')}")
        sub_md.append(f"- **Object Presence**: {j.get('object_presence')}")
        sub_md.append(f"- **Attribute Consistency**: {j.get('attribute_consistency')}")
        sub_md.append(f"- **Motion/State Change**: {j.get('motion_or_state_change')}")
        sub_md.append(f"- **Temporal Consistency**: {j.get('temporal_consistency')}")
        sub_md.append(f"- **Visual Quality**: {j.get('visual_quality')}")

        failures = j.get("major_failures", [])
        if failures:
            sub_md.append("- **Major Failures**:")
            for f in failures:
                sub_md.append(f"  - ❌ {f}")
        else:
            sub_md.append("- **Major Failures**: None reported ✅")

        sub_md.append(f"- **Evidence**: *{j.get('evidence')}*")
        sub_md.append(f"- **Recommendation**: {j.get('recommendation')}\n")
        return "\n".join(sub_md)

    # Final Video Section
    final_video = data.get("final_video")
    if final_video:
        md.append("## Final Full Video Judgment\n")
        md.append(format_judgment("Final Wan Video", final_video))
    else:
        md.append("## Final Full Video Judgment\n")
        md.append("*No final full video was generated or found.*\n")

    # Candidates Section
    candidates = data.get("candidates", {})
    if candidates:
        md.append("## Candidates Judgment\n")
        for cid, cand_data in sorted(candidates.items()):
            md.append(format_judgment(f"Candidate {cid}", cand_data))
    else:
        md.append("## Candidates Judgment\n")
        md.append("*No candidates found or evaluated.*\n")

    return "\n".join(md)

def visual_judge_run(run_dir: str | Path, model_name: str = "gemini-2.5-pro") -> int:
    run_dir = Path(run_dir).resolve()
    judge_json_path = run_dir / "visual_judge.json"
    judge_md_path = run_dir / "visual_judge.md"

    # 1. Read plan to get the original prompt
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        print(f"Error: plan.json not found in {run_dir}")
        return 1
    plan_data = read_json(plan_path)
    original_prompt = plan_data.get("original_prompt", "")

    # 2. Check API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set. Visual judge will fail gracefully and write status to json.")
        status_data = {
            "status": "unavailable",
            "reason": "GEMINI_API_KEY is not set",
            "run_id": run_dir.name,
            "original_prompt": original_prompt
        }
        write_json(judge_json_path, status_data)
        with open(judge_md_path, "w", encoding="utf-8") as f:
            f.write(generate_markdown_report(status_data))
        return 0

    client = GeminiClient(model=model_name, api_key=api_key)
    if not client.available:
        print("Gemini client initialization failed. Visual judge will fail gracefully.")
        status_data = {
            "status": "unavailable",
            "reason": "Gemini client could not be loaded",
            "run_id": run_dir.name,
            "original_prompt": original_prompt
        }
        write_json(judge_json_path, status_data)
        with open(judge_md_path, "w", encoding="utf-8") as f:
            f.write(generate_markdown_report(status_data))
        return 0

    result: dict[str, Any] = {
        "status": "success",
        "run_id": run_dir.name,
        "original_prompt": original_prompt,
        "candidates": {},
        "final_video": None
    }

    # 3. Process all candidates from round_* directories
    round_dirs = sorted(list(run_dir.glob("round_*")))
    for round_dir in round_dirs:
        cand_json_path = round_dir / "candidates.json"
        if not cand_json_path.exists():
            continue
        cands = read_json(cand_json_path) or []
        for cand in cands:
            cid = cand.get("candidate_id")
            storyboard_path = cand.get("storyboard")
            if not cid or not storyboard_path:
                continue
            storyboard_file = Path(storyboard_path)
            if not storyboard_file.exists():
                # Try relative to run_dir/round_dir
                storyboard_file = round_dir / "candidates" / cid / "storyboard_11.jpg"
                if not storyboard_file.exists():
                    continue

            print(f"Evaluating candidate {cid} using storyboard: {storyboard_file}")
            prompt_content = VISUAL_JUDGE_PROMPT.format(
                original_prompt=original_prompt,
                prompt=cand.get("prompt", original_prompt)
            )

            judgment, raw_text = client.generate_json(
                prompt=prompt_content,
                image_paths=[str(storyboard_file)]
            )

            result["candidates"][cid] = {
                "prompt": cand.get("prompt"),
                "seed": cand.get("seed"),
                "judgment": judgment,
                "raw_response": raw_text
            }

    # 4. Process final video if it exists
    final_video_dir = run_dir / "final"
    final_video_file = final_video_dir / "full_wan.mp4"
    final_json_file = final_video_dir / "final.json"

    if final_video_file.exists():
        print(f"Evaluating final full video: {final_video_file}")
        final_info = read_json(final_json_file) or {}
        final_prompt = final_info.get("prompt", original_prompt)
        final_seed = final_info.get("seed")

        # Extract keyframes to visual_judge_temp inside run_dir (satisfying Strict Workspace Limit)
        temp_dir = run_dir / "visual_judge_temp"
        image_paths = []
        try:
            image_paths = extract_keyframes(final_video_file, temp_dir, num_frames=8)
            if image_paths:
                print(f"Extracted {len(image_paths)} frames for final video evaluation.")
                prompt_content = VISUAL_JUDGE_PROMPT.format(
                    original_prompt=original_prompt,
                    prompt=final_prompt
                )
                judgment, raw_text = client.generate_json(
                    prompt=prompt_content,
                    image_paths=image_paths
                )
                result["final_video"] = {
                    "prompt": final_prompt,
                    "seed": final_seed,
                    "judgment": judgment,
                    "raw_response": raw_text
                }
        finally:
            # Clean up extracted temp frames (satisfying workspace hygiene)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    # 5. Write outputs
    write_json(judge_json_path, result)
    with open(judge_md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(result))

    print(f"Visual Judge complete. Results written to:")
    print(f"  JSON: {judge_json_path}")
    print(f"  MD:   {judge_md_path}")
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description="SparseMAC Visual Judge")
    parser.add_argument("--run-dir", required=True, help="Path to the run directory to judge")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name to use")
    args = parser.parse_args()
    return visual_judge_run(args.run_dir, model_name=args.model)

if __name__ == "__main__":
    import sys
    sys.exit(main())
