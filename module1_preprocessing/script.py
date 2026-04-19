import re

def vtt_to_text(filepath, output_path):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove VTT headers, timestamps, and tags
    content = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> .*', '', content)
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'^WEBVTT.*$', '', content, flags=re.MULTILINE)

    lines = [l.strip() for l in content.splitlines() if l.strip()]

    # Remove duplicate consecutive lines (VTT repeats lines)
    deduped = [lines[0]] if lines else []
    for line in lines[1:]:
        if line != deduped[-1]:
            deduped.append(line)

    text = " ".join(deduped)

    # ✅ Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text


# usage
text = vtt_to_text("transcript.vtt", "output.txt")