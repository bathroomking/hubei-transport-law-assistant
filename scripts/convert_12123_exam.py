"""Convert the numbered 12123 DOCX bank into JSONL for the text QA evaluator."""

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET
from zipfile import ZipFile

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
BLIP = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"

NUMBER = re.compile(r"^(\d+[.．]\s*\d+[.．]\s*\d+[.．]\s*\d+)\s*(.*)$")
SECTION = re.compile(r"^\d+([.．]\s*\d+){0,2}[.．]?\s*[^\d\s.．]")
ANSWER = re.compile(r"(?:答\s*案|答|案)\s*[:：]?\s*(正确|错误|[ABCD])$")
OPTION = re.compile(r"^([A-D])(?:\s*[.．、]\s*|(?=[\u3400-\u9fff]))(.*)$")
FULL_WIDTH_CHOICES = str.maketrans("ＡＢＣＤ", "ABCD")
IMAGE_HINT = re.compile(r"图中|如图|图示|下图|上图|图所示|图内")


def document_lines(docx):
    """Yield 1-based body paragraph positions, text lines and their source image parts."""
    with ZipFile(docx) as archive:
        rels = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        images = {
            item.get("Id"): "word/" + item.attrib["Target"]
            for item in rels
            if item.attrib["Type"].endswith("/image")
        }
        root = ET.fromstring(archive.read("word/document.xml"))
    for paragraph, node in enumerate(root.find(W + "body").iter(W + "p"), 1):
        text, media = [], []
        # Word renders underlined spaces as a fill-in blank, including at a line break.
        for run in node.iter(W + "r"):
            underline = run.find(W + "rPr/" + W + "u")
            if underline is not None and underline.get(W + "val") != "none":
                for word in run.iter(W + "t"):
                    word.text = re.sub(r"\s+", "____", word.text or "")
        for element in node.iter():
            if element.tag == W + "t":
                text.append(element.text or "")
            elif element.tag == W + "tab":
                text.append(" ")
            elif element.tag == BLIP:
                media.append(images[element.attrib[R + "embed"]])
                text.append("〔见原图〕")
            elif element.tag in {W + "br", W + "cr"}:
                yield paragraph, "".join(text).strip(), media
                text, media = [], []
        yield paragraph, "".join(text).strip(), media


def new_item(index, paragraph, source_number, stem):
    return {
        "id": f"12123-{index:04d}",
        "source_number": source_number,
        "paragraph_start": paragraph,
        "question": stem,
        "options": {},
        "images": [],
    }


def convert(docx):
    rows = []
    row = None
    for paragraph, line, images in document_lines(docx):
        line = line.translate(FULL_WIDTH_CHOICES)
        if not line:
            continue
        if match := NUMBER.fullmatch(line):
            if row is not None:
                raise ValueError(f"Missing answer before paragraph {paragraph}")
            number = re.sub(r"\s+", "", match[1]).replace("．", ".")
            row = new_item(len(rows) + 1, paragraph, number, match[2])
            line = ""
        elif row is None:
            if not rows or SECTION.match(line):
                continue
            # There is an unnumbered question between the final marked-lane item and section 3.
            row = new_item(len(rows) + 1, paragraph, None, line)
            line = ""
        row["images"].extend(images)
        answer = ANSWER.search(line)
        content = line[: answer.start()].strip() if answer else line
        if opt := OPTION.fullmatch(content):
            if opt[1] in row["options"]:
                raise ValueError(f"Duplicate choice {opt[1]} at paragraph {paragraph}")
            row["options"][opt[1]] = opt[2]
        elif content and content.strip(" _") != "〔见原图〕":
            if row["options"]:
                row["options"][next(reversed(row["options"]))] += content
            else:
                row["question"] += content
        if answer:
            finish_item(row, answer[1], paragraph)
            rows.append(row)
            row = None
    if row is not None or not rows:
        raise ValueError("Document ended without a complete question/answer set")
    return rows


def finish_item(row, answer, paragraph):
    # A single blank can span two runs or paragraphs.
    row["question"] = re.sub(r"_{4,}", "____", row["question"])
    options = row["options"]
    if answer in {"正确", "错误"} and not options:
        options.update(A="正确", B="错误")
        expected = "A" if answer == "正确" else "B"
        row["type"] = "true_false"
    elif list(options) == list("ABCD") and answer in options:
        expected = answer
        row["type"] = "single_choice"
    else:
        raise ValueError(f"Invalid choices/answer at paragraph {paragraph}: {options} / {answer}")

    if not row["question"] or not all(options.values()):
        raise ValueError(f"Empty stem or choice at paragraph {paragraph}")
    row.update(expected=expected, answer_text=options[expected], paragraph_end=paragraph)
    if row["images"] or IMAGE_HINT.search(row["question"] + "".join(options.values())):
        row["skip_reason"] = "题目依赖图示，当前问答入口仅接收文本"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", type=Path)
    parser.add_argument(
        "output", type=Path, help="JSONL file; images are saved in media/ beside it"
    )
    args = parser.parse_args()
    rows = convert(args.docx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.docx) as archive:
        for row in rows:
            media = []
            for part in row["images"]:
                path = Path("media", PurePosixPath(part).name)
                destination = args.output.parent / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(part))
                media.append(path.as_posix())
            row["images"] = media
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"{len(rows)} questions, {sum('skip_reason' in row for row in rows)} image-dependent")


if __name__ == "__main__":
    main()
