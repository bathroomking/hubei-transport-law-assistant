from pathlib import Path
from runpy import run_path
from zipfile import ZipFile

import pytest

convert = run_path(str(Path(__file__).parents[1] / "scripts" / "convert_12123_exam.py"))["convert"]


def write_document(path, paragraphs):
    body = "".join(f"<w:p><w:r>{runs}</w:r></w:p>" for runs in paragraphs)
    with ZipFile(path, "w") as docx:
        docx.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<w:body>{body}</w:body></w:document>",
        )
        docx.writestr(
            "word/_rels/document.xml.rels",
            '<Relationships><Relationship Id="rId1" Type="rel/image" Target="media/a.png"/>'
            "</Relationships>",
        )


def test_split_lines_duplicate_ids_and_unnumbered_picture(tmp_path):
    path = tmp_path / "exam.docx"
    write_document(
        path,
        [
            "<w:t>1.1.1.1 请选第一项。</w:t>",
            "<w:t>A.甲</w:t><w:br/><w:t>B.乙</w:t>",
            "<w:t>C丙</w:t><w:br/><w:t>D.丁</w:t><w:br/><w:t>答案：Ａ</w:t>",
            "<w:t>1.1.1.1 这是判断题。</w:t><w:br/><w:t>案：正确</w:t>",
            "<w:t>这是一个图示。</w:t>",
            '<a:blip r:embed="rId1"/><w:br/><w:t>答案：错误</w:t>',
        ],
    )
    choice, judgment, picture = convert(path)
    assert len({choice["id"], judgment["id"], picture["id"]}) == 3
    assert choice["source_number"] == judgment["source_number"] == "1.1.1.1"
    assert choice["options"] == dict(A="甲", B="乙", C="丙", D="丁")
    assert choice["question"] == "请选第一项。"
    assert choice["expected"] == judgment["expected"] == "A"
    assert picture["expected"] == "B"
    assert picture["source_number"] is None
    assert picture["skip_reason"] and picture["images"] == ["word/media/a.png"]


def test_missing_answer_stops_conversion(tmp_path):
    path = tmp_path / "incomplete.docx"
    write_document(path, ["<w:t>1.1.1.1 缺少答案。</w:t>", "<w:t>1.1.1.2 下一题。</w:t>"])
    with pytest.raises(ValueError, match="Missing answer"):
        convert(path)


def test_underlined_blank_crossing_paragraph_keeps_underlined_text(tmp_path):
    path = tmp_path / "blank.docx"
    underline = '<w:rPr><w:u w:val="single"/></w:rPr>'
    write_document(
        path,
        [
            f"<w:t>1.1.1.1 遇到</w:t></w:r><w:r>{underline}<w:t>   </w:t>",
            f"{underline}<w:t>   车流</w:t></w:r><w:r><w:t>时。</w:t>",
            "<w:t>A.甲</w:t><w:br/><w:t>B.乙</w:t><w:br/><w:t>C.丙</w:t><w:br/><w:t>D.丁</w:t>",
            "<w:t>答案：B</w:t>",
        ],
    )
    [row] = convert(path)
    assert row["question"] == "遇到____车流时。"
    assert row["options"] == dict(A="甲", B="乙", C="丙", D="丁")
